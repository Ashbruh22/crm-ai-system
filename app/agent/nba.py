"""Next-best-action rules (spec section 8).

The paper's hierarchical decision tree, written out as explicit rules. Each rule
has an id, a condition over the features / SHAP values / score, an action, a
priority, and a reason that **cites the drivers** rather than restating the
probability.

Why this replaced the previous implementation
---------------------------------------------
The old ``recommendation_service`` branched on ``win_prob`` alone into four
mutually exclusive bands and emitted one canned sentence each. A deal at 0.35
with a named champion and a meeting last week got exactly the same advice as a
silent one with no champion, because the only input was the number. Rules here
read the features that produced the score, so the advice differs when the
situation differs.

Design
------
* **Deterministic.** No randomness, no clock reads outside ``ctx``. The same
  context always yields the same recommendations, which is what makes the rules
  unit-testable and the audit trail meaningful.
* **Multiple rules fire.** Severity tiers are ordering, not exclusivity — a deal
  can be both single-threaded and going quiet, and a rep should hear both.
* **Reasons cite numbers.** "Win probability fell 18 points after 9 days of
  silence" is actionable; "this deal is at risk" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from app.features.build import FEATURE_LABELS, STAGES

# Priority ordering, most urgent first. Used for sorting and for the
# dashboard's top-action badge.
PRIORITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}

#: A drop of at least this many probability points since the previous score
#: counts as a material regression worth flagging.
SCORE_DROP_THRESHOLD = 0.10

#: Deals larger than this are expected to be multi-threaded.
SINGLE_THREAD_SIZE_K = 75.0

_LATE_STAGE = STAGES.index("Proposal")
_MID_STAGE = STAGES.index("Discovery")


@dataclass(frozen=True)
class RuleContext:
    """Everything the rules may read. Nothing else is in scope.

    Keeping this explicit (rather than passing the ORM objects around) is what
    lets a test construct a situation in three lines.
    """

    deal_id: str
    features: Mapping[str, float]
    win_prob: float
    days_to_close: float | None = None
    #: feature name -> SHAP contribution in log-odds.
    shap: Mapping[str, float] = field(default_factory=dict)
    #: The deal's previous win probability, if it has been scored before.
    previous_win_prob: float | None = None
    company: str = ""
    owner_rep: str = ""

    # --- convenience accessors ---------------------------------------------

    def f(self, name: str) -> float:
        return float(self.features.get(name, 0.0))

    @property
    def stage_index(self) -> int:
        return int(self.f("stage_ordinal"))

    @property
    def stage_name(self) -> str:
        i = self.stage_index
        return STAGES[i] if 0 <= i < len(STAGES) else "Unknown"

    @property
    def score_delta(self) -> float | None:
        """Change in win probability since the last score, in absolute points."""
        if self.previous_win_prob is None:
            return None
        return self.win_prob - self.previous_win_prob

    def top_negative_driver(self) -> tuple[str, float] | None:
        """The feature pushing this score down hardest."""
        negatives = [(k, v) for k, v in self.shap.items() if v < 0]
        if not negatives:
            return None
        return min(negatives, key=lambda kv: kv[1])

    def top_positive_driver(self) -> tuple[str, float] | None:
        positives = [(k, v) for k, v in self.shap.items() if v > 0]
        if not positives:
            return None
        return max(positives, key=lambda kv: kv[1])


@dataclass(frozen=True)
class Recommendation:
    rule_id: str
    action_type: str
    reason: str
    priority: str
    urgency_score: float

    def __repr__(self) -> str:
        return f"<Recommendation {self.rule_id} {self.priority}>"


@dataclass(frozen=True)
class Rule:
    id: str
    action_type: str
    priority: str
    #: One line on what the rule is for; surfaced in docs and tests.
    description: str
    condition: Callable[[RuleContext], bool]
    reason: Callable[[RuleContext], str]


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _pts(delta: float) -> str:
    """Format a probability delta in points, e.g. -0.18 -> '18 points'."""
    return f"{abs(delta) * 100:.0f} points"


def _driver_clause(ctx: RuleContext) -> str:
    """'largest drag is Days since last touch (33 days)' or ''."""
    driver = ctx.top_negative_driver()
    if driver is None:
        return ""
    name, _ = driver
    value = ctx.f(name)
    return f"{_label(name).lower()} ({value:g})"


# ---------------------------------------------------------------------------
# Rules, ordered by tier. Several may fire for one deal.
# ---------------------------------------------------------------------------

RULES: tuple[Rule, ...] = (
    # --- CRITICAL ----------------------------------------------------------
    Rule(
        id="R-SCORE-DROP",
        action_type="investigate_regression",
        priority="CRITICAL",
        description="Win probability fell materially since the last scoring.",
        condition=lambda c: (
            c.score_delta is not None and c.score_delta <= -SCORE_DROP_THRESHOLD
        ),
        reason=lambda c: (
            f"Win probability fell {_pts(c.score_delta)} "
            f"(from {c.previous_win_prob:.0%} to {c.win_prob:.0%}) since the last "
            f"scoring; the largest drag is {_driver_clause(c) or 'recent activity'}. "
            "Review what changed on this deal before the next touch."
        ),
    ),
    Rule(
        id="R-GONE-QUIET",
        action_type="schedule_check_in",
        priority="CRITICAL",
        description="Long silence on a deal that is not already won.",
        condition=lambda c: (
            c.f("days_since_last_activity") >= 21 and c.win_prob < 0.60
        ),
        reason=lambda c: (
            f"No contact for {c.f('days_since_last_activity'):.0f} days with win "
            f"probability at {c.win_prob:.0%}"
            + (
                f" and a champion already identified — reach out to them directly."
                if c.f("has_champion")
                else " and no champion identified — re-qualify before spending more time."
            )
        ),
    ),
    Rule(
        id="R-DISCOUNT-LATE",
        action_type="involve_deal_desk",
        priority="CRITICAL",
        description="Discount requested late in the cycle on a shaky deal.",
        condition=lambda c: (
            c.f("discount_requested_late") >= 1.0 and c.win_prob < 0.65
        ),
        reason=lambda c: (
            f"A discount was requested at {c.stage_name} stage while win "
            f"probability is only {c.win_prob:.0%}. Late discounting on an "
            "uncommitted deal usually signals a stalled decision, not a pricing "
            "objection — confirm budget authority before conceding margin."
        ),
    ),
    # --- HIGH --------------------------------------------------------------
    Rule(
        id="R-NO-CHAMPION-LATE",
        action_type="identify_champion",
        priority="HIGH",
        description="Past Proposal with nobody advocating internally.",
        condition=lambda c: (
            not c.f("has_champion") and c.stage_index >= _LATE_STAGE
        ),
        reason=lambda c: (
            f"This deal reached {c.stage_name} with no champion identified across "
            f"{c.f('n_stakeholders'):.0f} known contact(s). Champion presence is the "
            "single strongest driver in the model — name one before pushing to close."
        ),
    ),
    Rule(
        id="R-SINGLE-THREADED",
        action_type="multithread_stakeholders",
        priority="HIGH",
        description="Large deal resting on a single contact.",
        condition=lambda c: (
            c.f("n_stakeholders") <= 1 and c.f("deal_size_k") >= SINGLE_THREAD_SIZE_K
        ),
        reason=lambda c: (
            f"A ${c.f('deal_size_k'):,.0f}k deal is running through one contact. "
            "Add a second stakeholder — ideally in finance or the end-user team — "
            "so the deal survives that person changing role or going quiet."
        ),
    ),
    Rule(
        id="R-STALLED",
        action_type="schedule_check_in",
        priority="HIGH",
        description="Two weeks of silence, before it becomes critical.",
        condition=lambda c: (
            14 <= c.f("days_since_last_activity") < 21
        ),
        reason=lambda c: (
            f"{c.f('days_since_last_activity'):.0f} days since the last touch, with "
            f"{c.f('n_silence_gaps'):.0f} silent week(s) logged on this deal. "
            "Book a short check-in now, while the gap is still recoverable."
        ),
    ),
    Rule(
        id="R-NO-DEMO",
        action_type="schedule_demo",
        priority="HIGH",
        description="Mid-to-late stage without ever demonstrating the product.",
        condition=lambda c: (
            c.f("n_demos") == 0 and c.stage_index >= _MID_STAGE
        ),
        reason=lambda c: (
            f"At {c.stage_name} stage no demo has been delivered, despite "
            f"{c.f('n_meetings'):.0f} meeting(s). Deals that reach this stage "
            "without a demo close materially less often — schedule one."
        ),
    ),
    # --- MEDIUM ------------------------------------------------------------
    Rule(
        id="R-LOW-REPLY-RATE",
        action_type="change_outreach_channel",
        priority="MEDIUM",
        description="Plenty of outbound email, very little coming back.",
        condition=lambda c: (
            c.f("n_emails_sent") >= 4 and c.f("reply_rate") < 0.25
        ),
        reason=lambda c: (
            f"{c.f('n_emails_replied'):.0f} replies to {c.f('n_emails_sent'):.0f} "
            f"emails ({c.f('reply_rate'):.0%} reply rate). Email is not landing — "
            "switch channel or route through an existing relationship."
        ),
    ),
    Rule(
        id="R-NEGOTIATION-NO-PROPOSAL",
        action_type="send_proposal",
        priority="MEDIUM",
        description="Negotiating without a proposal on the table.",
        condition=lambda c: (
            c.stage_name == "Negotiation" and c.f("n_proposals") == 0
        ),
        reason=lambda c: (
            "The deal is marked Negotiation but no proposal has been sent. "
            "Put written terms in front of the buyer so the negotiation has a "
            "concrete anchor."
        ),
    ),
    Rule(
        id="R-SLOW-CYCLE",
        action_type="review_close_plan",
        priority="MEDIUM",
        description="Forecast close is far out for a deal this far along.",
        condition=lambda c: (
            c.days_to_close is not None
            and c.days_to_close >= 60
            and c.stage_index >= _LATE_STAGE
        ),
        reason=lambda c: (
            f"The model forecasts {c.days_to_close:.0f} more days to close from "
            f"{c.stage_name} stage, driven partly by "
            f"{c.f('n_stakeholders'):.0f} stakeholder(s) and a "
            f"${c.f('deal_size_k'):,.0f}k value. Agree a mutual close plan with "
            "dates, or the forecast quarter is optimistic."
        ),
    ),
    # --- LOW (healthy deals still get a next step) --------------------------
    Rule(
        id="R-READY-TO-CLOSE",
        action_type="send_closing_checklist",
        priority="LOW",
        description="Strong, well-supported deal — move to close.",
        condition=lambda c: (
            c.win_prob >= 0.75
            and c.f("has_champion") >= 1.0
            and c.f("days_since_last_activity") < 14
        ),
        reason=lambda c: (
            f"Win probability {c.win_prob:.0%} with a champion engaged and contact "
            f"{c.f('days_since_last_activity'):.0f} days ago"
            + (
                f"; strongest driver is {_label(c.top_positive_driver()[0]).lower()}. "
                if c.top_positive_driver()
                else ". "
            )
            + "Send the closing checklist and propose a contract review date."
        ),
    ),
    Rule(
        id="R-BUILD-ON-MOMENTUM",
        action_type="advance_stage",
        priority="LOW",
        description="Engaged deal that has not yet been asked to progress.",
        condition=lambda c: (
            0.55 <= c.win_prob < 0.75
            and c.f("days_since_last_activity") < 10
            and c.f("n_meetings") >= 1
        ),
        reason=lambda c: (
            f"Healthy engagement — {c.f('n_meetings'):.0f} meeting(s), last contact "
            f"{c.f('days_since_last_activity'):.0f} days ago, win probability "
            f"{c.win_prob:.0%}. Ask for the next commitment rather than waiting for "
            "the buyer to set the pace."
        ),
    ),
)

#: Rule ids must be unique — they are the audit key in the actions table.
assert len({r.id for r in RULES}) == len(RULES), "duplicate rule id"


def urgency(ctx: RuleContext, priority: str) -> float:
    """Value-at-risk style ranking score.

    Deliberately interpretable: money at risk, scaled by how much the model
    doubts the deal, weighted by tier, and damped by how long there is to act.
    A big shaky deal closing soon outranks a small shaky deal closing later.
    """
    value_at_risk = ctx.f("deal_size_k") * (1.0 - ctx.win_prob)
    tier_weight = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.15}[priority]
    horizon = max(float(ctx.days_to_close or 30.0), 7.0)
    return round(value_at_risk * tier_weight * (30.0 / horizon), 3)


def recommend(ctx: RuleContext, limit: int | None = None) -> list[Recommendation]:
    """Evaluate every rule against ``ctx``.

    Returns the ones that fired, most urgent first (priority tier, then urgency
    score). A rule raising inside its own condition or reason is skipped rather
    than failing the whole scoring request — one broken rule should not cost the
    deal its score.
    """
    fired: list[Recommendation] = []

    for rule in RULES:
        try:
            if not rule.condition(ctx):
                continue
            reason = rule.reason(ctx)
        except Exception:  # noqa: BLE001 - never fail scoring over one rule
            continue

        fired.append(
            Recommendation(
                rule_id=rule.id,
                action_type=rule.action_type,
                reason=reason,
                priority=rule.priority,
                urgency_score=urgency(ctx, rule.priority),
            )
        )

    fired.sort(key=lambda r: (PRIORITY_RANK[r.priority], -r.urgency_score))
    return fired[:limit] if limit else fired


def context_from_score(
    deal,
    features: Mapping[str, float],
    win_prob: float,
    days_to_close: float | None,
    shap_top: Sequence[Mapping],
    previous_win_prob: float | None = None,
) -> RuleContext:
    """Build a RuleContext from a scoring result."""
    return RuleContext(
        deal_id=deal.id,
        features=features,
        win_prob=win_prob,
        days_to_close=days_to_close,
        shap={d["feature"]: d["shap"] for d in shap_top},
        previous_win_prob=previous_win_prob,
        company=getattr(deal, "company", ""),
        owner_rep=getattr(deal, "owner_rep", ""),
    )
