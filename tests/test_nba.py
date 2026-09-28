"""NBA rule engine unit tests (spec sections 8, 11).

Fixture deals, one situation per rule. These are pure-function tests: no
database, no models, no clock — which is the point of RuleContext being a plain
dataclass.
"""

from __future__ import annotations

import pytest

from app.agent.nba import (
    PRIORITIES,
    RULES,
    Recommendation,
    RuleContext,
    recommend,
    urgency,
)
from app.features.build import FEATURE_NAMES, STAGES

RULES_BY_ID = {r.id: r for r in RULES}


def ctx(**overrides) -> RuleContext:
    """A healthy mid-pipeline deal; override to create each situation."""
    features = {name: 0.0 for name in FEATURE_NAMES}
    features.update(
        {
            "deal_size_k": 120.0,
            "days_open": 40.0,
            "days_since_last_activity": 3.0,
            "n_emails_sent": 6.0,
            "n_emails_replied": 3.0,
            "reply_rate": 0.5,
            "n_meetings": 2.0,
            "n_demos": 1.0,
            "n_proposals": 1.0,
            "n_stakeholders": 3.0,
            "has_champion": 1.0,
            "n_discount_requests": 0.0,
            "discount_requested_late": 0.0,
            "n_silence_gaps": 0.0,
            "activity_count": 12.0,
            "stage_ordinal": float(STAGES.index("Proposal")),
            "rep_win_rate": 0.45,
        }
    )
    features.update(overrides.pop("features", {}))

    defaults = {
        "deal_id": "LIVE-0001",
        "features": features,
        "win_prob": 0.62,
        "days_to_close": 25.0,
        "shap": {"has_champion": 0.5, "days_since_last_activity": -0.2},
        "previous_win_prob": None,
        "company": "Example Holdings",
        "owner_rep": "Dana Reed",
    }
    defaults.update(overrides)
    return RuleContext(**defaults)


def fired_ids(context: RuleContext) -> set[str]:
    return {r.rule_id for r in recommend(context)}


# --- structural ------------------------------------------------------------


def test_rule_ids_are_unique():
    assert len({r.id for r in RULES}) == len(RULES)


def test_every_rule_declares_a_valid_priority_and_description():
    for rule in RULES:
        assert rule.priority in PRIORITIES, rule.id
        assert rule.description.strip(), rule.id
        assert rule.action_type.strip(), rule.id


def test_engine_is_deterministic():
    context = ctx(win_prob=0.30, features={"days_since_last_activity": 30.0})
    assert [r.rule_id for r in recommend(context)] == [
        r.rule_id for r in recommend(context)
    ]


def test_results_are_ordered_by_priority_then_urgency():
    from app.agent.nba import PRIORITY_RANK

    results = recommend(
        ctx(
            win_prob=0.25,
            features={
                "days_since_last_activity": 30.0,
                "has_champion": 0.0,
                "n_stakeholders": 1.0,
                "deal_size_k": 400.0,
                "n_demos": 0.0,
            },
        )
    )
    assert len(results) >= 3
    ranks = [(PRIORITY_RANK[r.priority], -r.urgency_score) for r in results]
    assert ranks == sorted(ranks)


def test_a_broken_rule_does_not_fail_the_whole_engine(monkeypatch):
    """One bad rule must not cost the deal its recommendations."""
    import app.agent.nba as nba

    def explode(_c):
        raise RuntimeError("boom")

    broken = nba.Rule(
        id="R-BROKEN",
        action_type="x",
        priority="CRITICAL",
        description="always raises",
        condition=explode,
        reason=lambda c: "never reached",
    )
    monkeypatch.setattr(nba, "RULES", (broken,) + nba.RULES)

    results = nba.recommend(ctx(win_prob=0.20, features={"days_since_last_activity": 40.0}))
    assert "R-BROKEN" not in {r.rule_id for r in results}
    assert results, "other rules should still fire"


# --- individual rules ------------------------------------------------------


def test_score_drop_fires_and_quotes_the_delta():
    results = recommend(ctx(win_prob=0.42, previous_win_prob=0.60))
    drop = next(r for r in results if r.rule_id == "R-SCORE-DROP")

    assert drop.priority == "CRITICAL"
    assert "18 points" in drop.reason
    assert "60%" in drop.reason and "42%" in drop.reason


def test_score_drop_ignores_small_moves_and_improvements():
    assert "R-SCORE-DROP" not in fired_ids(ctx(win_prob=0.58, previous_win_prob=0.62))
    assert "R-SCORE-DROP" not in fired_ids(ctx(win_prob=0.80, previous_win_prob=0.60))
    # No history at all -> nothing to compare.
    assert "R-SCORE-DROP" not in fired_ids(ctx(win_prob=0.30, previous_win_prob=None))


def test_gone_quiet_names_the_champion_route_when_there_is_one():
    with_champion = recommend(
        ctx(win_prob=0.40, features={"days_since_last_activity": 28.0, "has_champion": 1.0})
    )
    reason = next(r for r in with_champion if r.rule_id == "R-GONE-QUIET").reason
    assert "28 days" in reason
    assert "champion" in reason.lower()

    without = recommend(
        ctx(win_prob=0.40, features={"days_since_last_activity": 28.0, "has_champion": 0.0})
    )
    reason = next(r for r in without if r.rule_id == "R-GONE-QUIET").reason
    assert "re-qualify" in reason.lower()


def test_gone_quiet_does_not_fire_on_a_strong_deal():
    assert "R-GONE-QUIET" not in fired_ids(
        ctx(win_prob=0.85, features={"days_since_last_activity": 30.0})
    )


def test_late_discount_fires_only_when_the_deal_is_shaky():
    assert "R-DISCOUNT-LATE" in fired_ids(
        ctx(win_prob=0.45, features={"discount_requested_late": 1.0})
    )
    # A discount on a strong deal is a pricing conversation, not a red flag.
    assert "R-DISCOUNT-LATE" not in fired_ids(
        ctx(win_prob=0.80, features={"discount_requested_late": 1.0})
    )


def test_no_champion_late_stage():
    results = recommend(
        ctx(
            features={
                "has_champion": 0.0,
                "stage_ordinal": float(STAGES.index("Negotiation")),
            }
        )
    )
    rec = next(r for r in results if r.rule_id == "R-NO-CHAMPION-LATE")
    assert "Negotiation" in rec.reason
    assert rec.priority == "HIGH"

    # Early stage without a champion is normal.
    assert "R-NO-CHAMPION-LATE" not in fired_ids(
        ctx(features={"has_champion": 0.0, "stage_ordinal": 0.0})
    )


def test_single_threaded_only_matters_on_large_deals():
    assert "R-SINGLE-THREADED" in fired_ids(
        ctx(features={"n_stakeholders": 1.0, "deal_size_k": 250.0})
    )
    assert "R-SINGLE-THREADED" not in fired_ids(
        ctx(features={"n_stakeholders": 1.0, "deal_size_k": 20.0})
    )


def test_stalled_and_gone_quiet_do_not_both_fire():
    """The silence rules partition the range; a deal gets one, not two."""
    at_16 = fired_ids(ctx(win_prob=0.40, features={"days_since_last_activity": 16.0}))
    assert "R-STALLED" in at_16 and "R-GONE-QUIET" not in at_16

    at_25 = fired_ids(ctx(win_prob=0.40, features={"days_since_last_activity": 25.0}))
    assert "R-GONE-QUIET" in at_25 and "R-STALLED" not in at_25


def test_no_demo_fires_from_discovery_onward():
    assert "R-NO-DEMO" in fired_ids(
        ctx(features={"n_demos": 0.0, "stage_ordinal": float(STAGES.index("Discovery"))})
    )
    assert "R-NO-DEMO" not in fired_ids(
        ctx(features={"n_demos": 0.0, "stage_ordinal": 0.0})
    )
    assert "R-NO-DEMO" not in fired_ids(ctx(features={"n_demos": 2.0}))


def test_low_reply_rate_quotes_the_counts():
    results = recommend(
        ctx(features={"n_emails_sent": 10.0, "n_emails_replied": 1.0, "reply_rate": 0.1})
    )
    rec = next(r for r in results if r.rule_id == "R-LOW-REPLY-RATE")
    assert "1 replies to 10 emails" in rec.reason
    assert "10%" in rec.reason


def test_low_reply_rate_needs_enough_emails_to_judge():
    assert "R-LOW-REPLY-RATE" not in fired_ids(
        ctx(features={"n_emails_sent": 2.0, "n_emails_replied": 0.0, "reply_rate": 0.0})
    )


def test_negotiation_without_a_proposal():
    assert "R-NEGOTIATION-NO-PROPOSAL" in fired_ids(
        ctx(
            features={
                "stage_ordinal": float(STAGES.index("Negotiation")),
                "n_proposals": 0.0,
            }
        )
    )


def test_slow_cycle_uses_the_lstm_forecast():
    assert "R-SLOW-CYCLE" in fired_ids(ctx(days_to_close=90.0))
    assert "R-SLOW-CYCLE" not in fired_ids(ctx(days_to_close=20.0))
    # No forecast available -> the rule stays silent rather than guessing.
    assert "R-SLOW-CYCLE" not in fired_ids(ctx(days_to_close=None))


def test_healthy_deals_still_get_a_next_step():
    results = recommend(ctx(win_prob=0.88, features={"days_since_last_activity": 2.0}))
    assert "R-READY-TO-CLOSE" in {r.rule_id for r in results}
    assert results, "a strong deal should still receive advice"


def test_momentum_rule_sits_between_the_bands():
    assert "R-BUILD-ON-MOMENTUM" in fired_ids(
        ctx(win_prob=0.65, features={"days_since_last_activity": 4.0, "n_meetings": 2.0})
    )
    assert "R-BUILD-ON-MOMENTUM" not in fired_ids(ctx(win_prob=0.90))


# --- reasons ---------------------------------------------------------------


def test_every_reason_cites_something_concrete():
    """Section 8's requirement: reasons name drivers, not just the score."""
    situations = [
        ctx(win_prob=0.40, previous_win_prob=0.62),
        ctx(win_prob=0.30, features={"days_since_last_activity": 40.0, "has_champion": 0.0}),
        ctx(win_prob=0.45, features={"discount_requested_late": 1.0}),
        ctx(features={"n_stakeholders": 1.0, "deal_size_k": 300.0}),
        ctx(features={"n_emails_sent": 9.0, "n_emails_replied": 1.0, "reply_rate": 0.11}),
        ctx(win_prob=0.88, features={"days_since_last_activity": 1.0}),
        ctx(days_to_close=120.0),
    ]
    seen = 0
    for context in situations:
        for rec in recommend(context):
            seen += 1
            assert len(rec.reason) > 40, rec
            assert any(ch.isdigit() for ch in rec.reason), (
                f"{rec.rule_id} reason cites no numbers: {rec.reason}"
            )
            assert rec.reason.strip().endswith("."), rec.rule_id
    assert seen >= 7


def test_multiple_rules_fire_for_a_genuinely_bad_deal():
    results = fired_ids(
        ctx(
            win_prob=0.18,
            previous_win_prob=0.44,
            features={
                "days_since_last_activity": 35.0,
                "has_champion": 0.0,
                "n_stakeholders": 1.0,
                "deal_size_k": 500.0,
                "n_demos": 0.0,
                "discount_requested_late": 1.0,
                "n_emails_sent": 12.0,
                "n_emails_replied": 1.0,
                "reply_rate": 0.083,
            },
        )
    )
    # The old win_prob-band engine could only ever return one.
    assert len(results) >= 5, results
    assert {"R-SCORE-DROP", "R-GONE-QUIET", "R-DISCOUNT-LATE"} <= results


# --- urgency ---------------------------------------------------------------


def test_urgency_rises_with_value_at_risk():
    big = urgency(ctx(win_prob=0.3, features={"deal_size_k": 500.0}), "CRITICAL")
    small = urgency(ctx(win_prob=0.3, features={"deal_size_k": 20.0}), "CRITICAL")
    assert big > small


def test_urgency_rises_as_confidence_falls():
    shaky = urgency(ctx(win_prob=0.2), "HIGH")
    solid = urgency(ctx(win_prob=0.9), "HIGH")
    assert shaky > solid


def test_urgency_respects_the_tier_weighting():
    base = ctx(win_prob=0.4)
    assert urgency(base, "CRITICAL") > urgency(base, "HIGH") > urgency(base, "LOW")


def test_urgency_is_damped_by_a_distant_close():
    soon = urgency(ctx(win_prob=0.4, days_to_close=10.0), "HIGH")
    later = urgency(ctx(win_prob=0.4, days_to_close=120.0), "HIGH")
    assert soon > later
