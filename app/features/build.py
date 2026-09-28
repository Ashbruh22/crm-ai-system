"""Deal + activity log -> model input.

Two representations are built here:

* a flat 32-feature vector for the XGBoost win-probability classifier
  (``build_feature_vector``), and
* a fixed (60, 7) activity sequence for the days-to-close LSTM
  (``build_sequence``).

The ordering in ``FEATURE_NAMES`` is a contract. It is frozen into
``artifacts/feature_schema.json`` at training time and checked at service
startup; the service refuses to boot on a mismatch. Append new features at the
end and retrain — never reorder.

Categorical vocabularies are hard-coded rather than fitted so that serving
needs no pickled encoder, which is what keeps the runtime image free of a
scikit-learn version pin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# --- vocabularies (order is part of the contract) ---------------------------

INDUSTRIES: tuple[str, ...] = (
    "Technology",
    "Financial Services",
    "Healthcare",
    "Manufacturing",
    "Retail",
    "Public Sector",
)

REGIONS: tuple[str, ...] = ("North America", "EMEA", "APAC", "LATAM")

SOURCES: tuple[str, ...] = ("Inbound", "Outbound", "Partner", "Event")

#: Open-pipeline stages in progression order. ``stage_ordinal`` is the index.
STAGES: tuple[str, ...] = (
    "Prospecting",
    "Qualification",
    "Discovery",
    "Proposal",
    "Negotiation",
)

#: Terminal stages, never scored.
CLOSED_STAGES: tuple[str, ...] = ("Closed Won", "Closed Lost")

# --- activity vocabulary ---------------------------------------------------

ACTIVITY_TYPES: tuple[str, ...] = (
    "email_sent",
    "email_replied",
    "meeting_held",
    "demo_done",
    "proposal_sent",
    "champion_identified",
    "discount_requested",
    "stakeholder_added",
    "no_activity_7d",
)

#: Buyer-side signals of real interest.
ENGAGEMENT_TYPES = frozenset({"email_replied", "meeting_held", "demo_done"})
#: Signals the deal is structurally advancing.
PROGRESSION_TYPES = frozenset(
    {"proposal_sent", "champion_identified", "stakeholder_added"}
)
#: Signals of risk.
RISK_TYPES = frozenset({"discount_requested", "no_activity_7d"})

#: Events that are not a contact with the buyer. ``no_activity_7d`` is a marker
#: the pipeline writes when a week passes with nothing happening, so counting it
#: as a "touch" would make a deal with eight logged weeks of silence look like it
#: was contacted seven days ago — exactly inverting the signal that
#: ``days_since_last_activity`` exists to carry.
NON_CONTACT_TYPES = frozenset({"no_activity_7d"})

# --- flat feature vector ---------------------------------------------------

NUMERIC_FEATURES: tuple[str, ...] = (
    "deal_size_k",
    "days_open",
    "days_since_last_activity",
    "n_emails_sent",
    "n_emails_replied",
    "reply_rate",
    "n_meetings",
    "n_demos",
    "n_proposals",
    "n_stakeholders",
    "has_champion",
    "n_discount_requests",
    "discount_requested_late",
    "n_silence_gaps",
    "activity_count",
    "activity_per_week",
    "stage_ordinal",
    "rep_win_rate",
)

FEATURE_NAMES: tuple[str, ...] = (
    NUMERIC_FEATURES
    + tuple(f"industry_{v}" for v in INDUSTRIES)
    + tuple(f"region_{v}" for v in REGIONS)
    + tuple(f"source_{v}" for v in SOURCES)
)

#: Plain-English labels for the dashboard's SHAP chart (spec section 9).
FEATURE_LABELS: Mapping[str, str] = {
    "deal_size_k": "Deal size ($k)",
    "days_open": "Days open",
    "days_since_last_activity": "Days since last touch",
    "n_emails_sent": "Emails sent",
    "n_emails_replied": "Emails replied to",
    "reply_rate": "Email reply rate",
    "n_meetings": "Meetings held",
    "n_demos": "Demos delivered",
    "n_proposals": "Proposals sent",
    "n_stakeholders": "Stakeholders involved",
    "has_champion": "Champion identified",
    "n_discount_requests": "Discount requests",
    "discount_requested_late": "Discount requested late in cycle",
    "n_silence_gaps": "7-day silence gaps",
    "activity_count": "Total activities",
    "activity_per_week": "Activities per week",
    "stage_ordinal": "Pipeline stage",
    "rep_win_rate": "Rep historical win rate",
}

# --- LSTM sequence ---------------------------------------------------------

SEQ_LEN = 60
SEQ_FEATURE_NAMES: tuple[str, ...] = (
    "days_since_prev_event",
    "cum_days_open",
    "is_engagement",
    "is_progression",
    "is_risk",
    "stage_ordinal",
    "log_deal_size",
)

#: Deals at or past this stage index treat a discount request as "late".
LATE_STAGE_INDEX = STAGES.index("Proposal")

_DEFAULT_REP_WIN_RATE = 0.4


def _as_datetime(value) -> datetime:
    """Coerce the assorted date shapes the CRM layer produces to a datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = pd.to_datetime(value).to_pydatetime()
    # Comparisons must not mix aware and naive datetimes.
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _stage_ordinal(stage: str) -> int:
    if stage in STAGES:
        return STAGES.index(stage)
    # A closed deal keeps the ordinal of the last open stage so that historical
    # rows and live rows stay on the same scale.
    return len(STAGES) - 1


def build_feature_vector(
    deal: Mapping,
    activities: Sequence[Mapping],
    rep_stats: Mapping[str, float] | None = None,
    as_of: datetime | None = None,
) -> np.ndarray:
    """Build the flat XGBoost feature vector for one deal.

    Args:
        deal: mapping with at least ``deal_size``, ``stage``, ``created_at``,
            ``industry``, ``region``, ``source``, ``owner_rep``.
        activities: that deal's activity rows, each with ``type`` and
            ``occurred_at``. Need not be sorted.
        rep_stats: rep name -> historical win rate, as computed on the training
            set and saved to ``artifacts/rep_stats.json``. Unknown reps fall
            back to the global base rate.
        as_of: scoring time. Defaults to now.

    Returns:
        float32 array of shape ``(len(FEATURE_NAMES),)``.
    """
    as_of = _as_datetime(as_of) if as_of is not None else datetime.utcnow()
    rep_stats = rep_stats or {}
    created_at = _as_datetime(deal["created_at"])

    events = sorted(
        (
            {"type": a["type"], "occurred_at": _as_datetime(a["occurred_at"])}
            for a in activities
        ),
        key=lambda e: e["occurred_at"],
    )

    days_open = max(0.0, (as_of - created_at).total_seconds() / 86400.0)

    counts = {t: 0 for t in ACTIVITY_TYPES}
    for e in events:
        if e["type"] in counts:
            counts[e["type"]] += 1

    stage = deal.get("stage", STAGES[0])
    stage_ordinal = _stage_ordinal(stage)

    contacts = [e for e in events if e["type"] not in NON_CONTACT_TYPES]
    if contacts:
        last_at = contacts[-1]["occurred_at"]
        days_since_last = max(0.0, (as_of - last_at).total_seconds() / 86400.0)
    else:
        # Never actually contacted: the whole life of the deal is silence.
        days_since_last = days_open

    n_sent = counts["email_sent"]
    n_replied = counts["email_replied"]
    reply_rate = (n_replied / n_sent) if n_sent else 0.0

    # A discount asked for at or after Proposal is the risk signal; asked for
    # early it is just noise in the qualification conversation.
    discount_late = 0.0
    if counts["discount_requested"] and stage_ordinal >= LATE_STAGE_INDEX:
        discount_late = 1.0

    activity_count = len(events)
    weeks_open = max(days_open / 7.0, 1.0)

    numeric = {
        "deal_size_k": float(deal["deal_size"]) / 1000.0,
        "days_open": days_open,
        "days_since_last_activity": days_since_last,
        "n_emails_sent": float(n_sent),
        "n_emails_replied": float(n_replied),
        "reply_rate": reply_rate,
        "n_meetings": float(counts["meeting_held"]),
        "n_demos": float(counts["demo_done"]),
        "n_proposals": float(counts["proposal_sent"]),
        # The deal starts with one known contact; each event adds another.
        "n_stakeholders": 1.0 + float(counts["stakeholder_added"]),
        "has_champion": 1.0 if counts["champion_identified"] else 0.0,
        "n_discount_requests": float(counts["discount_requested"]),
        "discount_requested_late": discount_late,
        "n_silence_gaps": float(counts["no_activity_7d"]),
        "activity_count": float(activity_count),
        "activity_per_week": float(activity_count) / weeks_open,
        "stage_ordinal": float(stage_ordinal),
        "rep_win_rate": float(
            rep_stats.get(deal.get("owner_rep", ""), _DEFAULT_REP_WIN_RATE)
        ),
    }

    values = [numeric[name] for name in NUMERIC_FEATURES]
    values += [1.0 if deal.get("industry") == v else 0.0 for v in INDUSTRIES]
    values += [1.0 if deal.get("region") == v else 0.0 for v in REGIONS]
    values += [1.0 if deal.get("source") == v else 0.0 for v in SOURCES]

    return np.asarray(values, dtype=np.float32)


def build_feature_frame(
    deals: pd.DataFrame,
    activities: pd.DataFrame,
    rep_stats: Mapping[str, float] | None = None,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Vectorised-enough wrapper over :func:`build_feature_vector`.

    ``as_of`` defaults per-row to the deal's ``closed_at`` when present (so
    historical rows are featurised at the moment they closed, not today) and to
    ``now`` otherwise.
    """
    rep_stats = rep_stats or {}
    by_deal: dict[object, list[dict]] = {}
    if len(activities):
        for deal_id, group in activities.groupby("deal_id", sort=False):
            by_deal[deal_id] = group[["type", "occurred_at"]].to_dict("records")

    rows = []
    for deal in deals.to_dict("records"):
        row_as_of = as_of
        if row_as_of is None and deal.get("closed_at") not in (None, "", float("nan")):
            if not pd.isna(deal.get("closed_at")):
                row_as_of = deal["closed_at"]
        rows.append(
            build_feature_vector(
                deal,
                by_deal.get(deal["id"], []),
                rep_stats=rep_stats,
                as_of=row_as_of,
            )
        )

    return pd.DataFrame(
        np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
        columns=list(FEATURE_NAMES),
        index=deals.index,
    )


def build_sequence(
    deal: Mapping,
    activities: Sequence[Mapping],
    max_len: int = SEQ_LEN,
) -> np.ndarray:
    """Build the fixed-shape activity sequence for the days-to-close LSTM.

    The sequence is **pre-padded** with zeros: the most recent event always sits
    at the last timestep, so a final-hidden-state readout sees it regardless of
    how many events the deal has. Deals with more than ``max_len`` events keep
    the most recent ``max_len``.

    Returns:
        float32 array of shape ``(max_len, len(SEQ_FEATURE_NAMES))``.
    """
    created_at = _as_datetime(deal["created_at"])
    stage_ordinal = _stage_ordinal(deal.get("stage", STAGES[0]))
    log_size = float(np.log1p(float(deal["deal_size"])) / 15.0)

    events = sorted(
        (
            {"type": a["type"], "occurred_at": _as_datetime(a["occurred_at"])}
            for a in activities
        ),
        key=lambda e: e["occurred_at"],
    )[-max_len:]

    seq = np.zeros((max_len, len(SEQ_FEATURE_NAMES)), dtype=np.float32)
    offset = max_len - len(events)
    prev_at = created_at

    for i, e in enumerate(events):
        gap_days = max(0.0, (e["occurred_at"] - prev_at).total_seconds() / 86400.0)
        cum_days = max(0.0, (e["occurred_at"] - created_at).total_seconds() / 86400.0)
        seq[offset + i] = (
            min(gap_days / 30.0, 3.0),
            min(cum_days / 180.0, 3.0),
            1.0 if e["type"] in ENGAGEMENT_TYPES else 0.0,
            1.0 if e["type"] in PROGRESSION_TYPES else 0.0,
            1.0 if e["type"] in RISK_TYPES else 0.0,
            stage_ordinal / max(1, len(STAGES) - 1),
            log_size,
        )
        prev_at = e["occurred_at"]

    return seq


def feature_schema() -> dict:
    """The contract written to ``artifacts/feature_schema.json``."""
    return {
        "version": 1,
        "n_features": len(FEATURE_NAMES),
        "features": [
            {
                "name": name,
                "dtype": "float32",
                "label": FEATURE_LABELS.get(name, name.replace("_", " ").capitalize()),
            }
            for name in FEATURE_NAMES
        ],
        "sequence": {
            "len": SEQ_LEN,
            "features": list(SEQ_FEATURE_NAMES),
            "padding": "pre",
        },
        "vocabularies": {
            "industry": list(INDUSTRIES),
            "region": list(REGIONS),
            "source": list(SOURCES),
            "stage": list(STAGES),
            "activity_type": list(ACTIVITY_TYPES),
        },
    }
