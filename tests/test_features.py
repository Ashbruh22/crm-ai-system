"""Feature builder contract tests (spec section 11)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from app.features.build import (
    FEATURE_NAMES,
    SEQ_FEATURE_NAMES,
    SEQ_LEN,
    STAGES,
    build_feature_vector,
    build_sequence,
    feature_schema,
)

CREATED = datetime(2026, 6, 1)


def _deal(**overrides) -> dict:
    deal = {
        "id": "D-00001",
        "company": "Example Holdings",
        "industry": "Technology",
        "region": "EMEA",
        "deal_size": 120_000.0,
        "stage": "Proposal",
        "source": "Inbound",
        "owner_rep": "Dana Reed",
        "created_at": CREATED,
    }
    deal.update(overrides)
    return deal


def _act(kind: str, day: int) -> dict:
    return {"type": kind, "occurred_at": CREATED + timedelta(days=day)}


def test_vector_length_matches_schema():
    vec = build_feature_vector(_deal(), [], as_of=CREATED + timedelta(days=10))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert vec.dtype == np.float32
    assert feature_schema()["n_features"] == len(FEATURE_NAMES)


def test_known_deal_produces_expected_values():
    activities = [
        _act("email_sent", 1),
        _act("email_sent", 3),
        _act("email_replied", 4),
        _act("meeting_held", 6),
        _act("champion_identified", 8),
        _act("stakeholder_added", 9),
        _act("no_activity_7d", 16),
    ]
    vec = build_feature_vector(
        _deal(),
        activities,
        rep_stats={"Dana Reed": 0.55},
        as_of=CREATED + timedelta(days=20),
    )
    f = dict(zip(FEATURE_NAMES, vec))

    assert f["deal_size_k"] == pytest.approx(120.0)
    assert f["days_open"] == pytest.approx(20.0)
    assert f["n_emails_sent"] == 2
    assert f["n_emails_replied"] == 1
    assert f["reply_rate"] == pytest.approx(0.5)
    assert f["n_meetings"] == 1
    assert f["has_champion"] == 1.0
    # One base contact plus the added stakeholder.
    assert f["n_stakeholders"] == 2
    assert f["n_silence_gaps"] == 1
    assert f["activity_count"] == 7
    # Last event was day 16, scored at day 20.
    assert f["days_since_last_activity"] == pytest.approx(4.0)
    assert f["stage_ordinal"] == STAGES.index("Proposal")
    assert f["rep_win_rate"] == pytest.approx(0.55)
    assert f["industry_Technology"] == 1.0
    assert f["industry_Retail"] == 0.0
    assert f["region_EMEA"] == 1.0
    assert f["source_Inbound"] == 1.0


def test_untouched_deal_counts_whole_life_as_silence():
    vec = build_feature_vector(_deal(), [], as_of=CREATED + timedelta(days=14))
    f = dict(zip(FEATURE_NAMES, vec))
    assert f["days_since_last_activity"] == pytest.approx(14.0)
    assert f["activity_count"] == 0
    assert f["reply_rate"] == 0.0


def test_unknown_rep_falls_back_to_default_rate():
    vec = build_feature_vector(
        _deal(owner_rep="Nobody At All"), [], rep_stats={"Dana Reed": 0.9}
    )
    f = dict(zip(FEATURE_NAMES, vec))
    assert f["rep_win_rate"] == pytest.approx(0.4)


def test_discount_is_only_risky_late_in_the_cycle():
    acts = [_act("proposal_sent", 5), _act("discount_requested", 6)]
    late = build_feature_vector(
        _deal(stage="Negotiation"), acts, as_of=CREATED + timedelta(days=10)
    )
    early = build_feature_vector(
        _deal(stage="Qualification"), acts, as_of=CREATED + timedelta(days=10)
    )
    assert dict(zip(FEATURE_NAMES, late))["discount_requested_late"] == 1.0
    assert dict(zip(FEATURE_NAMES, early))["discount_requested_late"] == 0.0
    # The raw count is recorded either way.
    assert dict(zip(FEATURE_NAMES, early))["n_discount_requests"] == 1.0


def test_one_hot_blocks_are_mutually_exclusive():
    vec = build_feature_vector(_deal(), [])
    f = dict(zip(FEATURE_NAMES, vec))
    for prefix in ("industry_", "region_", "source_"):
        block = [v for k, v in f.items() if k.startswith(prefix)]
        assert sum(block) == 1.0, prefix


def test_sequence_is_pre_padded_with_latest_event_last():
    activities = [_act("email_sent", 1), _act("email_replied", 2)]
    seq = build_sequence(_deal(), activities)

    assert seq.shape == (SEQ_LEN, len(SEQ_FEATURE_NAMES))
    assert seq.dtype == np.float32
    # Everything before the two real events is padding.
    assert np.all(seq[:-2] == 0.0)
    assert np.any(seq[-1] != 0.0)
    # The final timestep is the reply, which is an engagement event.
    engagement_idx = SEQ_FEATURE_NAMES.index("is_engagement")
    assert seq[-1][engagement_idx] == 1.0


def test_sequence_truncates_to_most_recent_events():
    activities = [_act("email_sent", d) for d in range(1, SEQ_LEN + 21)]
    activities.append(_act("champion_identified", SEQ_LEN + 21))
    seq = build_sequence(_deal(), activities)

    assert seq.shape == (SEQ_LEN, len(SEQ_FEATURE_NAMES))
    # No padding left: there were more events than slots.
    assert np.all(seq.any(axis=1))
    progression_idx = SEQ_FEATURE_NAMES.index("is_progression")
    assert seq[-1][progression_idx] == 1.0


def test_activities_need_not_arrive_sorted():
    ordered = [_act("email_sent", 1), _act("email_replied", 5)]
    shuffled = list(reversed(ordered))
    assert np.array_equal(
        build_sequence(_deal(), ordered), build_sequence(_deal(), shuffled)
    )
    assert np.array_equal(
        build_feature_vector(_deal(), ordered, as_of=CREATED + timedelta(days=9)),
        build_feature_vector(_deal(), shuffled, as_of=CREATED + timedelta(days=9)),
    )


def test_schema_declares_every_feature_with_a_label():
    schema = feature_schema()
    assert [f["name"] for f in schema["features"]] == list(FEATURE_NAMES)
    assert all(f["label"] for f in schema["features"])
    assert schema["sequence"]["len"] == SEQ_LEN
    assert schema["sequence"]["padding"] == "pre"
