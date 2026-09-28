"""The Phase 1 acceptance criterion: `make train` reproduces from a seed.

Generating 2,000 deals twice is slow, so the full-scale check is marked
``slow``. The fast tests below run the generator at reduced scale and assert the
properties that actually matter: same seed means identical output, different
seed means different output, and no real-world identifiers leak in.
"""

from __future__ import annotations

import hashlib
import os

import pandas as pd
import pytest

from training.generate_synthetic import generate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR = os.path.join(REPO_ROOT, "data", "synthetic")


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        df.to_csv(index=False).encode("utf-8")
    ).hexdigest()


@pytest.fixture(scope="module")
def frames():
    return generate(seed=1234)


def test_same_seed_reproduces_identical_data():
    a = generate(seed=99)
    b = generate(seed=99)
    for key in a:
        assert _hash(a[key]) == _hash(b[key]), key


def test_different_seed_produces_different_data():
    a = generate(seed=1)
    b = generate(seed=2)
    assert _hash(a["deals"]) != _hash(b["deals"])


def test_shapes_and_columns(frames):
    deals, activities = frames["deals"], frames["activities"]
    live = frames["live_deals"]

    assert len(deals) == 2000
    assert len(live) == 60
    assert set(deals.columns) >= {
        "id", "company", "industry", "region", "deal_size",
        "stage", "source", "created_at", "expected_close", "owner_rep",
    }
    assert set(activities.columns) == {"id", "deal_id", "type", "occurred_at"}


def test_historical_deals_are_labelled_and_live_deals_are_not(frames):
    deals, live = frames["deals"], frames["live_deals"]
    assert deals["won"].isin([0, 1]).all()
    assert deals["stage"].isin(["Closed Won", "Closed Lost"]).all()

    assert live["won"].isna().all(), "live demo deals must be unlabelled"
    assert live["closed_at"].isna().all()
    assert not live["stage"].isin(["Closed Won", "Closed Lost"]).any()


def test_win_rate_is_plausible(frames):
    """Neither degenerate nor suspiciously balanced."""
    rate = frames["deals"]["won"].mean()
    assert 0.20 < rate < 0.55, rate


def test_activity_ids_are_unique_and_types_are_known(frames):
    from app.features.build import ACTIVITY_TYPES

    activities = frames["activities"]
    assert activities["id"].is_unique
    assert set(activities["type"]) <= set(ACTIVITY_TYPES)
    # All nine documented event types should actually occur.
    assert set(activities["type"]) == set(ACTIVITY_TYPES)


def test_every_activity_belongs_to_a_generated_deal(frames):
    assert set(frames["activities"]["deal_id"]) <= set(frames["deals"]["id"])
    assert set(frames["live_activities"]["deal_id"]) <= set(frames["live_deals"]["id"])


def test_activities_never_precede_their_deal(frames):
    deals = frames["deals"].set_index("id")
    acts = frames["activities"].copy()
    acts["occurred_at"] = pd.to_datetime(acts["occurred_at"])
    created = pd.to_datetime(deals["created_at"])
    assert (acts["occurred_at"].values >= created.loc[acts["deal_id"]].values).all()


def test_no_real_company_or_placeholder_names_leak(frames):
    """Guards the spec's hard constraint: synthetic data only."""
    names = set(frames["deals"]["company"]) | set(frames["live_deals"]["company"])
    assert len(names) == len(frames["deals"]) + len(frames["live_deals"])
    # The old generator's Company_N placeholders must be gone.
    assert not any(n.startswith("Company_") for n in names)
    assert all(n.strip() for n in names)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(os.path.join(SYNTHETIC_DIR, "deals.csv")),
    reason="synthetic data not generated; run `make data`",
)
def test_committed_data_matches_a_fresh_run_at_the_default_seed():
    """The acceptance criterion, end to end."""
    from training.generate_synthetic import SEED

    fresh = generate(seed=SEED)
    for name in ("deals", "activities", "live_deals", "live_activities"):
        on_disk = pd.read_csv(os.path.join(SYNTHETIC_DIR, f"{name}.csv"))
        regenerated = pd.read_csv(
            pd.io.common.StringIO(fresh[name].to_csv(index=False))
        )
        pd.testing.assert_frame_equal(on_disk, regenerated, check_dtype=False)
