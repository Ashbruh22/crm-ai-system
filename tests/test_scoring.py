"""Phase 3 acceptance: /score and /explain shapes, and a real cache hit."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import artifacts_required, make_activity, make_deal

pytestmark = [pytest.mark.anyio, artifacts_required]


async def seed_deal(session, deal_id="LIVE-0001", activities=(("email_sent", 1),)):
    session.add(make_deal(deal_id))
    for kind, day in activities:
        session.add(make_activity(deal_id, kind, day))
    await session.commit()


ENGAGED = [
    ("email_sent", 1),
    ("email_replied", 2),
    ("meeting_held", 4),
    ("demo_done", 7),
    ("champion_identified", 9),
    ("proposal_sent", 12),
]


# --- scoring ---------------------------------------------------------------


async def test_score_returns_probability_days_and_drivers(client, session):
    await seed_deal(session, activities=ENGAGED)

    response = await client.post("/api/deals/LIVE-0001/score")
    assert response.status_code == 200
    body = response.json()

    assert body["deal_id"] == "LIVE-0001"
    assert 0.0 <= body["win_prob"] <= 1.0
    assert body["days_to_close"] > 0
    assert body["model_version"]
    assert body["feature_hash"]
    assert body["synthetic_data"] is True

    drivers = body["shap_top"]
    assert 1 <= len(drivers) <= 8
    for d in drivers:
        assert d["label"] and d["feature"]
        assert d["direction"] in ("increases", "decreases")
    # Sorted by absolute contribution.
    magnitudes = [abs(d["shap"]) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)


async def test_latency_is_broken_down_by_stage(client, session):
    await seed_deal(session, activities=ENGAGED)

    body = (await client.post("/api/deals/LIVE-0001/score")).json()
    latency = body["latency_ms"]

    for stage in ("features", "xgb", "lstm", "shap"):
        assert latency[stage] is not None, f"missing {stage} timing"
        assert latency[stage] >= 0
    assert latency["total"] > 0
    # Total covers the stages, allowing for rounding at 3 decimal places.
    stage_sum = sum(latency[s] for s in ("features", "xgb", "lstm", "shap", "cache"))
    assert stage_sum <= latency["total"] + 0.5


async def test_score_is_persisted_to_history(client, session):
    await seed_deal(session, activities=ENGAGED)

    await client.post("/api/deals/LIVE-0001/score")
    await client.post(
        "/api/deals/LIVE-0001/score", json={"bypass_cache": True}
    )

    detail = (await client.get("/api/deals/LIVE-0001")).json()
    assert len(detail["score_history"]) == 2
    assert detail["latest_score"]["win_prob"] is not None
    assert detail["latest_score"]["latency_ms"]["total"] > 0


async def test_scored_deal_appears_in_the_pipeline_list(client, session):
    await seed_deal(session, activities=ENGAGED)
    await client.post("/api/deals/LIVE-0001/score")

    body = (await client.get("/api/deals")).json()
    assert body["items"][0]["score"]["win_prob"] is not None


async def test_unknown_deal_is_404(client):
    assert (await client.post("/api/deals/NOPE/score")).status_code == 404
    assert (await client.get("/api/deals/NOPE/explain")).status_code == 404


# --- caching ---------------------------------------------------------------


async def test_second_call_is_a_cache_hit(client, session):
    """The phase 3 acceptance criterion."""
    await seed_deal(session, activities=ENGAGED)

    first = (
        await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": True})
    ).json()
    assert first["cache_hit"] is False

    second = (
        await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": False})
    ).json()
    assert second["cache_hit"] is True
    assert second["win_prob"] == pytest.approx(first["win_prob"])
    assert second["feature_hash"] == first["feature_hash"]
    # A cache hit must not re-run the models.
    assert second["latency_ms"].get("xgb") is None


async def test_cache_key_is_stable_across_processes(session):
    """Guards the bug this phase fixes.

    The old cache key used Python's builtin hash(), which is salted per
    interpreter, so identical features produced different keys in every worker
    and the cache never hit. The replacement must be pure content hashing.
    """
    import subprocess
    import sys

    from app.services.scoring import feature_hash

    vector = np.arange(32, dtype=np.float32)
    local = feature_hash(vector, "demo-v1")

    # PYTHONHASHSEED is randomised per process; a salted hash would differ here.
    import os

    code = (
        "import numpy as np;"
        "from app.services.scoring import feature_hash;"
        "print(feature_hash(np.arange(32, dtype=np.float32), 'demo-v1'))"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONHASHSEED="random", PYTHONPATH=repo_root)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == local


async def test_changing_features_changes_the_cache_key(client, session):
    from app.db.models import Activity
    from tests.conftest import make_activity

    await seed_deal(session, activities=ENGAGED)
    first = (await client.post("/api/deals/LIVE-0001/score")).json()

    session.add(make_activity("LIVE-0001", "discount_requested", 15))
    await session.commit()

    second = (
        await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": False})
    ).json()
    # New activity -> new features -> new key -> genuine miss, not a stale hit.
    assert second["feature_hash"] != first["feature_hash"]
    assert second["cache_hit"] is False


# --- explain ---------------------------------------------------------------


async def test_explain_returns_every_feature_and_is_additive(client, session):
    await seed_deal(session, activities=ENGAGED)

    body = (await client.get("/api/deals/LIVE-0001/explain")).json()

    assert len(body["shap_values"]) == 32, "explain must return the full attribution"
    assert body["synthetic_data"] is True

    # base_value + sum(shap) reconstructs the margin behind the probability.
    total = body["shap_base_value"] + sum(d["shap"] for d in body["shap_values"])
    assert total == pytest.approx(body["margin"], abs=1e-3)

    recovered = 1.0 / (1.0 + np.exp(-body["margin"]))
    assert recovered == pytest.approx(body["win_prob"], abs=2e-3)


async def test_explain_labels_are_plain_english(client, session):
    await seed_deal(session, activities=ENGAGED)

    body = (await client.get("/api/deals/LIVE-0001/explain")).json()
    labels = {d["feature"]: d["label"] for d in body["shap_values"]}
    assert labels["has_champion"] == "Champion identified"
    assert labels["days_since_last_activity"] == "Days since last touch"


# --- what-if ---------------------------------------------------------------


async def test_what_if_moves_the_score_without_saving(client, session):
    await seed_deal(session, activities=ENGAGED)
    before = (await client.post("/api/deals/LIVE-0001/score")).json()

    body = (
        await client.post(
            "/api/deals/LIVE-0001/what-if",
            json={"overrides": {"n_stakeholders": 6, "days_since_last_activity": 1}},
        )
    ).json()

    assert body["persisted"] is False
    assert body["baseline"]["win_prob"] == pytest.approx(before["win_prob"], abs=1e-6)
    # The service rounds the delta to 5 decimal places.
    assert body["delta"]["win_prob"] == pytest.approx(
        body["what_if"]["win_prob"] - body["baseline"]["win_prob"], abs=1e-5
    )
    assert body["applied_overrides"]["n_stakeholders"]["to"] == 6

    # Nothing new in the history: what-if is read-only.
    detail = (await client.get("/api/deals/LIVE-0001")).json()
    assert len(detail["score_history"]) == 1


async def test_silence_lowers_win_probability(client, session):
    """A behavioural check on the driver the generator encodes most strongly."""
    await seed_deal(session, activities=ENGAGED)

    body = (
        await client.post(
            "/api/deals/LIVE-0001/what-if",
            json={"overrides": {"days_since_last_activity": 90, "n_silence_gaps": 8}},
        )
    ).json()
    assert body["delta"]["win_prob"] < 0, body["delta"]


async def test_deal_size_also_moves_days_to_close(client, session):
    """deal_size feeds the LSTM sequence, so the cycle estimate must respond."""
    await seed_deal(session, activities=ENGAGED)

    body = (
        await client.post(
            "/api/deals/LIVE-0001/what-if",
            json={"overrides": {"deal_size_k": 4000}},
        )
    ).json()
    assert body["delta"]["days_to_close"] != 0.0


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"owner_rep": 1}, "not an adjustable feature"),
        ({"n_stakeholders": 999}, "above the bound"),
        ({"n_stakeholders": -1}, "below the bound"),
        ({"rep_win_rate": 0.9}, "not in the whitelist"),
    ],
)
async def test_what_if_rejects_bad_overrides(client, session, overrides, reason):
    await seed_deal(session, activities=ENGAGED)
    response = await client.post(
        "/api/deals/LIVE-0001/what-if", json={"overrides": overrides}
    )
    assert response.status_code == 422, reason


async def test_what_if_requires_at_least_one_override(client, session):
    await seed_deal(session, activities=ENGAGED)
    response = await client.post(
        "/api/deals/LIVE-0001/what-if", json={"overrides": {}}
    )
    assert response.status_code == 422


async def test_reply_rate_is_recomputed_not_left_inconsistent(session):
    """Overriding email counts must not leave a contradictory reply_rate."""
    from app.features.build import FEATURE_NAMES
    from app.services.scoring import apply_overrides

    vector = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    vector[idx["n_emails_sent"]] = 10
    vector[idx["n_emails_replied"]] = 1
    vector[idx["reply_rate"]] = 0.1

    modified, _ = apply_overrides(vector, {"n_emails_replied": 5})
    assert modified[idx["reply_rate"]] == pytest.approx(0.5)


async def test_repeated_scores_of_an_unchanged_deal_hash_identically(client, session):
    """Regression guard for the drifting-as_of bug.

    Time-derived features (days_open, days_since_last_activity) are built from
    "now". Unquantised, they drift between two calls a few milliseconds apart,
    the feature hash changes, and the cache never hits in production either.
    """
    import asyncio

    await seed_deal(session, activities=ENGAGED)

    first = (
        await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": True})
    ).json()
    await asyncio.sleep(0.15)
    second = (
        await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": True})
    ).json()

    assert second["feature_hash"] == first["feature_hash"]
    assert second["win_prob"] == pytest.approx(first["win_prob"])


def test_as_of_is_floored_to_the_minute():
    from datetime import datetime, timezone

    from app.services.scoring import AS_OF_QUANTUM_SECONDS, quantize_as_of

    moment = datetime(2026, 9, 28, 14, 37, 41, 812_000, tzinfo=timezone.utc)
    floored = quantize_as_of(moment)

    assert floored.second == 0 and floored.microsecond == 0
    assert (moment - floored).total_seconds() < AS_OF_QUANTUM_SECONDS
    # Two instants in the same minute quantise to the same value.
    assert quantize_as_of(moment.replace(second=3)) == floored


async def test_the_cache_write_does_not_block_the_response(client, session, fake_redis):
    """A score must not wait on the cache being written for the next caller.

    Measured against a managed Redis in another region, the write was ~118 ms
    against ~8 ms of actual model work. Nothing reads its result, and a failed
    write only costs the next request a miss.
    """
    import asyncio

    from app.models.registry import registry
    from app.services import scoring

    registry.load()
    await seed_deal(session, activities=ENGAGED)

    slow_writes = []

    async def slow_set(redis, key, result):
        slow_writes.append(key)
        await asyncio.sleep(0.4)

    original = scoring._cache_set
    scoring._cache_set = slow_set
    try:
        started = asyncio.get_event_loop().time()
        result = await scoring.score_deal(
            session, fake_redis, registry, "LIVE-0001", bypass_cache=True
        )
        elapsed = asyncio.get_event_loop().time() - started
    finally:
        scoring._cache_set = original

    assert slow_writes, "the cache write was never attempted"
    # It was spawned, not awaited: the 400 ms write cannot be inside this.
    assert elapsed < 0.35, f"scoring waited for the cache write ({elapsed:.2f}s)"
    assert result.win_prob > 0

    # Let the background task finish so it does not leak into the next test.
    await asyncio.sleep(0.5)
