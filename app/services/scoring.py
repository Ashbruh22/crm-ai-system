"""Scoring pipeline: features -> XGBoost -> LSTM -> SHAP (spec sections 3, 7).

One deal in, a win probability, a days-to-close estimate, the SHAP drivers
behind them, and a per-stage latency breakdown out.

Caching
-------
The cache key is ``deal_id + model_version + feature_hash``, where the hash is a
**SHA-256 of the feature bytes**. The previous implementation used Python's
builtin ``hash()`` on a string, which is salted per process (PYTHONHASHSEED):
every worker and every restart computed a different key for identical features,
so the cache silently never hit. Nothing errored — it just quietly did no work.

Latency
-------
Reported per stage (features / xgb / lstm / shap / total) rather than as one
number, because "which part is slow" is the question worth answering, and the
README quotes a measured figure or none at all.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import nba
from app.db.models import Action, ActionStatus, Activity, Deal, Score
from app.features.build import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    build_feature_vector,
    build_sequence,
)
from app.models.registry import ModelRegistry

#: How many SHAP drivers to keep on a stored score.
TOP_K = 8

#: Cached scores expire after this many seconds. Short, because a deal's score
#: changes whenever an activity lands, and the consumer invalidates explicitly.
CACHE_TTL = 900

CACHE_PREFIX = "score"

#: Scoring time is floored to this many seconds before features are built.
#:
#: Several features (days_open, days_since_last_activity) are derived from "now".
#: Left at full resolution they drift continuously, so two scores of an otherwise
#: unchanged deal produce different feature hashes and the cache never hits —
#: the same class of silent failure as the salted hash() key this replaced.
#: Flooring to the minute makes the vector stable between changes to the deal,
#: and a deal's score does not meaningfully move within one minute anyway.
AS_OF_QUANTUM_SECONDS = 60


def quantize_as_of(moment: datetime | None = None) -> datetime:
    """Floor a scoring instant to AS_OF_QUANTUM_SECONDS."""
    moment = moment or datetime.now(timezone.utc)
    epoch = int(moment.timestamp())
    floored = epoch - (epoch % AS_OF_QUANTUM_SECONDS)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


class DealNotFound(LookupError):
    """No such deal."""


@dataclass
class ScoreResult:
    deal_id: str
    win_prob: float
    days_to_close: float | None
    model_version: str
    shap_top: list[dict]
    shap_base_value: float
    feature_hash: str
    latency_ms: dict[str, float]
    cache_hit: bool
    features: dict[str, float] = field(default_factory=dict)
    recommendations: list = field(default_factory=list)
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "deal_id": self.deal_id,
            "win_prob": self.win_prob,
            "days_to_close": self.days_to_close,
            "model_version": self.model_version,
            "shap_top": self.shap_top,
            "shap_base_value": self.shap_base_value,
            "feature_hash": self.feature_hash,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "scored_at": self.scored_at.isoformat(),
            "actions": [
                {
                    "rule_id": r.rule_id,
                    "action_type": r.action_type,
                    "reason": r.reason,
                    "priority": r.priority,
                    "urgency_score": r.urgency_score,
                }
                for r in self.recommendations
            ],
        }


class _Timer:
    """Accumulates per-stage timings in milliseconds."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._start = time.perf_counter()
        self._mark = self._start

    def stage(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = round((now - self._mark) * 1000, 3)
        self._mark = now

    def total(self) -> dict[str, float]:
        self.stages["total"] = round((time.perf_counter() - self._start) * 1000, 3)
        return self.stages


def feature_hash(vector: np.ndarray, model_version: str) -> str:
    """Stable content hash of a feature vector.

    Deterministic across processes, restarts and machines — unlike ``hash()``,
    which is salted per interpreter. Values are rounded first so that float
    noise below the sixth decimal does not produce cache misses.
    """
    rounded = np.round(np.asarray(vector, dtype=np.float64), 6)
    digest = hashlib.sha256()
    digest.update(model_version.encode("utf-8"))
    digest.update(rounded.tobytes())
    return digest.hexdigest()[:32]


def cache_key(deal_id: str, fhash: str) -> str:
    return f"{CACHE_PREFIX}:{deal_id}:{fhash}"


async def load_deal(session: AsyncSession, deal_id: str) -> tuple[Deal, list[Activity]]:
    deal = (
        await session.execute(select(Deal).where(Deal.id == deal_id))
    ).scalar_one_or_none()
    if deal is None:
        raise DealNotFound(deal_id)

    activities = list(
        (
            await session.execute(
                select(Activity)
                .where(Activity.deal_id == deal_id)
                .order_by(Activity.occurred_at)
            )
        ).scalars()
    )
    return deal, activities


def deal_mapping(deal: Deal, deal_size: float | None = None) -> dict:
    """Deal ORM row -> the plain mapping the feature builder expects."""
    return {
        "id": deal.id,
        "company": deal.company,
        "industry": deal.industry,
        "region": deal.region,
        "deal_size": float(deal.deal_size) if deal_size is None else deal_size,
        "stage": deal.stage,
        "source": deal.source,
        "owner_rep": deal.owner_rep,
        "created_at": deal.created_at,
    }


def activity_mappings(activities: Sequence[Activity]) -> list[dict]:
    return [{"type": a.type, "occurred_at": a.occurred_at} for a in activities]


def shap_drivers(
    registry: ModelRegistry,
    vector: np.ndarray,
    top_k: int | None = TOP_K,
) -> tuple[list[dict], float]:
    """SHAP contributions for one feature vector, largest magnitude first.

    Uses XGBoost's own TreeSHAP, so values are exactly additive in log-odds:
    ``base + sum(shap)`` reproduces the model margin, and
    ``sigmoid(margin)`` reproduces the probability. The shap package's
    TreeExplainer did not hold that property against xgboost 3.2.
    """
    contribs, base = registry.shap_contribs(vector.reshape(1, -1))
    values = np.asarray(contribs).reshape(-1)

    order = np.argsort(np.abs(values))[::-1]
    if top_k is not None:
        order = order[:top_k]

    drivers = [
        {
            "feature": FEATURE_NAMES[i],
            "label": FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]),
            "value": round(float(vector[i]), 4),
            "shap": round(float(values[i]), 5),
            "direction": "increases" if values[i] >= 0 else "decreases",
        }
        for i in order
    ]
    return drivers, base


def score_vector(
    registry: ModelRegistry,
    vector: np.ndarray,
    sequence: np.ndarray | None,
    timer: _Timer,
    *,
    top_k: int | None = TOP_K,
) -> tuple[float, float | None, list[dict], float]:
    """Run both models plus SHAP over a prepared feature vector."""
    win_prob = float(registry.predict_win_prob(vector)[0])
    timer.stage("xgb")

    days_to_close: float | None = None
    if sequence is not None:
        days_to_close = float(registry.predict_days_to_close(sequence)[0])
    timer.stage("lstm")

    drivers, base = shap_drivers(registry, vector, top_k=top_k)
    timer.stage("shap")

    return win_prob, days_to_close, drivers, base


async def score_deal(
    session: AsyncSession,
    redis,
    registry: ModelRegistry,
    deal_id: str,
    *,
    bypass_cache: bool = False,
    persist: bool = True,
    as_of: datetime | None = None,
) -> ScoreResult:
    """Score one deal, using and populating the cache.

    Args:
        bypass_cache: recompute even if a cached result exists. The stored
            result is still refreshed.
        persist: append a row to ``scores``. False for read-only probes.
    """
    timer = _Timer()
    deal, activities = await load_deal(session, deal_id)

    # Quantised so repeated scores of an unchanged deal hash identically.
    as_of = quantize_as_of(as_of)
    vector = build_feature_vector(
        deal_mapping(deal),
        activity_mappings(activities),
        rep_stats=registry.rep_stats,
        as_of=as_of,
    )
    sequence = build_sequence(deal_mapping(deal), activity_mappings(activities))
    fhash = feature_hash(vector, registry.model_version)
    timer.stage("features")

    key = cache_key(deal_id, fhash)

    if not bypass_cache and redis is not None:
        cached = await _cache_get(redis, key)
        if cached is not None:
            # Report the cache lookup honestly: the model stages did not run.
            timer.stage("cache")
            latency = timer.total()
            return ScoreResult(
                deal_id=deal_id,
                win_prob=cached["win_prob"],
                days_to_close=cached.get("days_to_close"),
                model_version=cached["model_version"],
                shap_top=cached.get("shap_top", []),
                shap_base_value=cached.get("shap_base_value", 0.0),
                feature_hash=fhash,
                latency_ms=latency,
                cache_hit=True,
                features=dict(zip(FEATURE_NAMES, vector.tolist())),
            )

    # Captured before the new score is written, so R-SCORE-DROP can compare.
    previous = await latest_score(session, deal_id)
    previous_win_prob = previous.win_prob if previous else None

    win_prob, days_to_close, drivers, base = score_vector(
        registry, vector, sequence, timer
    )

    result = ScoreResult(
        deal_id=deal_id,
        win_prob=win_prob,
        days_to_close=days_to_close,
        model_version=registry.model_version,
        shap_top=drivers,
        shap_base_value=base,
        feature_hash=fhash,
        latency_ms={},
        cache_hit=False,
        features=dict(zip(FEATURE_NAMES, vector.tolist())),
    )

    recommendations = nba.recommend(
        nba.context_from_score(
            deal,
            dict(zip(FEATURE_NAMES, vector.tolist())),
            win_prob,
            days_to_close,
            drivers,
            previous_win_prob=previous_win_prob,
        )
    )
    result.recommendations = recommendations
    timer.stage("nba")

    if persist:
        score_row = Score(
            deal_id=deal_id,
            win_prob=win_prob,
            days_to_close=days_to_close,
            model_version=registry.model_version,
            shap_top=drivers,
            feature_hash=fhash,
            latency_ms=None,  # set below, once the total is known
            cache_hit=False,
            scored_at=result.scored_at,
        )
        session.add(score_row)
        # Flush so score_row.id exists for the actions' foreign key.
        await session.flush()
        await sync_actions(session, deal_id, recommendations, score_id=score_row.id)

    if redis is not None:
        # Fire and forget. The caller is waiting on a score, not on the cache
        # being warm for the next caller, and against a managed Redis in
        # another region that write was measured at ~118 ms -- half the entire
        # response time for ~8 ms of actual model work. Nothing reads the
        # result of this, and a failed write only costs the next request a
        # cache miss.
        _spawn(_cache_set(redis, key, result))
    timer.stage("cache")

    result.latency_ms = timer.total()

    if persist:
        # The Score row was created before the total was known.
        score_row.latency_ms = result.latency_ms
        await session.commit()

    return result


#: Strong references to in-flight background writes. asyncio only holds a weak
#: reference to a task, so without this the garbage collector can cancel one
#: mid-flight and the write silently never lands.
_background: set = set()


def _spawn(coro) -> None:
    """Run a coroutine without waiting for it."""
    try:
        task = asyncio.ensure_future(coro)
    except RuntimeError:  # no running loop (sync context in a test)
        coro.close()
        return
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _cache_get(redis, key: str) -> dict | None:
    try:
        raw = await redis.get(key)
    except Exception:  # noqa: BLE001 - a cache outage must not fail scoring
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


async def _cache_set(redis, key: str, result: ScoreResult) -> None:
    payload = {
        "win_prob": result.win_prob,
        "days_to_close": result.days_to_close,
        "model_version": result.model_version,
        "shap_top": result.shap_top,
        "shap_base_value": result.shap_base_value,
    }
    try:
        await redis.setex(key, CACHE_TTL, json.dumps(payload))
    except Exception:  # noqa: BLE001
        return


async def invalidate_deal(redis, deal_id: str) -> int:
    """Drop every cached score for a deal. Called when an activity lands."""
    if redis is None:
        return 0
    removed = 0
    try:
        async for key in redis.scan_iter(match=f"{CACHE_PREFIX}:{deal_id}:*"):
            await redis.delete(key)
            removed += 1
    except Exception:  # noqa: BLE001
        return removed
    return removed


async def sync_actions(
    session: AsyncSession,
    deal_id: str,
    recommendations: Sequence[nba.Recommendation],
    score_id=None,
) -> list[Action]:
    """Write newly-fired recommendations to the ledger.

    A rule that is already open for this deal is not re-inserted: re-scoring a
    deal every time an activity lands would otherwise pile up duplicate advice.
    The existing row stays, keeping its original created_at and whatever the
    user did with it.

    Nothing is auto-dismissed. A rule that stops firing leaves its action open,
    because the ledger is an audit trail of what was recommended and when — the
    user decides what happened to it.
    """
    open_rules = set(
        (
            await session.execute(
                select(Action.rule_id).where(
                    Action.deal_id == deal_id,
                    Action.status == ActionStatus.suggested,
                )
            )
        )
        .scalars()
        .all()
    )

    created: list[Action] = []
    for rec in recommendations:
        if rec.rule_id in open_rules:
            continue
        action = Action(
            deal_id=deal_id,
            score_id=score_id,
            rule_id=rec.rule_id,
            action_type=rec.action_type,
            reason=rec.reason,
            priority=rec.priority,
            urgency_score=rec.urgency_score,
            status=ActionStatus.suggested,
        )
        session.add(action)
        created.append(action)

    return created


async def latest_score(session: AsyncSession, deal_id: str) -> Score | None:
    return (
        await session.execute(
            select(Score)
            .where(Score.deal_id == deal_id)
            .order_by(desc(Score.scored_at))
            .limit(1)
        )
    ).scalar_one_or_none()


# --- what-if ---------------------------------------------------------------

#: Features a caller may override, with inclusive bounds (spec section 10).
#: Anything not listed here is rejected: the sliders exist to explore plausible
#: sales scenarios, not to probe the model with arbitrary vectors.
WHAT_IF_BOUNDS: Mapping[str, tuple[float, float]] = {
    "deal_size_k": (0.0, 10_000.0),
    "days_open": (0.0, 1_000.0),
    "days_since_last_activity": (0.0, 365.0),
    "n_emails_sent": (0.0, 200.0),
    "n_emails_replied": (0.0, 200.0),
    "n_meetings": (0.0, 50.0),
    "n_demos": (0.0, 20.0),
    "n_proposals": (0.0, 10.0),
    "n_stakeholders": (1.0, 20.0),
    "has_champion": (0.0, 1.0),
    "n_discount_requests": (0.0, 10.0),
    "discount_requested_late": (0.0, 1.0),
    "n_silence_gaps": (0.0, 50.0),
}


class WhatIfError(ValueError):
    """An override is unknown or out of bounds."""


def apply_overrides(
    vector: np.ndarray, overrides: Mapping[str, float]
) -> tuple[np.ndarray, dict[str, dict]]:
    """Return a copy of ``vector`` with the named features replaced."""
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    modified = vector.copy()
    applied: dict[str, dict] = {}

    for name, raw in overrides.items():
        if name not in WHAT_IF_BOUNDS:
            raise WhatIfError(
                f"{name!r} is not adjustable. Allowed: "
                f"{', '.join(sorted(WHAT_IF_BOUNDS))}"
            )
        low, high = WHAT_IF_BOUNDS[name]
        value = float(raw)
        if not (low <= value <= high):
            raise WhatIfError(f"{name} must be between {low} and {high}, got {value}")

        i = index[name]
        applied[name] = {"from": round(float(modified[i]), 4), "to": value}
        modified[i] = value

    # reply_rate is derived, so recompute it rather than let it contradict the
    # email counts the caller just changed.
    if "n_emails_sent" in overrides or "n_emails_replied" in overrides:
        sent = modified[index["n_emails_sent"]]
        replied = modified[index["n_emails_replied"]]
        modified[index["reply_rate"]] = float(replied / sent) if sent else 0.0

    return modified, applied


async def what_if(
    session: AsyncSession,
    registry: ModelRegistry,
    deal_id: str,
    overrides: Mapping[str, float],
    *,
    as_of: datetime | None = None,
) -> dict:
    """Score a hypothetical change. Never written to the database."""
    timer = _Timer()
    deal, activities = await load_deal(session, deal_id)

    as_of = quantize_as_of(as_of)
    deal_map = deal_mapping(deal)
    acts = activity_mappings(activities)

    baseline_vector = build_feature_vector(
        deal_map, acts, rep_stats=registry.rep_stats, as_of=as_of
    )

    # deal_size_k also feeds the LSTM sequence, so the days-to-close estimate
    # moves with that slider rather than sitting suspiciously still.
    sequence_size = (
        overrides["deal_size_k"] * 1000.0 if "deal_size_k" in overrides else None
    )
    baseline_sequence = build_sequence(deal_map, acts)
    modified_sequence = (
        build_sequence(deal_mapping(deal, deal_size=sequence_size), acts)
        if sequence_size is not None
        else baseline_sequence
    )

    modified_vector, applied = apply_overrides(baseline_vector, overrides)
    timer.stage("features")

    base_prob = float(registry.predict_win_prob(baseline_vector)[0])
    new_prob = float(registry.predict_win_prob(modified_vector)[0])
    timer.stage("xgb")

    base_days = float(registry.predict_days_to_close(baseline_sequence)[0])
    new_days = float(registry.predict_days_to_close(modified_sequence)[0])
    timer.stage("lstm")

    drivers, shap_base = shap_drivers(registry, modified_vector)
    timer.stage("shap")

    return {
        "deal_id": deal_id,
        "baseline": {"win_prob": base_prob, "days_to_close": base_days},
        "what_if": {"win_prob": new_prob, "days_to_close": new_days},
        "delta": {
            "win_prob": round(new_prob - base_prob, 5),
            "days_to_close": round(new_days - base_days, 3),
        },
        "applied_overrides": applied,
        "shap_top": drivers,
        "shap_base_value": shap_base,
        "model_version": registry.model_version,
        "latency_ms": timer.total(),
        "persisted": False,
    }
