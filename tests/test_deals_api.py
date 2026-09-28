"""Phase 2 acceptance: /healthz green and seeded deals listed via /api/deals."""

from __future__ import annotations

import pytest

from tests.conftest import (
    artifacts_required,
    make_action,
    make_activity,
    make_deal,
    make_score,
)

pytestmark = [pytest.mark.anyio, artifacts_required]


async def _seed(session, deals):
    for obj in deals:
        session.add(obj)
    await session.commit()


# --- health ----------------------------------------------------------------


async def test_healthz_reports_models_db_and_redis(client):
    response = await client.get("/healthz")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok", body
    assert body["db"] == "ok"
    assert body["redis"] == "ok"
    assert body["models"]["xgb_loaded"] is True
    assert body["models"]["lstm_loaded"] is True
    assert body["models"]["n_features"] == 32
    assert body["synthetic_data"] is True


@pytest.mark.parametrize("path", ["/health", "/ready"])
async def test_legacy_health_aliases_still_answer(client, path):
    assert (await client.get(path)).status_code == 200


# --- listing ---------------------------------------------------------------


async def test_empty_pipeline_returns_an_empty_page(client):
    body = (await client.get("/api/deals")).json()
    assert body["items"] == []
    assert body["total"] == 0


async def test_lists_seeded_deals_with_latest_score(client, session):
    await _seed(
        session,
        [
            make_deal("LIVE-0001", company="Alpha Ltd"),
            make_deal("LIVE-0002", company="Beta GmbH", stage="Negotiation"),
        ],
    )
    # Two scores for the same deal: only the newest should surface.
    await _seed(
        session,
        [
            make_score("LIVE-0001", win_prob=0.20, minutes_ago=60),
            make_score("LIVE-0001", win_prob=0.72, minutes_ago=1),
        ],
    )

    body = (await client.get("/api/deals")).json()
    assert body["total"] == 2
    assert body["synthetic_data"] is True

    scored = next(d for d in body["items"] if d["id"] == "LIVE-0001")
    assert scored["company"] == "Alpha Ltd"
    assert scored["score"]["win_prob"] == pytest.approx(0.72)

    unscored = next(d for d in body["items"] if d["id"] == "LIVE-0002")
    assert unscored["score"] is None


async def test_filter_by_stage_and_rep(client, session):
    await _seed(
        session,
        [
            make_deal("LIVE-0001", stage="Proposal", owner_rep="Dana Reed"),
            make_deal("LIVE-0002", stage="Negotiation", owner_rep="Dana Reed"),
            make_deal("LIVE-0003", stage="Proposal", owner_rep="Sam Okafor"),
        ],
    )

    body = (await client.get("/api/deals", params={"stage": "Proposal"})).json()
    assert {d["id"] for d in body["items"]} == {"LIVE-0001", "LIVE-0003"}

    body = (await client.get("/api/deals", params={"owner_rep": "Sam Okafor"})).json()
    assert {d["id"] for d in body["items"]} == {"LIVE-0003"}


async def test_sort_by_win_probability_descending(client, session):
    await _seed(session, [make_deal(f"LIVE-000{i}") for i in (1, 2, 3)])
    await _seed(
        session,
        [
            make_score("LIVE-0001", win_prob=0.30),
            make_score("LIVE-0002", win_prob=0.90),
            make_score("LIVE-0003", win_prob=0.60),
        ],
    )

    body = (await client.get("/api/deals", params={"sort": "win_prob"})).json()
    assert [d["id"] for d in body["items"]] == ["LIVE-0002", "LIVE-0003", "LIVE-0001"]


async def test_pagination_reports_total_beyond_the_page(client, session):
    await _seed(session, [make_deal(f"LIVE-{i:04d}") for i in range(1, 8)])

    body = (
        await client.get("/api/deals", params={"limit": 3, "offset": 0, "sort": "created_at"})
    ).json()
    assert len(body["items"]) == 3
    assert body["total"] == 7


async def test_closed_deals_are_not_in_the_open_pipeline(client, session):
    from datetime import timedelta

    from tests.conftest import BASE_TIME

    await _seed(
        session,
        [
            make_deal("LIVE-0001"),
            make_deal(
                "LIVE-0002",
                stage="Closed Won",
                closed_at=BASE_TIME - timedelta(days=1),
                won=True,
            ),
        ],
    )

    body = (await client.get("/api/deals")).json()
    assert [d["id"] for d in body["items"]] == ["LIVE-0001"]
    assert body["total"] == 1


async def test_top_action_prefers_the_highest_priority(client, session):
    await _seed(session, [make_deal("LIVE-0001")])
    await _seed(
        session,
        [
            make_action("LIVE-0001", priority="LOW", rule_id="R-LOW"),
            make_action("LIVE-0001", priority="CRITICAL", rule_id="R-CRIT"),
            make_action("LIVE-0001", priority="MEDIUM", rule_id="R-MED"),
        ],
    )

    body = (await client.get("/api/deals")).json()
    assert body["items"][0]["top_action"]["priority"] == "CRITICAL"


# --- detail ----------------------------------------------------------------


async def test_deal_detail_returns_timeline_history_and_actions(client, session):
    await _seed(session, [make_deal("LIVE-0001")])
    await _seed(
        session,
        [
            make_activity("LIVE-0001", "email_sent", 1),
            make_activity("LIVE-0001", "email_replied", 3),
            make_activity("LIVE-0001", "champion_identified", 9),
        ],
    )
    await _seed(
        session,
        [
            make_score("LIVE-0001", win_prob=0.40, minutes_ago=120),
            make_score("LIVE-0001", win_prob=0.65, minutes_ago=5),
        ],
    )
    await _seed(session, [make_action("LIVE-0001")])

    body = (await client.get("/api/deals/LIVE-0001")).json()

    assert body["deal"]["company"] == "Example Holdings"
    assert body["latest_score"]["win_prob"] == pytest.approx(0.65)
    assert len(body["timeline"]) == 3
    # History is oldest-first so the chart can plot it directly.
    assert [s["win_prob"] for s in body["score_history"]] == pytest.approx([0.40, 0.65])
    assert body["actions"][0]["status"] == "suggested"
    assert body["actions"][0]["reason"]


async def test_unknown_deal_is_404(client):
    response = await client.get("/api/deals/NOPE-9999")
    assert response.status_code == 404
    assert "NOPE-9999" in response.json()["detail"]


# --- meta ------------------------------------------------------------------


async def test_meta_metrics_keeps_paper_and_demo_apart(client):
    body = (await client.get("/api/meta/metrics")).json()

    assert body["demo"]["synthetic_data"] is True
    assert "synthetic" in body["demo"]["label"].lower()
    assert body["paper"]["reproducible_here"] is False
    assert "pilot" in body["paper"]["label"].lower()

    demo_auc = body["demo"]["win_probability"]["auc_roc"]
    paper_auc = body["paper"]["win_model_auc_roc"]
    assert demo_auc != paper_auc
    assert body["latency_ms"]["measured"] is None


async def test_meta_features_exposes_labels_for_the_chart(client):
    body = (await client.get("/api/meta/features")).json()
    assert body["n_features"] == 32
    assert body["labels"]["has_champion"] == "Champion identified"


async def test_model_card_is_served(client):
    body = (await client.get("/api/meta/model-card")).json()
    assert body["format"] == "markdown"
    assert "Synthetic data only" in body["content"]


# --- CORS ------------------------------------------------------------------


async def test_cors_allows_the_dashboard_origin_only(client):
    allowed = await client.get(
        "/healthz", headers={"Origin": "http://localhost:5173"}
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    denied = await client.get("/healthz", headers={"Origin": "https://evil.example"})
    # Starlette omits the header entirely for a disallowed origin.
    assert denied.headers.get("access-control-allow-origin") is None
