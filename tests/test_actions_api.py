"""Phase 4 acceptance: actions appear with reasons, and can be resolved."""

from __future__ import annotations

import pytest

from tests.conftest import artifacts_required, make_activity, make_deal

pytestmark = [pytest.mark.anyio, artifacts_required]


async def seed(session, deal_id="LIVE-0001", activities=(), **deal_kwargs):
    session.add(make_deal(deal_id, **deal_kwargs))
    for kind, day in activities:
        session.add(make_activity(deal_id, kind, day))
    await session.commit()


# A deal nobody has touched in a long time: silence plus no champion.
NEGLECTED = [("email_sent", 1), ("email_sent", 3)]

ENGAGED = [
    ("email_sent", 1),
    ("email_replied", 2),
    ("meeting_held", 4),
    ("demo_done", 7),
    ("champion_identified", 9),
    ("proposal_sent", 12),
]


async def test_scoring_creates_actions_with_reasons(client, session):
    await seed(session, activities=NEGLECTED)

    body = (await client.post("/api/deals/LIVE-0001/score")).json()
    assert body["actions"], "a neglected deal should draw recommendations"

    for action in body["actions"]:
        assert action["rule_id"].startswith("R-")
        assert action["priority"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        assert len(action["reason"]) > 40
        assert any(ch.isdigit() for ch in action["reason"]), action["reason"]


async def test_actions_are_persisted_and_visible_on_the_deal(client, session):
    await seed(session, activities=NEGLECTED)
    await client.post("/api/deals/LIVE-0001/score")

    detail = (await client.get("/api/deals/LIVE-0001")).json()
    assert detail["actions"]
    first = detail["actions"][0]
    assert first["status"] == "suggested"
    assert first["rule_id"]
    assert first["resolved_at"] is None


async def test_rescoring_does_not_duplicate_open_actions(client, session):
    """Every activity triggers a re-score; the ledger must not pile up."""
    await seed(session, activities=NEGLECTED)

    await client.post("/api/deals/LIVE-0001/score")
    first = (await client.get("/api/deals/LIVE-0001")).json()["actions"]

    for _ in range(3):
        await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": True})

    after = (await client.get("/api/deals/LIVE-0001")).json()["actions"]
    assert len(after) == len(first), "re-scoring duplicated recommendations"


async def test_top_action_surfaces_in_the_pipeline_list(client, session):
    await seed(session, activities=NEGLECTED)
    await client.post("/api/deals/LIVE-0001/score")

    body = (await client.get("/api/deals")).json()
    top = body["items"][0]["top_action"]
    assert top is not None
    assert top["reason"]
    assert top["status"] == "suggested"


# --- resolving -------------------------------------------------------------


async def _first_action_id(client) -> str:
    detail = (await client.get("/api/deals/LIVE-0001")).json()
    return detail["actions"][0]["id"]


@pytest.mark.parametrize("status", ["accepted", "dismissed"])
async def test_accept_or_dismiss_an_action(client, session, status):
    await seed(session, activities=NEGLECTED)
    await client.post("/api/deals/LIVE-0001/score")
    action_id = await _first_action_id(client)

    response = await client.patch(
        f"/api/actions/{action_id}", json={"status": status}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == status
    assert body["resolved_at"] is not None


async def test_resolved_actions_are_immutable(client, session):
    await seed(session, activities=NEGLECTED)
    await client.post("/api/deals/LIVE-0001/score")
    action_id = await _first_action_id(client)

    await client.patch(f"/api/actions/{action_id}", json={"status": "accepted"})
    second = await client.patch(f"/api/actions/{action_id}", json={"status": "dismissed"})

    assert second.status_code == 409
    assert "already accepted" in second.json()["detail"]


async def test_cannot_set_an_action_back_to_suggested(client, session):
    await seed(session, activities=NEGLECTED)
    await client.post("/api/deals/LIVE-0001/score")
    action_id = await _first_action_id(client)

    response = await client.patch(
        f"/api/actions/{action_id}", json={"status": "suggested"}
    )
    assert response.status_code == 422


async def test_dismissed_action_can_fire_again_on_the_next_score(client, session):
    """Dismissing is a decision about now, not a permanent mute."""
    await seed(session, activities=NEGLECTED)
    await client.post("/api/deals/LIVE-0001/score")

    action_id = await _first_action_id(client)
    dismissed = (
        await client.patch(f"/api/actions/{action_id}", json={"status": "dismissed"})
    ).json()

    await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": True})
    actions = (await client.get("/api/deals/LIVE-0001")).json()["actions"]

    same_rule = [a for a in actions if a["rule_id"] == dismissed["rule_id"]]
    assert len(same_rule) == 2, "a re-fired rule should create a new ledger entry"
    assert {a["status"] for a in same_rule} == {"dismissed", "suggested"}


async def test_unknown_action_is_404(client):
    response = await client.patch(
        "/api/actions/00000000-0000-0000-0000-000000000000",
        json={"status": "accepted"},
    )
    assert response.status_code == 404


# --- queue and catalogue ---------------------------------------------------


async def test_action_queue_is_ordered_by_priority(client, session):
    await seed(session, "LIVE-0001", NEGLECTED)
    await seed(session, "LIVE-0002", ENGAGED, company="Beta GmbH")
    await client.post("/api/deals/LIVE-0001/score")
    await client.post("/api/deals/LIVE-0002/score")

    body = (await client.get("/api/actions", params={"status": "suggested"})).json()
    assert body["items"]

    from app.agent.nba import PRIORITY_RANK

    ranks = [PRIORITY_RANK[i["priority"]] for i in body["items"]]
    assert ranks == sorted(ranks)
    # The queue joins the deal so a rep can triage without opening each one.
    assert body["items"][0]["company"]


async def test_action_queue_filters_by_status(client, session):
    await seed(session, activities=NEGLECTED)
    await client.post("/api/deals/LIVE-0001/score")
    action_id = await _first_action_id(client)
    await client.patch(f"/api/actions/{action_id}", json={"status": "accepted"})

    accepted = (await client.get("/api/actions", params={"status": "accepted"})).json()
    assert len(accepted["items"]) == 1
    assert accepted["items"][0]["id"] == action_id


async def test_rule_catalogue_is_exposed(client):
    body = (await client.get("/api/actions/rules")).json()
    assert body["count"] >= 10
    ids = {r["id"] for r in body["rules"]}
    assert "R-GONE-QUIET" in ids
    for rule in body["rules"]:
        assert rule["description"]
        assert rule["priority"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")


async def test_score_drop_rule_fires_across_two_real_scorings(client, session):
    """End-to-end check of the one rule that needs score history."""
    from sqlalchemy import delete

    from app.db.models import Activity

    await seed(session, activities=ENGAGED)
    first = (await client.post("/api/deals/LIVE-0001/score")).json()

    # Degrade the deal materially: the champion turns out not to be one, a
    # discount gets asked for late, and the deal goes quiet. Champion presence
    # is the model's strongest driver, so this is a large, genuine regression.
    await session.execute(
        delete(Activity).where(
            Activity.deal_id == "LIVE-0001",
            Activity.type == "champion_identified",
        )
    )
    session.add(make_activity("LIVE-0001", "discount_requested", 14))
    for day in range(15, 90, 7):
        session.add(make_activity("LIVE-0001", "no_activity_7d", day))
    await session.commit()

    second = (
        await client.post("/api/deals/LIVE-0001/score", json={"bypass_cache": True})
    ).json()

    drop = first["win_prob"] - second["win_prob"]
    assert drop > 0.10, f"expected a material regression, got {drop:.3f}"

    rules = {a["rule_id"] for a in second["actions"]}
    assert "R-SCORE-DROP" in rules, rules

    reason = next(
        a["reason"] for a in second["actions"] if a["rule_id"] == "R-SCORE-DROP"
    )
    assert "points" in reason
    # The reason must quote both ends of the move, not just say "it dropped".
    assert f"{first['win_prob']:.0%}" in reason
    assert f"{second['win_prob']:.0%}" in reason


async def test_silence_markers_do_not_disarm_the_go_quiet_rule(client, session):
    """Regression guard for the days-since-last-touch fix.

    no_activity_7d rows used to count as contact, so a deal with weeks of logged
    silence reported a recent last touch and R-GONE-QUIET stopped firing —
    exactly when it was most needed.
    """
    await seed(session, activities=ENGAGED)
    for day in range(15, 90, 7):
        session.add(make_activity("LIVE-0001", "no_activity_7d", day))
    await session.commit()

    body = (await client.post("/api/deals/LIVE-0001/score")).json()
    assert "R-GONE-QUIET" in {a["rule_id"] for a in body["actions"]}
