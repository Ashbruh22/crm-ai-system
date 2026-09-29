"""Seeded synthetic CRM generator (spec section 4).

Produces, under ``data/synthetic/``:

* ``deals.csv``       ~2,000 closed historical deals for training
* ``activities.csv``  ~40,000 activity events for those deals
* ``live_deals.csv``  ~60 open deals for the hosted demo
* ``live_activities.csv``

Nothing here touches real CRM data. Companies and reps come from Faker under a
fixed seed, so every run of ``make train`` reproduces byte-identical files.

Generative model
----------------
Each deal gets a latent ``quality`` (buyer intent) that is never exposed as a
feature. Quality drives which activities occur; the *observed* activities then
drive the outcome. That ordering matters: it means the features the model sees
genuinely carry the signal, rather than the label being written in directly.

Win log-odds rise with replies, meetings, demos and a named champion, and fall
with silence gaps and a late-cycle discount request. Cycle length grows with
deal size and stakeholder count. Noise is deliberate (``NOISE_SD``) so metrics
land in a believable band instead of looking suspiciously perfect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features.build import (  # noqa: E402
    INDUSTRIES,
    REGIONS,
    SOURCES,
    STAGES,
)

SEED = 42
N_HISTORICAL = 2000
N_LIVE = 60
N_REPS = 12

#: Residual noise on the win log-odds. Tuned so AUC lands near 0.85, not 0.99.
NOISE_SD = 1.15

#: "Today" for the generated world. Fixed so output is reproducible.
WORLD_NOW = datetime(2026, 9, 1)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "synthetic"
)

# Deal-size distribution per industry: (lognormal mean, sigma) over dollars.
INDUSTRY_SIZE = {
    "Technology": (10.6, 0.8),
    "Financial Services": (11.2, 0.75),
    "Healthcare": (10.9, 0.7),
    "Manufacturing": (10.8, 0.8),
    "Retail": (10.2, 0.85),
    "Public Sector": (11.4, 0.6),
}

# Lead source shifts baseline intent: inbound leads arrive warmer.
SOURCE_QUALITY = {
    "Inbound": 0.45,
    "Partner": 0.25,
    "Event": 0.0,
    "Outbound": -0.35,
}

# Win log-odds coefficients over OBSERVED features.
W_INTERCEPT = -2.70
W_REPLIES = 0.34
W_MEETINGS = 0.30
W_DEMOS = 0.42
W_PROPOSALS = 0.28
W_CHAMPION = 0.95
W_STAKEHOLDERS = 0.12
W_SILENCE = -0.46
W_DISCOUNT_LATE = -0.72
W_DAYS_SILENT = -0.030
W_REP_SKILL = 1.05
W_LOG_SIZE = -0.22


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _make_reps(fake: Faker, rng: np.random.Generator) -> pd.DataFrame:
    """Reps with a latent skill that shifts their deals' win odds."""
    names, seen = [], set()
    while len(names) < N_REPS:
        name = fake.name()
        if name not in seen:
            seen.add(name)
            names.append(name)
    return pd.DataFrame(
        {
            "owner_rep": names,
            # Centred so the cohort's average rep is neutral.
            "skill": rng.normal(0.0, 0.22, size=N_REPS),
        }
    )


def _simulate_activities(
    deal_id: str,
    created_at: datetime,
    quality: float,
    horizon_days: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Walk the deal forward a week at a time, emitting activity events.

    ``quality`` (roughly -1..+2) raises the chance of buyer-side engagement and
    lowers the chance of the deal going quiet.
    """
    events: list[dict] = []
    cursor = created_at
    week = 0
    champion = False
    proposal_sent = False
    engagement_run = 0

    p_engage = _sigmoid(0.45 * quality - 0.25)

    while (cursor - created_at).days < horizon_days:
        week += 1
        cursor = created_at + timedelta(days=7 * week)
        if (cursor - created_at).days > horizon_days:
            break

        # Rep outreach is near-constant; it says little about the deal.
        n_emails = rng.poisson(0.95)
        for _ in range(int(n_emails)):
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "email_sent",
                    "occurred_at": cursor
                    + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        touched = False

        # Buyer replies: the cheapest real signal of interest.
        if rng.random() < p_engage:
            touched = True
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "email_replied",
                    "occurred_at": cursor + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        if rng.random() < p_engage * 0.55:
            touched = True
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "meeting_held",
                    "occurred_at": cursor + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        # A demo needs some relationship first.
        if week >= 2 and rng.random() < p_engage * 0.30:
            touched = True
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "demo_done",
                    "occurred_at": cursor + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        if rng.random() < 0.10 + 0.14 * max(0.0, quality):
            touched = True
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "stakeholder_added",
                    "occurred_at": cursor + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        # A champion emerges only after sustained engagement.
        engagement_run = engagement_run + 1 if touched else 0
        if not champion and engagement_run >= 2 and rng.random() < 0.26 + 0.20 * quality:
            champion = True
            touched = True
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "champion_identified",
                    "occurred_at": cursor + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        if not proposal_sent and week >= 3 and rng.random() < 0.18 + 0.12 * quality:
            proposal_sent = True
            touched = True
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "proposal_sent",
                    "occurred_at": cursor + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        # Discounts get asked for once there is something to discount.
        if proposal_sent and rng.random() < 0.13:
            touched = True
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "discount_requested",
                    "occurred_at": cursor + timedelta(hours=float(rng.uniform(0, 120))),
                }
            )

        # A silent week is logged explicitly — it is a feature, not an absence.
        if not touched and rng.random() < 0.55:
            events.append(
                {
                    "deal_id": deal_id,
                    "type": "no_activity_7d",
                    "occurred_at": cursor,
                }
            )

    return events


def _summarise(events: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    return counts


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    reps = _make_reps(fake, rng)
    skill_by_rep = dict(zip(reps["owner_rep"], reps["skill"]))

    company_pool = []
    seen_companies = set()
    while len(company_pool) < N_HISTORICAL + N_LIVE:
        name = fake.company()
        if name not in seen_companies:
            seen_companies.add(name)
            company_pool.append(name)

    deal_rows: list[dict] = []
    activity_rows: list[dict] = []
    live_deal_rows: list[dict] = []
    live_activity_rows: list[dict] = []

    total = N_HISTORICAL + N_LIVE
    for i in range(total):
        is_live = i >= N_HISTORICAL
        deal_id = f"LIVE-{i - N_HISTORICAL + 1:04d}" if is_live else f"D-{i + 1:05d}"

        industry = str(rng.choice(INDUSTRIES))
        region = str(rng.choice(REGIONS, p=[0.42, 0.30, 0.18, 0.10]))
        source = str(rng.choice(SOURCES, p=[0.30, 0.34, 0.20, 0.16]))
        rep = str(rng.choice(reps["owner_rep"]))
        mu, sigma = INDUSTRY_SIZE[industry]
        deal_size = float(np.round(rng.lognormal(mu, sigma), 2))

        quality = (
            rng.normal(0.0, 0.85)
            + SOURCE_QUALITY[source]
            + 1.4 * skill_by_rep[rep]
        )

        # Live deals are young and still open; historical deals ran to a close.
        if is_live:
            age = int(rng.integers(9, 130))
            created_at = WORLD_NOW - timedelta(days=age)
            horizon = age
        else:
            created_at = WORLD_NOW - timedelta(days=int(rng.integers(150, 900)))
            horizon = int(np.clip(rng.normal(75, 30), 21, 260))

        events = _simulate_activities(deal_id, created_at, quality, horizon, rng)
        counts = _summarise(events)

        n_stakeholders = 1 + counts.get("stakeholder_added", 0)
        silence = counts.get("no_activity_7d", 0)

        # Days of silence at the end of the window.
        if events:
            last_at = max(e["occurred_at"] for e in events)
        else:
            last_at = created_at
        end_at = created_at + timedelta(days=horizon)
        days_silent = max(0.0, (end_at - last_at).total_seconds() / 86400.0)

        # Cycle length: bigger deals and more stakeholders take longer.
        cycle_days = (
            32.0
            + 9.5 * np.log1p(deal_size / 1000.0)
            + 6.2 * n_stakeholders
            + 2.6 * silence
            + rng.normal(0, 9.0)
        )
        cycle_days = float(np.clip(cycle_days, 12, 300))

        stage_idx = int(
            np.clip(
                round(
                    len(STAGES)
                    * min(1.0, horizon / max(cycle_days, 1.0))
                    * (0.55 + 0.30 * _sigmoid(quality))
                ),
                0,
                len(STAGES) - 1,
            )
        )
        stage = STAGES[stage_idx]

        discount_late = 1.0 if (
            counts.get("discount_requested", 0) and stage_idx >= STAGES.index("Proposal")
        ) else 0.0

        base = {
            "id": deal_id,
            "company": company_pool[i],
            "industry": industry,
            "region": region,
            "deal_size": deal_size,
            "source": source,
            "owner_rep": rep,
            "created_at": created_at.isoformat(timespec="seconds"),
            "expected_close": (
                created_at + timedelta(days=round(cycle_days))
            ).isoformat(timespec="seconds"),
        }

        if is_live:
            base["stage"] = stage
            base["closed_at"] = None
            base["won"] = None
            base["cycle_days"] = None
            live_deal_rows.append(base)
            live_activity_rows.extend(events)
            continue

        # Outcome from the OBSERVED activity counts, plus noise.
        logit = (
            W_INTERCEPT
            + W_REPLIES * counts.get("email_replied", 0)
            + W_MEETINGS * counts.get("meeting_held", 0)
            + W_DEMOS * counts.get("demo_done", 0)
            + W_PROPOSALS * counts.get("proposal_sent", 0)
            + W_CHAMPION * (1 if counts.get("champion_identified", 0) else 0)
            + W_STAKEHOLDERS * n_stakeholders
            + W_SILENCE * silence
            + W_DISCOUNT_LATE * discount_late
            + W_DAYS_SILENT * days_silent
            + W_REP_SKILL * skill_by_rep[rep]
            + W_LOG_SIZE * np.log1p(deal_size / 1000.0)
            + rng.normal(0, NOISE_SD)
        )
        won = bool(rng.random() < _sigmoid(logit))

        closed_at = created_at + timedelta(days=round(cycle_days))
        base["stage"] = "Closed Won" if won else "Closed Lost"
        base["closed_at"] = closed_at.isoformat(timespec="seconds")
        base["won"] = int(won)
        base["cycle_days"] = round(cycle_days, 2)
        deal_rows.append(base)
        activity_rows.extend(events)

    def _activity_frame(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if not len(df):
            return pd.DataFrame(columns=["id", "deal_id", "type", "occurred_at"])
        df = df.sort_values(["deal_id", "occurred_at"]).reset_index(drop=True)
        df["occurred_at"] = df["occurred_at"].apply(
            lambda d: d.isoformat(timespec="seconds")
        )
        df.insert(0, "id", [f"A-{i + 1:07d}" for i in range(len(df))])
        return df

    return {
        "deals": pd.DataFrame(deal_rows),
        "activities": _activity_frame(activity_rows),
        "live_deals": pd.DataFrame(live_deal_rows),
        "live_activities": _activity_frame(live_activity_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = generate(args.seed)

    for name, df in frames.items():
        path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {name + '.csv':22} {len(df):>7,} rows -> {path}")

    deals = frames["deals"]
    win_rate = deals["won"].mean()
    manifest = {
        "seed": args.seed,
        "world_now": WORLD_NOW.isoformat(),
        "n_historical_deals": int(len(deals)),
        "n_activities": int(len(frames["activities"])),
        "n_live_deals": int(len(frames["live_deals"])),
        "n_live_activities": int(len(frames["live_activities"])),
        "historical_win_rate": round(float(win_rate), 4),
        "mean_cycle_days": round(float(deals["cycle_days"].mean()), 2),
        "activities_per_deal": round(
            len(frames["activities"]) / max(1, len(deals)), 2
        ),
        "synthetic": True,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(
        f"\n  win rate {win_rate:.1%} | mean cycle "
        f"{manifest['mean_cycle_days']:.0f}d | "
        f"{manifest['activities_per_deal']:.1f} activities/deal"
    )


if __name__ == "__main__":
    main()
