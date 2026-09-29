"""Measure scoring latency on a running deployment (spec section 7).

    python training/measure_latency.py --url https://your-service.onrender.com

``artifacts/metrics.json`` ships with ``latency_ms.measured: null`` on purpose.
The paper quotes 180 ms, measured on different hardware against different data;
reusing it would be inventing a result. This script produces a real number from
a real deployment and writes it back, so the README and the architecture page
quote something that was actually observed.

What it reports, and why it is not one number:

* a **cold** first call, which is what a reviewer landing on a sleeping free
  tier actually experiences
* **warm** percentiles over repeated scoring, which is the figure worth quoting
* the **per-stage** breakdown (features / xgb / lstm / shap / nba), because
  "which part is slow" is the useful question
* a **cache hit**, for contrast

Cache is bypassed on the timed calls. A cached score measures Redis, not the
model.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")

STAGES = ("features", "xgb", "lstm", "shap", "nba", "cache")


def _get(url: str, timeout: float = 120.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def pick_deals(base: str, count: int) -> list[str]:
    body = _get(f"{base}/api/deals?limit={count}&sort=created_at")
    ids = [item["id"] for item in body.get("items", [])]
    if not ids:
        raise SystemExit("no deals in the pipeline; seed the service first")
    return ids


def score(base: str, deal_id: str, bypass_cache: bool = True) -> tuple[dict, float]:
    """Returns the service's own timings and the wall-clock round trip."""
    started = time.perf_counter()
    body = _post(
        f"{base}/api/deals/{deal_id}/score", {"bypass_cache": bypass_cache}
    )
    wall_ms = (time.perf_counter() - started) * 1000
    return body, wall_ms


def summarise(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "n": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="service base URL, no trailing slash")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--deals", type=int, default=10)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the result into artifacts/metrics.json",
    )
    args = parser.parse_args()
    base = args.url.rstrip("/")

    # --- cold start -------------------------------------------------------
    print(f"waking {base} ...")
    woken_at = time.perf_counter()
    try:
        health = _get(f"{base}/healthz", timeout=180)
    except urllib.error.URLError as exc:
        raise SystemExit(f"could not reach the service: {exc}")
    wake_ms = (time.perf_counter() - woken_at) * 1000

    print(f"  /healthz {health.get('status')} in {wake_ms:.0f} ms")
    if health.get("status") != "ok":
        print(f"  warning: service reports {health.get('status')}", file=sys.stderr)

    deal_ids = pick_deals(base, args.deals)
    print(f"  {len(deal_ids)} deals available\n")

    # --- first scoring ----------------------------------------------------
    first_body, first_wall = score(base, deal_ids[0])
    print(f"first score (cold): {first_wall:.0f} ms round trip")

    # --- warm ------------------------------------------------------------
    walls: list[float] = []
    totals: list[float] = []
    stages: dict[str, list[float]] = {s: [] for s in STAGES}

    for i in range(args.samples):
        deal_id = deal_ids[i % len(deal_ids)]
        body, wall = score(base, deal_id)
        walls.append(wall)
        latency = body.get("latency_ms", {})
        totals.append(float(latency.get("total", 0.0)))
        for stage in STAGES:
            value = latency.get(stage)
            if value is not None:
                stages[stage].append(float(value))

    server = summarise(totals)
    round_trip = summarise(walls)

    print(f"\nwarm, {args.samples} scorings across {len(deal_ids)} deals")
    print(f"  server-side  p50 {server['p50']:>7.1f} ms   p95 {server['p95']:>7.1f} ms")
    print(
        f"  round trip   p50 {round_trip['p50']:>7.1f} ms   p95 "
        f"{round_trip['p95']:>7.1f} ms   (includes network)"
    )
    print("\n  per stage (p50):")
    for stage in STAGES:
        if stages[stage]:
            print(f"    {stage:<10} {statistics.median(stages[stage]):>7.2f} ms")

    # --- cache hit --------------------------------------------------------
    cached, cached_wall = score(base, deal_ids[0], bypass_cache=False)
    hit = cached.get("cache_hit")
    print(
        f"\n  cache hit: {hit} — {cached.get('latency_ms', {}).get('total')} ms "
        f"server, {cached_wall:.0f} ms round trip"
    )

    result = {
        "measured": server["p50"],
        "measured_at": time.strftime("%Y-%m-%d"),
        "host": base,
        "server_side_ms": server,
        "round_trip_ms": round_trip,
        "per_stage_p50_ms": {
            s: round(statistics.median(v), 2) for s, v in stages.items() if v
        },
        "cold_start_ms": round(first_wall, 0),
        "note": (
            "Median server-side scoring time on the hosted free tier, cache "
            "bypassed. Round trip includes network. The paper's 180 ms was "
            "measured on different hardware and is not reused."
        ),
    }

    if args.write:
        path = os.path.join(ARTIFACT_DIR, "metrics.json")
        with open(path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        metrics["latency_ms"] = result
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"\nwrote latency_ms into {path}")
    else:
        print("\n(run again with --write to record this in artifacts/metrics.json)")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
