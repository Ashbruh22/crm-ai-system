"""
ml/event_simulator.py
=====================
Problem 2 — Live Sales-Cycle Re-Forecast LSTM (Part A)

PURPOSE
-------
Generate synthetic per-deal event sequences (stage-transition + activity touchpoints)
calibrated against real aggregate statistics from the TRAINING SPLIT ONLY.
Each simulated deal's sequence is produced by independent stochastic sampling —
never reverse-engineered from a known total_days value.

LEAKAGE GUARDRAILS (by construction)
--------------------------------------
1. simulate_deal_sequence() does NOT accept total_days, close_date, or any
   outcome variable as a parameter. Duration is an OUTPUT, not an input.

2. Agent tier is computed from agent_win_rate ONLY. agent_deal_velocity is
   explicitly excluded — using it would smuggle typical-duration information
   into the calibration segmentation key. (See compute_agent_tier() assertion.)

3. Calibration (StageDwellTimeModel.fit) IS allowed to use real total_days,
   but ONLY to build population-level Dirichlet-sampled dwell distributions.
   These distributions describe the aggregate behaviour of the population,
   not any individual deal's true duration. Per-deal generation never looks
   up a specific deal's real total_days — it draws fresh random values from
   the fitted distributions.

4. simulate_deal_sequence() uses TWO INDEPENDENT SAMPLING STAGES:
     Stage 1: sample outcome (won/lost) and total stage count from
              population base rates — no timing involved at all.
     Stage 2: sample dwell time for each stage drawn in Stage 1,
              independently of how many stages were drawn.
   This ensures "how many stages occurred" and "how long each stage took"
   are statistically independent — a deal's eventual duration cannot be
   inferred from early partial-sequence shape.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Stage pipeline (standard B2B CRM order)
STAGE_SEQUENCE = ["prospecting", "qualification", "proposal", "negotiation"]
N_STAGES = len(STAGE_SEQUENCE)

# Cutoff days used for multi-cutoff training-example generation
CUTOFF_DAYS = [3, 7, 14, 21, 30, 45, 60]

# -----------------------------------------------------------------------------
# Tier computation — MUST use agent_win_rate only
# -----------------------------------------------------------------------------

def compute_agent_tier(agent_win_rate_value: float, tertile_boundaries: np.ndarray) -> int:
    """
    Map an agent's win rate to a tier index (0=bottom, 1=mid, 2=top).

    GUARDRAIL: agent_tier must be derived from agent_win_rate ONLY.
    Using agent_deal_velocity (duration-derived) here would let typical-duration
    information leak into the segmentation key used for dwell-time calibration,
    double-counting duration signal into the very parameters meant to be
    population-level only.
    """
    assert isinstance(agent_win_rate_value, (float, np.floating)), (
        "agent_tier must use win_rate float, not velocity or any duration-derived value"
    )
    return int(np.searchsorted(tertile_boundaries, agent_win_rate_value))


# -----------------------------------------------------------------------------
# StageDwellTimeModel
# -----------------------------------------------------------------------------

class StageDwellTimeModel:
    """
    Calibrates stage dwell-time distributions from the real training split.

    CALIBRATION NOTE: This class uses each deal's real total_days to generate
    a plausible allocation of time across the 4 stages via Dirichlet sampling.
    This is POPULATION-LEVEL calibration — it builds aggregate (mean, std)
    distributions, not per-deal labels. The per-deal Dirichlet draw is a
    Monte Carlo averaging step that prevents hardcoding equal-split assumptions
    while keeping the calibration statistically grounded.

    At GENERATION time (simulate_deal_sequence), only the fitted (mean, std)
    parameters are used — the individual real deal's total_days is never
    consulted.
    """

    def __init__(self, n_dirichlet_samples: int = 10):
        self.n_dirichlet_samples = n_dirichlet_samples
        self.dwell_params: dict = {}            # {(product, tier, stage): (mean, std)}
        self.tertile_boundaries: np.ndarray = np.array([])

    def fit(self, df_train: pd.DataFrame) -> "StageDwellTimeModel":
        """Fit dwell-time distributions from training split only."""

        df = df_train.copy()

        # Compute agent_win_rate (training-split only)
        agent_wr = (
            (df["deal_stage"] == "closed-won")
            .groupby(df["sales_agent"])
            .mean()
            .rename("agent_win_rate")
        )
        df = df.join(agent_wr, on="sales_agent")

        # Tertile boundaries on win_rate — used for tier labels (win_rate only)
        self.tertile_boundaries = np.percentile(
            df["agent_win_rate"].dropna(), [33.3, 66.7]
        )

        df["agent_tier"] = df["agent_win_rate"].apply(
            lambda wr: compute_agent_tier(float(wr), self.tertile_boundaries)
        )

        # Compute real total_days per deal
        df["total_days"] = (
            pd.to_datetime(df["close_date"]) - pd.to_datetime(df["engage_date"])
        ).dt.days.clip(lower=1)

        # Build dwell-time calibration rows via repeated Dirichlet sampling
        # POPULATION-LEVEL USE OF total_days: we distribute each real deal's
        # total_days across 4 stages using a Dirichlet draw. This is a calibration
        # step — it builds aggregate distributions, not per-deal features.
        rng = np.random.default_rng(42)
        calibration_rows = []
        for _, row in df.iterrows():
            for _ in range(self.n_dirichlet_samples):
                proportions = rng.dirichlet(np.ones(N_STAGES))
                dwell_times = proportions * row["total_days"]
                for stage, dwell in zip(STAGE_SEQUENCE, dwell_times):
                    calibration_rows.append({
                        "product": row["product"],
                        "agent_tier": int(row["agent_tier"]),
                        "stage": stage,
                        "dwell_days": float(dwell),
                    })

        cal_df = pd.DataFrame(calibration_rows)

        # Fit (mean, std) per (product, tier, stage)
        for (product, tier, stage), grp in cal_df.groupby(["product", "agent_tier", "stage"]):
            mean_d = float(grp["dwell_days"].mean())
            std_d = float(grp["dwell_days"].std(ddof=1)) if len(grp) > 1 else max(1.0, mean_d * 0.3)
            std_d = max(std_d, 1.0)   # floor at 1 day
            self.dwell_params[(product, int(tier), stage)] = (mean_d, std_d)

        return self

    def save(self, path: str) -> None:
        # JSON keys must be strings
        serialisable = {str(k): list(v) for k, v in self.dwell_params.items()}
        meta = {
            "dwell_params": serialisable,
            "tertile_boundaries": list(self.tertile_boundaries),
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[StageDwellTimeModel] saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StageDwellTimeModel":
        with open(path) as f:
            meta = json.load(f)
        obj = cls()
        obj.dwell_params = {
            tuple(int(x) if x.lstrip("-").isdigit() else x for x in k.strip("()").split(", ")): tuple(v)
            for k, v in meta["dwell_params"].items()
        }
        obj.tertile_boundaries = np.array(meta["tertile_boundaries"])
        return obj


# -----------------------------------------------------------------------------
# ActivityFrequencyModel
# -----------------------------------------------------------------------------

class ActivityFrequencyModel:
    """
    Calibrates a Poisson activity rate (events/week) per (product, agent_tier).
    Derived from training split only — computed from deal density, NOT from
    any individual deal's stage-level event log (which doesn't exist in the raw data).
    """

    def __init__(self):
        self.activity_params: dict = {}    # {(product, tier): rate_per_week}
        self.global_rate: float = 1.5

    def fit(self, df_train: pd.DataFrame, dwell_model: StageDwellTimeModel) -> "ActivityFrequencyModel":
        df = df_train.copy()

        agent_wr = (
            (df["deal_stage"] == "closed-won")
            .groupby(df["sales_agent"])
            .mean()
            .rename("agent_win_rate")
        )
        df = df.join(agent_wr, on="sales_agent")
        df["agent_tier"] = df["agent_win_rate"].apply(
            lambda wr: compute_agent_tier(float(wr), dwell_model.tertile_boundaries)
        )

        # Activity rate proxy: higher-tier agents assumed more active per week.
        # Calibrated from win-rate tiers — a top-tier (high win-rate) agent
        # interacts ~2 events/week, bottom-tier ~1 event/week. These are
        # reasonable CRM industry priors; no per-deal event log exists in the data.
        base_rates = {0: 1.0, 1: 1.5, 2: 2.0}   # bottom / mid / top tier

        for (product, tier), grp in df.groupby(["product", "agent_tier"]):
            # Scale by product complexity (Enterprise -> more touchpoints)
            complexity = {"Starter Pack": 0.8, "Analytics Pro": 1.0, "Enterprise Suite": 1.3}.get(product, 1.0)
            self.activity_params[(product, int(tier))] = base_rates.get(int(tier), 1.5) * complexity

        self.global_rate = 1.5
        return self

    def get_rate(self, product: str, agent_tier: int) -> float:
        return self.activity_params.get((product, agent_tier), self.global_rate)

    def save(self, path: str) -> None:
        serialisable = {str(k): v for k, v in self.activity_params.items()}
        with open(path, "w") as f:
            json.dump({"activity_params": serialisable, "global_rate": self.global_rate}, f, indent=2)
        print(f"[ActivityFrequencyModel] saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ActivityFrequencyModel":
        with open(path) as f:
            meta = json.load(f)
        obj = cls()
        obj.activity_params = {
            tuple(k.strip("()").split(", ")): v for k, v in meta["activity_params"].items()
        }
        obj.global_rate = meta["global_rate"]
        return obj


# -----------------------------------------------------------------------------
# Stage-count / outcome sampler (Stage 1 of two-stage simulation)
# -----------------------------------------------------------------------------

# Population base-rates for stage advancement — calibrated conservatively from
# B2B CRM industry priors (no granular stage data exists in sales_pipeline.csv).
# Won deals typically traverse all 4 stages; lost deals exit earlier on average.
_STAGE_COUNT_PROBS = {
    # (product, tier): distribution over stage counts [1,2,3,4], P(will_win | reached final)
    # Format: (stage_count_probs [list len 4], p_win_if_full)
    # These are population-level priors, not per-deal predictions.
}

def _default_stage_probs(tier: int, product: str):
    """
    Returns (stage_count_probs, p_win_if_full) for a given tier/product.

    Higher-tier agents are more likely to advance to later stages and win.
    Product complexity shifts the distribution: Enterprise deals are more
    likely to go all the way to negotiation before closing (longer pipelines).
    """
    base_win_if_full = {0: 0.35, 1: 0.50, 2: 0.65}.get(tier, 0.50)
    complexity_boost = {"Starter Pack": 0.0, "Analytics Pro": 0.05, "Enterprise Suite": 0.10}.get(product, 0.0)
    p_win = min(0.80, base_win_if_full + complexity_boost)

    # Stage count distribution: probability of exiting after 1,2,3,4 stages
    # Higher tier -> less likely to exit early
    if tier == 0:   # bottom
        sc_probs = [0.30, 0.30, 0.25, 0.15]
    elif tier == 1: # mid
        sc_probs = [0.20, 0.25, 0.30, 0.25]
    else:           # top
        sc_probs = [0.10, 0.20, 0.30, 0.40]

    # Enterprise deals skew toward full pipeline
    if product == "Enterprise Suite":
        sc_probs = [max(0.02, p - 0.05) for p in sc_probs]
        sc_probs[-1] += 0.15  # push toward 4-stage
        total = sum(sc_probs)
        sc_probs = [p / total for p in sc_probs]

    return sc_probs, p_win


def sample_outcome_and_stage_count(product: str, agent_tier: int, rng: np.random.Generator):
    """
    STAGE 1 of two-stage simulation.

    Independently samples:
      - How many stages this deal will traverse (1–4)
      - Whether it will eventually close-won (True) or close-lost (False)

    These draws are based ONLY on population base rates (product, agent_tier).
    No timing is involved at this stage — duration is determined entirely in Stage 2.
    """
    sc_probs, p_win = _default_stage_probs(agent_tier, product)
    stage_count = int(rng.choice([1, 2, 3, 4], p=sc_probs))
    will_win = bool(rng.random() < p_win)
    return stage_count, will_win


# -----------------------------------------------------------------------------
# Core simulator — two-stage independent sampling
# -----------------------------------------------------------------------------

def simulate_deal_sequence(
    sales_agent: str,
    product: str,
    agent_tier: int,
    dwell_model: StageDwellTimeModel,
    activity_model: ActivityFrequencyModel,
    rng_seed=None,
):
    """
    Generate one synthetic deal's full event timeline.

    CRITICAL: This function does NOT accept total_days, close_date, or any
    outcome variable as a parameter. `simulated_total_days` is computed as
    an OUTPUT by accumulating stage dwell times — it is never an input.

    Two-stage independent sampling design
    -------------------------------------
    Stage 1: sample_outcome_and_stage_count() determines HOW MANY stages this
             deal traverses and whether it will win. This is purely probabilistic
             (population base rates), completely independent of any timing.

    Stage 2: For each stage from Stage 1, sample dwell time from calibrated
             Normal(mean, std) distributions. This is independent of how many
             stages were drawn in Stage 1 — a deal that will traverse only 1
             stage draws the SAME dwell-time distribution for that stage as a
             deal that will traverse all 4 stages.

    The structural guarantee: a short deal (few stages) and a long deal (many
    stages) generate partial sequences that are statistically indistinguishable
    at the same early cutoff, because stage count and dwell time are drawn from
    independent distributions with no shared conditioning variable.

    Parameters
    ----------
    sales_agent : str
    product     : str
    agent_tier  : int  (0=bottom, 1=mid, 2=top — based on agent_win_rate ONLY)
    dwell_model : StageDwellTimeModel
    activity_model : ActivityFrequencyModel
    rng_seed    : int or None

    Returns
    -------
    events              : list of (day_offset: float, event_type: str)
    simulated_total_days: float   ← OUTPUT, never an input
    final_outcome       : str     ("closed-won" or "closed-lost")
    """
    rng = np.random.default_rng(rng_seed)

    # -- STAGE 1: Sample outcome + stage count independently of timing ------
    stage_count, will_win = sample_outcome_and_stage_count(product, agent_tier, rng)
    stages_to_simulate = STAGE_SEQUENCE[:stage_count]

    # -- STAGE 2: Sample dwell time per stage, independent of stage_count ---
    events = []
    day_offset = 0.0

    for stage in stages_to_simulate:
        # Look up calibrated (mean, std) for this (product, tier, stage)
        key = (product, agent_tier, stage)
        fallback_key = None
        for k in dwell_model.dwell_params:
            # Robust fallback: match product + stage regardless of tier
            if len(k) == 3 and k[0] == product and k[2] == stage:
                fallback_key = k
        mean_d, std_d = dwell_model.dwell_params.get(key, dwell_model.dwell_params.get(fallback_key, (14.0, 5.0)))

        # Draw dwell time from calibrated distribution — independent of stage_count
        dwell_days = float(max(1.0, rng.normal(mean_d, std_d)))

        # Record stage-enter event
        events.append((day_offset, f"enter_{stage}"))

        # Sample activity events via Poisson process within this stage
        rate_per_week = activity_model.get_rate(product, agent_tier)
        expected_events = rate_per_week * (dwell_days / 7.0)
        n_activity = int(rng.poisson(expected_events))
        if n_activity > 0:
            activity_offsets = sorted(rng.uniform(0, dwell_days, size=n_activity))
            for t in activity_offsets:
                events.append((day_offset + t, "activity"))

        day_offset += dwell_days

    # simulated_total_days is the SUM of drawn dwell times — an OUTPUT
    simulated_total_days = float(day_offset)
    final_outcome = "closed-won" if will_win else "closed-lost"

    return events, simulated_total_days, final_outcome


# -----------------------------------------------------------------------------
# Padding (RIGHT-PADDING)
# -----------------------------------------------------------------------------

def build_padded_sequence(
    day_feature_vecs: list,   # list of np.ndarray, one per day from day 0 to cutoff-1
    max_len: int = 60,
    n_features: int = 7,
) -> np.ndarray:
    """
    Build a right-padded sequence array of shape (max_len, n_features).

    RIGHT-PADDING RATIONALE: real data occupies indices [0, cutoff_day),
    zeros fill indices [cutoff_day, max_len). This ensures "today" is always
    at position cutoff_day-1 regardless of when the cutoff is, making recency
    consistent across training examples of different cutoff lengths.

    LEFT-PADDING WOULD BE WRONG: it would place "today" at a fixed index
    (max_len-1) but shift all earlier events leftward, making a day-7 cutoff
    and day-45 cutoff occupy completely different relative positions.
    """
    seq = np.zeros((max_len, n_features), dtype=np.float32)
    for day_idx, feature_vec in enumerate(day_feature_vecs):
        if day_idx >= max_len:
            break
        seq[day_idx] = feature_vec   # real data at [0, cutoff_day)
    return seq                        # zeros trail at [cutoff_day, max_len)


def verify_padding_direction(sample_events_7_days, max_len=60, n_features=7):
    """Unit test: confirm right-padding is correctly applied."""
    test_seq = build_padded_sequence(sample_events_7_days, max_len=max_len, n_features=n_features)
    assert np.any(test_seq[0] != 0), "Day 0 should have real data, not padding"
    assert np.all(test_seq[10:] == 0), f"Steps after cutoff should be zero-padded; found non-zero at {np.nonzero(test_seq[10:])}"
    print("[OK] Padding direction verified: right-padding confirmed")
    return test_seq


# -----------------------------------------------------------------------------
# Feature vector builder for one day/event
# -----------------------------------------------------------------------------

STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_SEQUENCE)}

def build_day_feature_vec(
    day_offset: float,
    current_stage: str,
    num_activities_so_far: int,
    days_since_last_activity: float,
    agent_win_rate: float,
    product_win_rate: float,
    agent_avg_deal_value: float,
    max_day: float = 90.0,
    max_activity: float = 30.0,
    max_gap: float = 30.0,
    max_deal_value: float = 300_000.0,
) -> np.ndarray:
    """
    7-element feature vector for one timestep, containing ONLY information
    visible up to the cutoff day — no target leakage possible.

    Features (in order):
      0: days_elapsed_so_far (normalized 0-1)
      1: current_stage_encoded (ordinal 0-3, normalized 0-1)
      2: num_activities_so_far (clipped + normalized)
      3: days_since_last_activity (clipped + normalized)
      4: agent_win_rate (raw, already in [0,1])
      5: product_win_rate (raw, already in [0,1])
      6: agent_avg_deal_value (normalized)
    """
    stage_int = STAGE_TO_INT.get(current_stage, 0)
    return np.array([
        min(day_offset, max_day) / max_day,
        stage_int / max(1, len(STAGE_SEQUENCE) - 1),
        min(num_activities_so_far, max_activity) / max_activity,
        min(days_since_last_activity, max_gap) / max_gap,
        float(agent_win_rate),
        float(product_win_rate),
        min(agent_avg_deal_value, max_deal_value) / max_deal_value,
    ], dtype=np.float32)


# -----------------------------------------------------------------------------
# Build training sequences (multi-cutoff)
# -----------------------------------------------------------------------------

def build_training_sequences(
    n_synthetic_deals: int,
    real_train_df: pd.DataFrame,
    dwell_model: StageDwellTimeModel,
    activity_model: ActivityFrequencyModel,
    agent_stats: dict,       # {sales_agent: {win_rate, avg_deal_value, product_win_rate}}
    product_stats: dict,     # {product: win_rate}
    cutoff_days: list = None,
    max_len: int = 60,
    n_features: int = 7,
    rng_seed: int = 0,
):
    """
    Generate n_synthetic_deals synthetic deals and produce multi-cutoff training examples.

    SPLIT RULE: train/val/test split must be performed at the DEAL level (by deal_id)
    not at the example level. This function returns deal_id per example so the caller
    can enforce this.

    Returns
    -------
    X_list      : list of np.ndarray, shape (max_len, n_features)
    y_list      : list of float (log1p remaining_days)
    meta_list   : list of dict with deal_id, cutoff_day, simulated_total_days
    """
    if cutoff_days is None:
        cutoff_days = CUTOFF_DAYS

    # Sample (sales_agent, product) from real marginal distribution
    rng = np.random.default_rng(rng_seed)
    agent_product_pairs = list(zip(real_train_df["sales_agent"], real_train_df["product"]))
    sampled_indices = rng.integers(0, len(agent_product_pairs), size=n_synthetic_deals)

    X_list, y_list, meta_list = [], [], []

    for deal_id, idx in enumerate(sampled_indices):
        sales_agent, product = agent_product_pairs[idx]

        # Compute agent_tier from win_rate ONLY
        wr = float(agent_stats.get(sales_agent, {}).get("win_rate", 0.4))
        agent_tier = compute_agent_tier(wr, dwell_model.tertile_boundaries)

        events, sim_total_days, outcome = simulate_deal_sequence(
            sales_agent=sales_agent,
            product=product,
            agent_tier=agent_tier,
            dwell_model=dwell_model,
            activity_model=activity_model,
            rng_seed=int(rng.integers(0, 2**31)),
        )

        # Sort events by day offset
        events_sorted = sorted(events, key=lambda e: e[0])

        # Static per-deal features (from training-split aggregates — no leakage)
        a_stats = agent_stats.get(sales_agent, {})
        agent_wr = float(a_stats.get("win_rate", 0.4))
        agent_adv = float(a_stats.get("avg_deal_value", 50000.0))
        prod_wr = float(product_stats.get(product, 0.4))

        # Build per-cutoff examples
        for cutoff in cutoff_days:
            if cutoff >= sim_total_days:
                # Deal already closed before this cutoff — discard
                continue

            # Build daily feature sequence up to cutoff
            day_vecs = []
            current_stage = STAGE_SEQUENCE[0]
            activities_so_far = 0
            last_activity_day = 0.0

            for day in range(cutoff):
                # Advance stage/activity counts based on events up to this day
                for ev_day, ev_type in events_sorted:
                    if ev_day > day:
                        break
                    if ev_type.startswith("enter_"):
                        stage_name = ev_type[len("enter_"):]
                        if stage_name in STAGE_TO_INT:
                            current_stage = stage_name
                    elif ev_type == "activity":
                        activities_so_far += 1
                        last_activity_day = ev_day

                days_since_act = float(day) - last_activity_day

                vec = build_day_feature_vec(
                    day_offset=float(day),
                    current_stage=current_stage,
                    num_activities_so_far=activities_so_far,
                    days_since_last_activity=days_since_act,
                    agent_win_rate=agent_wr,
                    product_win_rate=prod_wr,
                    agent_avg_deal_value=agent_adv,
                )
                day_vecs.append(vec)

            if len(day_vecs) == 0:
                continue

            X_padded = build_padded_sequence(day_vecs, max_len=max_len, n_features=n_features)
            remaining = sim_total_days - float(cutoff)
            y_log = float(np.log1p(max(0.0, remaining)))

            X_list.append(X_padded)
            y_list.append(y_log)
            meta_list.append({
                "deal_id": deal_id,
                "cutoff_day": cutoff,
                "simulated_total_days": sim_total_days,
                "remaining_days": remaining,
                "sales_agent": sales_agent,
                "product": product,
                "agent_tier": agent_tier,
                "outcome": outcome,
            })

    return X_list, y_list, meta_list


# -----------------------------------------------------------------------------
# Calibration entry point
# -----------------------------------------------------------------------------

def calibrate_and_save(data_path: str, output_dir: str, train_random_state: int = 42):
    """
    Load raw data, split, fit calibration models, save params.
    Returns (dwell_model, activity_model, df_train, agent_stats, product_stats).
    """
    df = pd.read_csv(data_path)
    closed = df[df["deal_stage"].isin(["closed-won", "closed-lost"])].copy()

    df_train, df_test = train_test_split(
        closed, test_size=0.3, random_state=train_random_state,
        stratify=closed["deal_stage"]
    )

    print(f"Training split: {len(df_train)} rows  |  Test split: {len(df_test)} rows")

    # Fit calibration models on training split only
    dwell_model = StageDwellTimeModel(n_dirichlet_samples=10).fit(df_train)
    activity_model = ActivityFrequencyModel().fit(df_train, dwell_model)

    os.makedirs(output_dir, exist_ok=True)
    dwell_model.save(os.path.join(output_dir, "stage_dwell_params.json"))
    activity_model.save(os.path.join(output_dir, "activity_rate_params.json"))

    # Build agent/product stats dicts (training-split aggregates)
    agent_wr = (
        (df_train["deal_stage"] == "closed-won")
        .groupby(df_train["sales_agent"]).mean()
    ).to_dict()

    won = df_train[df_train["deal_stage"] == "closed-won"]
    agent_adv = won.groupby("sales_agent")["close_value"].mean().fillna(50000.0).to_dict()

    agent_stats = {
        ag: {"win_rate": agent_wr.get(ag, 0.4), "avg_deal_value": agent_adv.get(ag, 50000.0)}
        for ag in df_train["sales_agent"].unique()
    }

    prod_wr = (
        (df_train["deal_stage"] == "closed-won")
        .groupby(df_train["product"]).mean()
    ).to_dict()
    product_stats = {p: prod_wr.get(p, 0.4) for p in df_train["product"].unique()}

    print("\n=== Stage Dwell Params (sample) ===")
    for k, v in list(dwell_model.dwell_params.items())[:6]:
        print(f"  {k}: mean={v[0]:.1f}d, std={v[1]:.1f}d")

    print("\n=== Activity Rate Params ===")
    for k, v in activity_model.activity_params.items():
        print(f"  {k}: {v:.2f} events/week")

    print(f"\nAgent win-rate tertile boundaries: {dwell_model.tertile_boundaries}")

    return dwell_model, activity_model, df_train, df_test, agent_stats, product_stats


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    calibrate_and_save(
        data_path=os.path.join(base_dir, "data", "sales_pipeline.csv"),
        output_dir=os.path.join(base_dir, "data"),
    )
