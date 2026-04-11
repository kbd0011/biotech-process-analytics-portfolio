"""
Simulate multivariate time-series trajectories and end-of-run outcomes
for a CAR-T-like cell-expansion process.

Data provenance: All data is synthetic, seeded from published CAR-T QbD
studies and release-testing concepts. No proprietary data is used.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "design_space.yml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def simulate_single_batch(
    batch_id: str,
    factors: dict,
    failure_mode: str | None,
    rng: np.random.Generator,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Simulate one batch trajectory and return (trajectory_df, outcome_dict)."""
    proc = config["process"]
    n_days = int(proc["duration_days"])
    spd = int(proc["samples_per_day"])
    n_points = n_days * spd
    noise = float(proc["noise_level"])
    t = np.linspace(0, n_days, n_points)

    n_act = int(factors.get("n_activations", 2))
    seed_d = float(factors.get("seeding_density_e6_ml", 1.0))
    il2 = float(factors.get("il2_iu_ml", 200))
    hold = float(factors.get("hold_time_hrs", 2))
    feed = factors.get("feed_strategy", "bolus")

    # Growth rate influenced by factors
    mu_max = 0.3 + 0.02 * il2 / 200 - 0.05 * max(0, n_act - 2)
    carrying_cap = 5.0 * seed_d

    # VCD: logistic growth
    vcd = carrying_cap / (1 + (carrying_cap / seed_d - 1) * np.exp(-mu_max * t))
    vcd += rng.normal(0, noise * 0.3, n_points)
    vcd = np.clip(vcd, 0.05, 20)

    # Glucose: decreasing with growth
    glucose = 25 - 1.5 * vcd + rng.normal(0, noise * 2, n_points)
    if feed == "continuous":
        glucose += 3  # continuous feed maintains higher glucose
    glucose = np.clip(glucose, 0.5, 30)

    # Lactate: increasing with growth
    lactate = 2 + 1.2 * vcd + rng.normal(0, noise * 1.5, n_points)
    lactate = np.clip(lactate, 0.1, 25)

    # Viability: starts high, may decline
    viability = 95 - 0.3 * t - 0.5 * np.maximum(0, lactate - 12) + rng.normal(0, noise * 2, n_points)
    viability = np.clip(viability, 20, 99.9)

    # pH: slight decrease with lactate
    ph = 7.35 - 0.01 * lactate + rng.normal(0, noise * 0.05, n_points)
    ph = np.clip(ph, 6.5, 7.8)

    # DO: maintained but drops under high growth
    do_pct = 45 - 2 * vcd + rng.normal(0, noise * 3, n_points)
    do_pct = np.clip(do_pct, 10, 80)

    # Temperature: tightly controlled
    temperature = 37.0 + rng.normal(0, 0.2, n_points)

    # Cumulative feed
    feed_rate = 5.0 if feed == "continuous" else 0.0
    cum_feed = np.cumsum(np.full(n_points, feed_rate / spd)) + rng.normal(0, 0.5, n_points).cumsum() * 0.1
    cum_feed = np.clip(cum_feed, 0, 500)

    # Potency surrogate
    potency = 35 + 5 * np.log1p(vcd) - 3 * np.maximum(0, n_act - 2) + rng.normal(0, noise * 3, n_points)
    potency = np.clip(potency, 5, 90)

    # --- Apply failure modes ---
    if failure_mode == "over_activation":
        vcd *= 0.6
        viability -= 15 * (t / n_days)
        potency -= 20
    elif failure_mode == "nutrient_limitation":
        glucose[n_points // 2:] -= 8
        glucose = np.clip(glucose, 0.2, 30)
        lactate[n_points // 2:] += 5
        vcd[n_points // 2:] *= 0.7
    elif failure_mode == "delayed_harvest":
        # Last 20% shows viability crash
        crash_start = int(n_points * 0.8)
        viability[crash_start:] -= np.linspace(0, 25, n_points - crash_start)
        potency[crash_start:] -= np.linspace(0, 15, n_points - crash_start)
    elif failure_mode == "contamination_upset":
        event_pt = rng.integers(n_points // 3, 2 * n_points // 3)
        viability[event_pt:] -= np.linspace(0, 40, n_points - event_pt)
        vcd[event_pt:] *= np.linspace(1, 0.3, n_points - event_pt)
        ph[event_pt:] -= np.linspace(0, 0.5, n_points - event_pt)
    elif failure_mode == "low_transduction":
        potency -= 20

    viability = np.clip(viability, 10, 99.9)
    vcd = np.clip(vcd, 0.01, 20)
    potency = np.clip(potency, 2, 95)

    trajectory = pd.DataFrame({
        "batch_id": batch_id,
        "time_day": np.round(t, 3),
        "sample_idx": range(n_points),
        "glucose_mm": np.round(glucose, 2),
        "lactate_mm": np.round(lactate, 2),
        "viable_cell_density_e6_ml": np.round(vcd, 3),
        "viability_pct": np.round(viability, 1),
        "ph": np.round(ph, 3),
        "do_pct": np.round(do_pct, 1),
        "temperature_c": np.round(temperature, 2),
        "cumulative_feed_ml": np.round(cum_feed, 1),
        "potency_surrogate": np.round(potency, 1),
    })

    # End-of-run outcomes
    rc = config["release_criteria"]
    final_viability = float(viability[-1])
    final_vcd = float(vcd[-1])
    final_potency = float(potency[-1])
    release_pass = (
        final_viability >= float(rc["viability_min"]) and
        final_vcd >= float(rc["vcd_min_e6_ml"]) and
        final_potency >= float(rc["potency_min"])
    )

    outcome = {
        "batch_id": batch_id,
        "final_viability": round(final_viability, 1),
        "final_vcd_e6_ml": round(final_vcd, 3),
        "final_potency": round(final_potency, 1),
        "release_pass": release_pass,
        "failure_mode": failure_mode or "none",
        **factors,
    }

    return trajectory, outcome


def generate_batch_trajectories(
    n_batches: int = 500,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a cohort of batch trajectories with injected failure modes.

    Returns (trajectories_df, outcomes_df).
    """
    config = _load_config()
    seed = int(seed if seed is not None else config.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    fac = config["factors"]
    fm_config = config["failure_modes"]

    all_traj = []
    all_outcomes = []

    for i in range(n_batches):
        bid = f"GB-{i+1:04d}"

        # Sample factors
        factors = {
            "n_activations": int(rng.integers(int(fac["n_activations"]["range"][0]),
                                              int(fac["n_activations"]["range"][1]) + 1)),
            "seed_train_days": int(rng.integers(int(fac["seed_train_days"]["range"][0]),
                                                int(fac["seed_train_days"]["range"][1]) + 1)),
            "seeding_density_e6_ml": round(float(rng.uniform(
                float(fac["seeding_density_e6_ml"]["range"][0]),
                float(fac["seeding_density_e6_ml"]["range"][1]))), 2),
            "il2_iu_ml": round(float(rng.uniform(
                float(fac["il2_iu_ml"]["range"][0]),
                float(fac["il2_iu_ml"]["range"][1]))), 0),
            "feed_strategy": rng.choice(fac["feed_strategy"]["levels"]),
            "hold_time_hrs": round(float(rng.uniform(
                float(fac["hold_time_hrs"]["range"][0]),
                float(fac["hold_time_hrs"]["range"][1]))), 1),
        }

        # Determine failure mode
        fm = None
        if factors["n_activations"] >= 3:
            fm = "over_activation"
        elif factors["feed_strategy"] == "bolus" and factors["seeding_density_e6_ml"] > 1.5:
            if rng.random() < 0.5:
                fm = "nutrient_limitation"
        elif factors["hold_time_hrs"] > 8:
            fm = "delayed_harvest"

        # Random events
        if fm is None and rng.random() < float(fm_config["contamination_upset"]["probability"]):
            fm = "contamination_upset"
        if fm is None and rng.random() < float(fm_config["low_transduction"]["probability"]):
            fm = "low_transduction"

        traj, outcome = simulate_single_batch(bid, factors, fm, rng, config)
        all_traj.append(traj)
        all_outcomes.append(outcome)

    trajectories = pd.concat(all_traj, ignore_index=True)
    outcomes = pd.DataFrame(all_outcomes)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectories.to_parquet(output_dir / "trajectories.parquet", index=False)
        outcomes.to_csv(output_dir / "outcomes.csv", index=False)

    return trajectories, outcomes
