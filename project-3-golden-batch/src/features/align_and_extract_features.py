"""
Align batch trajectories by elapsed time and extract phase-level features
for monitoring and predictive modeling.
"""

import numpy as np
import pandas as pd


TRAJ_VARS = [
    "glucose_mm", "lactate_mm", "viable_cell_density_e6_ml",
    "viability_pct", "ph", "do_pct", "temperature_c",
    "cumulative_feed_ml", "potency_surrogate",
]


def extract_full_features(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Extract batch-level summary features from full trajectories."""
    records = []
    for bid, group in trajectories.groupby("batch_id"):
        group = group.sort_values("time_day")
        rec = {"batch_id": bid}
        for var in TRAJ_VARS:
            if var not in group.columns:
                continue
            vals = group[var].dropna().values
            if len(vals) == 0:
                continue
            rec[f"{var}_mean"] = np.mean(vals)
            rec[f"{var}_std"] = np.std(vals)
            rec[f"{var}_min"] = np.min(vals)
            rec[f"{var}_max"] = np.max(vals)
            rec[f"{var}_final"] = vals[-1]
            rec[f"{var}_initial"] = vals[0]
            # Slope over full trajectory
            if len(vals) >= 3:
                t = np.arange(len(vals))
                rec[f"{var}_slope"] = float(np.polyfit(t, vals, 1)[0])
            # Max rate of change
            if len(vals) >= 2:
                diffs = np.abs(np.diff(vals))
                rec[f"{var}_max_roc"] = float(np.max(diffs))
        records.append(rec)
    return pd.DataFrame(records)


def extract_partial_features(
    trajectories: pd.DataFrame,
    horizon_day: float,
) -> pd.DataFrame:
    """Extract features using only data up to horizon_day.

    This ensures no future leakage for early-risk prediction.
    """
    partial = trajectories[trajectories["time_day"] <= horizon_day].copy()
    if partial.empty:
        return pd.DataFrame()

    records = []
    for bid, group in partial.groupby("batch_id"):
        group = group.sort_values("time_day")
        rec = {"batch_id": bid, "horizon_day": horizon_day}
        for var in TRAJ_VARS:
            if var not in group.columns:
                continue
            vals = group[var].dropna().values
            if len(vals) == 0:
                continue
            rec[f"{var}_mean"] = np.mean(vals)
            rec[f"{var}_std"] = np.std(vals) if len(vals) > 1 else 0.0
            rec[f"{var}_last"] = vals[-1]
            rec[f"{var}_initial"] = vals[0]
            if len(vals) >= 3:
                t = np.arange(len(vals))
                rec[f"{var}_slope"] = float(np.polyfit(t, vals, 1)[0])
        records.append(rec)
    return pd.DataFrame(records)


def get_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric feature columns excluding identifiers."""
    exclude = {"batch_id", "horizon_day"}
    return [c for c in features.columns if c not in exclude and features[c].dtype in [np.float64, np.int64, float, int]]
