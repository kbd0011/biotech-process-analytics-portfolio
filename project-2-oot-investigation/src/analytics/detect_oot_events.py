"""
Detect out-of-trend (OOT) and out-of-specification (OOS) events
using rolling z-scores, run rules, and specification limits.

The input is always sorted by manufacturing date before applying any
rolling or sequential logic, ensuring results are independent of
the row order in which data is provided.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "spec_limits.yml"

# Columns added by detection that must never enter downstream modeling.
DETECTION_DERIVED_COLUMNS = frozenset({
    "oos_flag", "oot_zscore_flag", "oot_run_flag",
    "suspect", "reason_code", "severity", "_rolling_z",
})


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def detect_oot_events(
    features: pd.DataFrame,
    response_var: str = "dissolution",
    config: dict | None = None,
) -> pd.DataFrame:
    """Identify suspect lots using OOT/OOS logic from config.

    The input is sorted by manufacturing date (mfg_date) before any
    rolling or sequential detection logic is applied. If mfg_date is
    not present, the input index order is preserved with a warning.

    Returns a copy of the input with added columns:
    - oos_flag: lot is outside specification limits
    - oot_zscore_flag: lot exceeds rolling z-score threshold
    - oot_run_flag: lot is part of a run on one side of center
    - suspect: any flag is True
    - reason_code: text summary of triggered rules
    - severity: worst triggered severity level
    """
    if config is None:
        config = _load_config()

    var_cfg = config["response_variables"][response_var]
    lsl = float(var_cfg["lower_spec"])
    usl = float(var_cfg["upper_spec"])
    oot = var_cfg["oot_logic"]
    window = int(oot["window"])
    z_thresh = float(oot["z_threshold"])
    run_len = int(oot["run_rule_length"])

    df = features.copy()

    # --- Sort by manufacturing date before rolling logic ---
    if "mfg_date" in df.columns:
        df = df.sort_values("mfg_date").reset_index(drop=True)

    values = df[response_var].values.astype(float)

    # --- OOS detection ---
    df["oos_flag"] = (values < lsl) | (values > usl)

    # --- Rolling z-score OOT ---
    rolling_mean = pd.Series(values).rolling(window, min_periods=10).mean().values
    rolling_std = pd.Series(values).rolling(window, min_periods=10).std().values
    rolling_std = np.where(rolling_std < 1e-6, 1e-6, rolling_std)
    z_scores = (values - rolling_mean) / rolling_std
    # Prefixed with underscore to signal this is a detection artifact,
    # not a modeling feature. Also listed in DETECTION_DERIVED_COLUMNS.
    df["_rolling_z"] = np.round(z_scores, 3)
    df["oot_zscore_flag"] = np.abs(z_scores) > z_thresh

    # --- Run rule OOT ---
    grand_mean = np.nanmean(values[:window]) if len(values) >= window else np.nanmean(values)
    above = values > grand_mean
    run_flags = np.zeros(len(values), dtype=bool)
    for i in range(run_len - 1, len(values)):
        segment = above[i - run_len + 1: i + 1]
        if segment.all() or (~segment).all():
            run_flags[i - run_len + 1: i + 1] = True
    df["oot_run_flag"] = run_flags

    # --- Combine flags ---
    df["suspect"] = df["oos_flag"] | df["oot_zscore_flag"] | df["oot_run_flag"]

    # --- Reason codes and severity ---
    severities_map = var_cfg.get("severity_levels", {})
    reason_list = []
    severity_list = []
    for _, row in df.iterrows():
        codes = []
        sev = "none"
        if row["oos_flag"]:
            codes.append("OOS")
            sev = severities_map.get("oos", "critical")
        if row["oot_zscore_flag"]:
            codes.append("OOT-zscore")
            if sev == "none":
                sev = severities_map.get("oot_zscore", "major")
        if row["oot_run_flag"]:
            codes.append("OOT-run")
            if sev == "none":
                sev = severities_map.get("oot_run", "minor")
        reason_list.append("; ".join(codes) if codes else "")
        severity_list.append(sev)

    df["reason_code"] = reason_list
    df["severity"] = severity_list

    return df


def summarize_suspects(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of suspect lots."""
    suspects = df[df["suspect"]].copy()
    if suspects.empty:
        return pd.DataFrame(columns=["batch_id", "dissolution", "reason_code", "severity"])
    cols = ["batch_id", "dissolution", "_rolling_z", "reason_code", "severity"]
    return suspects[[c for c in cols if c in suspects.columns]].reset_index(drop=True)
