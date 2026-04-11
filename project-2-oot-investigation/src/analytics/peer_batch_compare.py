"""
Compare suspect lots against matched historical peer lots to quantify
what changed and generate evidence tables for investigation support.
"""

import numpy as np
import pandas as pd
from scipy import stats
import yaml
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "spec_limits.yml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def select_peer_lots(
    features: pd.DataFrame,
    suspect_batch_id: str,
    config: dict | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Select peer lots matched to the suspect lot on configured keys.

    Returns (suspect_row, peer_df, metadata) where metadata includes
    peer count and any warnings.
    """
    if config is None:
        config = _load_config()

    peer_cfg = config["peer_matching"]
    keys = peer_cfg["matching_keys"]
    min_peers = int(peer_cfg["min_peer_count"])
    warn_thresh = int(peer_cfg["low_peer_warning_threshold"])
    max_peers = int(peer_cfg["max_peer_count"])

    suspect_row = features[features["batch_id"] == suspect_batch_id].iloc[0]

    mask = pd.Series(True, index=features.index)
    for k in keys:
        mask &= features[k] == suspect_row[k]

    # Exclude suspect lot from peer set
    mask &= features["batch_id"] != suspect_batch_id

    # Exclude other suspect lots if flagged
    if "suspect" in features.columns and bool(peer_cfg.get("exclude_suspect", True)):
        mask &= ~features["suspect"]

    peers = features[mask].copy()

    # Limit to max peers (most recent)
    if len(peers) > max_peers:
        peers = peers.tail(max_peers)

    metadata = {
        "peer_count": len(peers),
        "low_peer_warning": len(peers) < warn_thresh,
        "insufficient_peers": len(peers) < min_peers,
        "matching_keys_used": keys,
    }
    return suspect_row, peers, metadata


def compute_comparisons(
    suspect_row: pd.Series,
    peers: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute standardized differences and effect sizes between suspect and peers.

    Returns a DataFrame with one row per feature, including:
    - suspect_value, peer_mean, peer_std
    - z_diff (standardized difference)
    - cohens_d (effect size treating suspect as a single-sample comparison)
    - p_value (two-sided z-test approximation)
    - direction (above/below/within)
    """
    records = []
    for col in feature_cols:
        s_val = float(suspect_row[col]) if pd.notna(suspect_row.get(col)) else np.nan
        peer_vals = peers[col].dropna()
        if len(peer_vals) < 3 or np.isnan(s_val):
            records.append({
                "feature": col, "suspect_value": s_val,
                "peer_mean": np.nan, "peer_std": np.nan,
                "z_diff": np.nan, "cohens_d": np.nan,
                "p_value": np.nan, "direction": "insufficient_data",
            })
            continue

        p_mean = peer_vals.mean()
        p_std = peer_vals.std()
        if p_std < 1e-9:
            p_std = 1e-9

        z_diff = (s_val - p_mean) / p_std
        cohens_d = z_diff  # single observation vs distribution
        p_value = 2 * (1 - stats.norm.cdf(abs(z_diff)))

        direction = "within"
        if z_diff > 1.0:
            direction = "above"
        elif z_diff < -1.0:
            direction = "below"

        records.append({
            "feature": col,
            "suspect_value": round(s_val, 4),
            "peer_mean": round(p_mean, 4),
            "peer_std": round(p_std, 4),
            "z_diff": round(z_diff, 3),
            "cohens_d": round(cohens_d, 3),
            "p_value": round(p_value, 4),
            "direction": direction,
        })

    result = pd.DataFrame(records)
    return result.sort_values("z_diff", key=abs, ascending=False).reset_index(drop=True)


def top_differences(comparisons: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top N features by absolute standardized difference."""
    valid = comparisons.dropna(subset=["z_diff"])
    return valid.head(n).reset_index(drop=True)
