"""
Build a lot-level feature store from laboratory and process data.

Joins raw-material attributes, process parameters, and engineered features
into a single modeling-ready table at one row per lot.
"""

import numpy as np
import pandas as pd


def build_feature_store(
    laboratory: pd.DataFrame,
    process: pd.DataFrame,
    deviation_log: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge lab, process, and optional deviation data into a lot-level feature table.

    Returns a DataFrame with one row per batch_id containing all raw-material,
    process, and engineered features suitable for investigation analytics.
    """
    # Merge lab and process on batch_id
    features = laboratory.merge(process, on="batch_id", how="left")

    # Engineered features
    features["rm_potency_x_moisture"] = features["rm_potency"] * features["rm_moisture"]
    features["compression_to_weight_ratio"] = (
        features["compression_force_kn"] / features["tablet_weight_mg"]
    )
    features["drying_intensity"] = features["drying_temp_c"] * features["drying_time_min"]
    features["mix_energy_proxy"] = features["mix_speed_rpm"] * features["mix_time_min"]

    # Deviation flag
    if deviation_log is not None and not deviation_log.empty:
        dev_flags = (
            deviation_log.groupby("batch_id")
            .agg(
                deviation_count=("deviation_type", "count"),
                has_major_deviation=("severity", lambda s: int((s.isin(["major", "critical"])).any())),
            )
            .reset_index()
        )
        features = features.merge(dev_flags, on="batch_id", how="left")
        features["deviation_count"] = features["deviation_count"].fillna(0).astype(int)
        features["has_major_deviation"] = features["has_major_deviation"].fillna(0).astype(int)
    else:
        features["deviation_count"] = 0
        features["has_major_deviation"] = 0

    # Supplier encoding (one-hot)
    if "rm_supplier" in features.columns:
        supplier_dummies = pd.get_dummies(features["rm_supplier"], prefix="supplier", dtype=int)
        features = pd.concat([features, supplier_dummies], axis=1)

    return features


def get_modeling_columns(features: pd.DataFrame, target: str = "dissolution") -> list[str]:
    """Return the list of numeric feature columns excluding identifiers,
    the target, and any detection-derived columns.

    Detection-derived columns (oos_flag, oot_*, suspect, _rolling_z, severity,
    reason_code) are excluded to prevent leakage from the detection step into
    downstream contributor-ranking models.
    """
    # Import here to avoid circular dependency at module level
    from analytics.detect_oot_events import DETECTION_DERIVED_COLUMNS

    exclude = {
        "batch_id", "product_code", "batch_size_category", "mfg_date",
        "rm_supplier", target,
    } | DETECTION_DERIVED_COLUMNS
    cols = [
        c for c in features.columns
        if c not in exclude
        and not c.startswith("_")  # exclude any underscore-prefixed internal columns
        and features[c].dtype in [np.float64, np.int64, np.int32, float, int]
    ]
    return cols


def get_traceability(features: pd.DataFrame) -> pd.DataFrame:
    """Return a mapping of engineered feature names to their source columns."""
    records = [
        {"feature": "rm_potency_x_moisture", "sources": "rm_potency, rm_moisture"},
        {"feature": "compression_to_weight_ratio", "sources": "compression_force_kn, tablet_weight_mg"},
        {"feature": "drying_intensity", "sources": "drying_temp_c, drying_time_min"},
        {"feature": "mix_energy_proxy", "sources": "mix_speed_rpm, mix_time_min"},
        {"feature": "deviation_count", "sources": "deviation_log.deviation_type"},
        {"feature": "has_major_deviation", "sources": "deviation_log.severity"},
    ]
    return pd.DataFrame(records)
