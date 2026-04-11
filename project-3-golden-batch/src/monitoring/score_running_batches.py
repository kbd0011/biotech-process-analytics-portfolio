"""
Score running batches against the golden-batch model and assign alert states.
"""

import numpy as np
import pandas as pd
from monitoring.build_golden_batch_model import compute_contributions


def score_batch(
    features_row: pd.Series,
    model: dict,
) -> dict:
    """Score a single batch against the golden-batch model.

    Returns dict with t2, spe, alert states, and contributions.
    """
    feature_cols = model["feature_cols"]
    scaler = model["scaler"]
    pca = model["pca"]

    x = features_row[feature_cols].values.astype(float)
    x = np.where(np.isnan(x), 0.0, x)
    # Pass as DataFrame with column names to avoid sklearn feature-name warnings
    x_df = pd.DataFrame([x], columns=feature_cols)
    x_scaled = scaler.transform(x_df).flatten()

    scores = pca.transform(x_scaled.reshape(1, -1))

    # T²
    t2 = float(np.sum((scores[0] / np.sqrt(pca.explained_variance_)) ** 2))

    # SPE
    residuals = x_scaled - (scores @ pca.components_).flatten()
    spe = float(np.sum(residuals ** 2))

    t2_alert = t2 > model["t2_limit"]
    spe_alert = spe > model["spe_limit"]

    alert_state = "normal"
    if t2_alert and spe_alert:
        alert_state = "critical"
    elif t2_alert:
        alert_state = "t2_warning"
    elif spe_alert:
        alert_state = "spe_warning"

    contribs = compute_contributions(x_scaled, model, feature_cols)

    return {
        "batch_id": features_row.get("batch_id", "unknown"),
        "t2": round(t2, 4),
        "spe": round(spe, 4),
        "t2_limit": model["t2_limit"],
        "spe_limit": model["spe_limit"],
        "t2_alert": t2_alert,
        "spe_alert": spe_alert,
        "alert_state": alert_state,
        "top_contributors": contribs.head(5).to_dict("records"),
    }


def score_cohort(
    features: pd.DataFrame,
    model: dict,
) -> pd.DataFrame:
    """Score all batches and return summary DataFrame."""
    records = []
    for _, row in features.iterrows():
        result = score_batch(row, model)
        records.append({
            "batch_id": result["batch_id"],
            "t2": result["t2"],
            "spe": result["spe"],
            "t2_alert": result["t2_alert"],
            "spe_alert": result["spe_alert"],
            "alert_state": result["alert_state"],
        })
    return pd.DataFrame(records)
