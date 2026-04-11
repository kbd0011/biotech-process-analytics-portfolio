"""
Predict end-of-run release risk from early (partial) trajectory information.

Uses only data available up to the scoring horizon — no future leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


def train_early_risk_model(
    partial_features: pd.DataFrame,
    outcomes: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
) -> dict:
    """Train logistic regression for early release-risk prediction.

    Args:
        partial_features: features extracted up to a scoring horizon
        outcomes: batch outcomes with release_pass column
        feature_cols: numeric feature columns to use
        seed: random seed

    Returns dict with model, scaler, metrics, and calibration data.
    """
    merged = partial_features.merge(outcomes[["batch_id", "release_pass"]], on="batch_id", how="inner")
    X = merged[feature_cols].fillna(0).values
    y = (~merged["release_pass"]).astype(int).values  # 1 = at risk

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Primary: logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=int(seed))

    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
    aucs, briers = [], []
    y_prob_all, y_true_all = [], []

    for train_idx, test_idx in cv.split(X_scaled, y):
        lr.fit(X_scaled[train_idx], y[train_idx])
        proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            aucs.append(roc_auc_score(y[test_idx], proba))
            briers.append(brier_score_loss(y[test_idx], proba))
        y_prob_all.extend(proba)
        y_true_all.extend(y[test_idx])

    # Final fit
    lr.fit(X_scaled, y)

    # Calibration
    y_prob_all = np.array(y_prob_all)
    y_true_all = np.array(y_true_all)
    if len(np.unique(y_true_all)) > 1 and len(y_true_all) > 20:
        frac_pos, mean_pred = calibration_curve(y_true_all, y_prob_all, n_bins=5, strategy="quantile")
    else:
        frac_pos, mean_pred = np.array([]), np.array([])

    # Benchmark: GBM
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=int(seed))
    gbm_aucs = []
    for train_idx, test_idx in cv.split(X_scaled, y):
        gbm.fit(X_scaled[train_idx], y[train_idx])
        proba = gbm.predict_proba(X_scaled[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            gbm_aucs.append(roc_auc_score(y[test_idx], proba))
    gbm.fit(X_scaled, y)

    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": np.round(lr.coef_[0], 4),
    }).sort_values("coefficient", key=abs, ascending=False)

    return {
        "primary_model": lr,
        "benchmark_model": gbm,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "coefficients": coefs,
        "metrics": {
            "cv_auc_mean": round(np.mean(aucs), 3) if aucs else None,
            "cv_brier_mean": round(np.mean(briers), 4) if briers else None,
            "benchmark_cv_auc_mean": round(np.mean(gbm_aucs), 3) if gbm_aucs else None,
        },
        "calibration": {
            "fraction_of_positives": frac_pos.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
        },
    }
