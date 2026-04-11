"""
Rank plausible contributing factors to OOT/OOS events using
interpretable models. The primary model is logistic regression;
a gradient boosting benchmark is included for comparison only.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, brier_score_loss,
)


def prepare_labels(
    features: pd.DataFrame,
    response_var: str = "dissolution",
    lower_spec: float = 75.0,
    upper_spec: float = 110.0,
    oot_percentile: float = 10.0,
) -> pd.Series:
    """Create binary labels: 1 = suspect (OOS or in bottom percentile), 0 = normal."""
    values = features[response_var].values
    threshold = np.percentile(values[~np.isnan(values)], oot_percentile)
    labels = ((values < lower_spec) | (values > upper_spec) | (values < threshold)).astype(int)
    return pd.Series(labels, index=features.index, name="suspect_label")


def train_contributor_model(
    features: pd.DataFrame,
    feature_cols: list[str],
    labels: pd.Series,
    seed: int = 42,
) -> dict:
    """Train logistic regression and optional GBM benchmark.

    Returns dict with:
    - 'primary_model': fitted LogisticRegression
    - 'scaler': fitted StandardScaler
    - 'coefficients': DataFrame of feature importances
    - 'metrics': dict of cross-validated performance
    - 'benchmark_model': fitted GBM (secondary)
    - 'benchmark_metrics': dict
    """
    X = features[feature_cols].copy()
    y = labels.values

    # Handle missing values
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Primary: logistic regression
    lr = LogisticRegression(
        l1_ratio=0, C=1.0, max_iter=1000,
        random_state=int(seed), solver="lbfgs",
    )

    # Time-aware cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    lr_aucs, lr_briers = [], []
    for train_idx, test_idx in tscv.split(X_scaled):
        lr.fit(X_scaled[train_idx], y[train_idx])
        proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            lr_aucs.append(roc_auc_score(y[test_idx], proba))
            lr_briers.append(brier_score_loss(y[test_idx], proba))

    # Final fit on all data
    lr.fit(X_scaled, y)

    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": np.round(lr.coef_[0], 4),
        "abs_coefficient": np.round(np.abs(lr.coef_[0]), 4),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    lr_metrics = {
        "cv_auc_mean": round(np.mean(lr_aucs), 3) if lr_aucs else None,
        "cv_brier_mean": round(np.mean(lr_briers), 4) if lr_briers else None,
    }

    # Benchmark: gradient boosting
    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=int(seed),
    )
    gbm_aucs = []
    for train_idx, test_idx in tscv.split(X_scaled):
        gbm.fit(X_scaled[train_idx], y[train_idx])
        proba = gbm.predict_proba(X_scaled[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            gbm_aucs.append(roc_auc_score(y[test_idx], proba))
    gbm.fit(X_scaled, y)

    gbm_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": np.round(gbm.feature_importances_, 4),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    gbm_metrics = {
        "cv_auc_mean": round(np.mean(gbm_aucs), 3) if gbm_aucs else None,
    }

    return {
        "primary_model": lr,
        "scaler": scaler,
        "coefficients": coefs,
        "metrics": lr_metrics,
        "benchmark_model": gbm,
        "benchmark_importance": gbm_importance,
        "benchmark_metrics": gbm_metrics,
    }


def rank_contributors(model_result: dict, top_n: int = 10) -> pd.DataFrame:
    """Return top contributing features from the primary model."""
    coefs = model_result["coefficients"].copy()
    coefs["rank"] = range(1, len(coefs) + 1)
    coefs["interpretation"] = coefs["coefficient"].apply(
        lambda c: "higher values increase suspect risk"
        if c > 0 else "lower values increase suspect risk"
    )
    return coefs.head(top_n)
