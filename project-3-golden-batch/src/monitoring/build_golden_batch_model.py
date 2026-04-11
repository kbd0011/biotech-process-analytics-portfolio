"""
Build the multivariate reference model for normal successful batches
using PCA, Hotelling T², and SPE/Q statistics.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats


def build_golden_batch_model(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    feature_cols: list[str],
    n_components: int | None = None,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Fit PCA monitoring model on successful baseline batches.

    Args:
        features: batch-level feature matrix
        outcomes: batch outcomes with release_pass column
        feature_cols: columns to use for PCA
        n_components: number of PCs (auto if None)
        confidence: confidence level for T² and SPE limits

    Returns dict with model artifacts.
    """
    # Select only successful batches for training
    good_ids = outcomes[outcomes["release_pass"] == True]["batch_id"].values
    train_mask = features["batch_id"].isin(good_ids)
    X_train = features.loc[train_mask, feature_cols].copy()

    # Handle missing values
    X_train = X_train.fillna(X_train.median())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # PCA
    if n_components is None:
        n_components = min(len(feature_cols), len(X_train) // 3, 10)
        n_components = max(n_components, 2)

    pca = PCA(n_components=n_components, random_state=int(seed))
    scores = pca.fit_transform(X_scaled)

    # Hotelling T² limit (F-distribution approximation)
    n = len(X_train)
    k = n_components
    t2_values = np.sum((scores / np.sqrt(pca.explained_variance_)) ** 2, axis=1)
    f_crit = stats.f.ppf(confidence, k, n - k)
    t2_limit = k * (n - 1) / (n - k) * f_crit

    # SPE/Q statistic
    X_reconstructed = scores @ pca.components_ + scaler.mean_
    residuals = X_scaled - (scores @ pca.components_)
    spe_values = np.sum(residuals ** 2, axis=1)
    # Chi-squared approximation for SPE limit
    spe_mean = np.mean(spe_values)
    spe_var = np.var(spe_values)
    if spe_var > 0:
        g = spe_var / (2 * spe_mean)
        h = 2 * spe_mean ** 2 / spe_var
        spe_limit = g * stats.chi2.ppf(confidence, h)
    else:
        spe_limit = spe_mean * 3

    model = {
        "pca": pca,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "n_components": n_components,
        "n_train": n,
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": float(sum(pca.explained_variance_ratio_)),
        "t2_limit": float(t2_limit),
        "spe_limit": float(spe_limit),
        "confidence": confidence,
        "train_t2": t2_values,
        "train_spe": spe_values,
        "loadings": pca.components_,
    }

    return model


def compute_contributions(
    x_scaled: np.ndarray,
    model: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute variable contributions to T² and SPE for a single observation.

    Returns DataFrame with feature-level contribution values.
    """
    pca = model["pca"]
    scores = pca.transform(x_scaled.reshape(1, -1))
    residuals = x_scaled - (scores @ pca.components_).flatten()

    # T² contributions (diagonal approximation)
    t2_contrib = np.zeros(len(feature_cols))
    for j in range(model["n_components"]):
        loading_j = pca.components_[j]
        t2_contrib += (scores[0, j] / np.sqrt(pca.explained_variance_[j])) ** 2 * loading_j ** 2

    # SPE contributions
    spe_contrib = residuals ** 2

    return pd.DataFrame({
        "feature": feature_cols,
        "t2_contribution": np.round(t2_contrib, 4),
        "spe_contribution": np.round(spe_contrib, 4),
        "total_contribution": np.round(t2_contrib + spe_contrib, 4),
    }).sort_values("total_contribution", ascending=False).reset_index(drop=True)
