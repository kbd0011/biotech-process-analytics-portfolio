"""
Apply SPC and capability logic to attribute and continuous KPI metrics.

Implements p-charts, I-MR charts, EWMA, point-outside-limits and run-of-seven rules,
and capability indices (Ppk/Cpk) where spec limits exist.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "kpi_definitions.yml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def p_chart(attribute_kpi: pd.DataFrame) -> pd.DataFrame:
    """Compute p-chart control limits for an attribute KPI time series.

    Expects columns: year_month, numerator, denominator, value.
    Returns the input with added columns: p_bar, ucl, lcl, signal.
    """
    df = attribute_kpi.sort_values("year_month").copy()
    total_n = df["numerator"].sum()
    total_d = df["denominator"].sum()
    p_bar = total_n / total_d if total_d > 0 else 0

    n_avg = df["denominator"].mean()
    se = np.sqrt(p_bar * (1 - p_bar) / n_avg) if n_avg > 0 and 0 < p_bar < 1 else 0

    df["p_bar"] = round(p_bar, 5)
    df["ucl"] = round(min(p_bar + 3 * se, 1.0), 5)
    df["lcl"] = round(max(p_bar - 3 * se, 0.0), 5)
    df["signal"] = (df["value"] > df["ucl"]) | (df["value"] < df["lcl"])
    return df


def i_mr_chart(values: pd.Series) -> pd.DataFrame:
    """Compute I-MR chart statistics for a continuous KPI.

    Returns DataFrame with columns: index, value, x_bar, mr, mr_bar,
    ucl_i, lcl_i, ucl_mr, signal_i, signal_mr.
    """
    vals = values.dropna().values.astype(float)
    n = len(vals)
    if n < 3:
        return pd.DataFrame()

    x_bar = np.mean(vals)
    mr = np.abs(np.diff(vals))
    mr_bar = np.mean(mr)

    # Constants for I-MR (d2=1.128 for n=2)
    d2 = 1.128
    sigma_est = mr_bar / d2

    ucl_i = x_bar + 3 * sigma_est
    lcl_i = x_bar - 3 * sigma_est
    ucl_mr = 3.267 * mr_bar  # D4 for n=2

    df = pd.DataFrame({
        "value": vals,
        "x_bar": x_bar,
        "mr": np.concatenate([[np.nan], mr]),
        "mr_bar": mr_bar,
        "ucl_i": ucl_i,
        "lcl_i": lcl_i,
        "ucl_mr": ucl_mr,
    })
    df["signal_i"] = (df["value"] > ucl_i) | (df["value"] < lcl_i)
    df["signal_mr"] = df["mr"] > ucl_mr
    return df


def ewma_chart(values: pd.Series, lam: float = 0.2) -> pd.DataFrame:
    """Compute EWMA chart for subtle drift detection.

    Returns DataFrame with columns: value, ewma, ucl, lcl, signal.
    """
    vals = values.dropna().values.astype(float)
    n = len(vals)
    if n < 5:
        return pd.DataFrame()

    mu0 = np.mean(vals)
    sigma = np.std(vals, ddof=1)

    ewma = np.zeros(n)
    ewma[0] = lam * vals[0] + (1 - lam) * mu0
    for i in range(1, n):
        ewma[i] = lam * vals[i] + (1 - lam) * ewma[i - 1]

    # Time-varying limits
    indices = np.arange(1, n + 1)
    factor = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * indices)))
    L = 2.7  # ~ARL=370
    ucl = mu0 + L * sigma * factor
    lcl = mu0 - L * sigma * factor

    df = pd.DataFrame({
        "value": vals,
        "ewma": np.round(ewma, 4),
        "ucl": np.round(ucl, 4),
        "lcl": np.round(lcl, 4),
        "center": mu0,
    })
    df["signal"] = (df["ewma"] > df["ucl"]) | (df["ewma"] < df["lcl"])
    return df


def capability_index(values: pd.Series, spec_lower: float | None, spec_upper: float | None) -> dict | None:
    """Compute Ppk and Cpk where specification limits are defined.

    Returns None if neither spec limit is provided.
    """
    if spec_lower is None and spec_upper is None:
        return None

    vals = values.dropna().values.astype(float)
    if len(vals) < 10:
        return None

    mu = np.mean(vals)
    sigma = np.std(vals, ddof=1)
    if sigma < 1e-9:
        return None

    ppk_values = []
    if spec_upper is not None:
        ppk_values.append((float(spec_upper) - mu) / (3 * sigma))
    if spec_lower is not None:
        ppk_values.append((mu - float(spec_lower)) / (3 * sigma))

    ppk = min(ppk_values) if ppk_values else None

    return {
        "mean": round(mu, 4),
        "std": round(sigma, 4),
        "spec_lower": spec_lower,
        "spec_upper": spec_upper,
        "ppk": round(ppk, 3) if ppk is not None else None,
        "n": len(vals),
    }


def run_cpv_analytics(
    attribute_kpis: pd.DataFrame,
    continuous_kpis: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """Run full CPV analytics across all KPIs.

    Returns dict with chart data and capability results.
    """
    if config is None:
        config = _load_config()

    results = {"p_charts": {}, "i_mr_charts": {}, "ewma_charts": {}, "capability": {}}

    # P-charts for attribute KPIs
    if not attribute_kpis.empty:
        for kpi_name in attribute_kpis["kpi_name"].unique():
            for site in attribute_kpis["site_id"].unique():
                subset = attribute_kpis[
                    (attribute_kpis["kpi_name"] == kpi_name) &
                    (attribute_kpis["site_id"] == site)
                ]
                if len(subset) >= 3:
                    key = f"{kpi_name}__{site}"
                    results["p_charts"][key] = p_chart(subset)

    # Continuous KPIs
    if not continuous_kpis.empty:
        for kpi_name in continuous_kpis["kpi_name"].unique():
            kpi_def = config["kpis"].get(kpi_name, {})
            chart_type = kpi_def.get("chart_type", "i_mr")

            for site in continuous_kpis["site_id"].unique():
                subset = continuous_kpis[
                    (continuous_kpis["kpi_name"] == kpi_name) &
                    (continuous_kpis["site_id"] == site)
                ].sort_values("year_month")

                if len(subset) < 5:
                    continue

                key = f"{kpi_name}__{site}"

                if chart_type == "ewma":
                    results["ewma_charts"][key] = ewma_chart(subset["value"])
                else:
                    results["i_mr_charts"][key] = i_mr_chart(subset["value"])

                # Capability only if specs exist
                sl = kpi_def.get("spec_lower")
                su = kpi_def.get("spec_upper")
                cap = capability_index(subset["value"], sl, su)
                if cap is not None:
                    results["capability"][key] = cap

    return results
