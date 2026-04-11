"""
Compute monthly and weekly KPI facts from the standardized lot mart.

KPI definitions are read from configs/kpi_definitions.yml.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "kpi_definitions.yml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_attribute_kpi(
    mart: pd.DataFrame,
    kpi_name: str,
    kpi_def: dict,
) -> pd.DataFrame:
    """Compute an attribute KPI (proportion) by site and month."""
    grp = mart.groupby(["site_id", "year_month"])
    records = []

    for (site, ym), group in grp:
        # Apply exclusions
        if kpi_def.get("exclusions") == "disposition == 'pending'":
            group = group[group["disposition"] != "pending"]

        denom = len(group)
        if denom == 0:
            continue

        # Compute numerator based on KPI name
        if kpi_name == "batch_success_rate":
            numer = (group["disposition"] == "released").sum()
        elif kpi_name == "oos_rate":
            numer = group["oos_flag"].sum()
        elif kpi_name == "termination_rate":
            numer = (group["disposition"] == "terminated").sum()
        elif kpi_name == "deviation_rate":
            numer = (group["deviation_count"] > 0).sum()
        else:
            continue

        rate = numer / denom
        threshold = float(kpi_def["threshold"])
        direction = kpi_def.get("alert_direction", "above")
        alert = (rate > threshold) if direction == "above" else (rate < threshold)

        records.append({
            "kpi_name": kpi_name,
            "display_name": kpi_def["display_name"],
            "site_id": site,
            "year_month": ym,
            "numerator": int(numer),
            "denominator": int(denom),
            "value": round(rate, 4),
            "threshold": threshold,
            "alert": bool(alert),
        })

    return pd.DataFrame(records)


def compute_continuous_kpi_lots(
    mart: pd.DataFrame,
    kpi_name: str,
    kpi_def: dict,
) -> pd.DataFrame:
    """Return lot-level values for a continuous KPI with spec metadata."""
    col_map = {
        "cycle_time_days": "cycle_time_days",
        "harvest_viability": "harvest_viability",
        "transduction_efficiency": "transduction_efficiency",
    }
    col = col_map.get(kpi_name)
    if col is None or col not in mart.columns:
        return pd.DataFrame()

    df = mart[["batch_id", "site_id", "year_month", col]].dropna(subset=[col]).copy()
    df.rename(columns={col: "value"}, inplace=True)
    df["kpi_name"] = kpi_name
    df["display_name"] = kpi_def["display_name"]
    df["spec_lower"] = kpi_def.get("spec_lower")
    df["spec_upper"] = kpi_def.get("spec_upper")
    df["target"] = kpi_def.get("target")
    return df


def compute_all_kpis(mart: pd.DataFrame, config: dict | None = None) -> dict[str, pd.DataFrame]:
    """Compute all configured KPIs. Returns dict with 'attribute_kpis' and 'continuous_kpis'."""
    if config is None:
        config = _load_config()

    attr_frames = []
    cont_frames = []

    for kpi_name, kpi_def in config["kpis"].items():
        mt = kpi_def.get("metric_type", "attribute")
        if mt == "attribute":
            df = compute_attribute_kpi(mart, kpi_name, kpi_def)
            if not df.empty:
                attr_frames.append(df)
        elif mt == "continuous":
            df = compute_continuous_kpi_lots(mart, kpi_name, kpi_def)
            if not df.empty:
                cont_frames.append(df)

    return {
        "attribute_kpis": pd.concat(attr_frames, ignore_index=True) if attr_frames else pd.DataFrame(),
        "continuous_kpis": pd.concat(cont_frames, ignore_index=True) if cont_frames else pd.DataFrame(),
    }
