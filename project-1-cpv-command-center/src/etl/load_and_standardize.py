"""
Load generated source CSVs and normalize into a standardized mart.
"""

import pandas as pd
from pathlib import Path


def load_and_standardize(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize and validate tables for KPI consumption.

    Returns a cleaned dict of DataFrames with consistent types and
    a data_quality report DataFrame.
    """
    clean = {}

    # Batch header
    bh = tables["batch_header"].copy()
    bh["lot_date"] = pd.to_datetime(bh["lot_date"])
    bh["year_month"] = bh["lot_date"].dt.to_period("M").astype(str)
    bh["disposition"] = bh["disposition"].str.lower().str.strip()
    clean["batch_header"] = bh

    # Batch step
    bs = tables["batch_step"].copy()
    bs["step_start"] = pd.to_datetime(bs["step_start"])
    bs["step_duration_hrs"] = pd.to_numeric(bs["step_duration_hrs"], errors="coerce")
    clean["batch_step"] = bs

    # In-process
    ip = tables["in_process_measurements"].copy()
    for col in ip.columns:
        if col != "batch_id":
            ip[col] = pd.to_numeric(ip[col], errors="coerce")
    clean["in_process_measurements"] = ip

    # Release tests
    rt = tables["release_tests"].copy()
    for col in ["potency_pct_killing", "endotoxin_eu_ml", "viability_pct"]:
        if col in rt.columns:
            rt[col] = pd.to_numeric(rt[col], errors="coerce")
    for col in ["sterility_pass", "identity_pass"]:
        if col in rt.columns:
            rt[col] = rt[col].astype(bool)
    clean["release_tests"] = rt

    # Deviations
    dev = tables["deviation_log"].copy()
    if "deviation_date" in dev.columns:
        dev["deviation_date"] = pd.to_datetime(dev["deviation_date"])
    clean["deviation_log"] = dev

    # Site master and calendar pass through
    clean["site_master"] = tables["site_master"].copy()
    clean["calendar"] = tables["calendar"].copy()

    # Data quality report
    dq_records = []
    for name, df in clean.items():
        missing = df.isnull().sum()
        for col, n_miss in missing.items():
            if n_miss > 0:
                dq_records.append({
                    "table": name, "column": col,
                    "missing_count": n_miss,
                    "missing_pct": round(n_miss / len(df) * 100, 2),
                })
        dupes = df.duplicated().sum()
        if dupes > 0:
            dq_records.append({
                "table": name, "column": "_duplicate_rows_",
                "missing_count": dupes, "missing_pct": round(dupes / len(df) * 100, 2),
            })
    clean["data_quality"] = pd.DataFrame(dq_records) if dq_records else pd.DataFrame(
        columns=["table", "column", "missing_count", "missing_pct"]
    )

    return clean


def build_lot_mart(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join batch header, in-process, release, and deviation counts into a single lot-level mart."""
    mart = clean["batch_header"].copy()
    mart = mart.merge(clean["in_process_measurements"], on="batch_id", how="left")
    mart = mart.merge(clean["release_tests"], on="batch_id", how="left")

    dev_counts = clean["deviation_log"].groupby("batch_id").size().reset_index(name="deviation_count")
    mart = mart.merge(dev_counts, on="batch_id", how="left")
    mart["deviation_count"] = mart["deviation_count"].fillna(0).astype(int)

    return mart
