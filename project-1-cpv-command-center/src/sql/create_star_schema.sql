-- Star schema for Cell Therapy CPV & KPI Command Center
-- This schema defines the warehouse structure for batch genealogy and KPI consumption.
-- Executed via Python using DuckDB or SQLite.

-- Dimension: Sites
CREATE TABLE IF NOT EXISTS dim_site (
    site_id TEXT PRIMARY KEY,
    site_name TEXT NOT NULL,
    region TEXT
);

-- Dimension: Calendar
CREATE TABLE IF NOT EXISTS dim_calendar (
    date DATE PRIMARY KEY,
    year INTEGER,
    month INTEGER,
    week INTEGER,
    quarter INTEGER
);

-- Fact: Batch header
CREATE TABLE IF NOT EXISTS fact_batch (
    batch_id TEXT PRIMARY KEY,
    donor_id TEXT,
    site_id TEXT REFERENCES dim_site(site_id),
    lot_date DATE,
    month_idx INTEGER,
    activation_strategy TEXT,
    disposition TEXT NOT NULL,
    cycle_time_days REAL,
    oos_flag BOOLEAN
);

-- Fact: Batch steps (one-to-many from batch)
CREATE TABLE IF NOT EXISTS fact_batch_step (
    batch_id TEXT REFERENCES fact_batch(batch_id),
    step_name TEXT,
    step_start DATE,
    step_duration_hrs REAL
);

-- Fact: In-process measurements
CREATE TABLE IF NOT EXISTS fact_in_process (
    batch_id TEXT PRIMARY KEY REFERENCES fact_batch(batch_id),
    apheresis_cd3_pct REAL,
    apheresis_viability REAL,
    seeding_density_e6_ml REAL,
    il2_iu_ml REAL,
    transduction_efficiency REAL,
    glucose_mm REAL,
    lactate_mm REAL,
    expansion_fold INTEGER,
    harvest_viability REAL,
    harvest_vcd_e6_ml REAL
);

-- Fact: Release tests
CREATE TABLE IF NOT EXISTS fact_release (
    batch_id TEXT PRIMARY KEY REFERENCES fact_batch(batch_id),
    potency_pct_killing REAL,
    sterility_pass BOOLEAN,
    endotoxin_eu_ml REAL,
    identity_pass BOOLEAN,
    viability_pct REAL
);

-- Fact: Deviations
CREATE TABLE IF NOT EXISTS fact_deviation (
    batch_id TEXT REFERENCES fact_batch(batch_id),
    deviation_type TEXT,
    severity TEXT,
    deviation_date DATE
);

-- View: Monthly KPI mart
CREATE VIEW IF NOT EXISTS v_monthly_lot_summary AS
SELECT
    b.batch_id,
    b.site_id,
    b.lot_date,
    b.disposition,
    b.cycle_time_days,
    b.oos_flag,
    ip.transduction_efficiency,
    ip.harvest_viability,
    ip.expansion_fold,
    r.potency_pct_killing,
    r.sterility_pass,
    r.endotoxin_eu_ml
FROM fact_batch b
LEFT JOIN fact_in_process ip ON b.batch_id = ip.batch_id
LEFT JOIN fact_release r ON b.batch_id = r.batch_id;
