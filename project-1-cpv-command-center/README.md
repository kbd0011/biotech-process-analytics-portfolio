# Cell Therapy CPV & KPI Command Center

A Stage 3 / CPV-style operating review environment for a multi-site autologous cell-therapy manufacturing case. Integrates lot-level process, in-process, release, and deviation data to compute recurring KPIs for management review and site-to-site comparison.

## What this project demonstrates

- KPI definition and ownership via config-driven logic
- Batch-genealogy-aware data model (one lot → multiple steps → multiple release tests)
- Monthly/weekly attribute and continuous KPI computation
- SPC logic: p-charts, I-MR, EWMA, run-of-seven rule
- Capability indices (Ppk) only where specification limits exist
- Site-to-site performance comparison
- Self-contained interactive Plotly dashboard with KPI cards, trend charts, SPC panels, capability table, and lot drill-through
- Automated monthly CPV review pack with narrative callouts

## Generated artifacts

After running the demo notebook, the following are produced in `artifacts/`:

- `dashboard/cpv_dashboard.html` — Interactive dashboard with executive KPI summary, site comparison trends, SPC control charts (p-charts, EWMA with signal highlighting), capability table, and lot-level drill-through. Open in any browser.
- `reports/cpv_review_*.html` — Monthly CPV review pack with threshold alerts and narrative callouts.
- `dashboard/*.csv` — Flat exports for Power BI / Tableau consumption.

## Data provenance

All data is literature-derived synthetic data inspired by published CAR-T CPP/CQA ranges and release-testing concepts. No proprietary manufacturing records are used. Four scenarios are deliberately injected for analytics validation: transduction-efficiency drift at Site B, temporary sterility spike after vendor change, improved cycle time after process improvement, and seasonal apheresis input variability.

## Quick start

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
cd notebooks && jupyter notebook demo.ipynb
```

## Project structure

```
configs/
  kpi_definitions.yml           # KPI formulas, thresholds, chart types, owners
src/
  data/
    generate_cell_therapy_batches.py  # Synthetic data generator
  sql/
    create_star_schema.sql      # Warehouse schema definition
  etl/
    load_and_standardize.py     # Data normalization and mart builder
  metrics/
    compute_kpis.py             # Monthly KPI fact computation
  analytics/
    cpv_rules.py                # SPC charts, EWMA, capability indices
  dashboard/
    export_dashboard_tables.py  # Power BI / Tableau export
  reports/
    generate_monthly_cpv_pack.py  # Automated CPV review pack
tests/
  test_cpv.py                   # 19 tests covering generation through reporting
notebooks/
  demo.ipynb                    # End-to-end demonstration
```

## Limitations

- Synthetic data is structurally realistic but does not replicate any specific manufacturing process
- Dashboard exports are flat files; no live BI connection is included
- Capability indices require genuine spec limits and are suppressed where specs are absent
