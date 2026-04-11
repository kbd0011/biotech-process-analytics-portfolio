# OOT Investigation Workbench for Batch Release Drift

A reproducible investigation-support workflow that identifies suspect lots, compares them against matched historical peers, ranks plausible contributing factors, and generates evidence-backed investigation memos.

## What this project demonstrates

- OOT/OOS detection using rolling z-scores, run rules, and specification limits
- Matched peer-batch comparison with standardized effect sizes
- Interpretable contributor ranking via logistic regression (primary) with gradient boosting benchmark
- Automated investigation memos with caveats, evidence tables, and follow-up recommendations
- Config-driven spec limits, peer-group logic, and investigation metadata

## Data provenance

The dataset structure mirrors the publicly available 1,005-batch pharmaceutical manufacturing dataset (DOI: 10.6084/m9.figshare.19228978). When the public CSV files are not locally available, a structurally equivalent synthetic dataset is generated with the same schema, variable distributions, and inter-variable relationships. All synthetic data is clearly labeled. The investigation workflow, statistical methods, and reporting format are designed to be operationally realistic regardless of data source.

## Quick start

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
```

Run the demo notebook:

```bash
cd notebooks && jupyter notebook demo.ipynb
```

## Project structure

```
configs/
  spec_limits.yml          # Spec limits, OOT logic, peer-group rules, memo templates
src/
  data/
    pull_public_dataset.py  # Data ingestion or generation
  features/
    build_feature_store.py  # Lot-level feature engineering
  analytics/
    detect_oot_events.py    # OOT/OOS detection
    peer_batch_compare.py   # Suspect-vs-peer statistical comparison
  models/
    rank_contributors.py    # Interpretable contributor ranking
  reports/
    generate_investigation_report.py  # HTML investigation memo
  app/
    case_review_dashboard.py  # Case review orchestration
tests/
  test_workbench.py         # Comprehensive test suite
notebooks/
  demo.ipynb                # End-to-end demonstration
```

## Causality discipline

This workbench separates trend detection, statistical differences, contributor ranking, and follow-up recommendations. It never claims "root cause proven." All contributor outputs are framed as candidate contributors or suggestive patterns requiring process review.

## Generated artifacts

After running the demo notebook, the following are produced in `artifacts/`:

- `reports/investigation_*.html` — Investigation memo with event summary, peer comparison table, contributor ranking, caveats, and recommended follow-up checks. Open in any browser.
- `data/*.csv` — Generated dataset exports (laboratory, process, deviation log).

## Limitations

- Investigation evidence reflects statistical association, not proven causality
- Low-peer-count cases produce caveats rather than confident conclusions
- The contributor model uses cross-validated logistic regression; the GBM benchmark is secondary
- Synthetic data is structurally realistic but does not replicate any specific manufacturing process
