# Golden Batch Early Warning System for Cell Expansion

An interpretable early-warning system for a CAR-T-like cell-expansion process that detects when a running batch is drifting away from the healthy operating envelope toward lower yield, viability, or potency-related outcomes. Framed as predictive process monitoring, not a replacement for release testing.

## What this project demonstrates

- DOE-informed simulation of multivariate batch trajectories with realistic failure modes
- Golden-batch reference model using PCA, Hotelling T², and SPE/Q statistics
- Contribution plots identifying which variables drive alerts
- Early release-risk prediction (day 2 / day 5) using only partial-history features
- Interpretable logistic regression as primary model with GBM benchmark
- Held-out cohort validation with separate false alarm rate and detection power assessment
- Model validation memo with intended use, limitations, and performance summary

## Generated artifacts

After running the demo notebook, the following are produced in `artifacts/`:

- `reports/validation_memo.html` — Model validation memo with intended use statement, training population summary, performance metrics, and limitations. Open in any browser.

## Data provenance

All data is synthetic, seeded from published CAR-T QbD studies and release-testing concepts. No proprietary data is used. Five failure modes are encoded: over-activation/exhaustion, nutrient limitation, delayed harvest, contamination-like process upset, and low transduction efficiency. The simulation is designed to produce trajectories with operationally realistic shapes and inter-variable relationships.

## Quick start

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
cd notebooks && jupyter notebook demo.ipynb
```

## Project structure

```
configs/
  design_space.yml              # Factor ranges, failure modes, release criteria
src/
  sim/
    generate_batch_trajectories.py  # Batch trajectory simulator
  features/
    align_and_extract_features.py   # Feature extraction (full and partial)
  monitoring/
    build_golden_batch_model.py     # PCA + T² + SPE model
    score_running_batches.py        # Batch scoring and alerting
  models/
    predict_release_risk.py         # Early risk prediction
  dashboard/
    batch_monitor_app.py            # Static monitoring dashboard
  reports/
    model_validation_memo.py        # Validation memo generator
tests/
  test_golden_batch.py              # 14 tests
notebooks/
  demo.ipynb                        # End-to-end demonstration
```

