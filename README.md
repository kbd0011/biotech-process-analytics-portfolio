# Biotech Process Analytics Portfolio

Three self-contained projects demonstrating applied manufacturing analytics for regulated biopharma and cell-therapy operations: investigation support, CPV and KPI ownership, and predictive multivariate monitoring.

## Project summary

| Project | Domain | Core contribution | Primary methods | Data provenance |
|---------|--------|-------------------|-----------------|-----------------|
| **1. CPV & KPI Command Center** | Multi-site cell-therapy manufacturing | KPI ownership, SPC, CPV review packs | p-charts, I-MR, EWMA, Ppk, Pareto | Literature-derived synthetic |
| **2. OOT Investigation Workbench** | Pharmaceutical batch release | Investigation support, contributor ranking | OOT detection, peer matching, logistic regression, coefficient-based ranking | Public real dataset (1,005 batches) |
| **3. Golden Batch Early Warning** | Cell-expansion monitoring | Predictive multivariate monitoring | PCA, T², SPE/Q, contribution plots, early-risk LR | Literature-derived synthetic |

## Portfolio progression

```
┌─────────────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────────┐
│  Project 2              │     │  Project 1               │     │  Project 3                  │
│  OOT Investigation      │────>│  CPV & KPI Command       │────>│  Golden Batch Early         │
│  Workbench              │     │  Center                  │     │  Warning System             │
│                         │     │                          │     │                             │
│  Real-data anchored     │     │  Operational KPI layer   │     │  Predictive multivariate    │
│  investigation support  │     │  and recurring reporting  │     │  monitoring                 │
└─────────────────────────┘     └──────────────────────────┘     └─────────────────────────────┘
```

## Which project to read first

| If you are reviewing for... | Start with |
|-----------------------------|------------|
| Process data analyst role | Project 1 (CPV Command Center) |
| CPV / technical operations | Project 1, then Project 2 |
| Manufacturing science / MSAT | Project 3, then Project 1 |
| Biostat / applied modeling | Project 3, then Project 2 |
| Operations analytics hiring manager | Project 2 (fastest credible demo), then Project 1 |

## Synthetic vs real data

- **Project 2** uses a publicly available 1,005-batch pharmaceutical manufacturing dataset as its primary data source. When the public files are not locally available, a structurally equivalent synthetic dataset is generated with the same schema and variable relationships.
- **Projects 1 and 3** use literature-derived synthetic data inspired by published CAR-T CPP/CQA ranges and release-testing concepts.
- No project implies access to proprietary GMP or commercial manufacturing systems.
- Credibility comes from realistic data models, operationally appropriate control logic, and reproducible reporting workflows — not from data access.

## Test summary

| Project | Tests | Status |
|---------|-------|--------|
| 1. CPV Command Center | 19 | Passing |
| 2. OOT Investigation Workbench | 24 | Passing |
| 3. Golden Batch Early Warning | 17 | Passing |
| **Total** | **60** | **All passing** |

## Why this portfolio is relevant

- **KPI ownership**: Project 1 defines, computes, and tracks KPIs with config-driven logic and recurring reporting
- **CPV / ongoing process verification**: Project 1 applies SPC methods and capability indices aligned with FDA Stage 3 and ICH Q10
- **OOT/OOS trending**: Projects 1 and 2 detect out-of-trend behavior using rolling z-scores, run rules, and control charts
- **Investigation support**: Project 2 provides reproducible peer-batch comparison and contributor ranking for deviation meetings
- **Predictive monitoring**: Project 3 builds multivariate early-warning models with contribution-based explainability

## Running locally

Each project is a self-contained Python package:

```bash
cd project-2-oot-investigation
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
cd notebooks && jupyter notebook demo.ipynb
```

These repositories are designed as runnable local baselines for portfolio review, not production deployment artifacts.

## References

See [references.md](references.md) for full citations.
