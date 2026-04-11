"""
Generate a validation-style summary memo for the golden-batch monitoring system.
"""

from pathlib import Path
from datetime import datetime
from jinja2 import Template
import pandas as pd

_MEMO_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }
h1 { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 8px; }
h2 { color: #2c5f8a; margin-top: 30px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
th { background: #f0f4f8; }
.warn { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 15px 0; }
.footer { margin-top: 40px; font-size: 0.85em; color: #888; border-top: 1px solid #ddd; padding-top: 10px; }
</style></head><body>
<h1>{{ title }}</h1>
<p><strong>Generated:</strong> {{ timestamp }}</p>

<h2>1. Intended Use</h2>
<p>This model provides early-warning alerts during the cell-expansion step of a CAR-T-like
manufacturing process. It is designed to flag batches drifting away from the healthy operating
envelope toward lower yield, viability, or potency-related outcomes. It is <strong>not</strong>
a replacement for release testing or automated batch disposition.</p>

<h2>2. Model Description</h2>
<p>The primary monitoring model uses PCA fitted on {{ n_train }} successful baseline batches,
retaining {{ n_components }} principal components explaining {{ var_explained }}% of total variance.
Hotelling T&sup2; and SPE/Q statistics are used with {{ confidence }}% confidence limits.</p>
<p>An early release-risk model (logistic regression at day {{ horizon }}) achieves
CV AUC of {{ risk_auc }} and Brier score of {{ risk_brier }}.</p>

<h2>3. Training Population</h2>
<p>{{ n_train }} batches met release criteria and were used to establish the normal operating envelope.
{{ n_total }} total batches were simulated, with {{ n_fail }} failing release criteria across
{{ n_failure_modes }} encoded failure modes.</p>

<h2>4. Performance Summary</h2>
{{ perf_table }}

<h2>5. Limitations</h2>
<div class="warn">
<ul>
<li>Data is synthetic and literature-derived; model performance on real manufacturing data may differ.</li>
<li>The model assumes training data represents a stable, in-control process. Retraining is needed after process changes.</li>
<li>Contribution plots identify statistically prominent variables, not proven root causes.</li>
<li>The early-risk model uses only partial-history features; accuracy improves closer to harvest.</li>
<li>Alerts should trigger manual review by qualified personnel, not automated actions.</li>
</ul>
</div>

<div class="footer">
<p>This memo is generated from literature-derived synthetic data for portfolio demonstration.
The monitoring framework, statistical methods, and validation structure are designed to reflect
operationally realistic multivariate process monitoring workflows.</p>
</div>
</body></html>
"""


def generate_model_validation_memo(
    model: dict,
    outcomes: pd.DataFrame,
    risk_model_result: dict | None = None,
    horizon: int = 5,
    output_path: Path | None = None,
) -> str:
    """Generate an HTML validation memo."""
    n_total = len(outcomes)
    n_pass = outcomes["release_pass"].sum()
    n_fail = n_total - n_pass
    fm_counts = outcomes[outcomes["failure_mode"] != "none"]["failure_mode"].nunique()

    perf_records = []
    if risk_model_result:
        m = risk_model_result["metrics"]
        perf_records.append({
            "Model": "PCA Monitoring (T² + SPE)",
            "Metric": "Confidence level",
            "Value": f"{model['confidence']*100:.0f}%",
        })
        perf_records.append({
            "Model": f"Early Risk (Day {horizon})",
            "Metric": "CV AUC",
            "Value": str(m.get("cv_auc_mean", "N/A")),
        })
        perf_records.append({
            "Model": f"Early Risk (Day {horizon})",
            "Metric": "CV Brier",
            "Value": str(m.get("cv_brier_mean", "N/A")),
        })
        perf_records.append({
            "Model": f"Benchmark GBM (Day {horizon})",
            "Metric": "CV AUC",
            "Value": str(m.get("benchmark_cv_auc_mean", "N/A")),
        })

    perf_df = pd.DataFrame(perf_records)
    perf_html = perf_df.to_html(index=False) if not perf_df.empty else "<p>No metrics available.</p>"

    template = Template(_MEMO_TEMPLATE)
    html = template.render(
        title="Model Validation Memo — Golden Batch Early Warning System",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_train=model["n_train"],
        n_components=model["n_components"],
        var_explained=round(model["total_variance_explained"] * 100, 1),
        confidence=round(model["confidence"] * 100, 0),
        horizon=horizon,
        risk_auc=risk_model_result["metrics"].get("cv_auc_mean", "N/A") if risk_model_result else "N/A",
        risk_brier=risk_model_result["metrics"].get("cv_brier_mean", "N/A") if risk_model_result else "N/A",
        n_total=n_total,
        n_fail=n_fail,
        n_failure_modes=fm_counts,
        perf_table=perf_html,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html
