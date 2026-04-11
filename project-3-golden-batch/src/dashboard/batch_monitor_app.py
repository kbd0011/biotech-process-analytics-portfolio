"""
Batch monitoring dashboard — generates static HTML review pages
showing golden-envelope comparison and alert summaries.
"""

from pathlib import Path
import pandas as pd
from jinja2 import Template

_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Batch Monitor</title>
<style>
body { font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; }
h1 { color: #1a3a5c; } h2 { color: #2c5f8a; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #f0f4f8; }
.normal { color: #28a745; } .warning { color: #ffc107; } .critical { color: #dc3545; }
</style></head><body>
<h1>Batch Monitoring Dashboard</h1>
<h2>Alert Summary</h2>
{{ alert_table }}
<h2>Score Distribution</h2>
<p>T&sup2; limit: {{ t2_limit }} | SPE limit: {{ spe_limit }}</p>
{{ score_table }}
</body></html>
"""


def generate_dashboard_html(
    score_results: pd.DataFrame,
    model: dict,
    output_path: Path | None = None,
) -> str:
    """Generate a static HTML dashboard from scoring results."""
    alert_summary = score_results["alert_state"].value_counts().reset_index()
    alert_summary.columns = ["Alert State", "Count"]

    template = Template(_DASHBOARD_TEMPLATE)
    html = template.render(
        alert_table=alert_summary.to_html(index=False),
        t2_limit=round(model["t2_limit"], 2),
        spe_limit=round(model["spe_limit"], 2),
        score_table=score_results.head(20).to_html(index=False, float_format="%.3f"),
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html)

    return html
