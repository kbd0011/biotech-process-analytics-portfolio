"""
Assemble an automated recurring CPV review pack with key charts and narrative callouts.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from jinja2 import Template


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{{ title }}</title>
<style>
body { font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #333; }
h1 { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 8px; }
h2 { color: #2c5f8a; margin-top: 30px; }
h3 { color: #3d7ab5; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
th { background: #f0f4f8; }
.alert { background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin: 10px 0; }
.ok { background: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; }
.info { background: #cce5ff; border-left: 4px solid #0d6efd; padding: 10px; margin: 10px 0; }
.footer { margin-top: 40px; font-size: 0.85em; color: #888; border-top: 1px solid #ddd; padding-top: 10px; }
</style></head>
<body>
<h1>{{ title }}</h1>
<p><strong>Reporting period:</strong> {{ period }}</p>
<p><strong>Generated:</strong> {{ timestamp }}</p>

<h2>Executive Summary</h2>
{% for callout in callouts %}
<div class="{{ callout.css_class }}">
<strong>{{ callout.kpi }}:</strong> {{ callout.message }}
</div>
{% endfor %}

<h2>Attribute KPI Summary</h2>
{{ attr_table }}

<h2>Capability Summary</h2>
{% if cap_table %}
{{ cap_table }}
{% else %}
<div class="info">No capability metrics computed for this period (spec limits required).</div>
{% endif %}

<h2>SPC Signals</h2>
{% if spc_signals %}
{{ spc_signals }}
{% else %}
<div class="ok">No SPC signals detected this period.</div>
{% endif %}

<div class="footer">
<p>This CPV review pack is generated from literature-derived synthetic data for portfolio demonstration.
All KPI definitions, control logic, and reporting structure are designed to reflect
operationally realistic Stage 3 / ongoing process verification workflows.</p>
</div>
</body></html>
"""


def generate_monthly_cpv_pack(
    period: str,
    attribute_kpis: pd.DataFrame,
    continuous_kpis: pd.DataFrame,
    cpv_results: dict,
    output_path: Path | None = None,
) -> str:
    """Generate an HTML CPV review pack for a specific period.

    Args:
        period: year-month string like "2024-01"
        attribute_kpis: computed attribute KPIs
        continuous_kpis: computed continuous KPIs
        cpv_results: dict from run_cpv_analytics
        output_path: optional path to save HTML file

    Returns the rendered HTML.
    """
    # Filter to period
    attr_period = attribute_kpis[attribute_kpis["year_month"] == period] if not attribute_kpis.empty else pd.DataFrame()

    # Generate callouts
    callouts = []
    for _, row in attr_period.iterrows():
        if row.get("alert"):
            callouts.append({
                "kpi": row["display_name"],
                "message": f"Site {row['site_id']}: {row['value']:.1%} (threshold: {row['threshold']:.1%})",
                "css_class": "alert",
            })

    # Check SPC signals
    spc_signal_records = []
    for chart_type in ["p_charts", "i_mr_charts", "ewma_charts"]:
        for key, df in cpv_results.get(chart_type, {}).items():
            signal_col = "signal" if "signal" in df.columns else "signal_i" if "signal_i" in df.columns else None
            if signal_col and df[signal_col].any():
                n_signals = df[signal_col].sum()
                spc_signal_records.append({
                    "chart": key,
                    "type": chart_type,
                    "signal_count": int(n_signals),
                })

    if not callouts:
        callouts.append({
            "kpi": "Overall",
            "message": "All attribute KPIs within thresholds for this period.",
            "css_class": "ok",
        })

    # Build tables
    attr_html = attr_period.to_html(index=False, float_format="%.4f") if not attr_period.empty else ""

    cap_data = cpv_results.get("capability", {})
    cap_html = ""
    if cap_data:
        cap_df = pd.DataFrame([{"metric": k, **v} for k, v in cap_data.items()])
        cap_html = cap_df.to_html(index=False, float_format="%.3f")

    spc_html = ""
    if spc_signal_records:
        spc_df = pd.DataFrame(spc_signal_records)
        spc_html = spc_df.to_html(index=False)

    template = Template(_REPORT_TEMPLATE)
    html = template.render(
        title=f"CPV Monthly Review — {period}",
        period=period,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        callouts=callouts,
        attr_table=attr_html,
        cap_table=cap_html,
        spc_signals=spc_html,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html
