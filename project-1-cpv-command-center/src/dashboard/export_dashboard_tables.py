"""
Export dashboard-ready tables and generate a self-contained interactive
HTML dashboard using Plotly for CPV review.

The dashboard includes:
- Executive KPI summary with threshold alerts
- Site-to-site comparison charts
- SPC control chart panels (p-chart, I-MR, EWMA)
- Lot-level drill-through table
- Capability summary
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def export_dashboard_tables(
    mart: pd.DataFrame,
    attribute_kpis: pd.DataFrame,
    continuous_kpis: pd.DataFrame,
    cpv_results: dict,
    output_dir: Path,
) -> list[str]:
    """Export all dashboard-ready tables to output_dir. Returns list of exported files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    if not attribute_kpis.empty:
        attribute_kpis.to_csv(output_dir / "exec_attribute_kpis.csv", index=False)
        exported.append("exec_attribute_kpis.csv")

    if not continuous_kpis.empty:
        continuous_kpis.to_csv(output_dir / "exec_continuous_kpis.csv", index=False)
        exported.append("exec_continuous_kpis.csv")

    mart.to_csv(output_dir / "lot_drill_through.csv", index=False)
    exported.append("lot_drill_through.csv")

    for chart_type in ["p_charts", "i_mr_charts", "ewma_charts"]:
        for key, df in cpv_results.get(chart_type, {}).items():
            fname = f"spc_{chart_type}_{key}.csv"
            df.to_csv(output_dir / fname, index=False)
            exported.append(fname)

    cap_data = cpv_results.get("capability", {})
    if cap_data:
        cap_df = pd.DataFrame([{"kpi_site": k, **v} for k, v in cap_data.items()])
        cap_df.to_csv(output_dir / "capability_summary.csv", index=False)
        exported.append("capability_summary.csv")

    return exported


def generate_interactive_dashboard(
    mart: pd.DataFrame,
    attribute_kpis: pd.DataFrame,
    continuous_kpis: pd.DataFrame,
    cpv_results: dict,
    output_path: Path | None = None,
) -> str | None:
    """Generate a self-contained interactive HTML dashboard.

    Returns the HTML string, or None if Plotly is not installed.
    """
    if not HAS_PLOTLY:
        return None

    html_parts = [_dashboard_header()]

    # --- Executive KPI cards ---
    html_parts.append(_render_exec_section(attribute_kpis))

    # --- Site comparison ---
    html_parts.append(_render_site_comparison(attribute_kpis))

    # --- SPC charts ---
    html_parts.append(_render_spc_section(cpv_results, attribute_kpis, continuous_kpis))

    # --- Capability ---
    html_parts.append(_render_capability_section(cpv_results))

    # --- Lot drill-through ---
    html_parts.append(_render_lot_table(mart))

    html_parts.append(_dashboard_footer())

    html = "\n".join(html_parts)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

    return html


# ---- internal rendering helpers ----

_CSS = """
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f4f6f9; color: #333; }
.header { background: linear-gradient(135deg, #1a3a5c 0%, #2c6faa 100%); color: #fff;
           padding: 30px 40px; }
.header h1 { margin: 0 0 6px 0; font-size: 28px; }
.header p { margin: 0; opacity: 0.85; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px 30px; }
.section { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
           padding: 24px 28px; margin-bottom: 24px; }
.section h2 { color: #1a3a5c; margin-top: 0; font-size: 20px; border-bottom: 2px solid #e8ecf1;
              padding-bottom: 8px; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.card { flex: 1; min-width: 180px; background: #f8f9fb; border-radius: 6px; padding: 16px;
        border-left: 4px solid #2c6faa; }
.card.alert { border-left-color: #dc3545; background: #fef2f2; }
.card .label { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 26px; font-weight: 700; color: #1a3a5c; margin: 4px 0; }
.card.alert .value { color: #dc3545; }
.card .detail { font-size: 12px; color: #888; }
table.drill { width: 100%; border-collapse: collapse; font-size: 13px; }
table.drill th { background: #f0f4f8; padding: 8px 10px; text-align: left; border-bottom: 2px solid #ddd;
                 position: sticky; top: 0; }
table.drill td { padding: 6px 10px; border-bottom: 1px solid #eee; }
table.drill tr:hover { background: #f5f8ff; }
.tab-bar { display: flex; gap: 0; border-bottom: 2px solid #e8ecf1; margin-bottom: 16px; }
.tab-bar button { background: none; border: none; padding: 10px 20px; font-size: 14px;
                  cursor: pointer; color: #666; border-bottom: 2px solid transparent; margin-bottom: -2px; }
.tab-bar button.active { color: #1a3a5c; border-bottom-color: #2c6faa; font-weight: 600; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.note { font-size: 12px; color: #999; margin-top: 12px; font-style: italic; }
</style>
"""

_JS_TABS = """
<script>
function switchTab(group, idx) {
  document.querySelectorAll('[data-group="'+group+'"]').forEach(function(el) {
    el.classList.remove('active');
  });
  document.querySelectorAll('[data-group="'+group+'"][data-idx="'+idx+'"]').forEach(function(el) {
    el.classList.add('active');
  });
}
</script>
"""


def _dashboard_header() -> str:
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>CPV & KPI Command Center</title>{_CSS}</head><body>
<div class="header">
<h1>Cell Therapy CPV &amp; KPI Command Center</h1>
<p>Multi-site manufacturing performance dashboard &bull; Literature-derived synthetic data for portfolio demonstration</p>
</div><div class="container">"""


def _dashboard_footer() -> str:
    return f"""{_JS_TABS}
<div class="note">Generated from literature-derived synthetic cell-therapy manufacturing data.
KPI definitions, SPC logic, and reporting structure reflect operationally realistic Stage 3 / CPV workflows.</div>
</div></body></html>"""


def _render_exec_section(attribute_kpis: pd.DataFrame) -> str:
    if attribute_kpis.empty:
        return ""

    # Latest period per KPI per site
    latest = attribute_kpis.sort_values("year_month").groupby(["kpi_name", "site_id"]).last().reset_index()

    cards_html = []
    for _, row in latest.iterrows():
        alert_cls = "card alert" if row.get("alert") else "card"
        val = row["value"]
        display = f"{val:.1%}" if val <= 1 else f"{val:.2f}"
        cards_html.append(f"""<div class="{alert_cls}">
<div class="label">{row['display_name']} — {row['site_id']}</div>
<div class="value">{display}</div>
<div class="detail">Threshold: {row['threshold']:.1%} | Period: {row['year_month']}</div>
</div>""")

    return f"""<div class="section"><h2>Executive KPI Summary (Latest Period)</h2>
<div class="cards">{"".join(cards_html)}</div></div>"""


def _render_site_comparison(attribute_kpis: pd.DataFrame) -> str:
    if attribute_kpis.empty or not HAS_PLOTLY:
        return ""

    kpi_names = attribute_kpis["kpi_name"].unique()
    figs_html = []

    for kpi in kpi_names[:4]:  # limit to top 4
        subset = attribute_kpis[attribute_kpis["kpi_name"] == kpi].copy()
        if subset.empty:
            continue
        display = subset.iloc[0]["display_name"]
        fig = go.Figure()
        for site in sorted(subset["site_id"].unique()):
            s = subset[subset["site_id"] == site].sort_values("year_month")
            fig.add_trace(go.Scatter(
                x=s["year_month"], y=s["value"], name=site, mode="lines+markers",
                line=dict(width=2), marker=dict(size=5),
            ))
        threshold = float(subset.iloc[0]["threshold"])
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text="Threshold", annotation_position="top left")
        fig.update_layout(
            title=display, height=300, margin=dict(l=50, r=30, t=50, b=40),
            yaxis_tickformat=".0%", template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
        )
        figs_html.append(fig.to_html(full_html=False, include_plotlyjs=False))

    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    charts = "</div><div class='section'>".join(figs_html)
    return f"""{plotly_js}<div class="section"><h2>Site-to-Site KPI Trends</h2>
{charts}</div>"""


def _render_spc_section(cpv_results: dict, attribute_kpis: pd.DataFrame,
                        continuous_kpis: pd.DataFrame) -> str:
    if not HAS_PLOTLY:
        return ""

    figs = []

    # P-charts
    for key, df in list(cpv_results.get("p_charts", {}).items())[:3]:
        kpi, site = key.split("__")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["value"], name="Rate",
            mode="lines+markers", marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["ucl"], name="UCL",
            mode="lines", line=dict(dash="dash", color="red", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["lcl"], name="LCL",
            mode="lines", line=dict(dash="dash", color="red", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["p_bar"], name="Center",
            mode="lines", line=dict(dash="dot", color="gray", width=1),
        ))
        # Highlight signals
        signals = df[df["signal"]]
        if not signals.empty:
            fig.add_trace(go.Scatter(
                x=signals.index.tolist(), y=signals["value"], name="Signal",
                mode="markers", marker=dict(size=10, color="red", symbol="x"),
            ))
        fig.update_layout(
            title=f"P-Chart: {kpi} ({site})", height=280,
            margin=dict(l=50, r=30, t=50, b=40), template="plotly_white",
            yaxis_tickformat=".1%", legend=dict(orientation="h", y=-0.2),
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    # EWMA charts
    for key, df in list(cpv_results.get("ewma_charts", {}).items())[:2]:
        kpi, site = key.split("__")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["value"], name="Observed",
            mode="markers", marker=dict(size=3, color="lightgray"),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["ewma"], name="EWMA",
            mode="lines", line=dict(width=2, color="#2c6faa"),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["ucl"], name="UCL",
            mode="lines", line=dict(dash="dash", color="red", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), y=df["lcl"], name="LCL",
            mode="lines", line=dict(dash="dash", color="red", width=1),
        ))
        signals = df[df["signal"]]
        if not signals.empty:
            fig.add_trace(go.Scatter(
                x=signals.index.tolist(), y=signals["ewma"], name="Signal",
                mode="markers", marker=dict(size=10, color="red", symbol="x"),
            ))
        fig.update_layout(
            title=f"EWMA: {kpi} ({site})", height=280,
            margin=dict(l=50, r=30, t=50, b=40), template="plotly_white",
            legend=dict(orientation="h", y=-0.2),
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    if not figs:
        return ""

    charts = "</div><div class='section'>".join(figs)
    return f"""<div class="section"><h2>SPC Control Charts</h2>
{charts}</div>"""


def _render_capability_section(cpv_results: dict) -> str:
    cap = cpv_results.get("capability", {})
    if not cap:
        return """<div class="section"><h2>Capability Summary</h2>
<p>No capability metrics computed (specification limits required).</p></div>"""

    rows = []
    for key, vals in cap.items():
        kpi, site = key.split("__")
        ppk = vals.get("ppk")
        ppk_class = ""
        if ppk is not None and ppk < 1.0:
            ppk_class = ' style="color:#dc3545;font-weight:700"'
        elif ppk is not None and ppk >= 1.33:
            ppk_class = ' style="color:#28a745;font-weight:700"'
        rows.append(f"""<tr><td>{kpi}</td><td>{site}</td>
<td>{vals.get('mean',''):.3f}</td><td>{vals.get('std',''):.3f}</td>
<td>{vals.get('spec_lower','—')}</td><td>{vals.get('spec_upper','—')}</td>
<td{ppk_class}>{ppk:.3f}</td><td>{vals.get('n','')}</td></tr>""")

    return f"""<div class="section"><h2>Capability Summary</h2>
<table class="drill"><thead><tr>
<th>KPI</th><th>Site</th><th>Mean</th><th>Std</th><th>LSL</th><th>USL</th><th>Ppk</th><th>N</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table>
<p class="note">Ppk &lt; 1.0 highlighted in red. Ppk ≥ 1.33 highlighted in green.
Capability computed only where specification limits are defined.</p></div>"""


def _render_lot_table(mart: pd.DataFrame) -> str:
    display_cols = [c for c in [
        "batch_id", "site_id", "year_month", "disposition", "cycle_time_days",
        "transduction_efficiency", "harvest_viability", "potency_pct_killing",
        "deviation_count",
    ] if c in mart.columns]

    sample = mart[display_cols].head(50)
    rows = []
    for _, r in sample.iterrows():
        cells = []
        for c in display_cols:
            v = r[c]
            style = ""
            if c == "disposition" and v == "rejected":
                style = ' style="color:#dc3545"'
            elif c == "disposition" and v == "terminated":
                style = ' style="color:#e67e22"'
            if isinstance(v, float):
                cells.append(f"<td{style}>{v:.2f}</td>")
            else:
                cells.append(f"<td{style}>{v}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    headers = "".join(f"<th>{c}</th>" for c in display_cols)
    return f"""<div class="section"><h2>Lot Drill-Through (first 50 lots)</h2>
<div style="max-height:400px;overflow-y:auto">
<table class="drill"><thead><tr>{headers}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></div>"""
