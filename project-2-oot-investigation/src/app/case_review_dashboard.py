"""
Lightweight case review dashboard for investigation outputs.

This module provides functions to render investigation cases
for interactive review. In a production setting this would be
a Streamlit or Panel app; here it produces standalone HTML
review pages per case.
"""

from pathlib import Path
import pandas as pd
from reports.generate_investigation_report import generate_investigation_report
from analytics.detect_oot_events import summarize_suspects
from analytics.peer_batch_compare import select_peer_lots, compute_comparisons, top_differences
from features.build_feature_store import get_modeling_columns
from models.rank_contributors import prepare_labels, train_contributor_model, rank_contributors


def run_case_review(
    features: pd.DataFrame,
    lot_id: str,
    response_var: str = "dissolution",
    output_dir: Path | None = None,
    config: dict | None = None,
) -> dict:
    """Run a full investigation case for a single lot and produce outputs.

    Returns dict with keys: suspect_row, peers, peer_metadata,
    comparisons, contributors, html_report.
    """
    suspect_row, peers, peer_meta = select_peer_lots(features, lot_id, config=config)

    feature_cols = get_modeling_columns(features, target=response_var)
    comparisons = compute_comparisons(suspect_row, peers, feature_cols)
    top_diffs = top_differences(comparisons, n=10)

    labels = prepare_labels(features, response_var=response_var)
    model_result = train_contributor_model(features, feature_cols, labels)
    contribs = rank_contributors(model_result, top_n=10)

    out_path = None
    if output_dir is not None:
        out_path = Path(output_dir) / f"investigation_{lot_id}.html"

    html = generate_investigation_report(
        lot_id=lot_id,
        suspect_row=suspect_row,
        peer_metadata=peer_meta,
        comparisons=top_diffs,
        contributors=contribs,
        response_var=response_var,
        config=config,
        output_path=out_path,
    )

    return {
        "suspect_row": suspect_row,
        "peers": peers,
        "peer_metadata": peer_meta,
        "comparisons": comparisons,
        "top_differences": top_diffs,
        "contributors": contribs,
        "model_result": model_result,
        "html_report": html,
    }
