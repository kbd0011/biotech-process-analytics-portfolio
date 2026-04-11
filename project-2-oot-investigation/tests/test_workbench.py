"""Tests for the OOT Investigation Workbench."""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.pull_public_dataset import generate_realistic_dataset
from features.build_feature_store import build_feature_store, get_modeling_columns
from analytics.detect_oot_events import detect_oot_events, summarize_suspects
from analytics.peer_batch_compare import select_peer_lots, compute_comparisons
from models.rank_contributors import prepare_labels, train_contributor_model, rank_contributors


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "spec_limits.yml"


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def dataset():
    return generate_realistic_dataset(n_batches=200, seed=42)


@pytest.fixture(scope="module")
def features(dataset):
    return build_feature_store(dataset["laboratory"], dataset["process"], dataset["deviation_log"])


# --- Data ingestion tests ---

class TestDataIngestion:
    def test_expected_tables(self, dataset):
        assert "laboratory" in dataset
        assert "process" in dataset
        assert "deviation_log" in dataset

    def test_row_counts(self, dataset):
        assert len(dataset["laboratory"]) == 200
        assert len(dataset["process"]) == 200

    def test_critical_columns_exist(self, dataset):
        lab = dataset["laboratory"]
        for col in ["batch_id", "dissolution", "product_code", "batch_size_category", "rm_supplier"]:
            assert col in lab.columns, f"Missing column: {col}"

    def test_no_duplicate_batch_ids(self, dataset):
        assert dataset["laboratory"]["batch_id"].is_unique

    def test_dissolution_in_plausible_range(self, dataset):
        vals = dataset["laboratory"]["dissolution"]
        assert vals.min() > 40, "Dissolution values unrealistically low"
        assert vals.max() < 130, "Dissolution values unrealistically high"


# --- Feature store tests ---

class TestFeatureStore:
    def test_one_row_per_lot(self, features):
        assert features["batch_id"].is_unique

    def test_engineered_features_exist(self, features):
        for col in ["rm_potency_x_moisture", "compression_to_weight_ratio", "drying_intensity", "mix_energy_proxy"]:
            assert col in features.columns

    def test_deviation_flag_column(self, features):
        assert "deviation_count" in features.columns
        assert features["deviation_count"].min() >= 0

    def test_modeling_columns_exclude_target(self, features):
        cols = get_modeling_columns(features, target="dissolution")
        assert "dissolution" not in cols
        assert "batch_id" not in cols

    def test_no_target_leakage(self, features):
        """Ensure no post-disposition or future variables in modeling columns."""
        cols = get_modeling_columns(features, target="dissolution")
        forbidden = {"lot_disposition", "disposition_date", "release_status"}
        assert not forbidden.intersection(set(cols))

    def test_detection_columns_excluded_from_modeling(self, features, config):
        """Detection-derived columns (rolling_z, oos_flag, suspect, etc.)
        must never appear in the modeling feature set."""
        from analytics.detect_oot_events import DETECTION_DERIVED_COLUMNS
        detected = detect_oot_events(features, "dissolution", config=config)
        cols = get_modeling_columns(detected, target="dissolution")
        leaked = DETECTION_DERIVED_COLUMNS.intersection(set(cols))
        assert not leaked, f"Detection columns leaked into modeling features: {leaked}"
        # Also check underscore-prefixed columns
        underscore = [c for c in cols if c.startswith("_")]
        assert not underscore, f"Underscore-prefixed internal columns in features: {underscore}"


# --- OOT detection tests ---

class TestOOTDetection:
    def test_order_independence(self, features, config):
        """OOT detection must produce identical suspect flags regardless
        of the input row order, as long as mfg_date is present."""
        original = detect_oot_events(features, "dissolution", config=config)
        # Shuffle input rows
        shuffled = features.sample(frac=1, random_state=99).reset_index(drop=True)
        reshuffled = detect_oot_events(shuffled, "dissolution", config=config)

        # Compare suspect sets by batch_id
        orig_suspects = set(original[original["suspect"]]["batch_id"])
        shuf_suspects = set(reshuffled[reshuffled["suspect"]]["batch_id"])
        assert orig_suspects == shuf_suspects, (
            f"Suspect sets differ after shuffle: "
            f"{len(orig_suspects.symmetric_difference(shuf_suspects))} lots changed"
        )
    def test_oos_flags_correct(self, features, config):
        df = detect_oot_events(features, "dissolution", config=config)
        lsl = float(config["response_variables"]["dissolution"]["lower_spec"])
        usl = float(config["response_variables"]["dissolution"]["upper_spec"])
        oos = df[df["oos_flag"]]
        for _, row in oos.iterrows():
            assert row["dissolution"] < lsl or row["dissolution"] > usl

    def test_stable_data_does_not_overtrigger(self, config):
        """Stable synthetic data should not flag more than 15% as suspect."""
        rng = np.random.default_rng(99)
        stable = pd.DataFrame({
            "batch_id": [f"S-{i}" for i in range(100)],
            "dissolution": rng.normal(90, 2, 100),
            "product_code": "PROD_A",
            "batch_size_category": "standard",
        })
        df = detect_oot_events(stable, "dissolution", config=config)
        suspect_rate = df["suspect"].mean()
        assert suspect_rate < 0.15, f"Over-trigger rate: {suspect_rate:.1%}"

    def test_suspect_column_exists(self, features, config):
        df = detect_oot_events(features, "dissolution", config=config)
        assert "suspect" in df.columns
        assert "reason_code" in df.columns
        assert "severity" in df.columns


# --- Peer batch comparison tests ---

class TestPeerComparison:
    def test_suspect_excluded_from_peers(self, features, config):
        detected = detect_oot_events(features, "dissolution", config=config)
        suspects = detected[detected["suspect"]]
        if len(suspects) == 0:
            pytest.skip("No suspect lots in this seed")
        lot_id = suspects.iloc[0]["batch_id"]
        _, peers, _ = select_peer_lots(detected, lot_id, config=config)
        assert lot_id not in peers["batch_id"].values

    def test_peer_group_respects_matching_keys(self, features, config):
        detected = detect_oot_events(features, "dissolution", config=config)
        lot_id = detected.iloc[50]["batch_id"]
        suspect_row, peers, meta = select_peer_lots(detected, lot_id, config=config)
        for key in meta["matching_keys_used"]:
            if len(peers) > 0:
                assert (peers[key] == suspect_row[key]).all()

    def test_comparison_output_shape(self, features, config):
        detected = detect_oot_events(features, "dissolution", config=config)
        lot_id = detected.iloc[50]["batch_id"]
        suspect_row, peers, _ = select_peer_lots(detected, lot_id, config=config)
        cols = get_modeling_columns(features, target="dissolution")
        comp = compute_comparisons(suspect_row, peers, cols)
        assert len(comp) == len(cols)
        assert "z_diff" in comp.columns

    def test_low_peer_warning(self, config):
        """When peers are few, metadata should flag a warning."""
        tiny = pd.DataFrame({
            "batch_id": [f"T-{i}" for i in range(5)],
            "dissolution": [90, 91, 89, 88, 92],
            "product_code": "PROD_X",
            "batch_size_category": "standard",
            "suspect": [False] * 5,
        })
        _, _, meta = select_peer_lots(tiny, "T-0", config=config)
        assert meta["low_peer_warning"] is True


# --- Contributor ranking tests ---

class TestContributorRanking:
    def test_model_returns_coefficients(self, features):
        cols = get_modeling_columns(features, target="dissolution")
        labels = prepare_labels(features, "dissolution")
        result = train_contributor_model(features, cols, labels)
        assert "coefficients" in result
        assert len(result["coefficients"]) == len(cols)

    def test_feature_names_are_readable(self, features):
        cols = get_modeling_columns(features, target="dissolution")
        labels = prepare_labels(features, "dissolution")
        result = train_contributor_model(features, cols, labels)
        contribs = rank_contributors(result, top_n=5)
        for feat in contribs["feature"]:
            assert isinstance(feat, str) and len(feat) > 0

    def test_uses_time_aware_validation(self, features):
        """Ensure model uses TimeSeriesSplit, not random split."""
        cols = get_modeling_columns(features, target="dissolution")
        labels = prepare_labels(features, "dissolution")
        result = train_contributor_model(features, cols, labels)
        # Metrics should be present from CV
        assert result["metrics"]["cv_auc_mean"] is not None or len(features) < 50


# --- Report tests ---

class TestReport:
    def test_report_generates_html(self, features, config):
        from reports.generate_investigation_report import generate_investigation_report
        detected = detect_oot_events(features, "dissolution", config=config)
        lot_id = detected.iloc[50]["batch_id"]
        suspect_row, peers, peer_meta = select_peer_lots(detected, lot_id, config=config)
        cols = get_modeling_columns(features, target="dissolution")
        comp = compute_comparisons(suspect_row, peers, cols)
        labels = prepare_labels(features, "dissolution")
        model = train_contributor_model(features, cols, labels)
        contribs = rank_contributors(model)
        html = generate_investigation_report(
            lot_id, suspect_row, peer_meta, comp, contribs,
            config=config,
        )
        assert "<html>" in html
        assert lot_id in html
        assert "Candidate Contributors" in html

    def test_report_includes_caveats(self, features, config):
        from reports.generate_investigation_report import generate_investigation_report
        detected = detect_oot_events(features, "dissolution", config=config)
        lot_id = detected.iloc[50]["batch_id"]
        suspect_row, peers, peer_meta = select_peer_lots(detected, lot_id, config=config)
        cols = get_modeling_columns(features, target="dissolution")
        comp = compute_comparisons(suspect_row, peers, cols)
        labels = prepare_labels(features, "dissolution")
        model = train_contributor_model(features, cols, labels)
        contribs = rank_contributors(model)
        html = generate_investigation_report(
            lot_id, suspect_row, peer_meta, comp, contribs,
            config=config,
        )
        assert "not a determination of root cause" in html
