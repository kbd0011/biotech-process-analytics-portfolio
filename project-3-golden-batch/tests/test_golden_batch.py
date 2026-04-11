"""Tests for the Golden Batch Early Warning System."""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sim.generate_batch_trajectories import generate_batch_trajectories
from features.align_and_extract_features import (
    extract_full_features, extract_partial_features, get_feature_columns,
)
from monitoring.build_golden_batch_model import build_golden_batch_model, compute_contributions
from monitoring.score_running_batches import score_batch, score_cohort
from models.predict_release_risk import train_early_risk_model


@pytest.fixture(scope="module")
def sim_data():
    traj, outcomes = generate_batch_trajectories(n_batches=150, seed=42)
    return traj, outcomes


@pytest.fixture(scope="module")
def features(sim_data):
    traj, _ = sim_data
    return extract_full_features(traj)


@pytest.fixture(scope="module")
def model(features, sim_data):
    _, outcomes = sim_data
    fcols = get_feature_columns(features)
    return build_golden_batch_model(features, outcomes, fcols)


# --- Simulation tests ---

class TestSimulation:
    def test_trajectory_shape(self, sim_data):
        traj, outcomes = sim_data
        assert len(outcomes) == 150
        assert len(traj) > 0

    def test_outcomes_have_failure_modes(self, sim_data):
        _, outcomes = sim_data
        fm = outcomes["failure_mode"].value_counts()
        assert "none" in fm.index
        assert len(fm) > 1, "Expected multiple failure modes"

    def test_values_in_biological_range(self, sim_data):
        traj, _ = sim_data
        assert traj["viability_pct"].min() >= 5
        assert traj["viability_pct"].max() <= 100
        assert traj["glucose_mm"].min() >= 0
        assert traj["ph"].min() >= 6.0
        assert traj["ph"].max() <= 8.0

    def test_failure_modes_produce_intended_signatures(self, sim_data):
        traj, outcomes = sim_data
        # Over-activation should lower potency
        over_act = outcomes[outcomes["failure_mode"] == "over_activation"]
        normal = outcomes[outcomes["failure_mode"] == "none"]
        if len(over_act) > 0 and len(normal) > 5:
            assert over_act["final_potency"].mean() < normal["final_potency"].mean()


# --- Feature extraction tests ---

class TestFeatures:
    def test_one_row_per_batch(self, features, sim_data):
        _, outcomes = sim_data
        assert len(features) == len(outcomes)
        assert features["batch_id"].is_unique

    def test_partial_features_no_leakage(self, sim_data):
        traj, _ = sim_data
        partial = extract_partial_features(traj, horizon_day=2.0)
        assert not partial.empty
        # Should not have 'final' features
        assert all("final" not in c for c in partial.columns)

    def test_partial_uses_only_early_data(self, sim_data):
        traj, _ = sim_data
        partial = extract_partial_features(traj, horizon_day=2.0)
        # Verify horizon_day column
        assert "horizon_day" in partial.columns
        assert (partial["horizon_day"] == 2.0).all()


# --- Monitoring model tests ---

class TestMonitoring:
    def test_model_trained_on_good_batches_only(self, model, sim_data):
        _, outcomes = sim_data
        n_good = outcomes["release_pass"].sum()
        assert model["n_train"] == n_good

    def test_healthy_batches_mostly_in_control(self, features, sim_data, model):
        _, outcomes = sim_data
        good_ids = outcomes[outcomes["release_pass"]]["batch_id"].values
        good_feats = features[features["batch_id"].isin(good_ids)]
        scores = score_cohort(good_feats, model)
        normal_rate = (scores["alert_state"] == "normal").mean()
        assert normal_rate > 0.80, f"Too many false alarms on good batches: {1-normal_rate:.1%}"

    def test_fault_batches_trigger_alerts(self, features, sim_data, model):
        _, outcomes = sim_data
        fault_ids = outcomes[outcomes["failure_mode"] != "none"]["batch_id"].values
        if len(fault_ids) == 0:
            pytest.skip("No fault batches in this seed")
        fault_feats = features[features["batch_id"].isin(fault_ids)]
        scores = score_cohort(fault_feats, model)
        alert_rate = (scores["alert_state"] != "normal").mean()
        assert alert_rate > 0.30, f"Fault detection rate too low: {alert_rate:.1%}"

    def test_contributions_identify_perturbed_vars(self, features, sim_data, model):
        _, outcomes = sim_data
        # Pick an over-activation fault batch
        oa = outcomes[outcomes["failure_mode"] == "over_activation"]
        if len(oa) == 0:
            pytest.skip("No over_activation batches")
        bid = oa.iloc[0]["batch_id"]
        row = features[features["batch_id"] == bid].iloc[0]

        fcols = model["feature_cols"]
        x = row[fcols].values.astype(float)
        x = np.where(np.isnan(x), 0.0, x)
        x_df = pd.DataFrame([x], columns=fcols)
        x_scaled = model["scaler"].transform(x_df).flatten()
        contribs = compute_contributions(x_scaled, model, fcols)

        # Top contributors should include viability or potency related features
        top5 = contribs.head(5)["feature"].tolist()
        relevant = [f for f in top5 if "viability" in f or "potency" in f or "vcd" in f]
        assert len(relevant) > 0, f"Expected relevant contributors in top 5, got: {top5}"


# --- Early risk prediction tests ---

class TestEarlyRisk:
    def test_model_trains_and_returns_metrics(self, sim_data):
        traj, outcomes = sim_data
        partial = extract_partial_features(traj, horizon_day=5.0)
        fcols = get_feature_columns(partial)
        result = train_early_risk_model(partial, outcomes, fcols)
        assert result["metrics"]["cv_auc_mean"] is not None
        assert len(result["coefficients"]) == len(fcols)

    def test_no_future_leakage_in_partial_features(self, sim_data):
        traj, _ = sim_data
        partial = extract_partial_features(traj, horizon_day=2.0)
        # No column should contain 'final'
        for col in partial.columns:
            assert "final" not in col, f"Leakage detected: {col}"


# --- Memo tests ---

class TestMemo:
    def test_memo_includes_intended_use_and_limitations(self, model, sim_data):
        from reports.model_validation_memo import generate_model_validation_memo
        _, outcomes = sim_data
        html = generate_model_validation_memo(model, outcomes)
        assert "Intended Use" in html
        assert "Limitations" in html
        assert "not" in html.lower()  # "not a replacement"


# --- Held-out validation tests ---

class TestHeldOutValidation:
    """Score an independent challenge cohort (different seed) against the
    model trained on the original cohort. This prevents the optimistic
    evaluation that comes from scoring training data against its own model."""

    @pytest.fixture(scope="class")
    def challenge_data(self):
        traj, outcomes = generate_batch_trajectories(n_batches=100, seed=999)
        return traj, outcomes

    @pytest.fixture(scope="class")
    def challenge_features(self, challenge_data):
        traj, _ = challenge_data
        return extract_full_features(traj)

    def test_false_alarm_rate_on_heldout_good(self, challenge_features, challenge_data, model):
        """False alarm rate on held-out healthy batches should stay below 25%."""
        _, outcomes = challenge_data
        good_ids = outcomes[outcomes["release_pass"]]["batch_id"].values
        good_feats = challenge_features[challenge_features["batch_id"].isin(good_ids)]
        if len(good_feats) < 5:
            pytest.skip("Not enough held-out good batches")
        scores = score_cohort(good_feats, model)
        false_alarm_rate = (scores["alert_state"] != "normal").mean()
        assert false_alarm_rate < 0.25, (
            f"False alarm rate on held-out good batches too high: {false_alarm_rate:.1%}"
        )

    def test_detection_rate_on_heldout_faults(self, challenge_features, challenge_data, model):
        """Detection rate on held-out fault batches should exceed 25%."""
        _, outcomes = challenge_data
        fault_ids = outcomes[outcomes["failure_mode"] != "none"]["batch_id"].values
        if len(fault_ids) < 3:
            pytest.skip("Not enough held-out fault batches")
        fault_feats = challenge_features[challenge_features["batch_id"].isin(fault_ids)]
        scores = score_cohort(fault_feats, model)
        detection_rate = (scores["alert_state"] != "normal").mean()
        assert detection_rate > 0.25, (
            f"Detection rate on held-out faults too low: {detection_rate:.1%}"
        )

    def test_heldout_separation(self, challenge_features, challenge_data, model):
        """Alert rate on fault batches should exceed alert rate on good batches."""
        _, outcomes = challenge_data
        good_ids = set(outcomes[outcomes["release_pass"]]["batch_id"])
        fault_ids = set(outcomes[outcomes["failure_mode"] != "none"]["batch_id"])

        good_feats = challenge_features[challenge_features["batch_id"].isin(good_ids)]
        fault_feats = challenge_features[challenge_features["batch_id"].isin(fault_ids)]
        if len(good_feats) < 5 or len(fault_feats) < 3:
            pytest.skip("Not enough held-out batches for separation test")

        good_scores = score_cohort(good_feats, model)
        fault_scores = score_cohort(fault_feats, model)

        good_alert = (good_scores["alert_state"] != "normal").mean()
        fault_alert = (fault_scores["alert_state"] != "normal").mean()
        assert fault_alert > good_alert, (
            f"No separation: fault alert={fault_alert:.1%}, good alert={good_alert:.1%}"
        )
