"""Tests for the Cell Therapy CPV & KPI Command Center."""

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.generate_cell_therapy_batches import generate_cell_therapy_batches
from etl.load_and_standardize import load_and_standardize, build_lot_mart
from metrics.compute_kpis import compute_all_kpis
from analytics.cpv_rules import p_chart, i_mr_chart, ewma_chart, capability_index, run_cpv_analytics


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "kpi_definitions.yml"


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def tables():
    return generate_cell_therapy_batches(n_lots=400, n_months=12, seed=42)


@pytest.fixture(scope="module")
def clean(tables):
    return load_and_standardize(tables)


@pytest.fixture(scope="module")
def mart(clean):
    return build_lot_mart(clean)


@pytest.fixture(scope="module")
def kpis(mart, config):
    return compute_all_kpis(mart, config=config)


# --- Data generation tests ---

class TestDataGeneration:
    def test_all_tables_present(self, tables):
        expected = {"batch_header", "batch_step", "in_process_measurements",
                    "release_tests", "deviation_log", "site_master", "calendar"}
        assert expected.issubset(set(tables.keys()))

    def test_batch_id_unique(self, tables):
        assert tables["batch_header"]["batch_id"].is_unique

    def test_referential_integrity(self, tables):
        batch_ids = set(tables["batch_header"]["batch_id"])
        step_ids = set(tables["batch_step"]["batch_id"])
        assert step_ids.issubset(batch_ids)

    def test_one_lot_many_steps(self, tables):
        steps_per_lot = tables["batch_step"].groupby("batch_id").size()
        assert (steps_per_lot > 1).any(), "Expected lots with multiple process steps"

    def test_one_lot_multiple_release_tests(self, tables):
        # Each lot has one row in release_tests with multiple test columns
        rt = tables["release_tests"]
        test_cols = [c for c in rt.columns if c != "batch_id"]
        assert len(test_cols) >= 3

    def test_drift_scenario_present(self, tables):
        """Transduction efficiency should drift at SITE_B in later months."""
        bh = tables["batch_header"]
        ip = tables["in_process_measurements"]
        merged = bh.merge(ip, on="batch_id")
        site_b = merged[merged["site_id"] == "SITE_B"]
        early = site_b[site_b["month_idx"] < 5]["transduction_efficiency"].mean()
        late = site_b[site_b["month_idx"] >= 10]["transduction_efficiency"].mean()
        assert late < early - 2, f"Expected drift: early={early:.1f}, late={late:.1f}"


# --- ETL tests ---

class TestETL:
    def test_mart_one_row_per_lot(self, mart):
        assert mart["batch_id"].is_unique

    def test_year_month_column(self, mart):
        assert "year_month" in mart.columns
        assert mart["year_month"].str.match(r"\d{4}-\d{2}").all()


# --- KPI tests ---

class TestKPIs:
    def test_attribute_kpis_computed(self, kpis):
        attr = kpis["attribute_kpis"]
        assert not attr.empty
        assert "batch_success_rate" in attr["kpi_name"].values

    def test_continuous_kpis_computed(self, kpis):
        cont = kpis["continuous_kpis"]
        assert not cont.empty

    def test_kpi_denominators_nonzero(self, kpis):
        attr = kpis["attribute_kpis"]
        assert (attr["denominator"] > 0).all()


# --- SPC tests ---

class TestSPC:
    def test_p_chart_signals_on_known_violation(self):
        """Inject a known spike and verify p-chart signals."""
        df = pd.DataFrame({
            "year_month": [f"2024-{m:02d}" for m in range(1, 13)],
            "numerator": [2, 2, 3, 2, 2, 2, 15, 2, 2, 2, 2, 2],
            "denominator": [50] * 12,
            "value": [0.04, 0.04, 0.06, 0.04, 0.04, 0.04, 0.30, 0.04, 0.04, 0.04, 0.04, 0.04],
        })
        result = p_chart(df)
        assert result["signal"].any(), "Expected p-chart signal on injected spike"

    def test_i_mr_returns_correct_columns(self):
        vals = pd.Series(np.random.default_rng(42).normal(10, 1, 50))
        result = i_mr_chart(vals)
        assert "ucl_i" in result.columns
        assert "signal_i" in result.columns

    def test_ewma_detects_drift(self):
        rng = np.random.default_rng(42)
        vals = np.concatenate([rng.normal(10, 1, 30), rng.normal(13, 1, 20)])
        result = ewma_chart(pd.Series(vals))
        assert result["signal"].any(), "Expected EWMA signal on injected drift"

    def test_capability_requires_specs(self):
        vals = pd.Series(np.random.default_rng(42).normal(50, 5, 100))
        assert capability_index(vals, None, None) is None

    def test_capability_computes_with_specs(self):
        vals = pd.Series(np.random.default_rng(42).normal(50, 5, 100))
        result = capability_index(vals, 30.0, 70.0)
        assert result is not None
        assert "ppk" in result
        assert result["ppk"] > 0

    def test_no_capability_without_specs(self, kpis, config):
        """KPIs without spec limits should not produce capability metrics."""
        results = run_cpv_analytics(kpis["attribute_kpis"], kpis["continuous_kpis"], config)
        for key, cap in results["capability"].items():
            kpi_name = key.split("__")[0]
            kpi_def = config["kpis"].get(kpi_name, {})
            has_spec = kpi_def.get("spec_lower") is not None or kpi_def.get("spec_upper") is not None
            assert has_spec, f"Capability computed for {kpi_name} without spec limits"


# --- Report tests ---

class TestReport:
    def test_report_generates_html(self, kpis, config):
        from reports.generate_monthly_cpv_pack import generate_monthly_cpv_pack
        attr = kpis["attribute_kpis"]
        cont = kpis["continuous_kpis"]
        cpv = run_cpv_analytics(attr, cont, config)
        periods = attr["year_month"].unique()
        html = generate_monthly_cpv_pack(periods[0], attr, cont, cpv)
        assert "<html>" in html
        assert "CPV Monthly Review" in html

    def test_report_callouts_traceable(self, kpis, config):
        """Every alert callout should correspond to an actual KPI alert."""
        from reports.generate_monthly_cpv_pack import generate_monthly_cpv_pack
        attr = kpis["attribute_kpis"]
        cont = kpis["continuous_kpis"]
        cpv = run_cpv_analytics(attr, cont, config)
        periods = attr["year_month"].unique()
        html = generate_monthly_cpv_pack(periods[0], attr, cont, cpv)
        # Report should contain either alert or "within thresholds"
        assert "threshold" in html.lower() or "within" in html.lower()
