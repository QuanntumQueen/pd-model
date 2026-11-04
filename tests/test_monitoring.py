# Тесты мониторинга дрейфа (src/monitoring/psi.py)
import numpy as np

from src.config import TARGET
from src.monitoring.psi import FEATURES, psi, psi_report, psi_status, simulate_drift


def test_psi_same_data_is_zero():
    x = np.random.default_rng(42).normal(size=5000)
    assert psi(x, x) < 0.001


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    assert psi(rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)) > 0.25


def test_psi_works_for_discrete_values():
    value = psi([0, 0, 0, 1, 2] * 100, [0, 1, 1, 2, 2] * 100)
    assert np.isfinite(value) and value > 0


def test_psi_status_thresholds():
    assert psi_status(0.05) == "стабильно"
    assert psi_status(0.15) == "умеренный сдвиг"
    assert psi_status(0.40) == "значительный сдвиг"


def test_report_stable_on_same_data(sample_df):
    X = sample_df.drop(columns=[TARGET])
    report = psi_report(X, X)
    assert list(report["feature"]) == FEATURES
    assert (report["status"] == "стабильно").all()


def test_report_finds_simulated_drift(sample_df):
    X = sample_df.drop(columns=[TARGET])
    report = psi_report(X, simulate_drift(X)).set_index("feature")
    assert report.loc["LIMIT_BAL", "status"] == "значительный сдвиг"
    assert report.loc["AGE", "status"] == "стабильно"
