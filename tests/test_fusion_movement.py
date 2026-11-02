import numpy as np

from src.anomaly.movement import MovementAnomalyDetector
from src.fusion.risk_score import fuse
from src.schemas import Finding, Modality, Severity


def f(m, s):
    return Finding("P1", m, "x", s, "teste")


def test_no_findings_is_low():
    assert fuse("P1", []).severity == Severity.LOW


def test_agreement_raises_severity():
    single = fuse("P1", [f(Modality.VITALS, 0.7)])
    multi = fuse("P1", [f(Modality.VITALS, 0.7), f(Modality.AUDIO, 0.7)])
    assert multi.score > single.score


def test_idle_and_spike():
    series = [0.05] * 40 + [0.0] * 40
    kinds = {x.kind for x in MovementAnomalyDetector().detect(series, "P1")}
    assert "imobilidade_prolongada" in kinds
    rng = np.random.default_rng(0)
    spike = list(rng.normal(0.05, 0.005, 60)) + [0.8]
    assert any(x.kind == "movimento_brusco" for x in MovementAnomalyDetector().detect(spike, "P1"))
