from scripts.generate_synthetic_data import make_vitals
from src.anomaly.vitals import VitalsAnomalyDetector


def test_detects_desaturation():
    df = make_vitals()
    findings = VitalsAnomalyDetector().detect(df, "P001")
    kinds = {(f.kind, f.details.get("signal")) for f in findings}
    assert ("fora_do_limite_clinico", "spo2") in kinds
    assert any(f.score >= 0.9 for f in findings)  # faixa crítica de SpO2
