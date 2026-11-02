from scripts.generate_synthetic_data import make_prescriptions
from src.anomaly.prescriptions import PrescriptionAnomalyDetector


def test_prescription_anomalies():
    kinds = {f.kind for f in PrescriptionAnomalyDetector().detect(make_prescriptions(), "P001")}
    assert {"mudanca_de_dose", "suspensao_abrupta", "interacao_medicamentosa"} <= kinds
