"""Anomalias na evolução de prescrições (regras + interações simples)."""
from __future__ import annotations

import pandas as pd

from ..schemas import Finding, Modality

# Pares de interação ilustrativos. Substituir por base farmacológica validada.
RISKY_PAIRS = {frozenset({"varfarina", "aas"}), frozenset({"varfarina", "ibuprofeno"})}


class PrescriptionAnomalyDetector:
    """Espera DataFrame com colunas: timestamp, drug, dose_mg."""

    def __init__(self, dose_change_ratio: float = 0.5):
        self.dose_change_ratio = dose_change_ratio

    def detect(self, df: pd.DataFrame, patient_id: str) -> list[Finding]:
        df = df.sort_values("timestamp")
        df = df.assign(drug=df["drug"].str.lower())
        findings: list[Finding] = []

        for drug, g in df.groupby("drug"):
            doses = g["dose_mg"].tolist()
            for prev, cur in zip(doses, doses[1:]):
                if prev > 0 and cur == 0:
                    findings.append(Finding(patient_id, Modality.PRESCRIPTION, "suspensao_abrupta",
                                            0.6, f"{drug}: suspenso (era {prev} mg)", {"drug": drug}))
                elif prev > 0:
                    ratio = abs(cur - prev) / prev
                    if ratio >= self.dose_change_ratio:
                        findings.append(Finding(
                            patient_id, Modality.PRESCRIPTION, "mudanca_de_dose",
                            min(1.0, 0.4 + ratio / 2),
                            f"{drug}: dose {prev} → {cur} mg ({ratio:+.0%})",
                            {"drug": drug, "prev": prev, "cur": cur}))

        active = set(df[df["dose_mg"] > 0]["drug"])
        for pair in RISKY_PAIRS:
            if pair <= active:
                findings.append(Finding(patient_id, Modality.PRESCRIPTION, "interacao_medicamentosa",
                                        0.8, f"Combinação de risco: {' + '.join(sorted(pair))}",
                                        {"drugs": sorted(pair)}))
        return findings
