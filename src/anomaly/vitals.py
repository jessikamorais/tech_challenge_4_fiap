"""Anomalias em sinais vitais: limites clínicos + z-score móvel + Isolation Forest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..schemas import Finding, Modality

VITAL_COLS = ["hr", "sbp", "dbp", "spo2"]
# Limites de triagem (ilustrativos). Validar/ajustar com a equipe clínica.
LIMITS = {"hr": (50, 120), "sbp": (90, 180), "dbp": (50, 110), "spo2": (92, 100)}
CRITICAL_LIMITS = {"hr": (40, 150), "sbp": (80, 200), "dbp": (40, 120), "spo2": (88, 100)}


class VitalsAnomalyDetector:
    """Espera DataFrame com colunas: timestamp, hr, sbp, dbp, spo2."""

    def __init__(self, contamination: float = 0.03, window: int = 12, z_thresh: float = 3.0,
                 random_state: int = 42):
        self.window = window
        self.z_thresh = z_thresh
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self._fitted = False

    def fit(self, baseline: pd.DataFrame) -> "VitalsAnomalyDetector":
        self.model.fit(baseline[VITAL_COLS].dropna())
        self._fitted = True
        return self

    @staticmethod
    def _ratio_score(n: int, total: int) -> float:
        return float(min(1.0, 0.4 + 6 * n / max(total, 1)))

    def detect(self, df: pd.DataFrame, patient_id: str) -> list[Finding]:
        df = df.sort_values("timestamp").reset_index(drop=True)
        findings: list[Finding] = []

        # 1) Limites clínicos
        for col in VITAL_COLS:
            lo, hi = LIMITS[col]
            out = df[(df[col] < lo) | (df[col] > hi)]
            if out.empty:
                continue
            clo, chi = CRITICAL_LIMITS[col]
            critical = bool(((out[col] < clo) | (out[col] > chi)).any())
            findings.append(Finding(
                patient_id, Modality.VITALS, "fora_do_limite_clinico",
                1.0 if critical else self._ratio_score(len(out), len(df)),
                f"{col}: {len(out)} leituras fora de [{lo}, {hi}]"
                + (" (faixa crítica)" if critical else ""),
                {"signal": col, "n": len(out), "min": float(out[col].min()),
                 "max": float(out[col].max())}))

        # 2) Z-score móvel (mudanças bruscas em relação ao passado recente)
        for col in VITAL_COLS:
            roll = df[col].shift(1).rolling(self.window, min_periods=self.window // 2)
            z = (df[col] - roll.mean()) / (roll.std() + 1e-6)
            n = int((z.abs() > self.z_thresh).sum())
            if n:
                findings.append(Finding(
                    patient_id, Modality.VITALS, "mudanca_brusca",
                    self._ratio_score(n, len(df)),
                    f"{col}: {n} pontos com |z| > {self.z_thresh} (janela={self.window})",
                    {"signal": col, "max_abs_z": round(float(z.abs().max()), 2)}))

        # 3) Isolation Forest (padrão multivariado incomum)
        if not self._fitted:
            self.fit(df)
        X = df[VITAL_COLS].ffill().bfill()
        pred = self.model.predict(X)
        n = int((pred == -1).sum())
        if n:
            findings.append(Finding(
                patient_id, Modality.VITALS, "padrao_multivariado_anomalo",
                self._ratio_score(n, len(df)),
                f"Isolation Forest sinalizou {n} de {len(df)} leituras",
                {"n": n, "idx": np.where(pred == -1)[0][:10].tolist()}))
        return findings
