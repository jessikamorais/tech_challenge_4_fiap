"""Gera dados sintéticos (sinais vitais e prescrições) com anomalias injetadas.

Útil para desenvolver/demonstrar sem depender do PhysioNet. Para dados reais:
  - PhysioNet (MIMIC, Challenge): https://physionet.org/
  - AudioSet (áudio): https://research.google.com/audioset/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "samples"


def make_vitals(patient_id="P001", n=288, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-10-26 00:00", periods=n, freq="5min")
    df = pd.DataFrame({
        "timestamp": ts,
        "hr": rng.normal(78, 4, n),
        "sbp": rng.normal(122, 6, n),
        "dbp": rng.normal(78, 4, n),
        "spo2": np.clip(rng.normal(97, 0.8, n), 90, 100),
    })
    df.loc[200:215, "spo2"] = np.linspace(95, 87, 16)        # dessaturação progressiva
    df.loc[200:215, "hr"] += np.linspace(5, 45, 16)          # taquicardia associada
    df.loc[120:122, "sbp"] += 55                              # pico pressórico
    num = ["hr", "sbp", "dbp", "spo2"]
    df[num] = df[num].round(1)
    return df.assign(patient_id=patient_id)


def make_prescriptions(patient_id="P001") -> pd.DataFrame:
    rows = [
        ("2026-10-24 08:00", "varfarina", 5), ("2026-10-25 08:00", "varfarina", 5),
        ("2026-10-26 08:00", "varfarina", 10),                       # dobrou a dose
        ("2026-10-25 08:00", "aas", 100),                            # interação com varfarina
        ("2026-10-24 08:00", "losartana", 50), ("2026-10-26 08:00", "losartana", 0),  # suspensão
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "drug", "dose_mg"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.assign(patient_id=patient_id)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gera vitais/prescrições SINTÉTICOS para um ID de paciente.")
    ap.add_argument("--patient", default="P001", help="ID do paciente (os dados não são reais)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    make_vitals(a.patient).to_csv(OUT / f"vitals_{a.patient}.csv", index=False)
    make_prescriptions(a.patient).to_csv(OUT / f"prescriptions_{a.patient}.csv", index=False)
    print(f"Arquivos sintéticos do paciente {a.patient} gerados em {OUT}")
