"""Padrões de movimentação: picos bruscos (possível queda) e imobilidade prolongada.

A série vem de `PoseResult.movement_series` (vídeo de enfermaria/internação) ou de
sensores (acelerômetro). Cada ponto = movimento normalizado entre amostras.
"""
from __future__ import annotations

import numpy as np

from ..schemas import Finding, Modality


class MovementAnomalyDetector:
    def __init__(self, spike_z: float = 4.0, idle_eps: float = 0.005,
                 idle_min_samples: int = 30, window: int = 20):
        self.spike_z = spike_z
        self.idle_eps = idle_eps
        self.idle_min = idle_min_samples
        self.window = window

    def detect(self, series: list[float], patient_id: str) -> list[Finding]:
        x = np.asarray(series, dtype=float)
        if len(x) < self.window:
            return []
        findings: list[Finding] = []

        mu = np.convolve(x, np.ones(self.window) / self.window, mode="valid")[:-1]
        sd = np.array([x[i:i + self.window].std() for i in range(len(mu))]) + 1e-6
        z = (x[self.window:] - mu) / sd
        spikes = np.where(z > self.spike_z)[0]
        if len(spikes):
            findings.append(Finding(patient_id, Modality.MOVEMENT, "movimento_brusco",
                                    min(1.0, 0.6 + 0.05 * len(spikes)),
                                    f"{len(spikes)} picos de movimento (possível queda/agitação)",
                                    {"max_z": round(float(z.max()), 2)}))

        run = longest = 0
        for v in x:
            run = run + 1 if v < self.idle_eps else 0
            longest = max(longest, run)
        if longest >= self.idle_min:
            findings.append(Finding(patient_id, Modality.MOVEMENT, "imobilidade_prolongada",
                                    min(1.0, longest / (3 * self.idle_min)),
                                    f"Imobilidade por {longest} amostras consecutivas",
                                    {"samples": longest}))
        return findings
