"""Contratos de dados compartilhados entre todos os módulos."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Modality(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    VITALS = "vitals"
    PRESCRIPTION = "prescription"
    MOVEMENT = "movement"


class Severity(str, Enum):
    LOW = "baixa"
    MEDIUM = "media"
    HIGH = "alta"
    CRITICAL = "critica"

    @property
    def rank(self) -> int:
        return ["baixa", "media", "alta", "critica"].index(self.value)


@dataclass
class Finding:
    """Um achado individual de qualquer modalidade (score normalizado 0..1)."""
    patient_id: str
    modality: Modality
    kind: str
    score: float
    description: str
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modality"] = self.modality.value
        return d


@dataclass
class RiskAssessment:
    """Resultado da fusão multimodal para um paciente."""
    patient_id: str
    score: float
    severity: Severity
    findings: list[Finding]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "score": round(self.score, 3),
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "modalities": sorted({f.modality.value for f in self.findings}),
            "findings": [f.to_dict() for f in self.findings],
        }
