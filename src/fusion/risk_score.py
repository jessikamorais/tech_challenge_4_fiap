"""Fusão multimodal tardia (late fusion): findings -> score de risco -> severidade."""
from __future__ import annotations

from ..schemas import Finding, Modality, RiskAssessment, Severity

WEIGHTS = {
    Modality.VITALS: 0.30, Modality.VIDEO: 0.25, Modality.AUDIO: 0.25,
    Modality.PRESCRIPTION: 0.10, Modality.MOVEMENT: 0.10,
}
THRESHOLDS = [(0.80, Severity.CRITICAL), (0.60, Severity.HIGH), (0.35, Severity.MEDIUM)]


def severity_from_score(score: float) -> Severity:
    for thr, sev in THRESHOLDS:
        if score >= thr:
            return sev
    return Severity.LOW


def fuse(patient_id: str, findings: list[Finding]) -> RiskAssessment:
    """score = 0.6·pico + 0.4·média ponderada + bônus por concordância entre modalidades."""
    per_modality: dict[Modality, float] = {}
    for f in findings:
        per_modality[f.modality] = max(per_modality.get(f.modality, 0.0), f.score)

    if not per_modality:
        return RiskAssessment(patient_id, 0.0, Severity.LOW, [])

    peak = max(per_modality.values())
    weighted = sum(WEIGHTS[m] * s for m, s in per_modality.items())
    bonus = 0.10 * (len(per_modality) - 1)  # mais de uma modalidade concordando
    score = min(1.0, 0.6 * peak + 0.4 * weighted + bonus)
    return RiskAssessment(patient_id, score, severity_from_score(score),
                          sorted(findings, key=lambda f: -f.score))
