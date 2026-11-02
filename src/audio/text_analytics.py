"""Azure AI Language (Text Analytics): sentimento, frases-chave e termos críticos."""
from __future__ import annotations

from ..config import settings
from ..schemas import Finding, Modality

# Dicionário clínico inicial (peso = gravidade 0..1). Expandir com a equipe médica.
CRITICAL_TERMS = {
    "dor no peito": 0.9, "dificuldade para respirar": 0.9, "falta de ar": 0.8,
    "desmaio": 0.8, "desmaiei": 0.8, "sangramento": 0.8, "confusão": 0.6,
    "tontura": 0.5, "fraqueza": 0.5, "cansaço": 0.4, "cansado": 0.4,
}


class TextAnalyzer:
    def __init__(self, endpoint: str | None = None, key: str | None = None):
        self.endpoint = endpoint or settings.language_endpoint
        self.key = key or settings.language_key

    def _client(self):
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
        return TextAnalyticsClient(self.endpoint, AzureKeyCredential(self.key))

    @staticmethod
    def find_critical_terms(text: str) -> list[tuple[str, float]]:
        t = text.lower()
        return [(term, w) for term, w in CRITICAL_TERMS.items() if term in t]

    def analyze(self, text: str, patient_id: str, use_azure: bool = True) -> list[Finding]:
        findings: list[Finding] = []
        for term, weight in self.find_critical_terms(text):
            findings.append(Finding(patient_id, Modality.AUDIO, "termo_critico", weight,
                                    f"Termo crítico na fala: '{term}'", {"term": term}))
        if use_azure and self.key and text:
            client = self._client()
            sent = client.analyze_sentiment([text], language="pt")[0]
            neg = sent.confidence_scores.negative
            if neg >= 0.7:
                findings.append(Finding(patient_id, Modality.AUDIO, "sentimento_negativo",
                                        min(1.0, 0.3 + 0.4 * neg),
                                        f"Sentimento predominantemente negativo ({neg:.0%})",
                                        {"sentiment": sent.sentiment}))
            kp = client.extract_key_phrases([text], language="pt")[0]
            if findings:
                findings[0].details["key_phrases"] = list(kp.key_phrases)[:10]
        return findings
