"""Despacho de alertas: log local (JSONL) + Azure Event Grid + Teams (opcionais)."""
from __future__ import annotations

import json
import logging

from ..config import settings
from ..schemas import RiskAssessment, Severity

log = logging.getLogger(__name__)


class AlertDispatcher:
    def __init__(self, min_severity: str | None = None):
        self.min_severity = Severity(min_severity or settings.alert_min_severity)

    def should_alert(self, a: RiskAssessment) -> bool:
        return a.severity.rank >= self.min_severity.rank

    def dispatch(self, a: RiskAssessment) -> bool:
        if not self.should_alert(a):
            return False
        payload = a.to_dict()
        self._log_local(payload)
        self._publish_event_grid(payload)
        self._notify_teams(a)
        return True

    @staticmethod
    def _log_local(payload: dict) -> None:
        settings.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with settings.alerts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _publish_event_grid(payload: dict) -> None:
        if not (settings.eventgrid_endpoint and settings.eventgrid_key):
            return
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.eventgrid import EventGridEvent, EventGridPublisherClient
            client = EventGridPublisherClient(settings.eventgrid_endpoint,
                                              AzureKeyCredential(settings.eventgrid_key))
            client.send([EventGridEvent(
                event_type="Hospital.PatientRisk", subject=f"patients/{payload['patient_id']}",
                data=payload, data_version="1.0")])
        except Exception:
            log.exception("Falha ao publicar no Event Grid")

    @staticmethod
    def _notify_teams(a: RiskAssessment) -> None:
        if not settings.teams_webhook_url:
            return
        import requests
        top = "\n".join(f"- [{f.modality.value}] {f.description}" for f in a.findings[:5])
        text = (f"**ALERTA {a.severity.value.upper()}** — paciente {a.patient_id} "
                f"(risco {a.score:.2f})\n\n{top}")
        try:
            requests.post(settings.teams_webhook_url, json={"text": text}, timeout=10)
        except requests.RequestException:
            log.exception("Falha ao notificar Teams")
