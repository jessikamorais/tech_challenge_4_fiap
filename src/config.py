"""Configuração central: lê variáveis de ambiente (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv é opcional em testes
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    speech_key: str = os.getenv("AZURE_SPEECH_KEY", "")
    speech_region: str = os.getenv("AZURE_SPEECH_REGION", "brazilsouth")
    language_endpoint: str = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
    language_key: str = os.getenv("AZURE_LANGUAGE_KEY", "")
    eventgrid_endpoint: str = os.getenv("AZURE_EVENTGRID_ENDPOINT", "")
    eventgrid_key: str = os.getenv("AZURE_EVENTGRID_KEY", "")
    teams_webhook_url: str = os.getenv("TEAMS_WEBHOOK_URL", "")
    pose_model: str = os.getenv("POSE_MODEL", "yolov8n-pose.pt")
    detect_model: str = os.getenv("DETECT_MODEL", "yolov8n.pt")
    surgical_model: str = os.getenv("SURGICAL_MODEL", str(ROOT / "models" / "cholect50_tools.pt"))
    alert_min_severity: str = os.getenv("ALERT_MIN_SEVERITY", "media")
    alerts_path: Path = ROOT / "data" / "alerts" / "alerts.jsonl"
    reports_dir: Path = ROOT / "data" / "reports"


settings = Settings()
