"""Transcrição com Azure Speech to Text (reconhecimento contínuo, pt-BR)."""
from __future__ import annotations

import threading

from ..config import settings


class SpeechTranscriber:
    def __init__(self, key: str | None = None, region: str | None = None, language: str = "pt-BR"):
        self.key = key or settings.speech_key
        self.region = region or settings.speech_region
        self.language = language

    def transcribe(self, wav_path: str, timeout_s: int = 600) -> str:
        """`wav_path`: WAV PCM (idealmente 16 kHz, mono)."""
        import azure.cognitiveservices.speech as speechsdk
        if not self.key:
            raise RuntimeError("AZURE_SPEECH_KEY não configurada")
        cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        cfg.speech_recognition_language = self.language
        audio = speechsdk.audio.AudioConfig(filename=wav_path)
        rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)

        parts: list[str] = []
        done = threading.Event()
        rec.recognized.connect(lambda e: parts.append(e.result.text) if e.result.text else None)
        rec.session_stopped.connect(lambda e: done.set())
        rec.canceled.connect(lambda e: done.set())
        rec.start_continuous_recognition()
        done.wait(timeout=timeout_s)
        rec.stop_continuous_recognition()
        return " ".join(parts).strip()
