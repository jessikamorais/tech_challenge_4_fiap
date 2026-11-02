"""Features acústicas simples para sinais de fadiga/disartria.

ATENÇÃO: são heurísticas de triagem, não diagnóstico. Calibre os limiares com
dados reais (baseline por paciente) antes de usar em produção.
"""
from __future__ import annotations

import numpy as np

from ..schemas import Finding, Modality


class VoiceAnalyzer:
    def __init__(self, pause_ratio_thr: float = 0.5, speech_rate_thr_wpm: float = 80.0,
                 rms_thr: float = 0.01, jitter_thr: float = 0.05):
        self.pause_ratio_thr = pause_ratio_thr
        self.speech_rate_thr = speech_rate_thr_wpm
        self.rms_thr = rms_thr
        self.jitter_thr = jitter_thr

    def extract(self, wav_path: str, transcript: str = "") -> dict:
        import librosa
        y, sr = librosa.load(wav_path, sr=16000)
        dur = len(y) / sr
        intervals = librosa.effects.split(y, top_db=30)
        speech = sum(e - s for s, e in intervals) / sr
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"),
                                fmax=librosa.note_to_hz("C6"), sr=sr)
        f0v = f0[~np.isnan(f0)]
        jitter = float(np.mean(np.abs(np.diff(f0v))) / np.mean(f0v)) if len(f0v) > 2 else 0.0
        words = len(transcript.split())
        return {
            "duration_s": round(dur, 2),
            "pause_ratio": round(1 - speech / dur, 3) if dur else 0.0,
            "rms_mean": float(librosa.feature.rms(y=y)[0].mean()),
            "jitter_f0": round(jitter, 4),
            "speech_rate_wpm": round(words / dur * 60, 1) if dur and words else None,
        }

    def evaluate(self, feats: dict, patient_id: str) -> list[Finding]:
        out: list[Finding] = []

        def add(kind, score, desc):
            out.append(Finding(patient_id, Modality.AUDIO, kind, score, desc, {"features": feats}))

        if feats["pause_ratio"] > self.pause_ratio_thr:
            add("fala_com_muitas_pausas", min(1.0, feats["pause_ratio"]),
                f"Proporção de pausas alta ({feats['pause_ratio']:.0%}) — possível fadiga/dispneia")
        rate = feats.get("speech_rate_wpm")
        if rate is not None and rate < self.speech_rate_thr:
            add("fala_lenta", 0.5, f"Taxa de fala baixa ({rate} palavras/min)")
        if feats["rms_mean"] < self.rms_thr:
            add("voz_fraca", 0.5, "Energia vocal baixa")
        if feats["jitter_f0"] > self.jitter_thr:
            add("instabilidade_vocal", 0.6, "Variação de pitch elevada — possível disartria")
        return out

    def analyze(self, wav_path: str, patient_id: str, transcript: str = ""):
        feats = self.extract(wav_path, transcript)
        return feats, self.evaluate(feats, patient_id)
