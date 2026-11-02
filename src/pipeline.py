"""Orquestra o fluxo multimodal: entradas -> módulos -> fusão -> alerta.

Exemplo:
  python -m src.pipeline --patient P001 --video data/raw/fisio.mp4 \
      --audio data/raw/consulta.wav --vitals data/samples/vitals_P001.csv \
      --prescriptions data/samples/prescriptions_P001.csv
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from .alerts.dispatcher import AlertDispatcher
from .anomaly.movement import MovementAnomalyDetector
from .anomaly.prescriptions import PrescriptionAnomalyDetector
from .anomaly.vitals import VitalsAnomalyDetector
from .fusion.risk_score import fuse
from .schemas import Finding, RiskAssessment

log = logging.getLogger("pipeline")


def analyze_detections_from(analyzer, video, patient_id, timestamps):
    from .video.surgical_tools import analyze_detections
    return analyze_detections(analyzer.detect(video, timestamps=timestamps), patient_id,
                              time_unit="frame", **analyzer.rule_kwargs)


def run(patient_id: str, video: str | None = None, audio: str | None = None,
        vitals: str | None = None, prescriptions: str | None = None,
        dispatch: bool = True, video_type: str = "physio",
        video_timestamps: list[float] | None = None) -> RiskAssessment:
    findings: list[Finding] = []

    if video and video_type == "surgery":
        # Vídeo laparoscópico: YOLOv8 treinado no CholecT50 (instrumentos). Sem pose/OpenPose.
        from .video.report import generate_video_report
        from .video.surgical_tools import SurgicalToolAnalyzer
        analyzer = SurgicalToolAnalyzer()
        if video_timestamps is not None:   # vídeo montado de frames esparsos: tempo = nº do frame original
            res = analyze_detections_from(analyzer, video, patient_id, video_timestamps)
        else:
            res = analyzer.analyze(video, patient_id)
        findings += res.findings
        log.info("Resumo cirúrgico: %s", res.summary)
        report = generate_video_report(patient_id, video, res.findings, summary=res.summary)
        log.info("Relatório de vídeo: %s", report)
    elif video:
        from .video.object_detector import CriticalObjectDetector
        from .video.pose_analyzer import PoseAnalyzer
        from .video.report import generate_video_report
        pose = PoseAnalyzer().analyze(video, patient_id)
        findings += pose.findings
        findings += CriticalObjectDetector().analyze(video, patient_id)
        findings += MovementAnomalyDetector().detect(pose.movement_series, patient_id)
        video_findings = [f for f in findings if f.modality.value in ("video", "movement")]
        report = generate_video_report(patient_id, video, video_findings)
        log.info("Relatório de vídeo: %s", report)

    if audio:
        from .audio.text_analytics import TextAnalyzer
        from .audio.transcriber import SpeechTranscriber
        from .audio.voice_features import VoiceAnalyzer
        transcript = SpeechTranscriber().transcribe(audio)
        log.info("Transcrição: %s", transcript[:200])
        findings += TextAnalyzer().analyze(transcript, patient_id)
        _, voice_findings = VoiceAnalyzer().analyze(audio, patient_id, transcript)
        findings += voice_findings

    if vitals:
        df = pd.read_csv(vitals, parse_dates=["timestamp"])
        findings += VitalsAnomalyDetector().detect(df, patient_id)

    if prescriptions:
        df = pd.read_csv(prescriptions, parse_dates=["timestamp"])
        findings += PrescriptionAnomalyDetector().detect(df, patient_id)

    assessment = fuse(patient_id, findings)
    if dispatch:
        AlertDispatcher().dispatch(assessment)
    return assessment


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Monitoramento multimodal de pacientes")
    p.add_argument("--patient", required=True)
    p.add_argument("--video")
    p.add_argument("--video-type", choices=["physio", "surgery"], default="physio",
                   help="physio: pose (fisioterapia); surgery: instrumentos (CholecT50)")
    p.add_argument("--audio")
    p.add_argument("--vitals")
    p.add_argument("--prescriptions")
    p.add_argument("--no-dispatch", action="store_true")
    a = p.parse_args()
    res = run(a.patient, a.video, a.audio, a.vitals, a.prescriptions, dispatch=not a.no_dispatch,
              video_type=a.video_type)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
