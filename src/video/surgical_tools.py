"""Análise de vídeo cirúrgico com YOLOv8 treinado no CholecT50 (instrumentos).

Em vídeo laparoscópico não há corpo em pose para o OpenPose; o que dá para medir com
um detector de instrumentos são INDICADORES operacionais:
  - instrumento fora do campo central da imagem;
  - movimentos bruscos do instrumento (deslocamento anormal entre amostras);
  - ausência prolongada de instrumentos com a câmera dentro do corpo.

IMPORTANTE: o CholecT50 não traz rótulos de "desvio/erro cirúrgico". Estes indicadores
são heurísticas para triagem e revisão humana, não julgamento clínico; só a detecção
dos instrumentos (mAP) pode ser validada com os rótulos do dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import settings
from ..schemas import Finding, Modality


@dataclass
class ToolDetection:
    cls: str
    conf: float
    cx: float  # centro, normalizado [0,1]
    cy: float
    w: float
    h: float


@dataclass
class SampledFrame:
    t: float                       # tempo (s) OU número do frame, conforme `time_unit` da análise
    visible: bool                  # False = frame preto (fora do corpo)
    dets: list[ToolDetection] = field(default_factory=list)


@dataclass
class SurgicalAnalysis:
    findings: list[Finding]
    summary: dict


def analyze_detections(frames: list[SampledFrame], patient_id: str,
                       allowed_zone=(0.08, 0.05, 0.92, 0.95), min_outside_ratio: float = 0.15,
                       speed_min: float = 0.25, speed_z: float = 3.5,
                       max_idle_s: float = 15.0, max_gap: float = 1.5,
                       time_unit: str = "s") -> SurgicalAnalysis:
    """Lógica pura (sem modelo): detecções por frame -> Findings + resumo.

    `t` é a coordenada temporal de cada amostra, em `time_unit` ("s" ou "frame"). Duas amostras só
    são tratadas como ADJACENTES se `b.t - a.t <= max_gap`; deslocamento e ausência prolongada nunca
    atravessam uma lacuna (frames espaçados irregularmente não têm significado temporal entre si).
    """
    visible = [f for f in frames if f.visible]
    summary: dict = {"frames_amostrados": len(frames), "frames_dentro_do_corpo": len(visible)}
    ordered = sorted(frames, key=lambda f: f.t)
    summary["lacunas_na_amostragem"] = sum(1 for a, b in zip(ordered, ordered[1:]) if b.t - a.t > max_gap)
    findings: list[Finding] = []
    if not visible:
        return SurgicalAnalysis(findings, summary)

    # Uso de instrumentos (% dos frames visíveis em que cada classe aparece)
    usage: dict[str, int] = {}
    for f in visible:
        for c in {d.cls for d in f.dets}:
            usage[c] = usage.get(c, 0) + 1
    summary["uso_instrumentos_pct"] = {c: round(100 * n / len(visible), 1) for c, n in sorted(usage.items())}

    # 1) Instrumento fora do campo permitido
    x1, y1, x2, y2 = allowed_zone
    all_dets = [(f.t, d) for f in visible for d in f.dets]
    if all_dets:
        out = [(t, d) for t, d in all_dets if not (x1 <= d.cx <= x2 and y1 <= d.cy <= y2)]
        ratio = len(out) / len(all_dets)
        summary["deteccoes_fora_do_campo_pct"] = round(100 * ratio, 1)
        if ratio >= min_outside_ratio:
            findings.append(Finding(
                patient_id, Modality.VIDEO, "instrumento_fora_do_campo", min(1.0, ratio / 0.5),
                f"{ratio:.0%} das detecções de instrumentos fora do campo central {allowed_zone}",
                {"ratio": round(ratio, 3), "examples": [{"t": round(t, 1), "class": d.cls} for t, d in out[:5]]}))

    # 2) Movimentos bruscos: deslocamento do mesmo instrumento entre amostras consecutivas
    disp: list[tuple[float, str, float]] = []  # (t, classe, deslocamento normalizado)
    vis_sorted = sorted(visible, key=lambda f: f.t)
    for a, b in zip(vis_sorted, vis_sorted[1:]):
        if b.t - a.t > max_gap:  # lacuna: amostras não adjacentes
            continue
        for cls in {d.cls for d in a.dets} & {d.cls for d in b.dets}:
            pa = [(d.cx, d.cy) for d in a.dets if d.cls == cls]
            pb = [(d.cx, d.cy) for d in b.dets if d.cls == cls]
            # pareamento guloso pelo vizinho mais próximo
            dmin = min(np.hypot(xa - xb, ya - yb) for xa, ya in pa for xb, yb in pb)
            disp.append((b.t, cls, float(dmin)))
    if len(disp) >= 10:
        v = np.array([d for _, _, d in disp])
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-6
        spikes = [(t, c, d) for (t, c, d) in disp if d >= speed_min and (d - med) / mad >= speed_z]
        summary["deslocamento_mediano"] = round(med, 3)
        if spikes:
            findings.append(Finding(
                patient_id, Modality.VIDEO, "movimento_brusco_instrumento",
                min(1.0, 0.4 + 0.1 * len(spikes)),
                f"{len(spikes)} deslocamentos bruscos de instrumento entre amostras consecutivas",
                {"examples": [{"t": round(t, 1), "class": c, "disp": round(d, 2)} for t, c, d in spikes[:5]]}))

    # 3) Ausência prolongada de instrumentos (câmera dentro do corpo), só entre amostras adjacentes
    longest, run_start, best_start, prev_t = 0.0, None, None, None
    for f in ordered:
        if prev_t is not None and f.t - prev_t > max_gap:
            run_start = None                      # lacuna interrompe a sequência
        prev_t = f.t
        if f.visible and not f.dets:
            run_start = f.t if run_start is None else run_start
            if f.t - run_start > longest:
                longest, best_start = f.t - run_start, run_start
        else:
            run_start = None
    summary["maior_periodo_sem_instrumentos_s"] = round(longest, 1)
    summary["unidade_de_tempo"] = time_unit
    if longest >= max_idle_s:
        findings.append(Finding(
            patient_id, Modality.VIDEO, "ausencia_prolongada_de_instrumentos",
            min(1.0, longest / (3 * max_idle_s)),
            f"{longest:.0f} {time_unit} consecutivos sem instrumentos visíveis (a partir de t={best_start:.0f} {time_unit})",
            {"start_s": round(best_start or 0.0, 1), "duration_s": round(longest, 1), "unit": time_unit}))

    return SurgicalAnalysis(findings, summary)


class SurgicalToolAnalyzer:
    """Roda o YOLOv8 treinado no CholecT50 em um vídeo e aplica `analyze_detections`."""

    def __init__(self, weights: str | None = None, conf: float = 0.35,
                 sample_every: int | None = None, black_thr: float = 3.0, **rule_kwargs):
        self.weights = weights or settings.surgical_model
        self.conf = conf
        self.sample_every = sample_every  # None = ~1 fps (mesma taxa do treino)
        self.black_thr = black_thr
        self.rule_kwargs = rule_kwargs
        self._model = None

    def detect(self, video_path: str, timestamps: list[float] | None = None) -> list[SampledFrame]:
        """`timestamps`: coordenada temporal de cada frame do vídeo (ex.: números dos frames originais
        quando o vídeo foi montado a partir de imagens esparsas). Sem ela, usa idx/fps."""
        import cv2
        from ultralytics import YOLO
        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"Pesos não encontrados em {self.weights}. Treine com "
                "scripts/train_yolo_cholect50.py (veja docs/cholect50.md) ou ajuste SURGICAL_MODEL.")
        self._model = self._model or YOLO(self.weights)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = 1 if timestamps is not None else (self.sample_every or max(1, round(fps)))
        frames: list[SampledFrame] = []
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % step == 0:
                if timestamps is not None and idx >= len(timestamps):
                    break
                t = float(timestamps[idx]) if timestamps is not None else idx / fps
                if float(img.mean()) < self.black_thr:      # fora do corpo / tela preta
                    frames.append(SampledFrame(t, False))
                else:
                    r = self._model(img, conf=self.conf, verbose=False)[0]
                    dets = []
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxyn[0].tolist()
                        dets.append(ToolDetection(r.names[int(b.cls)], float(b.conf),
                                                  (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
                    frames.append(SampledFrame(t, True, dets))
            idx += 1
        cap.release()
        return frames

    def analyze(self, video_path: str, patient_id: str) -> SurgicalAnalysis:
        return analyze_detections(self.detect(video_path), patient_id, **self.rule_kwargs)
