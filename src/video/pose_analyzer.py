"""Análise postural: keypoints COCO -> ângulos articulares -> desvios.

Backend padrão: YOLOv8-pose (17 keypoints COCO, mesmo formato do OpenPose COCO).
Para usar o OpenPose "puro", implemente `_infer_keypoints` com a sua build do
OpenPose e mantenha o restante igual (o formato de saída é o mesmo).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from ..schemas import Finding, Modality

COCO = {
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
JOINTS = {
    "joelho_esq": ("l_hip", "l_knee", "l_ankle"),
    "joelho_dir": ("r_hip", "r_knee", "r_ankle"),
    "cotovelo_esq": ("l_shoulder", "l_elbow", "l_wrist"),
    "cotovelo_dir": ("r_shoulder", "r_elbow", "r_wrist"),
    "quadril_esq": ("l_shoulder", "l_hip", "l_knee"),
    "quadril_dir": ("r_shoulder", "r_hip", "r_knee"),
}
# Amplitude esperada (graus) por articulação. AJUSTAR por exercício/procedimento.
DEFAULT_RANGES = {j: (60.0, 180.0) for j in JOINTS}


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo em graus no vértice b formado por a-b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@dataclass
class PoseResult:
    findings: list[Finding] = field(default_factory=list)
    movement_series: list[float] = field(default_factory=list)  # deslocamento normalizado
    fps: float = 0.0
    sample_every: int = 1


class PoseAnalyzer:
    def __init__(self, model_path: str | None = None, ranges: dict | None = None,
                 min_conf: float = 0.5, violation_ratio: float = 0.15, sample_every: int = 3):
        self.model_path = model_path or settings.pose_model
        self.ranges = ranges or DEFAULT_RANGES
        self.min_conf = min_conf
        self.violation_ratio = violation_ratio
        self.sample_every = sample_every
        self._model = None

    def _infer_keypoints(self, frame):
        """Retorna (keypoints[17,2], conf[17]) da primeira pessoa ou None."""
        if self._model is None:
            from ultralytics import YOLO  # import tardio: só quando usado
            self._model = YOLO(self.model_path)
        r = self._model(frame, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        return r.keypoints.xy[0].cpu().numpy(), r.keypoints.conf[0].cpu().numpy()

    def analyze(self, video_path: str, patient_id: str) -> PoseResult:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        valid = {j: 0 for j in JOINTS}
        viol: dict[str, list[tuple[float, float]]] = {j: [] for j in JOINTS}
        movement: list[float] = []
        prev = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % self.sample_every == 0:
                out = self._infer_keypoints(frame)
                if out is not None:
                    kp, conf = out
                    t = idx / fps
                    for joint, (a, b, c) in JOINTS.items():
                        ia, ib, ic = COCO[a], COCO[b], COCO[c]
                        if min(conf[ia], conf[ib], conf[ic]) < self.min_conf:
                            continue
                        valid[joint] += 1
                        ang = joint_angle(kp[ia], kp[ib], kp[ic])
                        lo, hi = self.ranges[joint]
                        if not lo <= ang <= hi:
                            viol[joint].append((round(t, 2), round(ang, 1)))
                    if prev is not None:
                        torso = np.linalg.norm(kp[COCO["l_shoulder"]] - kp[COCO["l_hip"]]) + 1e-6
                        movement.append(float(np.mean(np.linalg.norm(kp - prev, axis=1)) / torso))
                    prev = kp
            idx += 1
        cap.release()

        findings = []
        for joint, events in viol.items():
            if valid[joint] == 0:
                continue
            ratio = len(events) / valid[joint]
            if ratio >= self.violation_ratio:
                findings.append(Finding(
                    patient_id=patient_id, modality=Modality.VIDEO, kind="desvio_postural",
                    score=min(1.0, ratio / 0.5),
                    description=f"{joint}: {ratio:.0%} dos frames fora da amplitude {self.ranges[joint]}",
                    details={"joint": joint, "ratio": round(ratio, 3), "examples": events[:5]},
                ))
        return PoseResult(findings, movement, fps, self.sample_every)
