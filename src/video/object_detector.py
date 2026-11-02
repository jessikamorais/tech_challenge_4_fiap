"""YOLOv8: detecção de objetos/instrumentos e verificação de área crítica."""
from __future__ import annotations

from ..config import settings
from ..schemas import Finding, Modality


class CriticalObjectDetector:
    """Sinaliza instrumentos (watch_classes) detectados FORA da zona permitida.

    `allowed_zone` é uma caixa normalizada (x1, y1, x2, y2) em [0, 1] — por exemplo,
    o campo operatório. Para cirurgias reais, treine/ajuste um modelo com classes
    de instrumentos cirúrgicos (as classes COCO são só ponto de partida).
    """

    def __init__(self, model_path: str | None = None, watch_classes=("scissors", "knife"),
                 allowed_zone=(0.25, 0.25, 0.75, 0.75), min_conf: float = 0.4,
                 sample_every: int = 5, min_ratio: float = 0.1):
        self.model_path = model_path or settings.detect_model
        self.watch_classes = set(watch_classes)
        self.zone = allowed_zone
        self.min_conf = min_conf
        self.sample_every = sample_every
        self.min_ratio = min_ratio
        self._model = None

    def _inside(self, cx: float, cy: float) -> bool:
        x1, y1, x2, y2 = self.zone
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def analyze(self, video_path: str, patient_id: str) -> list[Finding]:
        import cv2
        from ultralytics import YOLO
        self._model = self._model or YOLO(self.model_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total, outside, examples = 0, 0, []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % self.sample_every == 0:
                h, w = frame.shape[:2]
                r = self._model(frame, verbose=False, conf=self.min_conf)[0]
                for box in r.boxes:
                    name = r.names[int(box.cls)]
                    if name not in self.watch_classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    total += 1
                    if not self._inside((x1 + x2) / 2 / w, (y1 + y2) / 2 / h):
                        outside += 1
                        if len(examples) < 5:
                            examples.append({"t": round(idx / fps, 2), "class": name})
            idx += 1
        cap.release()

        if total and outside / total >= self.min_ratio:
            ratio = outside / total
            return [Finding(patient_id, Modality.VIDEO, "objeto_fora_da_area_critica",
                            min(1.0, ratio / 0.5),
                            f"{ratio:.0%} das detecções de instrumentos fora do campo permitido",
                            {"ratio": round(ratio, 3), "examples": examples})]
        return []
