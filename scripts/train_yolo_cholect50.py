"""Fine-tuning do YOLOv8 para detecção de instrumentos (CholecT50) + avaliação no teste.

Uso:
  python scripts/train_yolo_cholect50.py --data data/cholect50_yolo/cholect50_yolo.yaml \
      --model yolov8n.pt --epochs 60 --imgsz 640 --device 0

Saídas:
  <pasta do treino>/                (curvas, pesos; o Ultralytics pode aninhar em runs/detect/...)
  models/cholect50_tools.pt         (melhores pesos — usado pelo pipeline)
  models/test_metrics.json          (mAP geral e por classe, se houver split de teste)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt (pré-treinado em COCO)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="0 para GPU, 'cpu' para CPU")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--project", default="runs/cholect50")
    p.add_argument("--name", default="yolov8_tools")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", default="models/cholect50_tools.pt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-plots", action="store_true", help="Não gerar gráficos (útil offline)")
    a = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        patience=a.patience, project=a.project, name=a.name, seed=a.seed, exist_ok=True,
        workers=a.workers, plots=not a.no_plots,
        # Imagens endoscópicas não têm "cima" fixo: flips/rotações são seguros e ajudam
        # muito porque o conjunto anotado é pequeno.
        fliplr=0.5, flipud=0.5, degrees=15.0, hsv_v=0.3, mosaic=1.0,
    )

    best = Path(model.trainer.best)
    dst = Path(a.export)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    print(f"Melhores pesos copiados para {dst}")

    # Avaliação no conjunto de teste (vídeos NUNCA vistos no treino/val), se existir.
    import yaml
    cfg = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    if cfg.get("test"):
        m = YOLO(str(best)).val(data=a.data, split="test", imgsz=a.imgsz, device=a.device)
        names = cfg["names"]
        per_class = {names[int(c)]: round(float(ap), 4) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
        out = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
               "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
               "AP50_por_classe": per_class}
        text = json.dumps(out, ensure_ascii=False, indent=2)
        (Path(model.trainer.save_dir) / "test_metrics.json").write_text(text, encoding="utf-8")
        stable = dst.parent / "test_metrics.json"       # local previsível, ao lado dos pesos
        stable.write_text(text, encoding="utf-8")
        print(text)
        print(f"Métricas salvas em {stable}")
    else:
        print("Sem split de teste no yaml; apenas a validação do treino foi usada.")


if __name__ == "__main__":
    main()
