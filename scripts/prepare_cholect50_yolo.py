"""Converte o CholecT50 para o formato YOLO (detecção de instrumentos cirúrgicos).

Aceita DOIS tipos de entrada, descobrindo a estrutura sozinho:
  1) CholecT50 completo (formulário do CAMMA): videos/VIDxx/000000.png + labels/VIDxx.json
  2) Pacote oficial "CholecT50 Challenge Validation" (download direto): clipes com caixas.

Formatos de rótulo reconhecidos (por instância, dentro de cada frame):
  - lista de 15 itens (CholecT50): [0]=tripleto, [1:7]=[ID, SC, BX, BY, BW, BH], [7]=verbo, ...
  - lista de 7 itens (desafio):    [tripletoID, instrumentoID, confiança, x, y, w, h]
  - dict (desafio):                {"triplet": id, "instrument": [instrumentoID, prob, x, y, w, h]}
  Caixas normalizadas em [0,1], canto superior esquerdo + largura/altura; -1 = ausente.
Os frames podem estar como imagens (png/jpg) ou dentro de vídeos (extraídos a 1 fps).

ATENÇÃO: no CholecT50 completo, as caixas existem só para 5 dos 50 vídeos. O script usa
apenas as fontes que têm caixas e divide treino/val/teste POR CLIPE/VÍDEO (nunca por
frame, para evitar vazamento entre frames quase idênticos).

Uso:
  python scripts/prepare_cholect50_yolo.py --src data/cholect50_raw --inspect
  python scripts/prepare_cholect50_yolo.py --src data/cholect50_raw --out data/cholect50_yolo --dry-run
  python scripts/prepare_cholect50_yolo.py --src data/cholect50_raw --out data/cholect50_yolo
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Usado só se o JSON não trouxer os nomes das categorias.
FALLBACK_INSTRUMENTS = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
IMG_EXT = {".png", ".jpg", ".jpeg"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# ----------------------------------------------------------------------------------------------
# Rótulos
# ----------------------------------------------------------------------------------------------
def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def instrument_names(data) -> list[str]:
    """Lê os nomes das classes de instrumento do JSON (aceita dict ou lista em `categories`)."""
    cats = data.get("categories") if isinstance(data, dict) else None
    inst = None
    if isinstance(cats, dict):
        inst = cats.get("instrument")
    elif isinstance(cats, list):
        for c in cats:
            if isinstance(c, dict) and "instrument" in c:
                inst = c["instrument"]
                break
    if not isinstance(inst, dict) or not inst:
        return list(FALLBACK_INSTRUMENTS)
    try:
        pairs = sorted((int(k), str(v)) for k, v in inst.items())
    except ValueError:
        return list(FALLBACK_INSTRUMENTS)
    names = [""] * (pairs[-1][0] + 1)
    for k, v in pairs:
        names[k] = v
    return [n or f"class_{i}" for i, n in enumerate(names)]


def _inst_fields(inst):
    """Instância -> (id_do_instrumento, caixa|None). None se o formato não for reconhecido."""
    try:
        if isinstance(inst, dict):
            ins = inst.get("instrument")
            if not isinstance(ins, (list, tuple)) or not ins:
                return None
            box = tuple(float(v) for v in ins[2:6]) if len(ins) >= 6 else None
            return int(ins[0]), box
        if isinstance(inst, (list, tuple)) and len(inst) > 1:
            box = tuple(float(v) for v in inst[3:7]) if len(inst) >= 7 else None
            return int(inst[1]), box
    except (TypeError, ValueError):
        return None
    return None


def extract_instrument_boxes(instances) -> list[tuple[int, float, float, float, float]]:
    """Instâncias de um frame -> caixas YOLO únicas (classe, cx, cy, w, h), normalizadas.

    Um mesmo instrumento pode aparecer em vários tripletos do mesmo frame (a caixa se
    repete), então deduplicamos por (classe, caixa).
    """
    seen: set[tuple] = set()
    boxes: list[tuple[int, float, float, float, float]] = []
    for inst in instances or []:
        fields = _inst_fields(inst)
        if fields is None or fields[1] is None:
            continue
        cid, (bx, by, bw, bh) = fields
        if cid < 0 or min(bx, by, bw, bh) < 0 or bw <= 0 or bh <= 0:
            continue  # instrumento ausente ou sem caixa (-1)
        x1, y1 = max(0.0, bx), max(0.0, by)
        x2, y2 = min(1.0, bx + bw), min(1.0, by + bh)
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            continue
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        key = (cid, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
        if key in seen:
            continue
        seen.add(key)
        boxes.append((cid, cx, cy, w, h))
    return boxes


def classify_frame(instances):
    """-> (caixas, status): 'positive' | 'negative' | 'ambiguous'.

    - positive: há caixas e TODO instrumento presente tem caixa;
    - negative: nenhum instrumento no frame (imagem de fundo legítima);
    - ambiguous: há instrumento presente SEM caixa (rótulo incompleto). Usar como fundo
      ensinaria o modelo que instrumento visível é "nada", então o frame é descartado.
    """
    boxes = extract_instrument_boxes(instances)
    present = set()
    for inst in instances or []:
        f = _inst_fields(inst)
        if f is not None and f[0] >= 0:
            present.add(f[0])
    boxed = {b[0] for b in boxes}
    if present - boxed:
        return boxes, "ambiguous"
    return boxes, ("positive" if boxes else "negative")


def frame_annotations(data) -> dict[int, list]:
    """JSON -> {número_do_frame: [instâncias]} (aceita `annotations`, dict direto ou lista)."""
    ann = None
    if isinstance(data, dict):
        ann = data.get("annotations")
        if ann is None:
            ann = {k: v for k, v in data.items() if isinstance(v, list)}
    elif isinstance(data, list):
        ann = {str(i): v for i, v in enumerate(data)}
    if isinstance(ann, list):
        ann = {str(i): v for i, v in enumerate(ann)}
    if not isinstance(ann, dict):
        return {}
    out: dict[int, list] = {}
    for k, v in ann.items():
        nums = re.findall(r"\d+", str(k))
        if nums and isinstance(v, list):
            out[int(nums[-1])] = v
    return out


# ----------------------------------------------------------------------------------------------
# Descoberta da estrutura em disco
# ----------------------------------------------------------------------------------------------
@dataclass
class ClipSource:
    name: str
    labels_path: Path
    annotations: dict[int, list]
    images: dict[int, Path]
    names: list[str] = field(default_factory=lambda: list(FALLBACK_INSTRUMENTS))

    def boxes(self, frame: int):
        return classify_frame(self.annotations.get(frame, []))


def _visible(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).parts
    return not any(p.startswith(".") or p == "__MACOSX" for p in rel)


def _images_in(d: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            nums = re.findall(r"\d+", p.stem)
            if nums:
                out[int(nums[-1])] = p
    return out


def extract_video_frames(video: Path, out_dir: Path, fps: float = 1.0) -> dict[int, Path]:
    """Extrai frames a `fps` (padrão 1 fps, a taxa do CholecT50) -> {índice: png}."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / fps))
    frames: dict[int, Path] = {}
    i = k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            p = out_dir / f"{k:06d}.png"
            if not p.exists():
                cv2.imwrite(str(p), img)
            frames[k] = p
            k += 1
        i += 1
    cap.release()
    return frames


def _score_dir(name: str, d: Path) -> int:
    n = name.lower()
    tokens = {d.name.lower(), d.parent.name.lower()}
    if n in tokens:
        return 3
    if any(len(t) >= 3 and (n in t or t in n) for t in tokens):
        return 2
    return 0


def discover_sources(root, fps: float = 1.0) -> list[ClipSource]:
    """Procura rótulos JSON e os frames correspondentes (imagens ou vídeos) sob `root`."""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(f"Pasta inexistente: {root}")
    json_files = [p for p in sorted(root.rglob("*.json")) if _visible(p, root)]
    img_dirs: dict[Path, dict[int, Path]] = {}
    for d in {p.parent for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and _visible(p, root)}:
        imgs = _images_in(d)
        if imgs:
            img_dirs[d] = imgs
    videos = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in VID_EXT and _visible(p, root)]

    sources: list[ClipSource] = []
    used_names: Counter = Counter()
    for jf in json_files:
        try:
            data = load_json(jf)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        ann = frame_annotations(data)
        if not ann:
            continue

        def coverage(imgs: dict[int, Path]) -> float:
            return sum(1 for k in ann if k in imgs) / len(ann)

        best: dict[int, Path] | None = None
        ranked = sorted(((_score_dir(jf.stem, d), coverage(imgs), d) for d, imgs in img_dirs.items()),
                        key=lambda t: (t[0], t[1]), reverse=True)
        if ranked and ranked[0][0] > 0 and ranked[0][1] >= 0.9:
            best = img_dirs[ranked[0][2]]
        elif len(json_files) == 1 and len(img_dirs) == 1:
            only = next(iter(img_dirs.values()))
            best = only if coverage(only) >= 0.9 else None
        if best is None:
            for v in videos:
                if jf.stem.lower() in v.stem.lower() or v.stem.lower() in jf.stem.lower():
                    frames = extract_video_frames(v, root / "_frames" / v.stem, fps)
                    if frames and coverage(frames) >= 0.9:
                        best = frames
                        break
        if best is None:
            continue

        used_names[jf.stem] += 1
        name = jf.stem if used_names[jf.stem] == 1 else f"{jf.stem}_{used_names[jf.stem]}"
        sources.append(ClipSource(name, jf, ann, best, instrument_names(data)))

    if not sources:
        raise RuntimeError(
            "Não consegui casar rótulos (JSON) com frames (imagens/vídeos). Estrutura encontrada:\n"
            + inspect_dataset(root))
    return sources


def inspect_dataset(root, max_dirs: int = 25) -> str:
    """Resumo legível da estrutura baixada + amostra de um JSON de rótulos."""
    root = Path(root)
    lines = [f"Pasta: {root}"]
    by_dir: dict[Path, Counter] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and _visible(p, root):
            by_dir.setdefault(p.parent, Counter())[p.suffix.lower() or "(sem ext.)"] += 1
    for d, c in list(by_dir.items())[:max_dirs]:
        lines.append(f"  {d.relative_to(root) or '.'}/  " + ", ".join(f"{n}×{e}" for e, n in sorted(c.items())))
    if len(by_dir) > max_dirs:
        lines.append(f"  ... e mais {len(by_dir) - max_dirs} pastas")
    js = [p for p in sorted(root.rglob("*.json")) if _visible(p, root)]
    if js:
        lines.append(f"\nAmostra de rótulos: {js[0].relative_to(root)}")
        try:
            data = load_json(js[0])
            if isinstance(data, dict):
                lines.append("  chaves: " + ", ".join(list(data.keys())[:8]))
            ann = frame_annotations(data)
            if ann:
                k = sorted(ann)[len(ann) // 2]
                lines.append(f"  frames anotados: {len(ann)} | exemplo frame {k}: {json.dumps(ann[k])[:300]}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"  (não consegui ler: {e})")
    else:
        lines.append("\n(nenhum arquivo .json encontrado)")
    return "\n".join(lines)


def frames_to_video(frames: dict[int, Path] | list[Path], out_path, fps: float = 1.0) -> Path:
    """Monta um vídeo mp4 a partir de frames (ordem numérica). 1 fps = tempo real do dataset."""
    import cv2
    paths = [frames[k] for k in sorted(frames)] if isinstance(frames, dict) else list(frames)
    first = cv2.imread(str(paths[0]))
    h, w = first.shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in paths:
        img = cv2.imread(str(p))
        vw.write(img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h)))
    vw.release()
    return out_path


# ----------------------------------------------------------------------------------------------
# Conversão
# ----------------------------------------------------------------------------------------------
def is_black_frame(path: Path, thr: float = 3.0) -> bool:
    """Frames fora do corpo/anonimizados são totalmente pretos."""
    from PIL import Image
    import numpy as np
    with Image.open(path) as im:
        return float(np.asarray(im.convert("L")).mean()) < thr


def choose_split(names: list[str], val: list[str] | None, test: list[str] | None):
    """Divide as fontes com caixa em treino/val/teste (por clipe/vídeo)."""
    vids = sorted(names)
    if val is not None or test is not None:
        val, test = val or [], test or []
        unknown = (set(val) | set(test)) - set(vids)
        if unknown:
            raise RuntimeError(f"Clipes sem caixas (ou inexistentes): {sorted(unknown)}. Disponíveis: {vids}")
        train = [v for v in vids if v not in val and v not in test]
        return train, list(val), list(test)
    if len(vids) >= 3:
        return vids[:-2], [vids[-2]], [vids[-1]]
    if len(vids) == 2:
        print("AVISO: só 2 clipes com caixas — sem conjunto de teste.")
        return vids[:1], vids[1:], []
    raise RuntimeError(
        f"Encontrei {len(vids)} clipe(s) com caixas de instrumento; preciso de pelo menos 2 "
        "para separar treino e validação por clipe.")


def prepare(src: Path, out: Path, val=None, test=None, neg_ratio: float = 0.1,
            skip_black: bool = True, symlink: bool = False, dry_run: bool = False, seed: int = 0) -> dict:
    sources = discover_sources(src)
    with_boxes = [s for s in sources
                  if any(s.boxes(f)[1] == "positive" for f in s.annotations)]
    print(f"{len(sources)} fontes de rótulos; {len(with_boxes)} com caixas de instrumento: "
          f"{sorted(s.name for s in with_boxes)}")
    if not with_boxes:
        raise RuntimeError("Nenhuma fonte com caixas de instrumento. Estrutura:\n" + inspect_dataset(src))

    by_name = {s.name: s for s in with_boxes}
    names = with_boxes[0].names
    train, val_v, test_v = choose_split(list(by_name), val, test)
    splits = {"train": train, "val": val_v, "test": test_v}
    print(splits)

    rng = random.Random(seed)
    stats: dict = {"classes": names, "splits": {}}
    for split, clip_names in splits.items():
        if not clip_names:
            continue
        pos, neg, ambiguous = [], [], 0  # (clip, frame, imagem, caixas)
        for cn in clip_names:
            s = by_name[cn]
            for fid in sorted(s.annotations):
                img = s.images.get(fid)
                if img is None or (skip_black and is_black_frame(img)):
                    continue
                boxes, status = s.boxes(fid)
                if status == "ambiguous":
                    ambiguous += 1
                    continue
                (pos if status == "positive" else neg).append((cn, fid, img, boxes))
        rng.shuffle(neg)
        neg = neg[: int(len(pos) * neg_ratio)]
        items = pos + neg
        counts = Counter(b[0] for *_, bxs in pos for b in bxs)
        stats["splits"][split] = {
            "videos": clip_names, "images": len(items), "positives": len(pos), "negatives": len(neg),
            "ambiguos_descartados": ambiguous,
            "boxes_per_class": {names[c] if c < len(names) else f"class_{c}": n
                                for c, n in sorted(counts.items())},
        }
        if dry_run:
            continue
        img_out, lbl_out = out / "images" / split, out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for cn, fid, img, boxes in items:
            stem = f"{cn}_{fid:06d}"
            dst = img_out / f"{stem}{img.suffix.lower()}"
            if not dst.exists():
                if symlink:
                    dst.symlink_to(img.resolve())
                else:
                    shutil.copy2(img, dst)
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not dry_run:
        n_cls = max([len(names)] + [c + 1 for s in with_boxes for f in s.annotations
                                    for c, *_ in s.boxes(f)[0]])
        names = names + [f"class_{i}" for i in range(len(names), n_cls)]
        out.mkdir(parents=True, exist_ok=True)
        yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val"]
        if stats["splits"].get("test"):
            yaml.append("test: images/test")
        yaml.append("names:")
        yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
        (out / "cholect50_yolo.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
        stats["classes"] = names
        (out / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Dataset YOLO pronto em {out} (yaml: {out / 'cholect50_yolo.yaml'})")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="Pasta com o dataset baixado/extraído")
    p.add_argument("--out", type=Path, default=Path("data/cholect50_yolo"))
    p.add_argument("--inspect", action="store_true", help="Só mostra a estrutura encontrada")
    p.add_argument("--val-videos", nargs="*", help="Ex.: VID96")
    p.add_argument("--test-videos", nargs="*", help="Ex.: VID111")
    p.add_argument("--neg-ratio", type=float, default=0.1, help="Frames sem instrumento = fração dos positivos")
    p.add_argument("--keep-black", action="store_true", help="Não descartar frames pretos (fora do corpo)")
    p.add_argument("--symlink", action="store_true", help="Links simbólicos em vez de copiar (Linux/macOS)")
    p.add_argument("--dry-run", action="store_true", help="Só imprime as estatísticas")
    a = p.parse_args()
    try:
        if a.inspect:
            print(inspect_dataset(a.src))
            return
        prepare(a.src, a.out, a.val_videos, a.test_videos, a.neg_ratio,
                skip_black=not a.keep_black, symlink=a.symlink, dry_run=a.dry_run)
    except RuntimeError as e:
        raise SystemExit(f"ERRO: {e}")


if __name__ == "__main__":
    main()
