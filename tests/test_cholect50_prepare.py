"""Testa a conversão com uma réplica minúscula da estrutura oficial do CholecT50."""
import json

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_cholect50_yolo import classify_frame, extract_instrument_boxes, prepare

NULL = [-1] * 15


def inst(triplet, tool, box, verb=2, target=0, phase=1):
    """Vetor de 15 itens: [tripleto, ID, SC, BX, BY, BW, BH, verbo, alvo(6), fase]."""
    return [triplet, tool, 1.0, *box, verb, target, 1.0, -1, -1, -1, -1, phase]


def make_dataset(root, box_videos, no_box_videos=()):
    cats = {"instrument": {"0": "grasper", "1": "bipolar", "2": "hook",
                           "3": "scissors", "4": "clipper", "5": "irrigator"}}
    rng = np.random.default_rng(0)
    for vid in list(box_videos) + list(no_box_videos):
        (root / "videos" / vid).mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(exist_ok=True)
        ann = {}
        for fid in range(4):
            arr = rng.integers(20, 255, (36, 64, 3), dtype=np.uint8)
            if fid == 3:
                arr[:] = 0                                   # frame fora do corpo (preto)
            Image.fromarray(arr).save(root / "videos" / vid / f"{fid:06d}.png")
            if vid in box_videos:
                if fid == 0:   # mesmo instrumento em 2 tripletos (caixa duplicada) + 1 outro
                    ann[str(fid)] = [inst(1, 0, [0.2, 0.3, 0.2, 0.4]), inst(2, 0, [0.2, 0.3, 0.2, 0.4]),
                                     inst(40, 2, [0.6, 0.5, 0.1, 0.2])]
                elif fid == 1:  # instrumento presente, mas sem caixa (-1)
                    ann[str(fid)] = [inst(5, 1, [-1, -1, -1, -1])]
                elif fid == 2:  # nenhum instrumento
                    ann[str(fid)] = [NULL]
                else:
                    ann[str(fid)] = [inst(1, 0, [0.1, 0.1, 0.2, 0.2])]
            else:
                ann[str(fid)] = [inst(5, 1, [-1, -1, -1, -1])]
        (root / "labels" / f"{vid}.json").write_text(
            json.dumps({"video": 1, "fps": 1, "categories": cats, "annotations": ann}))


def test_extract_boxes_dedup_and_yolo_format():
    boxes = extract_instrument_boxes([inst(1, 0, [0.2, 0.3, 0.2, 0.4]),
                                      inst(2, 0, [0.2, 0.3, 0.2, 0.4]),
                                      inst(40, 2, [0.6, 0.5, 0.1, 0.2]), NULL,
                                      inst(5, 1, [-1, -1, -1, -1])])
    assert len(boxes) == 2                                   # duplicata removida, -1 ignorado
    cid, cx, cy, w, h = boxes[0]
    assert cid == 0 and (cx, cy, w, h) == pytest.approx((0.3, 0.5, 0.2, 0.4))  # canto sup. esq. -> centro


def test_box_clipping_to_image():
    (_, cx, cy, w, h), = extract_instrument_boxes([inst(1, 3, [0.9, 0.9, 0.3, 0.3])])
    assert abs(w - 0.1) < 1e-9 and abs(cx - 0.95) < 1e-9


def test_prepare_splits_by_video(tmp_path):
    src, out = tmp_path / "CholecT50", tmp_path / "yolo"
    make_dataset(src, ["VID01", "VID02", "VID03", "VID04"], ["VID05"])
    stats = prepare(src, out, neg_ratio=0.0)
    assert stats["splits"]["train"]["videos"] == ["VID01", "VID02"]
    assert stats["splits"]["val"]["videos"] == ["VID03"]
    assert stats["splits"]["test"]["videos"] == ["VID04"]
    # VID05 (sem caixas) nunca entra; o frame preto (fid 3) e os sem caixa são descartados
    all_files = [p.name for p in (out / "images").rglob("*.png")]
    assert not any(n.startswith("VID05") for n in all_files)
    assert not any(n.endswith("_000003.png") for n in all_files)
    assert stats["splits"]["train"]["positives"] == 2        # só o frame 0 de cada vídeo
    lbl = (out / "labels" / "train" / "VID01_000000.txt").read_text().strip().splitlines()
    assert lbl[0].startswith("0 0.300000 0.500000 0.200000 0.400000")
    assert len(lbl) == 2
    yaml = (out / "cholect50_yolo.yaml").read_text()
    assert "test: images/test" in yaml and "0: grasper" in yaml and "5: irrigator" in yaml
    # cada imagem tem seu .txt
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (out / "images" / split).glob("*.png")}
        lbls = {p.stem for p in (out / "labels" / split).glob("*.txt")}
        assert imgs == lbls


def test_classify_frame():
    ok = [inst(1, 0, [0.2, 0.3, 0.2, 0.4])]
    assert classify_frame(ok)[1] == "positive"
    assert classify_frame([NULL])[1] == "negative"
    assert classify_frame([inst(5, 1, [-1, -1, -1, -1])])[1] == "ambiguous"       # presente, sem caixa
    partial = ok + [inst(6, 2, [-1, -1, -1, -1])]                                  # 1 com caixa, 1 sem
    assert classify_frame(partial)[1] == "ambiguous"


def test_prepare_explicit_split_and_negatives(tmp_path):
    src, out = tmp_path / "c", tmp_path / "y"
    make_dataset(src, ["VID10", "VID11", "VID12"])
    stats = prepare(src, out, val=["VID10"], test=["VID11"], neg_ratio=1.0)
    assert stats["splits"]["train"]["videos"] == ["VID12"]
    assert stats["splits"]["train"]["negatives"] == 1        # só o frame 2 (realmente vazio)
    assert stats["splits"]["train"]["ambiguos_descartados"] == 1   # frame 1: instrumento sem caixa
    assert (out / "labels" / "train" / "VID12_000002.txt").read_text() == ""
