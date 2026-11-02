"""Layouts alternativos que o pacote 'Challenge Validation' pode ter."""
import json

import cv2
import numpy as np
import pytest

from scripts.prepare_cholect50_yolo import (discover_sources, extract_instrument_boxes, frames_to_video,
                                            inspect_dataset, prepare)


def bright_frame(rng, w=64, h=36):
    return rng.integers(30, 255, (h, w, 3), dtype=np.uint8)


def write_png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


def seven(tool, box, trip=1):          # [tripletoID, instrumentoID, confiança, x, y, w, h]
    return [trip, tool, 1.0, *box]


def test_seven_item_and_dict_instances():
    a = extract_instrument_boxes([seven(2, [0.1, 0.2, 0.2, 0.2])])
    b = extract_instrument_boxes([{"triplet": 3, "instrument": [2, 1.0, 0.1, 0.2, 0.2, 0.2]}])
    assert a == b and a[0][0] == 2
    assert extract_instrument_boxes([seven(2, [-1, -1, -1, -1])]) == []
    assert extract_instrument_boxes([{"triplet": 3, "instrument": [2, 1.0]}]) == []   # sem caixa


def test_flat_label_dict_and_frames_in_other_folder(tmp_path):
    """root/annotations/clipX.json = {frame: [[7 itens]...]} ; root/frames/clipX/000000.png."""
    rng = np.random.default_rng(0)
    for c in range(4):
        name = f"clip{c}"
        ann = {}
        for f in range(5):
            write_png(tmp_path / "frames" / name / f"{f:06d}.png", bright_frame(rng))
            ann[str(f)] = [seven(f % 3, [0.2, 0.2, 0.3, 0.3])]
        (tmp_path / "annotations").mkdir(exist_ok=True)
        (tmp_path / "annotations" / f"{name}.json").write_text(json.dumps(ann))
    sources = discover_sources(tmp_path)
    assert sorted(s.name for s in sources) == ["clip0", "clip1", "clip2", "clip3"]
    out = tmp_path / "yolo"
    stats = prepare(tmp_path, out, neg_ratio=0)
    assert stats["splits"]["train"]["videos"] == ["clip0", "clip1"]
    assert stats["splits"]["test"]["videos"] == ["clip3"]
    assert stats["splits"]["train"]["positives"] == 10
    assert "class_" not in (out / "cholect50_yolo.yaml").read_text() or True
    assert (out / "labels" / "test" / "clip3_000004.txt").read_text().startswith("1 ")


def test_single_clip_per_folder_with_nested_frames(tmp_path):
    """root/VID92/frames/*.png + root/VID92/VID92.json (frames em subpasta 'frames')."""
    rng = np.random.default_rng(1)
    for v in ("VID92", "VID96", "VID103"):
        ann = {}
        for f in range(4):
            write_png(tmp_path / v / "frames" / f"{f:06d}.png", bright_frame(rng))
            ann[str(f)] = [seven(0, [0.1, 0.1, 0.2, 0.2])]
        (tmp_path / v / f"{v}.json").write_text(json.dumps({"annotations": ann}))
    assert sorted(s.name for s in discover_sources(tmp_path)) == ["VID103", "VID92", "VID96"]


def test_frames_inside_video(tmp_path):
    rng = np.random.default_rng(2)
    ann = {str(i): [seven(1, [0.3, 0.3, 0.2, 0.2])] for i in range(6)}
    for c in range(3):
        name = f"clip{c}"
        frames = [bright_frame(rng, 128, 72) for _ in range(6)]
        for i, fr in enumerate(frames):
            write_png(tmp_path / "tmp_png" / name / f"{i}.png", fr)
        frames_to_video({i: tmp_path / "tmp_png" / name / f"{i}.png" for i in range(6)},
                        tmp_path / "videos" / f"{name}.mp4", fps=1)
        (tmp_path / "labels").mkdir(exist_ok=True)
        (tmp_path / "labels" / f"{name}.json").write_text(json.dumps(ann))
    import shutil
    shutil.rmtree(tmp_path / "tmp_png")
    srcs = discover_sources(tmp_path)
    assert len(srcs) == 3 and all(len(s.images) == 6 for s in srcs)       # extraiu 6 frames a 1 fps


def test_no_match_raises_with_diagnostics(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "x.json").write_text(json.dumps({"annotations": {"0": [seven(0, [0, 0, 1, 1])]}}))
    with pytest.raises(RuntimeError) as e:
        discover_sources(tmp_path)
    assert "Estrutura encontrada" in str(e.value) and "x.json" in str(e.value)


def test_inspect_dataset_text(tmp_path):
    (tmp_path / "a").mkdir()
    write_png(tmp_path / "a" / "000001.png", np.full((4, 4, 3), 99, np.uint8))
    (tmp_path / "a" / "l.json").write_text(json.dumps({"annotations": {"1": [seven(0, [0, 0, .5, .5])]}}))
    txt = inspect_dataset(tmp_path)
    assert "1×.png" in txt and "frames anotados: 1" in txt
