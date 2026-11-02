from src.video.surgical_tools import SampledFrame, ToolDetection, analyze_detections


def det(cls="grasper", cx=0.5, cy=0.5):
    return ToolDetection(cls, 0.9, cx, cy, 0.1, 0.1)


def kinds(res):
    return {f.kind for f in res.findings}


def test_normal_procedure_has_no_findings():
    frames = [SampledFrame(float(t), True, [det(cx=0.5 + 0.01 * (t % 3))]) for t in range(60)]
    res = analyze_detections(frames, "P1")
    assert res.findings == []
    assert res.summary["uso_instrumentos_pct"]["grasper"] == 100.0


def test_instrument_outside_field():
    frames = [SampledFrame(float(t), True, [det(cx=0.97)]) for t in range(20)]
    assert "instrumento_fora_do_campo" in kinds(analyze_detections(frames, "P1"))


def test_abrupt_movement():
    frames = [SampledFrame(float(t), True, [det(cx=0.5, cy=0.5 + 0.005 * (t % 2))]) for t in range(40)]
    frames[20] = SampledFrame(20.0, True, [det(cx=0.9, cy=0.1)])    # salto grande
    res = analyze_detections(frames, "P1")
    assert "movimento_brusco_instrumento" in kinds(res)


def test_long_absence_ignores_out_of_body_frames():
    # 20 s sem instrumento dentro do corpo => alerta
    frames = [SampledFrame(float(t), True, [det()]) for t in range(5)]
    frames += [SampledFrame(float(t), True, []) for t in range(5, 30)]
    assert "ausencia_prolongada_de_instrumentos" in kinds(analyze_detections(frames, "P1"))
    # os mesmos 25 s, mas fora do corpo (frame preto) => sem alerta
    frames = [SampledFrame(float(t), True, [det()]) for t in range(5)]
    frames += [SampledFrame(float(t), False) for t in range(5, 30)]
    assert "ausencia_prolongada_de_instrumentos" not in kinds(analyze_detections(frames, "P1"))


def test_no_visible_frames():
    res = analyze_detections([SampledFrame(0.0, False)], "P1")
    assert res.findings == [] and res.summary["frames_dentro_do_corpo"] == 0


def test_gaps_break_absence_and_movement():
    # frames esparsos (números 0,1,2, 100,101,...): a lacuna não conta como ausência contínua nem deslocamento
    fr = [SampledFrame(float(t), True, []) for t in (0, 1, 2)] + [SampledFrame(float(t), True, []) for t in (200, 201, 202)]
    res = analyze_detections(fr, "P1", max_idle_s=5, time_unit="frame")
    assert "ausencia_prolongada_de_instrumentos" not in kinds(res)
    assert res.summary["lacunas_na_amostragem"] == 1 and res.summary["maior_periodo_sem_instrumentos_s"] == 2.0
    fr = [SampledFrame(float(t), True, [det(cx=0.5, cy=0.5 + 0.005 * (t % 2))]) for t in range(40)]
    fr += [SampledFrame(float(t), True, [det(cx=0.9, cy=0.1)]) for t in range(500, 505)]  # salto por lacuna
    assert "movimento_brusco_instrumento" not in kinds(analyze_detections(fr, "P1"))
