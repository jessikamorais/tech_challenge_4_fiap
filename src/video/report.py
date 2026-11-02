"""Relatório automático (HTML) com os desvios detectados no vídeo."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import settings
from ..schemas import Finding


def generate_video_report(patient_id: str, video_path: str, findings: list[Finding],
                          out_dir: Path | None = None, summary: dict | None = None) -> Path:
    out_dir = Path(out_dir or settings.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{f.kind}</td><td>{f.score:.2f}</td><td>{f.description}</td>"
        f"<td><code>{f.details}</code></td></tr>" for f in findings
    ) or "<tr><td colspan='4'>Nenhum desvio detectado.</td></tr>"
    summ = ""
    if summary:
        items = "".join(f"<tr><td>{k}</td><td><code>{v}</code></td></tr>" for k, v in summary.items())
        summ = f"<h2>Resumo</h2><table>{items}</table>"
    html = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<title>Relatório de vídeo — {patient_id}</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style>
<h1>Relatório de análise de vídeo</h1>
<p>Paciente: <b>{patient_id}</b> · Vídeo: <code>{video_path}</code> · Gerado em {datetime.now():%d/%m/%Y %H:%M}</p>
{summ}<h2>Achados</h2>
<table><tr><th>Tipo</th><th>Score</th><th>Descrição</th><th>Detalhes</th></tr>{rows}</table>
</html>"""
    path = out_dir / f"video_report_{patient_id}.html"
    path.write_text(html, encoding="utf-8")
    return path
