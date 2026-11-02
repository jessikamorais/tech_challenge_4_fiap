"""Painel da equipe médica: streamlit run dashboard/app.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ALERTS = Path(__file__).resolve().parent.parent / "data" / "alerts" / "alerts.jsonl"
st.set_page_config(page_title="Monitoramento multimodal", layout="wide")
st.title("Alertas de risco — monitoramento multimodal")

if not ALERTS.exists():
    st.info("Nenhum alerta ainda. Execute o pipeline para gerar alertas.")
    st.stop()

rows = [json.loads(l) for l in ALERTS.read_text(encoding="utf-8").splitlines() if l.strip()]
df = pd.DataFrame(rows)
sev = st.multiselect("Severidade", ["critica", "alta", "media", "baixa"],
                     default=["critica", "alta", "media"])
view = df[df["severity"].isin(sev)].sort_values("timestamp", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("Alertas", len(view))
c2.metric("Pacientes", view["patient_id"].nunique())
c3.metric("Críticos", int((view["severity"] == "critica").sum()))

st.dataframe(view[["timestamp", "patient_id", "severity", "score", "modalities"]],
             use_container_width=True)
for _, r in view.head(5).iterrows():
    with st.expander(f"{r['patient_id']} — {r['severity']} ({r['score']})"):
        for f in r["findings"]:
            st.write(f"**[{f['modality']}]** {f['description']}")
