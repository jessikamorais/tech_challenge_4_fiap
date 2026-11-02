# Monitoramento multimodal de pacientes — Tech Challenge Fase 4 (FIAP IA para Devs)

Sistema que funde **vídeo, áudio e dados estruturados** (sinais vitais, prescrições) para
identificar sinais precoces de risco e alertar a equipe médica, usando **Azure** para
transcrição e análise de texto.

## Fluxo

```
Vídeo ──► OpenPose/YOLOv8 ──────────┐
Áudio ──► Azure Speech + Language ──┼─► Fusão multimodal ─► Motor de alertas ─► Equipe médica
Vitais/Prescrições ─► Anomalias ────┘      (score de risco)   (Event Grid/Teams)
```

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `src/video/` | `pose_analyzer.py` (fisioterapia), `object_detector.py` (YOLOv8 genérico), `surgical_tools.py` (YOLOv8 + CholecT50, cirurgia), `report.py` (HTML) |
| `src/audio/` | `transcriber.py` (Azure Speech), `text_analytics.py` (Azure Language), `voice_features.py` (librosa) |
| `src/anomaly/` | `vitals.py`, `prescriptions.py`, `movement.py` |
| `src/fusion/` | `risk_score.py` (late fusion + severidade) |
| `src/alerts/` | `dispatcher.py` (JSONL, Event Grid, Teams) |
| `src/pipeline.py` | Orquestração e CLI |
| `notebooks/` | `main.ipynb` (CholecT50 + YOLOv8, Colab) e `pipeline_multimodal.ipynb` (áudio, fisioterapia, vitais) |
| `dashboard/` | Painel Streamlit |
| `scripts/` | Dados sintéticos, `prepare_cholect50_yolo.py` (conversão), `train_yolo_cholect50.py` (treino/avaliação) |
| `docs/` | Template do relatório técnico e roteiro do vídeo |

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # preencha as chaves Azure
python scripts/generate_synthetic_data.py
pytest

# Somente dados estruturados (não precisa de Azure nem GPU)
python -m src.pipeline --patient P001 \
  --vitals data/samples/vitals_P001.csv \
  --prescriptions data/samples/prescriptions_P001.csv

# Fluxo completo
python -m src.pipeline --patient P001 --video data/raw/fisio.mp4 --audio data/raw/consulta.wav \
  --vitals data/samples/vitals_P001.csv --prescriptions data/samples/prescriptions_P001.csv

streamlit run dashboard/app.py
```

## Google Colab

- `notebooks/main.ipynb` — **foco no CholecT50**: baixa o pacote oficial *Challenge Validation* sozinho, converte
  para YOLO, treina e avalia o YOLOv8, analisa o clipe de teste (linha do tempo, vídeo anotado, relatório) e
  integra com o alerta. Não exige enviar vídeo.
- `notebooks/pipeline_multimodal.ipynb` — pipeline completo com áudio (Azure), fisioterapia e sinais vitais.

Como usar:
1. Suba este projeto para o GitHub (veja abaixo) e abra `notebooks/main.ipynb` no Colab: *Arquivo → Abrir
   notebook → GitHub*, ou o link
   `https://colab.research.google.com/github/<usuario>/<repositorio>/blob/main/notebooks/main.ipynb`.
2. Escolha GPU (T4) em *Ambiente de execução → Alterar tipo*.
3. Na primeira célula, preencha `REPO_URL` (e, se o repositório for **privado**, crie o segredo `GITHUB_TOKEN`
   em 🔑 *Segredos*, com um token de leitura). O notebook clona o repositório e, nas próximas execuções,
   atualiza o código com `git pull`. Não há upload de arquivos.
4. Execute as células em ordem. Detalhes em [`docs/cholect50.md`](docs/cholect50.md).

### Publicar no GitHub
```bash
git init -b main
git add .
git status            # confira: não deve aparecer .env, dados do dataset nem pesos (*.pt)
git commit -m "Projeto inicial"
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```
As chaves Azure ficam fora do repositório (`.env` está no `.gitignore`; no Colab use *Segredos*). O dataset
CholecT50 também não é versionado (licença CC BY-NC-SA 4.0): o notebook o baixa do servidor oficial.

## Vídeo cirúrgico (CholecT50 + YOLOv8)

Detecção de instrumentos laparoscópicos com YOLOv8 ajustado no CholecT50. Passo a passo, limitações
(poucos dados, restrições do pacote) e licença em [`docs/cholect50.md`](docs/cholect50.md).

```bash
python scripts/prepare_cholect50_yolo.py --src /caminho/CholecT50 --out data/cholect50_yolo --dry-run
python scripts/prepare_cholect50_yolo.py --src /caminho/CholecT50 --out data/cholect50_yolo
python scripts/train_yolo_cholect50.py --data data/cholect50_yolo/cholect50_yolo.yaml --device 0
python -m src.pipeline --patient P001 --video-type surgery --video data/raw/cirurgia.mp4
```

## Serviços Azure necessários

1. Azure AI Speech (`AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`)
2. Azure AI Language (`AZURE_LANGUAGE_ENDPOINT`, `AZURE_LANGUAGE_KEY`)
3. Event Grid Topic (opcional) e webhook do Teams (opcional)

Sem as chaves, o módulo de áudio falha com mensagem clara; vídeo e anomalias funcionam localmente.

## Decisões e limitações

- **Pose**: usa YOLOv8-pose (17 keypoints COCO, mesmo formato do OpenPose). Para OpenPose "puro",
  troque `_infer_keypoints` em `pose_analyzer.py`; o resto do código não muda.
- **Limiares clínicos e vocais** são ilustrativos e precisam de validação clínica/calibração por paciente.
- As features de voz são heurísticas de triagem, **não diagnóstico**.
- Dados sintéticos servem para demonstração; para os resultados do relatório, use PhysioNet e AudioSet.
- Dados de saúde são sensíveis (LGPD): anonimize e não versione dados reais (`data/raw` está no `.gitignore`).
