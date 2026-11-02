# Análise de vídeo cirúrgico com YOLOv8 (CholecT50)

Tech Challenge Fase 4 · FIAP IA para Devs. Detecta instrumentos cirúrgicos em vídeo laparoscópico com um
**YOLOv8** ajustado no dataset **CholecT50** e calcula indicadores operacionais de triagem a partir das
detecções, com relatório HTML e vídeo anotado.

O fluxo roda no Google Colab pelo notebook `notebooks/main.ipynb`, que baixa o dataset oficial sozinho.
Nenhum vídeo precisa ser enviado.

## O que o código faz

1. **Baixa e converte** o pacote CholecT50 Challenge Validation (5 clipes) para o formato YOLO, dividindo por clipe.
2. **Treina** o YOLOv8 (fine-tuning a partir de pesos COCO) e avalia num clipe de teste nunca visto.
3. **Analisa** um clipe cirúrgico: detecta os instrumentos em cada frame e calcula três indicadores:
   - `instrumento_fora_do_campo`: detecções fora da zona central da imagem;
   - `movimento_brusco_instrumento`: deslocamento abrupto de um instrumento entre frames adjacentes;
   - `ausencia_prolongada_de_instrumentos`: muitos frames seguidos sem instrumento, com a câmera dentro do corpo.
4. **Gera** um relatório HTML e um vídeo anotado com as detecções.

Cada clipe é tratado como um paciente separado, com ID `CHOLECT-VIDxx`. O pacote não traz identificação de
paciente; o ID é só uma convenção do projeto.

## Estrutura (vídeo)

| Caminho | Conteúdo |
|---|---|
| `notebooks/main.ipynb` | Fluxo completo no Colab: download, conversão, treino, avaliação e análise do clipe |
| `src/video/surgical_tools.py` | Detector (`SurgicalToolAnalyzer`) e regras de indicadores (`analyze_detections`) |
| `src/video/report.py` | Relatório HTML |
| `src/pipeline.py` | Execução por linha de comando (`--video-type surgery`) |
| `scripts/prepare_cholect50_yolo.py` | Descobre o layout do pacote, converte rótulos para YOLO e divide por clipe |
| `scripts/train_yolo_cholect50.py` | Treino e avaliação no clipe de teste |
| `tests/` | Testes (`pytest`) dos indicadores e da conversão |
| `docs/cholect50.md` | Dados, licença, limitações e status de validação |

## Como rodar no Google Colab

1. Abra `notebooks/main.ipynb` no Colab, por *Arquivo → Abrir notebook → GitHub* ou pelo link
   `https://colab.research.google.com/github/<usuario>/<repositorio>/blob/main/notebooks/main.ipynb`.
2. Escolha GPU (T4) em *Ambiente de execução → Alterar tipo*.
3. Preencha `REPO_URL` na primeira célula com o endereço deste repositório. Se o repositório for privado,
   crie o segredo `GITHUB_TOKEN` em 🔑 *Segredos*, com um token de leitura, e ative *Acesso ao notebook*.
   Nas próximas execuções, o código é atualizado com `git pull`.
4. Execute as células em ordem.

## Como rodar localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest

# Conversão, treino e análise (requer o pacote CholecT50 baixado e GPU recomendada)
python scripts/prepare_cholect50_yolo.py --src /caminho/CholecT50 --out data/cholect50_yolo --dry-run
python scripts/prepare_cholect50_yolo.py --src /caminho/CholecT50 --out data/cholect50_yolo
python scripts/train_yolo_cholect50.py --data data/cholect50_yolo/cholect50_yolo.yaml --device 0
python -m src.pipeline --patient CHOLECT-VID75 --video-type surgery --video data/raw/cirurgia.mp4
```

O pesos treinados ficam em `models/` (ignorado pelo Git). Ajuste `SURGICAL_MODEL` no `.env` se usar outro arquivo.

## Publicar no GitHub

```bash
git init -b main
git add .
git status            # confira que não aparecem .env, dados do dataset nem pesos (*.pt)
git commit -m "Análise de vídeo cirúrgico com YOLOv8 e CholecT50"
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

## Dados, licença e limitações

- **Dataset:** CholecT50, licença **CC BY-NC-SA 4.0** (uso não comercial). Não é versionado neste repositório;
  o notebook o baixa do servidor oficial do CAMMA.
  Cite: C.I. Nwoye et al., *Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in
  Endoscopic Videos*, Medical Image Analysis, 78 (2022) 102433.
- **Restrição de publicação:** a página do desafio CholecTriplet2022 informa que publicar resultados sobre o
  conjunto de validação não era permitido até a publicação conjunta do desafio. Verifique a situação atual
  antes de divulgar o trabalho fora da disciplina.
- **Poucos dados:** o pacote tem 5 clipes com frames **não contínuos** (espalhados no tempo). As métricas
  variam muito conforme o clipe de teste.
- **Sem rótulo de erro cirúrgico:** o CholecT50 só traz caixas de instrumentos. Os indicadores são heurísticas
  de triagem para revisão humana, não diagnóstico nem julgamento do procedimento. Só a detecção dos
  instrumentos (mAP) tem métrica validável contra o dataset.
- **Tempo:** a coordenada temporal é o número do frame original. Tempo em segundos só é interpretável em vídeo
  contínuo.
- **Pose:** não há análise de postura nesta versão. OpenPose não se aplica a imagens laparoscópicas.
- **Dados de saúde:** o repositório não versiona vídeos nem dados de pacientes (`data/raw` está no `.gitignore`).
  Mantenha assim se trabalhar com dados reais (LGPD).

Detalhes, resultados medidos e status de validação em [`docs/cholect50.md`](docs/cholect50.md).
