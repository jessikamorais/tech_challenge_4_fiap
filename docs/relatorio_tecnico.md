# Relatório técnico — Tech Challenge Fase 4

## 1. Introdução e objetivo
## 2. Fluxo multimodal
Inserir o diagrama de arquitetura e descrever entrada → processamento → fusão → alerta.

## 3. Modelos aplicados por tipo de dado
| Dado | Modelo/serviço | Saída |
|---|---|---|
| Vídeo | YOLOv8-pose / OpenPose; YOLOv8 detecção | Ângulos, desvios, objetos fora da área |
| Áudio | Azure Speech to Text; Azure Language; librosa | Transcrição, sentimento, termos críticos, features vocais |
| Vitais | Limites clínicos, z-score móvel, Isolation Forest | Anomalias por sinal e multivariadas |
| Prescrições | Regras (dose, suspensão, interação) | Alterações inesperadas |
| Movimentação | Z-score móvel, imobilidade | Picos bruscos, imobilidade |
| Fusão | Late fusion ponderada + bônus de concordância | Score e severidade |

## 4. Datasets (PhysioNet, AudioSet, vídeos usados) e pré-processamento
## 5. Resultados
- Exemplos de anomalias detectadas (prints, tabelas, relatório HTML do vídeo)
- Métricas, se houver rótulos (precisão/recall nas anomalias injetadas)
## 6. Fluxo de alerta (Event Grid → Teams/painel)
## 7. Limitações, ética e LGPD
## 8. Próximos passos

# Roteiro do vídeo (até 15 min)
1. (1 min) Problema e arquitetura
2. (3 min) Vídeo: upload, pose, desvios, relatório
3. (3 min) Áudio: transcrição Azure, termos críticos, sentimento, features vocais
4. (3 min) Anomalias: vitais com SpO2 caindo, prescrição alterada
5. (3 min) Fusão, alerta no Teams e painel Streamlit
6. (2 min) Resultados e limitações
