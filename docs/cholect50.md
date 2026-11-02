# Vídeo cirúrgico: CholecT50 + YOLOv8 (Google Colab)

Detecção de instrumentos em vídeo laparoscópico (colecistectomia) com YOLOv8 ajustado no CholecT50, mais
indicadores operacionais derivados das detecções. O notebook `notebooks/main.ipynb` executa tudo no Colab,
**baixando o dataset automaticamente** (sem enviar vídeo).

## 1. De onde vêm os dados (e por que usamos o pacote de validação)

| Fonte | Acesso | Caixas de instrumento? | Serve para treinar o YOLOv8? |
|---|---|---|---|
| CholecT50 completo (50 vídeos, ~100,9 mil frames) | **formulário de solicitação** (sem link direto) | só em 5 vídeos | sim, mas exige o formulário |
| **CholecT50 Challenge Validation** (5 clipes, ~1,1 mil frames) | **download direto** (página *Downloads* do repositório) | em todos os clipes | **sim — é o que o notebook baixa** |
| Espelho Hugging Face `Voxel51/cholect50` (FiftyOne) | download via `fiftyone` | não (só presença por frame) | não (serve a classificação, não a detecção) |

Fonte oficial: <https://github.com/CAMMA-public/cholect50> (página *Downloads*). Licença **CC BY-NC-SA 4.0**
(uso não comercial; derivados sob a mesma licença). Cite no relatório:

> C.I. Nwoye et al. *Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in
> Endoscopic Videos.* Medical Image Analysis, 78 (2022) 102433.

**Restrições do pacote de validação** (segundo a página do desafio CholecTriplet2022):
- as caixas vêm do dataset *m2cai16-tool-location*;
- a página informava que **publicar resultados sobre esse conjunto não era permitido até a publicação conjunta
  do desafio** — verifique a situação atual antes de divulgar fora da disciplina;
- imagens com rostos foram zeradas (frames pretos); o código os descarta.

Se o grupo obtiver o CholecT50 completo pelo formulário, o mesmo conversor funciona (aponte `LOCAL_ZIP`/`RAW_DIR`
para a pasta com `videos/` e `labels/`).

## 2. Limitação central: poucos dados
O pacote tem **5 clipes curtos**. Divide-se **por clipe** (nunca por frame): 3 treino, 1 validação, 1 teste.
Consequências a declarar no relatório: conjunto pequeno e muito correlacionado (frames a 1 fps), classes raras
com poucos exemplos e métricas que variam bastante conforme o clipe de teste.

## 2b. Identificação de paciente
Cada clipe é tratado como **um paciente separado**, com ID `CHOLECT-VIDxx` (ex.: `CHOLECT-VID75`). O pacote não traz
identificação de paciente; o ID é só uma convenção do projeto para não misturar achados entre clipes. Os sinais vitais
e as prescrições da fusão (seção 7 do notebook) são **sintéticos**, gerados para esse mesmo ID apenas para demonstrar
o mecanismo de alerta.

## 3. Como o conversor lê o pacote
`scripts/prepare_cholect50_yolo.py` **descobre a estrutura sozinho** (não depende de nomes de pasta fixos):
- rótulos JSON em três formatos de instância: lista de 15 itens (CholecT50), lista de 7 itens
  `[tripleto, instrumento, confiança, x, y, w, h]` e dict `{"instrument": [id, prob, x, y, w, h]}`;
- frames como imagens (pastas) ou dentro de vídeos (extraídos a 1 fps);
- remove caixas duplicadas, descarta frames pretos e **frames ambíguos** (instrumento presente sem caixa) e usa
  como fundo só frames realmente vazios.

Se nada casar, o erro mostra a estrutura encontrada (`--inspect`). Uso manual:
```bash
python scripts/prepare_cholect50_yolo.py --src data/cholect50_raw --inspect
python scripts/prepare_cholect50_yolo.py --src data/cholect50_raw --out data/cholect50_yolo
python scripts/train_yolo_cholect50.py --data data/cholect50_yolo/cholect50_yolo.yaml --device 0
```

## 4. Saídas mostradas no notebook
Estrutura baixada · distribuição de caixas por classe/split · amostras com anotações · curvas de treino ·
matriz de confusão · mAP50/mAP50-95/precisão/recall e AP50 por classe no clipe de teste · linha do tempo dos
instrumentos (previsto × anotado) · frames com detecções · vídeo anotado · relatório HTML · fusão e alerta.

> **Atenção ao tempo:** os frames do pacote de validação **não são contínuos** (ex.: VID75 tem 141 imagens
> espalhadas por mais de 1.800 posições, com lacunas irregulares). Não existe "1 s de vídeo = 1 s de cirurgia":
> a coordenada temporal é o **número do frame original** e as lacunas interrompem as sequências. Tempo e
> ausência só ficam interpretáveis em vídeo contínuo (ex.: CholecT50 completo, via formulário, a 1 fps).

## 5. Indicadores de "fora do padrão" (heurísticas)
| Achado | Regra |
|---|---|
| `instrumento_fora_do_campo` | % de detecções fora da zona central acima do limiar |
| `movimento_brusco_instrumento` | deslocamento entre frames **adjacentes** acima de limiar robusto (mediana/MAD); lacunas não contam |
| `ausencia_prolongada_de_instrumentos` | ≥ `MAX_IDLE_S` frames adjacentes sem instrumento com a câmera dentro do corpo |

**O CholecT50 não tem rótulos de "erro/desvio cirúrgico".** Esses achados são triagem para revisão humana e
não podem ser validados contra o dataset; só a **detecção** dos instrumentos tem métrica objetiva (mAP).

## 6. Status de validação
Validado com um pacote **sintético** que imita o dataset (rótulos de 7 itens, frames em pastas, frames zerados),
em CPU, `ultralytics 8.4.194`: download por URL, extração, conversão, treino, avaliação, inferência, linha do
tempo, vídeo anotado, relatório, fusão/alerta e a execução completa do notebook (`nbconvert`).
Testes automáticos: `tests/test_cholect50_prepare.py`, `tests/test_cholect50_layouts.py`,
`tests/test_surgical_tools.py`.

**Não validado:**
- o ZIP **real** (o ambiente de desenvolvimento não alcança o servidor do CAMMA): o layout real foi inferido da
  documentação oficial; se o conversor não casar rótulos e frames, rode a seção 3.1 do notebook e use a saída;
- métricas em dados reais: o mAP do teste sintético não diz nada sobre imagens cirúrgicas;
- execução no Colab propriamente dito (`userdata`, Drive, `files.upload` e `%pip` não rodam fora dele).
