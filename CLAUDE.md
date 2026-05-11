# CLAUDE.md — Memória do Projeto

Este arquivo é lido automaticamente pelo Claude Code em cada sessão. Contém o contexto necessário para trabalhar neste repositório sem precisar reexplicar tudo.

---

## 1. Sobre este projeto

**Nome:** `techchallenge-fase4-saude-mulher`
**Autor:** Diego Valentim Cordeiro (RM369026) — `diego_xr3@yahoo.com.br`
**Curso:** Pós-Tech IADT (IA Para Devs) — FIAP
**Fase:** 4 — Tech Challenge final
**Entrega:** até 20/05/2026
**Modalidade:** Solo

Tech Challenge Fase 4 — Sistema multimodal de monitoramento clínico para saúde da mulher. Processa vídeo, áudio e documentos de uma consulta médica e detecta sinais precoces de risco psicológico ou de violência doméstica usando uma combinação de visão computacional, transcrição de fala e NLP, orquestrados por um grafo LangGraph e com decisão final tomada por um LLM (GPT-4).

---

## 2. Escopo decidido (NÃO alterar sem consultar o autor)

### Funcionalidades atendidas (2 — mínimo do enunciado):
- **Análise de vídeo clínico** (DeepFace para emoções + MediaPipe para postura + YOLOv8 para objetos em cena)
- **Análise de áudio de consultas** (Whisper local para transcrição + AWS Comprehend para análise NLP)

### Objetivos atendidos (3 — mínimo do enunciado):
- **(ii) Triagem de violência doméstica** — fusão de emoções faciais + postura defensiva + análise lexical da fala
- **(iii) Monitoramento de bem-estar psicológico** — detecção de sinais de depressão pós-parto, ansiedade, fadiga
- **(iv) Integração com serviços em nuvem especializados** — AWS Textract (laudos PDF) + AWS Comprehend (NLP da transcrição) + OpenAI GPT-4 (síntese)

### Funcionalidades/objetivos DESCARTADOS (não tentar reintroduzir):
- Detecção de anomalias em sinais vitais (não há dados disponíveis)
- Monitoramento em tempo real / streaming (complexidade incompatível com o prazo)
- Detecção precoce de riscos materna/ginecológica (exigiria dataset clínico real)
- YOLOv8 **treinado** customizado — usaremos apenas YOLOv8 **pré-treinado** (Ultralytics, modelo `yolov8n.pt` padrão COCO)

---

## 3. Stack tecnológica (decidida e congelada)

### Bibliotecas Python obrigatórias
- `opencv-python` — captura de vídeo e desenho de anotações
- `deepface` + `tf-keras` — análise de emoções faciais
- `mediapipe` — keypoints de pose corporal
- `ultralytics` — YOLOv8 pré-treinado
- `openai-whisper` — transcrição de áudio local (FOSS)
- `moviepy` — extração de áudio do vídeo
- `boto3` — clientes AWS (Textract + Comprehend)
- `openai` — cliente da API OpenAI para GPT-4
- `langgraph` + `langchain-openai` — orquestração do pipeline
- `python-dotenv` — gerenciamento de credenciais via `.env`
- `tqdm` — barras de progresso
- `pydantic` — validação de schemas entre nós
- `reportlab` — geração de PDF do relatório final

### Serviços externos
- **AWS** (escolhido sobre Azure que o enunciado cita) — Textract + Comprehend via boto3
- **OpenAI API** — GPT-4 para fusão multimodal e geração do relatório clínico
- **Whisper local** (não API) — roda na GPU do autor

### Ferramentas/abordagens DESCARTADAS
- **Não usar** Azure Cognitive Services (o enunciado cita como exemplo, mas o curso ensinou AWS)
- **Não usar** SpeechRecognition (Google Web API) — o curso ensinou isso na Aula 4, mas Whisper local é tecnicamente superior, gratuito, e funciona offline; vamos justificar essa substituição no relatório
- **Não treinar** modelos do zero — só inferência com pesos pré-treinados
- **Não usar** LangChain `LLMChain` ou `.run()` (deprecated) — apenas `prompt | llm | .invoke()` (pipeline moderno)
- **Não usar** Azure Face API, Amazon Rekognition (mencionados nas aulas mas não obrigatórios)

---

## 4. Ambiente de execução (CRITICAL)

### Hardware
- **GPU:** NVIDIA RTX 5070 (arquitetura **Blackwell, compute capability sm_120**)
- **OS:** Windows 11
- **Python:** 3.11+

### PEGADINHA CRÍTICA — PyTorch em Blackwell
O autor já passou por isso na Fase 3 (projeto OncoSUS): **PyTorch estável NÃO funciona em sm_120**. Para qualquer biblioteca que use PyTorch (Whisper, DeepFace com backend TF que também tem dependências), instalar a build nightly com CUDA 12.8:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Sintoma se errar:** `RuntimeError: CUDA error: no kernel image is available for execution on the device` ou similar mencionando `sm_120` ou `Blackwell`. Se aparecer, reinstalar PyTorch nightly cu128.

### TensorFlow / DeepFace
DeepFace usa TensorFlow por padrão. Em Windows + RTX 5070, pode dar problema. Estratégia: **deixar DeepFace usar CPU** (é rápido o suficiente para o caso de uso — análise de frames amostrados, não vídeo em tempo real). Para forçar:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # antes de importar deepface
```

### Caches de modelos (importante para não baixar tudo de novo)
Todos os modelos baixados pelas bibliotecas vão para `./.cache/` (relativo à raiz do projeto):
- Whisper: `WHISPER_CACHE_DIR=./.cache/whisper`
- HuggingFace: `HF_HOME=./.cache/huggingface`
- DeepFace: `DEEPFACE_HOME=./.cache/deepface` (envia para `~/.deepface` por padrão; sobrescrever)
- Ultralytics: salva em `./.cache/ultralytics`

Configurar via `.env` e carregar com `python-dotenv` ANTES de importar as bibliotecas.

---

## 5. Estrutura do repositório (alvo)

```
techchallenge-fase4-saude-mulher/
├── CLAUDE.md                    # este arquivo
├── PROJECT_SPEC.md              # arquitetura detalhada + prompts sequenciais
├── README.md                    # documentação pública do projeto
├── RELATORIO_TECNICO.md         # relatório técnico final (entregável)
├── requirements.txt
├── .env.example                 # template das credenciais (sem secrets)
├── .env                         # credenciais reais (gitignored)
├── .gitignore
├── data/
│   ├── input/                   # vídeos e PDFs de entrada (gitignored)
│   ├── intermediate/            # outputs dos nós (JSON, áudio extraído)
│   └── output/                  # vídeo anotado + relatório final
├── src/
│   ├── __init__.py
│   ├── config.py                # carrega .env, paths, constantes
│   ├── schemas.py               # Pydantic models do State LangGraph
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── video_node.py        # DeepFace + MediaPipe + YOLOv8
│   │   ├── audio_node.py        # MoviePy + Whisper local
│   │   ├── nlp_node.py          # AWS Comprehend + classificador
│   │   ├── doc_node.py          # AWS Textract para laudos PDF
│   │   └── fusion_node.py       # GPT-4 sintetiza e classifica risco
│   ├── graph.py                 # construção do StateGraph LangGraph
│   └── report.py                # geração do PDF final do relatório clínico
├── scripts/
│   ├── 01_test_video_node.py    # testa o nó VIDEO isolado
│   ├── 02_test_audio_node.py    # testa o nó AUDIO isolado
│   ├── 03_test_nlp_node.py
│   ├── 04_test_doc_node.py
│   ├── 05_test_fusion_node.py
│   └── run_pipeline.py          # CLI principal — roda o grafo completo
├── notebooks/
│   └── exploracao.ipynb         # opcional, para debug
└── docs/
    ├── arquitetura.png          # diagrama do grafo (gerar com LangGraph)
    └── exemplos_saida/          # screenshots e amostras de output
```

---

## 6. Convenções de código

- **Idioma:** código em inglês, docstrings e comentários em **português brasileiro**, mensagens de log em português
- **Formatação:** PEP 8, linhas até 100 chars
- **Type hints:** obrigatório em assinaturas de função pública
- **Validação:** todos os contratos entre nós usam Pydantic models definidos em `src/schemas.py`
- **Logging:** usar `logging` stdlib com nível INFO por padrão; nunca `print()` em código de produção (só em scripts de demo)
- **Paths:** sempre via `pathlib.Path`; nunca strings hardcoded
- **Caminhos de arquivo:** sempre relativos à raiz do projeto, carregados de `src/config.py`

---

## 7. Mapeamento aulas-do-curso ↔ requisitos-do-enunciado

Para o relatório técnico, é importante mostrar que cada requisito do enunciado foi atendido com uma técnica vista em alguma aula. Quando o Claude Code escrever o relatório, deve referenciar:

| Requisito do enunciado | Aula | Stack usada |
|---|---|---|
| Detecção de sinais não-verbais (medo, desconforto) em consultas | Bloco 1 — Aula 2 | DeepFace `analyze(actions=['emotion'])` |
| Linguagem corporal indicativa de abuso | Bloco 1 — Aula 3 | MediaPipe Pose + heurística sobre landmarks |
| Detecção de objetos em cena clínica (cumpre "YOLOv8 customizado") | Não ensinado em aula — proposta do aluno | Ultralytics YOLOv8 pré-treinado COCO |
| Transcrição de áudio de consultas | Bloco 1 — Aula 4 (substituído por Whisper) | openai-whisper local |
| Análise de sentimento e entidades na fala | Bloco 2 — Aula 3 | AWS Comprehend `detect_sentiment`, `detect_entities`, `detect_key_phrases` |
| Classificação de tópicos de risco na fala | Bloco 1 — Aula 5 | scikit-learn TF-IDF + Naive Bayes |
| Processamento de laudos em PDF | Bloco 2 — Aula 2 | AWS Textract `analyze_document` |
| Síntese multimodal e relatórios automáticos | Bloco 3 — Aulas 2-3 + Bloco 1 — Aula 6 | OpenAI GPT-4 `chat.completions` (substitui pipeline de sumarização Transformers) |
| Orquestração de fluxo multimodal | Disciplina paralela LangGraph | LangGraph StateGraph |

A substituição de Whisper sobre SpeechRecognition e de GPT-4 sobre Transformers `pipeline("summarization")` é tecnicamente justificável (qualidade superior + raciocínio multimodal) — o relatório deve explicar essas decisões.

---

## 8. Restrições de privacidade e ética (importante para o relatório)

Domínio é saúde da mulher — extremamente sensível. O relatório técnico deve discutir:
- Dados sensíveis nunca são logados em texto plano
- Credenciais AWS/OpenAI via `.env`, nunca em código
- O sistema é uma **ferramenta de apoio à decisão**, não substitui avaliação médica
- Vídeo de demo usa material disponibilizado pela instituição (FIAP), não pacientes reais
- Em produção real, exigiria LGPD compliance, consentimento explícito, criptografia em trânsito e repouso

---

## 9. Vídeo-fonte de demonstração

A FIAP disponibilizou um vídeo institucional para uso no Tech Challenge:
- URL: `https://docs.google.com/videos/d/1XVjkncR0HZGVhLZD9M1jPf8oO0wvhILqN_-xTpvR8Kk/edit?scene=id.p`
- **Esse vídeo precisa ser baixado MANUALMENTE pelo autor** (Google Docs Videos não suporta download direto via API)
- Salvar em `data/input/video_consulta.mp4`
- Se o vídeo não tiver áudio de qualidade suficiente, o autor grava 30-60s adicionais simulando uma consulta

---

## 10. Como o Claude Code deve agir

- **Sempre ler PROJECT_SPEC.md primeiro** para entender qual fase do projeto está sendo executada
- **Antes de cada sessão**, perguntar "qual seção do PROJECT_SPEC.md vamos executar?"
- **Nunca instalar bibliotecas sem confirmar** — RTX 5070 tem caso especial com PyTorch
- **Sempre fazer commit Git** ao final de cada subtarefa concluída e testada
- **Pedir ao autor** para fornecer arquivos de input antes de tentar baixar/gerar
- **Em caso de erro de CUDA/sm_120**, sugerir reinstalar PyTorch nightly cu128 (ver seção 4)
- **Mensagens de progresso e logs em português**
- **Não criar testes unitários elaborados** — o Tech Challenge é avaliado por demonstração funcional em vídeo, não por cobertura de teste. Scripts em `scripts/0X_test_*.py` são suficientes para validar cada nó manualmente
