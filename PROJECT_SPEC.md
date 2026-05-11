# PROJECT_SPEC.md — Especificação Técnica e Roteiro de Execução

Este documento contém a arquitetura completa do projeto e os **prompts sequenciais** que devem ser passados ao Claude Code em cada sessão de trabalho. Cada seção numerada (S1, S2, ...) corresponde a uma sessão.

> **Como usar:** Abra o Claude Code (`claude` no terminal). Copie e cole o bloco "PROMPT PARA O CLAUDE CODE" da sessão atual. Acompanhe o que ele faz, aprove edições, e ao final faça commit Git.

---

## Visão geral da arquitetura

```
                          ┌─────────────────────────┐
                          │  Entrada do usuário     │
                          │  - video_consulta.mp4   │
                          │  - laudo.pdf (opcional) │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   LangGraph Orquestrador│
                          │   (StateGraph)          │
                          └────────────┬────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       ┌─────────────┐         ┌─────────────┐          ┌─────────────┐
       │   VIDEO     │         │   AUDIO     │          │    DOC      │
       │             │         │             │          │             │
       │ DeepFace    │         │ MoviePy →   │          │ AWS Textract│
       │ MediaPipe   │         │ Whisper     │          │             │
       │ YOLOv8      │         │             │          │             │
       └─────┬───────┘         └──────┬──────┘          └──────┬──────┘
             │                        │                        │
             │                        ▼                        │
             │                ┌─────────────┐                  │
             │                │     NLP     │                  │
             │                │             │                  │
             │                │ AWS         │                  │
             │                │ Comprehend  │                  │
             │                │ + Classifier│                  │
             │                └──────┬──────┘                  │
             │                       │                         │
             └───────────────────────┼─────────────────────────┘
                                     ▼
                          ┌─────────────────────────┐
                          │       FUSION            │
                          │                         │
                          │  GPT-4 recebe JSON      │
                          │  de todos os nós,       │
                          │  classifica risco,      │
                          │  gera relatório         │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  Saídas                 │
                          │  - video_anotado.mp4    │
                          │  - relatorio.pdf        │
                          │  - alerta.json          │
                          └─────────────────────────┘
```

---

## Schema do State LangGraph (contrato entre nós)

Definido em `src/schemas.py` usando Pydantic + TypedDict. Cada nó lê campos do State e devolve um delta (dict parcial).

```python
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel
from datetime import datetime

# === Modelos de evento (Pydantic) ===

class FacialEmotionEvent(BaseModel):
    timestamp_seconds: float
    dominant_emotion: str  # angry, disgust, fear, happy, sad, surprise, neutral
    confidence: float
    all_emotions: dict[str, float]

class PostureEvent(BaseModel):
    timestamp_seconds: float
    posture_label: str  # "defensiva", "aberta", "encolhida", "neutra"
    landmarks_summary: dict  # ângulos relevantes (ombros, braços, cabeça)

class ObjectDetectionEvent(BaseModel):
    timestamp_seconds: float
    detected_objects: List[dict]  # [{class, confidence, bbox}]

class VideoAnalysis(BaseModel):
    duration_seconds: float
    fps: float
    total_frames_analyzed: int
    emotion_events: List[FacialEmotionEvent]
    posture_events: List[PostureEvent]
    object_events: List[ObjectDetectionEvent]
    dominant_emotion_overall: str
    emotion_distribution: dict[str, float]
    defensive_posture_ratio: float  # 0.0–1.0

class AudioAnalysis(BaseModel):
    duration_seconds: float
    transcription_full: str
    transcription_segments: List[dict]  # [{start, end, text}]
    language_detected: str

class NLPAnalysis(BaseModel):
    aws_sentiment: str  # POSITIVE, NEGATIVE, NEUTRAL, MIXED
    aws_sentiment_scores: dict[str, float]
    aws_entities: List[dict]  # [{text, type, score}]
    aws_key_phrases: List[str]
    custom_risk_classification: str  # "violencia", "depressao", "ansiedade", "normal"
    custom_risk_confidence: float
    risk_keywords_found: List[str]

class DocAnalysis(BaseModel):
    source_pdf_path: str
    extracted_text: str
    tables_extracted: List[dict]
    forms_extracted: List[dict]
    nlp_summary: Optional[NLPAnalysis]

class FusionResult(BaseModel):
    risk_level: str  # "ALTO", "MEDIO", "BAIXO", "INDETERMINADO"
    risk_categories: List[str]  # ["violencia_domestica", "depressao_pos_parto", ...]
    confidence_overall: float
    clinical_summary: str  # texto gerado pelo GPT-4
    recommended_actions: List[str]
    alert_triggered: bool
    reasoning: str  # raciocínio do GPT-4

# === State principal do LangGraph ===

class PipelineState(TypedDict, total=False):
    # Entradas
    video_path: str
    pdf_path: Optional[str]
    
    # Artefatos intermediários
    audio_path: str  # extraído do vídeo
    
    # Resultados dos nós
    video_analysis: VideoAnalysis
    audio_analysis: AudioAnalysis
    nlp_analysis: NLPAnalysis
    doc_analysis: Optional[DocAnalysis]
    fusion: FusionResult
    
    # Metadados
    pipeline_started_at: str
    pipeline_completed_at: str
    errors: List[str]
```

---

## S1 — Setup inicial do projeto

**Objetivo:** criar estrutura de pastas, requirements.txt, configurar AWS CLI e .env, fazer o primeiro commit.

**Tempo estimado:** 30 min

### Pré-requisitos do autor (fazer ANTES de abrir o Claude Code)
- [ ] Conta AWS criada com credenciais IAM com permissões para Textract e Comprehend
- [ ] Chave da API OpenAI gerada em platform.openai.com
- [ ] Vídeo da FIAP baixado para `~/Downloads/video_consulta.mp4`
- [ ] Pasta vazia criada: `C:\Users\DIEGO\Documents\git\techchallenge-fase4-saude-mulher`
- [ ] Terminal aberto nessa pasta

### PROMPT PARA O CLAUDE CODE

```
Estou começando o Tech Challenge da Fase 4 da minha pós-graduação. Os arquivos CLAUDE.md e PROJECT_SPEC.md já estão na raiz deste diretório — leia ambos antes de fazer qualquer coisa.

Vamos executar a sessão S1 do PROJECT_SPEC.md (Setup inicial).

Tarefas:

1. Inicialize git neste diretório.

2. Crie a estrutura de pastas exatamente como descrita na seção 5 do CLAUDE.md (data/, src/, scripts/, notebooks/, docs/).

3. Crie um .gitignore Python padrão + adicione:
   - .env
   - data/input/
   - data/intermediate/
   - data/output/
   - .cache/
   - *.mp4, *.wav, *.mp3 em data/

4. Crie .env.example com as variáveis necessárias:
   - OPENAI_API_KEY
   - OPENAI_MODEL (default gpt-4o-mini)
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION (sa-east-1 ou us-east-1)
   - WHISPER_MODEL_SIZE (default large-v3)
   - WHISPER_CACHE_DIR (default ./.cache/whisper)
   - HF_HOME (default ./.cache/huggingface)
   - DEEPFACE_HOME (default ./.cache/deepface)

5. Crie requirements.txt com as bibliotecas listadas na seção 3 do CLAUDE.md. NÃO inclua torch ainda — vou instalar manualmente nightly cu128 para o Blackwell sm_120.

6. Crie src/config.py que:
   - Carrega .env via python-dotenv
   - Define paths absolutos baseados em pathlib.Path
   - Define constantes (PROJECT_ROOT, DATA_DIR, etc)
   - Configura logging

7. Crie src/schemas.py com os Pydantic models e o TypedDict PipelineState exatamente como definido no PROJECT_SPEC.md.

8. Crie um README.md inicial mínimo (preenchemos depois) com nome do projeto, autor, instruções de instalação.

9. NÃO instale dependências ainda — apenas crie os arquivos. Vou rodar pip install manualmente para controlar versões.

10. Faça o primeiro commit: "S1: estrutura inicial do projeto"

Antes de começar, me mostre o plano e pergunte se algo está unclear. Depois execute passo a passo, mostrando o que está editando.
```

### Validação pós-sessão
- [ ] `tree` mostra a estrutura esperada
- [ ] `git log` mostra o primeiro commit
- [ ] `.env.example` tem todas as variáveis
- [ ] `src/schemas.py` importa sem erro: `python -c "from src.schemas import PipelineState"`

---

## S2 — Nó VIDEO

**Objetivo:** script standalone que processa um vídeo e produz um JSON com eventos de emoção, postura e objetos detectados, mais um vídeo anotado.

**Tempo estimado:** 1-2h

### Pré-requisitos
- [ ] S1 concluída
- [ ] PyTorch nightly cu128 instalado: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
- [ ] `pip install -r requirements.txt`
- [ ] Vídeo de teste em `data/input/video_consulta.mp4`

### PROMPT PARA O CLAUDE CODE

```
Vamos executar S2 do PROJECT_SPEC.md — Nó VIDEO.

Crie src/nodes/video_node.py implementando uma função:

    def analyze_video(video_path: Path, output_video_path: Path, sample_every_n_frames: int = 15) -> VideoAnalysis

Esta função:

1. Abre o vídeo com cv2.VideoCapture
2. Para cada frame amostrado (a cada N frames, default 15, para não processar tudo):
   a) Roda DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
      → registra FacialEmotionEvent
   b) Roda MediaPipe Pose
      → calcula ângulos entre ombros/braços/cabeça
      → classifica postura: "defensiva" (braços cruzados/encolhidos), "aberta", "encolhida", "neutra"
      → registra PostureEvent
   c) Roda YOLOv8 (modelo yolov8n.pt) e filtra apenas as classes relevantes para contexto clínico:
      person, knife, scissors, bottle, cell phone, book, laptop, chair
      → registra ObjectDetectionEvent

3. Para CADA frame (não só amostrado), desenha sobre o frame:
   - Retângulo verde em torno da face com a emoção dominante (estilo Aula 2)
   - Esqueleto MediaPipe (estilo Aula 3)
   - Bounding boxes YOLO com label e confiança
   - Texto no canto: "tempo: XX.Xs | emocao: Y | postura: Z"

4. Escreve o vídeo anotado em output_video_path

5. Calcula estatísticas agregadas:
   - dominant_emotion_overall (moda das emoções)
   - emotion_distribution (% de cada emoção)
   - defensive_posture_ratio (% de frames classificados como defensiva ou encolhida)

6. Retorna VideoAnalysis preenchido

Detalhes de implementação:

- Use tqdm para barra de progresso
- Configure os caches para ./.cache/ (DEEPFACE_HOME, HF_HOME) ANTES de importar deepface
- Force DeepFace para CPU (CUDA_VISIBLE_DEVICES=-1) — explicado no CLAUDE.md
- Whisper, YOLO e MediaPipe rodam em GPU ok
- Logging em português, INFO no console e DEBUG num arquivo data/intermediate/video_node.log

Crie também scripts/01_test_video_node.py que:
- Carrega .env
- Lê o vídeo de data/input/video_consulta.mp4
- Chama analyze_video()
- Salva o JSON em data/intermediate/video_analysis.json
- Salva o vídeo anotado em data/intermediate/video_anotado.mp4
- Imprime as estatísticas agregadas no terminal

Antes de codar, me mostre o plano e os ângulos que você vai calcular para definir "postura defensiva". Depois execute, me mostrando cada arquivo que cria/edita.

Ao terminar, rode `python scripts/01_test_video_node.py` e me mostre a saída. Se der erro de CUDA, sugira correções.

Faça commit ao final: "S2: implementacao do no VIDEO"
```

### Validação pós-sessão
- [ ] `data/intermediate/video_analysis.json` existe e tem dados coerentes
- [ ] `data/intermediate/video_anotado.mp4` é reproduzível e mostra anotações sobrepostas
- [ ] Console mostrou estatísticas agregadas
- [ ] Git tem o commit S2

---

## S3 — Nó AUDIO (Whisper local)

**Objetivo:** extrair áudio do vídeo e transcrever com Whisper local.

**Tempo estimado:** 30-45 min

### PROMPT PARA O CLAUDE CODE

```
Vamos executar S3 — Nó AUDIO.

Crie src/nodes/audio_node.py com:

    def extract_audio(video_path: Path, audio_path: Path) -> Path
    def transcribe_audio(audio_path: Path, model_size: str = None) -> AudioAnalysis

extract_audio:
- Usa moviepy.editor.VideoFileClip para extrair o áudio
- Salva como WAV mono 16kHz (ideal para Whisper)
- Retorna o path

transcribe_audio:
- Carrega Whisper local (default config WHISPER_MODEL_SIZE do .env, fallback "large-v3")
- Configura cache em WHISPER_CACHE_DIR
- Roda model.transcribe(audio_path, language="pt", task="transcribe")
- Retorna AudioAnalysis com:
  - transcription_full: texto completo
  - transcription_segments: list de {start, end, text} dos segmentos retornados pelo Whisper
  - duration_seconds: duração total
  - language_detected: pt

Detalhes:
- Use logging em pt-BR
- Verifique se o áudio existe antes de transcrever
- Se a placa não suportar large-v3 por VRAM, fallback automático para medium

Crie scripts/02_test_audio_node.py que:
- Lê data/input/video_consulta.mp4
- Extrai áudio para data/intermediate/audio_consulta.wav
- Transcreve
- Salva data/intermediate/audio_analysis.json com toda a AudioAnalysis serializada
- Imprime a transcrição completa no terminal

Antes de codar, me confirme que vamos usar whisper-1 oficial da OpenAI (pacote openai-whisper, não faster-whisper). Faça commit: "S3: implementacao do no AUDIO com Whisper local"
```

### Validação pós-sessão
- [ ] `data/intermediate/audio_consulta.wav` existe
- [ ] `data/intermediate/audio_analysis.json` tem transcrição em PT
- [ ] Transcrição lê coerente (não é gibberish)

---

## S4 — Nó NLP (AWS Comprehend + classificador customizado)

**Objetivo:** receber a transcrição e produzir análise de sentimento, entidades, frases-chave, mais classificação de risco psicológico/violência.

**Tempo estimado:** 1h

### PROMPT PARA O CLAUDE CODE

```
Vamos executar S4 — Nó NLP.

Crie src/nodes/nlp_node.py com:

    def analyze_text_with_comprehend(text: str) -> dict
    def classify_risk(text: str) -> tuple[str, float, list[str]]
    def analyze_nlp(text: str) -> NLPAnalysis

analyze_text_with_comprehend:
- Cria cliente boto3 ('comprehend')
- Chama detect_sentiment, detect_entities, detect_key_phrases com LanguageCode='pt'
- IMPORTANTE: textos com mais de 5000 bytes precisam ser cortados ou divididos (limite do Comprehend)
- Retorna dict com sentiment, sentiment_scores, entities, key_phrases

classify_risk:
- Recebe o texto da transcrição
- Aplica um classificador baseado em listas de palavras-chave (heurística simples mas defensável):
  
  RISK_KEYWORDS = {
      "violencia": ["bate", "machuca", "agride", "empurra", "grita comigo", "tenho medo dele", 
                    "ele não deixa", "controla", "ameaça", "isolada", "não posso", "ele me proíbe",
                    "sozinha em casa", "me sinto presa"],
      "depressao": ["triste", "vazio", "sem esperança", "nada faz sentido", "chorando", "não consigo",
                    "exausta", "cansada o tempo todo", "sem energia", "não durmo", "não tenho vontade",
                    "não consigo cuidar", "péssima mãe", "falhei"],
      "ansiedade": ["preocupada", "ansiosa", "pânico", "coração disparado", "falta de ar", "agitada",
                    "não consigo relaxar", "medo de", "e se", "tenho medo que aconteça"]
  }
  
- Conta ocorrências de cada categoria (case insensitive, normalize com unicodedata)
- Retorna a categoria com maior contagem, confidence proporcional, e a lista de keywords encontradas
- Se nenhuma keyword bater, retorna ("normal", 1.0, [])

analyze_nlp:
- Chama os dois acima e monta NLPAnalysis

Crie scripts/03_test_nlp_node.py que:
- Lê data/intermediate/audio_analysis.json
- Pega o transcription_full
- Chama analyze_nlp
- Salva data/intermediate/nlp_analysis.json
- Imprime o resumo no terminal (sentimento, top 5 entidades, top 5 frases-chave, classificação de risco)

Notas:
- Se as credenciais AWS não estiverem configuradas, dê uma mensagem clara
- Custo AWS Comprehend é por unidade de 100 caracteres — uma consulta inteira custa centavos
- Para o classificador customizado, NÃO chame OpenAI ainda — apenas keywords. O GPT-4 entra só no FUSION

Faça commit: "S4: implementacao do no NLP com AWS Comprehend + classificador heuristico"
```

### Validação pós-sessão
- [ ] `data/intermediate/nlp_analysis.json` tem todos os campos
- [ ] AWS Comprehend respondeu sem erro (verificar conta tem permissões)
- [ ] Classificador de risco identifica algo coerente com o conteúdo do vídeo

---

## S5 — Nó DOC (AWS Textract)

**Objetivo:** extrair texto de um PDF de laudo médico (opcional no pipeline) e rodar Comprehend em cima.

**Tempo estimado:** 30 min

### PROMPT PARA O CLAUDE CODE

```
Vamos executar S5 — Nó DOC.

Crie src/nodes/doc_node.py com:

    def upload_pdf_to_s3(pdf_path: Path, bucket: str, key: str) -> str
    def extract_text_with_textract(bucket: str, key: str) -> dict
    def analyze_doc(pdf_path: Path) -> DocAnalysis

Implementação:

1. upload_pdf_to_s3:
   - Usa boto3 s3.upload_file
   - Bucket name vem de variável de ambiente AWS_S3_BUCKET (adicionar no .env.example)
   - Retorna o URI s3://...

2. extract_text_with_textract:
   - Cliente boto3 'textract'
   - Chama analyze_document com Document={'S3Object': {...}} e FeatureTypes=['TABLES', 'FORMS']
   - Processa os Blocks: extrai LINEs em texto corrido, TABELAS estruturadas, FORMS como key-value

3. analyze_doc:
   - Faz upload do PDF
   - Extrai texto com Textract
   - Roda analyze_nlp do nlp_node sobre o texto extraído
   - Retorna DocAnalysis

Crie scripts/04_test_doc_node.py que:
- Requer um PDF de teste em data/input/laudo_exemplo.pdf
- SE não existir, GERA um automaticamente usando reportlab (simule um laudo ginecológico fictício com dados como "Paciente: Maria Silva, idade 32 anos, queixa: dor pélvica recorrente, exame físico: ..." — use a Aula 2 do bloco AWS como referência)
- Chama analyze_doc
- Salva data/intermediate/doc_analysis.json
- Imprime resumo no terminal

Antes de codar, me confirme o nome do bucket S3 que vamos criar. Sugestão: "techchallenge-fase4-diego-369026" (adicione AWS_S3_BUCKET ao .env.example).

Faça commit: "S5: implementacao do no DOC com AWS Textract"
```

### Validação pós-sessão
- [ ] PDF foi gerado (se não existia) e enviado ao S3
- [ ] Textract retornou texto coerente
- [ ] `data/intermediate/doc_analysis.json` tem extracted_text + nlp_summary

---

## S6 — Nó FUSION (GPT-4)

**Objetivo:** receber os JSONs dos 3 nós anteriores e produzir o relatório clínico + decisão de alerta.

**Tempo estimado:** 1h

### PROMPT PARA O CLAUDE CODE

```
Vamos executar S6 — Nó FUSION.

Crie src/nodes/fusion_node.py com:

    def fuse_multimodal(
        video: VideoAnalysis,
        audio: AudioAnalysis,
        nlp: NLPAnalysis,
        doc: Optional[DocAnalysis] = None,
    ) -> FusionResult

Implementação:

1. Monta um prompt para o GPT-4 (modelo do .env, fallback gpt-4o-mini) que:
   - System message: "Você é um assistente clínico especializado em saúde da mulher, particularmente em triagem de violência doméstica e bem-estar psicológico pós-parto. Sua função é analisar dados multimodais de uma consulta e produzir um relatório clínico estruturado. VOCÊ NÃO SUBSTITUI AVALIAÇÃO MÉDICA — você apenas sinaliza padrões para que a equipe especializada decida intervenções."
   - User message: JSON estruturado com:
     - emotion_distribution e dominant_emotion_overall do vídeo
     - defensive_posture_ratio
     - transcricao_completa (audio.transcription_full)
     - aws_sentiment, key_phrases, entities, custom_risk_classification do NLP
     - resumo do laudo (se houver)
   - Pede uma resposta em JSON estrito:
     {
       "risk_level": "ALTO|MEDIO|BAIXO|INDETERMINADO",
       "risk_categories": ["violencia_domestica", "depressao_pos_parto", "ansiedade", ...],
       "confidence_overall": 0.0-1.0,
       "clinical_summary": "texto narrativo de 150-300 palavras",
       "recommended_actions": ["acao1", "acao2", ...],
       "alert_triggered": true|false,
       "reasoning": "explicacao do raciocinio em 2-3 paragrafos"
     }

2. Use response_format={"type": "json_object"} na chamada à API

3. Faz o parse do JSON e retorna FusionResult

4. Logging detalhado: prompt enviado (sem dados sensiveis em log), tokens consumidos, custo estimado

Crie scripts/05_test_fusion_node.py que:
- Lê os 3 JSONs anteriores (video, audio→nlp, doc opcional)
- Chama fuse_multimodal
- Salva data/intermediate/fusion_result.json
- Imprime o clinical_summary e o risk_level no terminal

Boas praticas:
- temperature=0.2 (queremos consistencia, nao criatividade)
- max_tokens razoavel (~1500)
- Tratamento de erro se OpenAI retornar erro de API

Faça commit: "S6: implementacao do no FUSION com GPT-4"
```

### Validação pós-sessão
- [ ] `data/intermediate/fusion_result.json` tem todos os campos
- [ ] `clinical_summary` é coerente com o conteúdo do vídeo
- [ ] `risk_level` faz sentido dado o sentimento e classificação

---

## S7 — Grafo LangGraph + CLI principal

**Objetivo:** amarrar os 5 nós em um StateGraph executável de ponta a ponta via CLI.

**Tempo estimado:** 1-1.5h

### PROMPT PARA O CLAUDE CODE

```
Vamos executar S7 — orquestracao com LangGraph.

Crie src/graph.py com:

1. Wrappers de cada nó que recebem o State e retornam um delta:

    def video_node_fn(state: PipelineState) -> PipelineState:
        result = analyze_video(Path(state["video_path"]), Path(state["video_output_path"]))
        return {"video_analysis": result}

    def audio_node_fn(state: PipelineState) -> PipelineState:
        audio_path = state.get("audio_path") or extract_audio(Path(state["video_path"]), Path(...))
        result = transcribe_audio(audio_path)
        return {"audio_path": str(audio_path), "audio_analysis": result}

    def nlp_node_fn(state: PipelineState) -> PipelineState:
        text = state["audio_analysis"].transcription_full
        result = analyze_nlp(text)
        return {"nlp_analysis": result}

    def doc_node_fn(state: PipelineState) -> PipelineState:
        if not state.get("pdf_path"):
            return {}  # noop se nao houver PDF
        result = analyze_doc(Path(state["pdf_path"]))
        return {"doc_analysis": result}

    def fusion_node_fn(state: PipelineState) -> PipelineState:
        result = fuse_multimodal(
            state["video_analysis"],
            state["audio_analysis"],
            state["nlp_analysis"],
            state.get("doc_analysis"),
        )
        return {"fusion": result}

2. Construa o grafo:

    grafo = StateGraph(PipelineState)
    grafo.add_node("video", video_node_fn)
    grafo.add_node("audio", audio_node_fn)
    grafo.add_node("nlp", nlp_node_fn)
    grafo.add_node("doc", doc_node_fn)
    grafo.add_node("fusion", fusion_node_fn)
    
    grafo.set_entry_point("video")
    # video roda primeiro, depois em paralelo audio e doc
    grafo.add_edge("video", "audio")
    grafo.add_edge("audio", "nlp")
    grafo.add_edge("video", "doc")  # paralelo
    
    # fusion espera nlp E doc
    grafo.add_edge("nlp", "fusion")
    grafo.add_edge("doc", "fusion")
    grafo.add_edge("fusion", END)
    
    app = grafo.compile()

3. Salve o diagrama ASCII do grafo em docs/arquitetura.txt usando app.get_graph().draw_ascii()

4. Crie scripts/run_pipeline.py — CLI principal com argparse:

    python scripts/run_pipeline.py --video data/input/video_consulta.mp4 [--pdf data/input/laudo.pdf]

    Executa o grafo completo, salva todos os intermediarios em data/intermediate/, e produz:
    - data/output/video_anotado.mp4
    - data/output/relatorio.json (toda a State serializada)
    - data/output/alerta.txt (resumo legivel humano)

5. Crie src/report.py com gerar_relatorio_pdf(state: PipelineState, output_path: Path) usando reportlab — produz um PDF clinico bem formatado com:
   - Capa (titulo, data, autor)
   - Resumo executivo (FusionResult.clinical_summary, risk_level, alert_triggered)
   - Secao Video (estatisticas de emocao, postura)
   - Secao Audio (transcricao completa)
   - Secao NLP (sentimento, entidades, classificacao)
   - Secao Doc (se houver)
   - Disclaimer ("este relatorio NAO substitui avaliacao medica...")

6. Adicione chamada a gerar_relatorio_pdf no final do run_pipeline.py

Faça commit: "S7: orquestracao LangGraph + CLI run_pipeline + relatorio PDF"
```

### Validação pós-sessão
- [ ] `python scripts/run_pipeline.py --video data/input/video_consulta.mp4` roda end-to-end
- [ ] `data/output/relatorio.pdf` é gerado e abre corretamente
- [ ] `docs/arquitetura.txt` mostra o grafo ASCII

---

## S8 — Vídeo de demonstração

**Objetivo:** gravar vídeo de até 15 min mostrando o pipeline funcionando, conforme entregável obrigatório do Tech Challenge.

**Tempo estimado:** 2-3h (gravação + edição)

### Roteiro sugerido (você grava com OBS, edita no DaVinci Resolve)

- **00:00–01:30** Apresentação pessoal + apresentação do problema (saúde da mulher, multimodalidade)
- **01:30–03:00** Visão geral da arquitetura (mostra `docs/arquitetura.txt` no terminal ou um slide com o diagrama)
- **03:00–04:30** Demonstração do nó VIDEO (`scripts/01_test_video_node.py` rodando, mostrar o vídeo anotado)
- **04:30–06:00** Demonstração do nó AUDIO + NLP (`scripts/02_test_audio_node.py` + `scripts/03_test_nlp_node.py`)
- **06:00–07:30** Demonstração do nó DOC com Textract (`scripts/04_test_doc_node.py`)
- **07:30–09:00** Demonstração do nó FUSION com GPT-4 (`scripts/05_test_fusion_node.py`)
- **09:00–11:00** Pipeline completo via LangGraph (`scripts/run_pipeline.py`) + abrir o `relatorio.pdf`
- **11:00–13:00** Como o sistema dispararia alerta numa situação real (mostrar o `alerta.txt` e o `alert_triggered: true`)
- **13:00–14:30** Discussão de limitações éticas e privacidade
- **14:30–15:00** Conclusão e próximos passos

Upload no YouTube como **não listado**, anexar link no Git README.md.

### PROMPT PARA O CLAUDE CODE (preparação de demo)

```
Vamos executar S8 — preparacao do video de demonstracao.

Crie scripts/demo.py que roda toda a pipeline com prints bonitos no terminal usando rich (instale rich):

- Header com nome do projeto e autor
- Cada nó tem uma section com tempo de execucao
- Mostra "loading spinners" durante operacoes longas
- Ao final, mostra o relatorio clinico formatado em um Panel rich

Este script vai ser a base da minha gravação. Quero que pareça uma demo profissional, não um output de log cru.

Adicione tambem um docs/roteiro_video.md com o roteiro sugerido (00:00–01:30 apresentação etc).

Faça commit: "S8: script demo.py + roteiro video"
```

---

## S9 — Relatório técnico final

**Objetivo:** redigir o RELATORIO_TECNICO.md exigido pelo Tech Challenge.

**Tempo estimado:** 1.5-2h

### PROMPT PARA O CLAUDE CODE

```
Vamos executar S9 — relatorio tecnico final.

Crie RELATORIO_TECNICO.md cobrindo o que o enunciado exige (PROJECT_SPEC.md secao 7 do CLAUDE.md tem o mapeamento):

1. Capa (titulo, autor, RM, data)
2. Resumo executivo (1 paragrafo)
3. Descricao do desafio
4. Escopo escolhido (funcionalidades e objetivos selecionados, com justificativas)
5. Arquitetura da solucao
   - Diagrama do grafo LangGraph (texto/ASCII de docs/arquitetura.txt)
   - Descricao de cada nó com tecnologia, entrada, saida
6. Fluxo multimodal (como os dados fluem)
7. Modelos aplicados em cada tipo de dado (tabela exatamente como na secao 7 do CLAUDE.md)
8. Justificativas tecnicas das substituicoes:
   - Por que Whisper local em vez de SpeechRecognition (qualidade, privacidade, offline)
   - Por que GPT-4 em vez de Transformers pipeline (raciocinio multimodal, sintese estruturada)
   - Por que YOLOv8 pre-treinado em vez de fine-tuning (escopo de tempo + COCO ja cobre objetos relevantes)
   - Por que AWS em vez de Azure (curso ensinou AWS)
9. Resultados obtidos
   - Estatisticas do video processado (durar, frames, emocao dominante, etc) — puxar de fusion_result.json
   - Exemplos concretos de anomalias detectadas
10. Limitacoes
    - Sem sinais vitais
    - Sem real-time
    - Heuristica de risco baseada em keywords (nao validada clinicamente)
11. Consideracoes eticas e LGPD
12. Conclusao e proximos passos

Use linguagem academica formal mas direta. Inclua trechos de codigo apenas onde for didatico. Imagens podem ser referenciadas mas nao geradas agora.

Atualize tambem README.md para ficar publicavel: instalacao, uso, link do video YouTube (placeholder), arquitetura, autor.

Faça commit: "S9: relatorio tecnico final + README publicavel"
```

---

## Checklist final de entrega

- [ ] Repositório Git público (GitHub) com todos os commits
- [ ] `RELATORIO_TECNICO.md` completo
- [ ] `README.md` claro com instruções de uso
- [ ] Vídeo no YouTube (não listado) com link no README
- [ ] Pipeline executável end-to-end com um comando
- [ ] `data/output/` populado com saídas reais de uma execução
- [ ] Diagrama da arquitetura em `docs/`

---

## Estimativa total de tempo

| Sessão | Tempo |
|---|---|
| S1 Setup | 30min |
| S2 VIDEO | 1.5h |
| S3 AUDIO | 45min |
| S4 NLP | 1h |
| S5 DOC | 30min |
| S6 FUSION | 1h |
| S7 LangGraph + CLI | 1.5h |
| S8 Vídeo demo | 2.5h |
| S9 Relatório | 1.5h |
| **Total** | **~11h** |

Espalhe em 1-2 semanas — não tente fazer tudo num dia, há muita troca de contexto entre as sessões.
