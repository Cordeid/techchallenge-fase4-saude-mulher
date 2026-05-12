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
