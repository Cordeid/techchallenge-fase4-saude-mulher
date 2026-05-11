"""
Schemas Pydantic e TypedDict do State LangGraph.
Estes modelos são o contrato entre todos os nós do pipeline.
"""

from typing import Annotated, List, Optional, TypedDict

from pydantic import BaseModel


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
    nlp_summary: Optional[NLPAnalysis] = None


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
    video_output_path: str  # caminho do vídeo anotado

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
