"""
Nó AUDIO do pipeline: extrai áudio de um vídeo e transcreve com Whisper local.

Usa openai-whisper (pacote FOSS, roda na GPU via PyTorch), não a API da OpenAI.
Fallback automático de large-v3 para medium em caso de OOM.
"""

import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    INTERMEDIATE_DIR,
    WHISPER_CACHE_DIR,
    WHISPER_MODEL_SIZE,
    configurar_logging,
)
from src.schemas import AudioAnalysis

logger = logging.getLogger(__name__)


# ── Extração de áudio ─────────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extrai o áudio de um vídeo e salva como WAV mono 16kHz.

    O formato WAV mono 16kHz é o ideal para Whisper: corresponde ao sample rate
    interno do modelo e elimina redundância de canais.

    Args:
        video_path: Caminho para o vídeo de entrada.
        audio_path: Caminho de destino para o arquivo WAV.

    Returns:
        Path do arquivo WAV gerado.

    Raises:
        FileNotFoundError: Se o vídeo não existir.
        RuntimeError: Se o vídeo não contiver trilha de áudio.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Extraindo áudio de: %s", video_path)

    from moviepy import VideoFileClip  # import tardio para não penalizar inicialização

    clip = VideoFileClip(str(video_path))
    try:
        if clip.audio is None:
            raise RuntimeError(f"O vídeo não contém trilha de áudio: {video_path}")

        clip.audio.write_audiofile(
            str(audio_path),
            fps=16000,      # sample rate ideal para Whisper
            nbytes=2,       # 16-bit PCM
            ffmpeg_params=["-ac", "1"],  # forçar mono
            logger=None,    # suprimir output do ffmpeg
        )
    finally:
        clip.close()

    logger.info("Áudio extraído: %s (%.1f MB)", audio_path, audio_path.stat().st_size / 1e6)
    return audio_path


# ── Transcrição com Whisper ───────────────────────────────────────────────────

def _carregar_audio_numpy(audio_path: Path) -> np.ndarray:
    """Carrega WAV como array float32 sem depender de ffmpeg no PATH.

    Whisper aceita numpy array em model.transcribe(), evitando a chamada
    de subprocesso ffmpeg que falha quando o executável não está no PATH do sistema.
    Assume WAV mono 16kHz (produzido por extract_audio).
    """
    with wave.open(str(audio_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        n_frames   = wf.getnframes()
        raw        = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= float(np.iinfo(dtype).max)  # normaliza para [-1, 1]

    if n_channels > 1:  # garante mono
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio


def _carregar_whisper(model_size: str):
    """Carrega o modelo Whisper com cache local. Retorna (model, model_size_usado)."""
    import whisper

    WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Carregando Whisper '%s' (cache: %s)...", model_size, WHISPER_CACHE_DIR)
    try:
        model = whisper.load_model(model_size, download_root=str(WHISPER_CACHE_DIR))
        logger.info("Modelo Whisper '%s' carregado", model_size)
        return model, model_size
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
            fallback = "medium"
            logger.warning(
                "OOM ao carregar '%s' — fazendo fallback para '%s'", model_size, fallback
            )
            model = whisper.load_model(fallback, download_root=str(WHISPER_CACHE_DIR))
            logger.info("Modelo Whisper '%s' carregado (fallback)", fallback)
            return model, fallback
        raise


def transcribe_audio(
    audio_path: Path,
    model_size: Optional[str] = None,
) -> AudioAnalysis:
    """Transcreve um arquivo de áudio em português usando Whisper local.

    Args:
        audio_path: Caminho para o arquivo WAV (idealmente mono 16kHz).
        model_size: Tamanho do modelo Whisper. Se None, usa WHISPER_MODEL_SIZE
            do .env (default large-v3). Fallback automático para medium em OOM.

    Returns:
        AudioAnalysis com transcrição completa, segmentos e metadados.

    Raises:
        FileNotFoundError: Se o arquivo de áudio não existir.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    tamanho_solicitado = model_size or WHISPER_MODEL_SIZE
    model, tamanho_usado = _carregar_whisper(tamanho_solicitado)

    logger.info("Transcrevendo: %s (modelo: %s)", audio_path, tamanho_usado)

    # Carrega como numpy array — evita chamada de subprocesso ffmpeg que falha
    # quando o executável não está no PATH do sistema (caso comum no Windows com venv)
    audio_array = _carregar_audio_numpy(audio_path)
    logger.debug("Áudio carregado: %d amostras (%.1fs)", len(audio_array), len(audio_array) / 16000)

    import whisper as _w
    result = model.transcribe(
        audio_array,
        language="pt",
        task="transcribe",
        verbose=False,
        fp16=True,   # FP16 na GPU para economizar VRAM; ignorado em CPU
    )

    transcription_full: str = result["text"].strip()
    segments_raw: list = result.get("segments", [])

    transcription_segments = [
        {
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "text":  seg["text"].strip(),
        }
        for seg in segments_raw
    ]

    # Duração: último end dos segmentos ou calculada pelo tamanho do array
    if transcription_segments:
        duration_seconds = transcription_segments[-1]["end"]
    else:
        duration_seconds = round(len(audio_array) / _w.audio.SAMPLE_RATE, 2)

    logger.info(
        "Transcrição concluída | %.1fs | %d segmentos | %d caracteres",
        duration_seconds, len(transcription_segments), len(transcription_full),
    )

    return AudioAnalysis(
        duration_seconds=round(duration_seconds, 2),
        transcription_full=transcription_full,
        transcription_segments=transcription_segments,
        language_detected="pt",
    )
