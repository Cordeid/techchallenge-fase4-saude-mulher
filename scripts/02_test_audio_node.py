"""
Script de teste para o nó AUDIO.
Extrai áudio do vídeo de consulta e transcreve com Whisper local.

Uso:
    python scripts/02_test_audio_node.py
    python scripts/02_test_audio_node.py --model medium
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import INPUT_DIR, INTERMEDIATE_DIR, configurar_logging
from src.nodes.audio_node import extract_audio, transcribe_audio

configurar_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa o nó AUDIO do pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        default=INPUT_DIR / "video_consulta.mp4",
        help="Caminho para o vídeo de entrada",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Tamanho do modelo Whisper (sobrescreve .env). Ex: large-v3, medium",
    )
    args = parser.parse_args()

    video_path:  Path = args.video
    audio_path:  Path = INTERMEDIATE_DIR / "audio_consulta.wav"
    output_json: Path = INTERMEDIATE_DIR / "audio_analysis.json"

    if not video_path.exists():
        print(f"ERRO: Vídeo não encontrado: {video_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Nó AUDIO — Extração e Transcrição")
    print(f"{'='*60}")
    print(f"  Entrada : {video_path}")
    print(f"  WAV     : {audio_path}")
    print(f"  JSON    : {output_json}")
    print(f"{'='*60}\n")

    # Extração do áudio
    print("[ 1/2 ] Extraindo áudio...")
    extract_audio(video_path, audio_path)
    print(f"        Salvo: {audio_path} ({audio_path.stat().st_size / 1e6:.1f} MB)\n")

    # Transcrição
    print("[ 2/2 ] Transcrevendo com Whisper...")
    resultado = transcribe_audio(audio_path, model_size=args.model)

    # Salvar JSON
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        f.write(resultado.model_dump_json(indent=2))

    # Imprimir resultados
    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"  Duração       : {resultado.duration_seconds:.1f}s")
    print(f"  Idioma        : {resultado.language_detected}")
    print(f"  Segmentos     : {len(resultado.transcription_segments)}")
    print(f"  Caracteres    : {len(resultado.transcription_full)}")
    print()

    # Primeiros 5 segmentos
    print("  Primeiros segmentos:")
    for seg in resultado.transcription_segments[:5]:
        print(f"    [{seg['start']:6.1f}s -> {seg['end']:6.1f}s]  {seg['text']}")
    if len(resultado.transcription_segments) > 5:
        print(f"    ... ({len(resultado.transcription_segments) - 5} segmentos restantes)")

    print()
    print("  Transcrição completa:")
    print("  " + "-" * 56)
    # Imprime em blocos de ~80 chars para facilitar leitura
    texto = resultado.transcription_full
    largura = 76
    for i in range(0, len(texto), largura):
        print(f"  {texto[i:i+largura]}")

    print(f"\n{'='*60}")
    print(f"  JSON salvo : {output_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
