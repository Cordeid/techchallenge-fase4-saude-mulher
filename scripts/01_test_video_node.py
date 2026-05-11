"""
Script de teste para o nó VIDEO.
Executa analyze_video() no vídeo de entrada e salva JSON + vídeo anotado.

Uso:
    python scripts/01_test_video_node.py
    python scripts/01_test_video_node.py --sample-rate 30
"""

import argparse
import sys
from pathlib import Path

# Garante que o pacote src é encontrado independente de onde o script é chamado
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import INPUT_DIR, INTERMEDIATE_DIR, configurar_logging
from src.nodes.video_node import analyze_video

configurar_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa o nó VIDEO do pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        default=INPUT_DIR / "video_consulta.mp4",
        help="Caminho para o vídeo de entrada",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=15,
        help="Analisar 1 frame a cada N frames (padrão: 15)",
    )
    args = parser.parse_args()

    video_path: Path = args.video
    output_video_path: Path = INTERMEDIATE_DIR / "video_anotado.mp4"
    output_json_path: Path = INTERMEDIATE_DIR / "video_analysis.json"

    if not video_path.exists():
        print(f"ERRO: Vídeo não encontrado: {video_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Nó VIDEO — Análise de Consulta Clínica")
    print(f"{'='*60}")
    print(f"  Entrada : {video_path}")
    print(f"  Saída   : {output_video_path}")
    print(f"  JSON    : {output_json_path}")
    print(f"  Sample  : 1 frame a cada {args.sample_rate}")
    print(f"{'='*60}\n")

    resultado = analyze_video(video_path, output_video_path, sample_every_n_frames=args.sample_rate)

    # Salvar JSON
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(resultado.model_dump_json(indent=2))

    # Imprimir estatísticas
    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"  Duração total       : {resultado.duration_seconds:.1f}s")
    print(f"  FPS                 : {resultado.fps}")
    print(f"  Frames analisados   : {resultado.total_frames_analyzed}")
    print(f"  Emoção dominante    : {resultado.dominant_emotion_overall}")
    print(f"  Postura defensiva   : {resultado.defensive_posture_ratio:.1%} dos frames")
    print()
    print("  Distribuição de emoções:")
    for emocao, pct in sorted(resultado.emotion_distribution.items(), key=lambda x: -x[1]):
        barra = "█" * int(pct * 30)
        print(f"    {emocao:<12} {pct:>6.1%}  {barra}")
    print()
    print(f"  Eventos de emoção   : {len(resultado.emotion_events)}")
    print(f"  Eventos de postura  : {len(resultado.posture_events)}")
    print(f"  Eventos de objetos  : {len(resultado.object_events)}")

    # Resumo de objetos detectados
    todos_objetos: list[str] = []
    for ev in resultado.object_events:
        todos_objetos.extend(o["class"] for o in ev.detected_objects)
    if todos_objetos:
        from collections import Counter
        print("\n  Objetos detectados (total de ocorrências):")
        for cls, cnt in Counter(todos_objetos).most_common():
            print(f"    {cls:<15} {cnt}x")

    print(f"\n{'='*60}")
    print(f"  JSON salvo    : {output_json_path}")
    print(f"  Vídeo salvo   : {output_video_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
