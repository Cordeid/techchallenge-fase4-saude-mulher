from pathlib import Path
import os
from dotenv import load_dotenv
import json

load_dotenv()

# =========================
# CONFIGURAÇÕES
# =========================

DATA_RAW = Path(os.getenv("DATA_RAW"))
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED"))

VIDEO_EXTENSIONS = [".mp4"]
PDF_EXTENSIONS = [".pdf"]

# =========================
# FUNÇÕES
# =========================

def encontrar_arquivo(caminho: Path, extensoes: list[str]):
    for arquivo in caminho.iterdir():
        if arquivo.suffix.lower() in extensoes:
            return arquivo
    return None


def criar_estrutura_processada(caso_id: str):

    pasta_caso = DATA_PROCESSED / caso_id

    subpastas = [
        "analise_prontuario",
        "analise_comunicacao",
        "analise_emocional_audio",
        "analise_video",
        "data_audio",
        "data_frames",
        "relatorio_final"
    ]

    pasta_caso.mkdir(parents=True, exist_ok=True)

    for sub in subpastas:
        (pasta_caso / sub).mkdir(exist_ok=True)

    return pasta_caso


def gerar_metadata(
    caso_id: str,
    video,
    pdf
):

    metadata = {
        "caso": caso_id,
        "video_encontrado": video is not None,
        "pdf_encontrado": pdf is not None,
        "video_nome": video.name if video else None,
        "pdf_nome": pdf.name if pdf else None,
    }

    return metadata


def salvar_metadata(
    pasta_processada: Path,
    metadata: dict
):

    caminho = pasta_processada / "metadata.json"

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )


def salvar_resumo(
    pasta_processada: Path,
    metadata: dict
):

    resumo = f"""
Caso: {metadata['caso']}

Arquivos encontrados:
- Vídeo: {metadata['video_nome']}
- PDF: {metadata['pdf_nome']}

Status:
- Caso pronto para processamento
"""

    caminho = pasta_processada / "resumo.txt"

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(resumo)


# =========================
# LISTAR CASOS
# =========================

def listar_casos():

    casos = [
        pasta for pasta in DATA_RAW.iterdir()
        if pasta.is_dir()
    ]

    print(f"\n✔ Casos encontrados: {len(casos)}")

    return casos


# =========================
# PREPARAR CASO
# =========================

def preparar_caso(caso: Path):

    caso_id = caso.name

    print("\n======================")
    print(f"Preparando: {caso_id}")

    video = encontrar_arquivo(caso, VIDEO_EXTENSIONS)

    pdf = encontrar_arquivo(caso, PDF_EXTENSIONS)

    if not video:
        print("❌ Vídeo não encontrado")
        return None

    pasta_processada = criar_estrutura_processada(caso_id)

    metadata = gerar_metadata(
        caso_id,
        video,
        pdf
    )

    salvar_metadata(pasta_processada, metadata)

    salvar_resumo(pasta_processada, metadata)

    print("✔ Estrutura criada")
    print("✔ Metadata salva")
    print("✔ Resumo salvo")

    return {
        "caso_id": caso_id,
        "raw_path": caso,
        "processed_path": pasta_processada,
        "video": video,
        "pdf": pdf,
        "metadata": metadata
    }


# =========================
# PROCESSAR TODOS
# =========================

def processar_casos():
    """
    Responsável pela etapa de ingestão e preparação dos casos clínicos. 
    O script identifica automaticamente os arquivos disponíveis (vídeo e prontuário PDF), valida a estrutura mínima necessária para processamento e cria a organização padronizada de diretórios do caso no ambiente processado. 
    Também gera arquivos de metadata e um resumo inicial do caso, permitindo rastreabilidade e padronização do pipeline multimodal. 
    Atua como ponto de entrada do Sistema Hórus, estruturando os dados que serão utilizados pelos agentes especializados de prontuário, comunicação, análise vocal, vídeo e fusão multimodal.

    """
    casos = listar_casos()
    casos_processados = []

    for caso in casos:
        resultado = preparar_caso(caso)
        if resultado:
            casos_processados.append(resultado)

    print("\n✔ Ingestão concluída")

    return casos_processados


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    processar_casos()
