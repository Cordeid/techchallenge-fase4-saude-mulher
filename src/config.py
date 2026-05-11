"""
Configuração central do projeto: carrega variáveis de ambiente, define paths e
configura o sistema de logging. Deve ser importado antes de qualquer outra parte
do projeto.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Raiz do projeto = diretório pai deste arquivo (src/)
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

# Carrega .env da raiz antes de qualquer acesso às variáveis de ambiente
load_dotenv(PROJECT_ROOT / ".env")

# === Diretórios principais ===
DATA_DIR: Path = PROJECT_ROOT / "data"
INPUT_DIR: Path = DATA_DIR / "input"
INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"
OUTPUT_DIR: Path = DATA_DIR / "output"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Garante que os diretórios de runtime existem (dados nunca são commitados)
for _dir in (INPUT_DIR, INTERMEDIATE_DIR, OUTPUT_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# === Caminhos de arquivo padrão ===
DEFAULT_VIDEO_PATH: Path = INPUT_DIR / "video_consulta.mp4"
DEFAULT_PDF_PATH: Path = INPUT_DIR / "laudo_exemplo.pdf"
DEFAULT_AUDIO_PATH: Path = INTERMEDIATE_DIR / "audio_consulta.wav"
DEFAULT_ANNOTATED_VIDEO_PATH: Path = INTERMEDIATE_DIR / "video_anotado.mp4"
OUTPUT_VIDEO_PATH: Path = OUTPUT_DIR / "video_anotado.mp4"
OUTPUT_REPORT_PDF: Path = OUTPUT_DIR / "relatorio.pdf"
OUTPUT_REPORT_JSON: Path = OUTPUT_DIR / "relatorio.json"
OUTPUT_ALERT_TXT: Path = OUTPUT_DIR / "alerta.txt"

# === OpenAI ===
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# === AWS ===
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "techchallenge-fase4-diego-369026")

# === Whisper ===
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_CACHE_DIR: Path = Path(os.getenv("WHISPER_CACHE_DIR", str(CACHE_DIR / "whisper")))

# === Caches de modelos (definir ANTES de importar deepface / HuggingFace) ===
HF_HOME: str = os.getenv("HF_HOME", str(CACHE_DIR / "huggingface"))
DEEPFACE_HOME: str = os.getenv("DEEPFACE_HOME", str(CACHE_DIR / "deepface"))

# Propaga para variáveis de ambiente para que as bibliotecas as leiam
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("DEEPFACE_HOME", DEEPFACE_HOME)

# === Logging ===
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configurar_logging(nivel: int = logging.INFO, arquivo_log: Path | None = None) -> None:
    """Configura o logging global do projeto.

    Args:
        nivel: Nível de log para o console (padrão INFO).
        arquivo_log: Se fornecido, escreve DEBUG neste arquivo além do console.
    """
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
    ]
    if arquivo_log is not None:
        arquivo_log.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(arquivo_log, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level=nivel,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Configuração padrão ao importar o módulo
configurar_logging()
