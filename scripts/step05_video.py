from pathlib import Path
import cv2
import json
import os
from dotenv import load_dotenv

from ultralytics import YOLO
from deepface import DeepFace
from openai import OpenAI

load_dotenv()

# =========================
# CONFIGURAÇÕES
# =========================

DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent

PROTOCOLO_PATH = BASE_DIR / "protocolos" / "protocolo_postura_comunicacao_naoverbal.md"

def carregar_protocolo():
    with open(PROTOCOLO_PATH, "r", encoding="utf-8") as f:
        return f.read()

# =========================
# YOLOv8
# =========================

print("Carregando YOLOv8...")
model = YOLO("yolov8n.pt")

# =========================
# EXTRAÇÃO DE FRAMES
# =========================

def extrair_frames(video_path, output_dir, intervalo_segundos=3):

    print("Extraindo frames...")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * intervalo_segundos)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = (output_dir / f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(str(frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"✔ {saved_count} frames extraídos")


# =========================
# YOLO
# =========================

def detectar_pessoas(frame_path):
    results = model(str(frame_path))
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])
            detections.append({
                "label": label,
                "confidence": confidence
            })

    return detections


# =========================
# EMOÇÕES
# =========================

def analisar_emocoes(frame_path):

    try:
        result = DeepFace.analyze(
            img_path=str(frame_path),
            actions=['emotion'],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        return result["dominant_emotion"]

    except Exception as e:
        print(f"Erro emoção: {e}")

        return "desconhecido"


# =========================
# GPT
# =========================

def analise_video(analises, protocolo):

    prompt = f"""
Você é um auditor especializado em:

- comportamento humano;
- comunicação não verbal;
- postura corporal;
- análise emocional visual;
- humanização clínica;
- saúde da mulher.

Sua tarefa é avaliar criticamente
a comunicação não verbal da consulta.

Você DEVE utilizar o protocolo abaixo
como referência oficial.

==================================
PROTOCOLO OFICIAL
==================================

{protocolo}

==================================
INSTRUÇÕES
==================================

Avalie criticamente:

- postura corporal;
- sinais de acolhimento;
- distanciamento emocional;
- tensão visual;
- retraimento;
- expressões faciais;
- interação visual;
- segurança emocional;
- humanização não verbal;
- possível frieza relacional;
- sinais de desconforto emocional;
- ambiente emocional da consulta.

Considere NEGATIVO quando houver:

- postura mecanizada;
- tensão persistente;
- expressões negativas recorrentes;
- ausência de acolhimento visual;
- ambiente frio;
- distanciamento relacional;
- baixa conexão humana.

IMPORTANTE:

- Não faça diagnóstico psicológico.
- Não confirme violência.
- Não invente informações.
- Baseie-se SOMENTE nos dados visuais.
- Seja técnico e conservador.

==================================
SCORE
==================================

0-20 → muito ruim
21-40 → ruim
41-60 → regular
61-80 → adequado
81-100 → excelente

==================================
FORMATO
==================================

Retorne APENAS JSON válido.

{{
    "score_video": 0,
    "indicadores_positivos": [],
    "indicadores_negativos": [],
    "analise_video": ""
}}

==================================
DADOS VISUAIS
==================================

{json.dumps(analises, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um auditor clínico "
                    "especializado em comunicação "
                    "não verbal e humanização."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    resposta = response.choices[0].message.content

    resposta = (
        resposta
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        return json.loads(resposta)

    except Exception:

        return {
            "erro": "JSON inválido",
            "resposta_bruta": resposta
        }


# =========================
# PROCESSAMENTO
# =========================
def salvar_json(dados, caminho):

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dados,
            f,
            ensure_ascii=False,
            indent=4
        )

def processar_caso(caso_info):
    """
    Responsável pela análise visual e comportamental da consulta.
    O script extrai frames do vídeo, realiza detecção de pessoas com YOLOv8 e análise emocional facial com DeepFace, produzindo sinais visuais relacionados à postura corporal, expressões faciais e comunicação não verbal.
    Os dados extraídos são interpretados por LLM com base em um protocolo especializado de humanização e postura clínica, gerando score visual, indicadores positivos e negativos e auditoria crítica do ambiente relacional da consulta.
    """

    caso_id = caso_info["caso_id"]
    video = caso_info["video"]
    pasta_processada = caso_info["processed_path"]

    protocolo = carregar_protocolo()

    print("\n======================")
    print(f"Processando vídeo: {caso_id}")

    # =========================
    # PASTAS
    # =========================
    pasta_frames = (pasta_processada / "data_frames")
    pasta_analise = (pasta_processada / "analise_video")
    pasta_analise.mkdir(parents=True,exist_ok=True)

    # =========================
    # EXTRAÇÃO FRAMES
    # =========================
    extrair_frames(video, pasta_frames)
    analises = []
    frames = list(pasta_frames.glob("*.jpg"))

    # =========================
    # ANÁLISE FRAMES
    # =========================
    for frame in frames:
        print(f"Analisando {frame.name}")

        detections = detectar_pessoas(frame)
        emocao = analisar_emocoes(frame)

        analise = {
            "frame": frame.name,
            "detections": detections,
            "dominant_emotion": emocao
        }

        analises.append(analise)

    # =========================
    # SALVAR JSON
    # =========================
    deteccoes_path = (pasta_analise / "deteccoes.json")

    with open(deteccoes_path, "w", encoding="utf-8") as f:
        json.dump(
            analises,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("✔ JSON salvo")

    # =========================
    # ANALISE IA
    # =========================
    analise = analise_video(analises, protocolo)
    analise_path = (pasta_analise /"analise_video.json")
    salvar_json(analise, analise_path)
