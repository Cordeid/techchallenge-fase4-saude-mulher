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

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

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

def gerar_resumo_visual(analises):

    prompt = f"""
Você é um auditor especializado em:

- comportamento humano;
- comunicação não verbal;
- análise emocional visual;
- humanização de consultas médicas;
- desconforto psicológico em saúde da mulher.

Sua tarefa é analisar criticamente os dados visuais gerados pela IA.

==================================
CRITÉRIOS DE ANÁLISE
==================================

Avalie:

- emoções predominantes;
- sinais de ansiedade;
- medo;
- tristeza;
- desconforto emocional;
- tensão;
- retraimento;
- insegurança;
- expressões faciais negativas;
- interação entre pessoas;
- possível distanciamento emocional;
- consulta mecanizada;
- baixa humanização;
- ausência de acolhimento;
- sinais de pressão psicológica;
- comunicação não verbal negativa;
- ambiente emocionalmente desconfortável.

Também avalie:

- coerência emocional geral da consulta;
- predominância de emoções negativas;
- postura comportamental do ambiente;
- possíveis sinais de frieza relacional;
- ausência de conexão humana.

Considere NEGATIVO quando houver:

- predominância de tristeza, medo ou raiva;
- expressões recorrentes de tensão;
- sinais de desconforto persistente;
- interação fria;
- ausência de sinais de acolhimento;
- comportamento mecanizado;
- ambiente emocionalmente distante;
- sinais repetidos de sofrimento emocional.

==================================
NÍVEL DE ATENÇÃO
==================================

Use:

- "baixo"
- "moderado"
- "elevado"

Considere "elevado" quando houver múltiplos sinais visuais relevantes.

==================================
SCORE
==================================

O campo "score_visual" deve ser um número inteiro de 0 a 100.

Use esta escala:

- 90-100:
  Ambiente visual acolhedor, seguro e humanizado.

- 75-89:
  Interação visual adequada com pequenas falhas.

- 60-74:
  Ambiente razoável, mas com desconfortos perceptíveis.

- 40-59:
  Ambiente visual frio, mecanizado ou emocionalmente inadequado.

- 0-39:
  Ambiente visual crítico, desconfortável ou potencialmente danoso.

IMPORTANTE:
- Nunca utilize escala de 0 a 10.
- Não atribua score alto apenas por ausência de agressividade.
- Consultas frias ou mecanizadas devem receber score reduzido.
- O score deve refletir criticamente a qualidade emocional do ambiente visual.

==================================
RESTRIÇÕES
==================================

- Não faça diagnóstico psicológico.
- Não confirme violência.
- Apenas identifique sinais de atenção.
- Seja técnico, crítico e cauteloso.

==================================
FORMATO DE SAÍDA
==================================

Retorne APENAS JSON válido.

Formato:

{{
    "emocao_predominante": "",
    "nivel_atencao": "",
    "avaliacao_humanizacao": "",
    "avaliacao_comportamental": "",
    "indicadores_visuais": [],
    "sinais_desconforto": [],
    "alertas": [],
    "pontos_positivos": [],
    "pontos_negativos": [],
    "score_visual": 0,
    "resumo_critico": ""
}}

==================================
DADOS VISUAIS
==================================

{json.dumps(analises, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
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
    caso_id = caso_info["caso_id"]
    video = caso_info["video"]
    pasta_processada = caso_info["processed_path"]

    print("\n======================")
    print(f"Processando vídeo: {caso_id}")

    # =========================
    # PASTAS
    # =========================
    pasta_frames = (pasta_processada / "frames")
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
    # RESUMO IA
    # =========================
    resumo = gerar_resumo_visual(analises)
    resumo_path = (pasta_analise /"resumo_video_ia.json")
    salvar_json(resumo, resumo_path)
