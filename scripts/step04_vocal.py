from pathlib import Path
import json
import os

import librosa
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent

PROTOCOLO_PATH = BASE_DIR / "protocolos" / "protocolo_comunicacao_vocal.md"


def carregar_protocolo():
    with open(PROTOCOLO_PATH, "r", encoding="utf-8") as f:
        return f.read()


def extrair_features_audio(audio_path):

    y, sr = librosa.load(str(audio_path), sr=16000)

    pitch = librosa.yin(y, fmin=50, fmax=300)

    energia = librosa.feature.rms(y=y)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    silences = librosa.effects.split(y, top_db=25)

    duracao_total = librosa.get_duration(y=y, sr=sr)

    tempo_falado = sum(
        (end - start)
        for start, end in silences
    ) / sr

    percentual_silencio = (
        1 - (tempo_falado / duracao_total)
    )

    return {
        "pitch_medio": round(float(np.mean(pitch)), 2),
        "pitch_variacao": round(float(np.std(pitch)), 2),
        "energia_media": round(float(np.mean(energia)), 4),
        "energia_variacao": round(float(np.std(energia)), 4),
        "ritmo_fala": round(float(tempo), 2),
        "percentual_silencio": round(float(percentual_silencio), 2),
        "duracao_audio": round(float(duracao_total), 2)
    }


def analise_vocal(features, protocolo):

    prompt = f"""
Você é um auditor especializado em:

- comportamento vocal;
- comunicação clínica;
- humanização;
- saúde da mulher;
- análise emocional de voz.

Utilize o protocolo abaixo como referência oficial.

==================================
PROTOCOLO
==================================

{protocolo}

==================================
FEATURES ACÚSTICAS
==================================

{json.dumps(features, ensure_ascii=False, indent=2)}

==================================
INSTRUÇÕES
==================================

Avalie:

- possível agressividade;
- tensão vocal;
- frieza emocional;
- acolhimento;
- empatia percebida;
- ritmo acelerado;
- estabilidade emocional;
- possíveis sinais de impaciência;
- segurança emocional transmitida.

IMPORTANTE:

- Não invente informações.
- Não faça diagnóstico psicológico.
- Não conclua violência.
- Seja conservador e técnico.
- Baseie-se apenas nos sinais acústicos.

==================================
FORMATO
==================================

Retorne APENAS JSON válido.

{{
    "score_vocal": 0,
    "sinais_positivos": [],
    "sinais_negativos": [],
    "analise_vocal": ""
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é especialista em "
                    "auditoria vocal clínica."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(
        response.choices[0].message.content
    )


def salvar_json(dados, caminho):

    caminho.parent.mkdir(parents=True,exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(
            dados,
            f,
            ensure_ascii=False,
            indent=4
        )


def processar_caso(caso_info):
    """
    Responsável pela análise emocional e comportamental da voz.
    O script utiliza processamento acústico para extrair features como pitch, energia, ritmo de fala e pausas, permitindo inferir possíveis sinais de tensão, agressividade, frieza emocional ou acolhimento vocal.
    As métricas acústicas são interpretadas por LLM com base em um protocolo especializado de comunicação vocal humanizada, gerando score emocional, análise crítica e indicadores positivos e negativos da comunicação não verbal sonora.
    """

    caso_id = caso_info["caso_id"]
    pasta_processada = (caso_info["processed_path"])

    print("\n======================")
    print(f"Analisando emoção vocal: {caso_id}")

    audio_path = pasta_processada/ "data_audio"/ "extracted_audio.wav"
    pasta_saida = pasta_processada / "analise_vocal"

    protocolo = carregar_protocolo()

    features = extrair_features_audio(audio_path)

    salvar_json(features, pasta_saida / "features_audio.json")

    resultado = analise_vocal(features, protocolo)

    salvar_json(resultado, pasta_saida / "analise_vocal.json")

    print("✔ Análise vocal concluída")