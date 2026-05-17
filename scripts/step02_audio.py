from pathlib import Path
import moviepy as mp
import json
import speech_recognition as sr
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv

import shutil
import os

load_dotenv()

# =========================
# CONFIGURAÇÕES
# =========================

DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED"))

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# EXTRAÇÃO DE ÁUDIO
# =========================

def extract_audio_from_video(
    video_path,
    audio_path
):
    print("Extraindo áudio...")
    video = mp.VideoFileClip(str(video_path))
    video.audio.write_audiofile(
        str(audio_path)
    )
    print("✔ Áudio extraído")


# =========================
# CHUNKS
# =========================
def split_audio(audio_path, chunks_dir, chunk_length_ms=60000):
    
    print("Dividindo áudio em chunks...")
    
    audio = AudioSegment.from_wav(str(audio_path))

    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = []

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_name = (chunks_dir / f"chunk_{i}.wav")
        chunk.export(str(chunk_name), format="wav")
        chunks.append(chunk_name)

    print(f"✔ {len(chunks)} chunks criados")

    return chunks


# =========================
# TRANSCRIÇÃO
# =========================

def transcribe_audio_chunks(chunks):

    recognizer = sr.Recognizer()
    full_text = ""

    print("Iniciando transcrição...")

    for chunk_file in chunks:
        with sr.AudioFile(str(chunk_file)) as source:
            audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data, language='pt-BR')
                print(f"[OK] {chunk_file.name}")
                full_text += text + " "

            except sr.UnknownValueError:
                print(f"[ERRO] Não entendeu {chunk_file.name}")

            except sr.RequestError as e:
                print(f"[ERRO API] {e}")

    print("✔ Transcrição concluída")

    return full_text


# =========================
# OPENAI
# =========================

def gerar_resumo(texto):

    print("Gerando resumo IA...")

    prompt = f"""
Você é um auditor clínico especializado em qualidade comunicacional e humanização de consultas médicas.

Analise criticamente a consulta abaixo.

Avalie:
- empatia;
- clareza;
- humanização;
- escuta ativa;
- adaptação ao perfil da paciente;
- excesso de termos técnicos;
- qualidade comunicacional;
- sinais de desconforto emocional.

IMPORTANTE:
- Não faça diagnóstico clínico.
- Não confirme violência.
- Seja técnico e crítico.
- Consultas frias ou excessivamente técnicas devem receber avaliação negativa.

Retorne APENAS JSON válido.

Formato:

{{
    "contexto_geral": "",
    "empatia": "",
    "linguagem_tecnica": "",
    "humanizacao": "",
    "escuta_ativa": "",
    "adaptacao_paciente": "",
    "pontos_positivos": [],
    "pontos_negativos": [],
    "alertas": [],
    "score_comunicacao": 0
}}

TRANSCRIÇÃO:

{texto}
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


def gerar_analise_emocional(texto):

    print("Gerando análise emocional...")

    prompt = f"""
Você é um auditor especializado em análise emocional e humanização de consultas médicas.

Analise criticamente:
- medo;
- ansiedade;
- insegurança;
- hesitação;
- desconforto psicológico;
- submissão;
- constrangimento;
- dificuldade de compreensão;
- sinais de intimidação;
- baixa empatia;
- excesso de tecnicismo;
- comunicação inadequada.

IMPORTANTE:
- Não faça diagnóstico psicológico.
- Não confirme violência.
- Apenas identifique sinais de atenção.

Retorne APENAS JSON válido.

Formato:

{{
    "emocoes_predominantes": [],
    "sinais_emocionais": [],
    "evidencias": [],
    "nivel_atencao": "",
    "avaliacao_comunicacao": "",
    "avaliacao_empatia": "",
    "fatores_desconforto": [],
    "alertas": [],
    "score_emocional": 0,
    "resumo_critico": ""
}}

TRANSCRIÇÃO:

{texto}
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
# UTIL
# =========================

def salvar_texto(texto, caminho):

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)

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

# =========================
# PROCESSAMENTO
# =========================

def processar_caso(caso_info):
    caso_id = caso_info["caso_id"]
    video = caso_info["video"]
    pasta_processada = caso_info["processed_path"]

    print("\n======================")
    print(f"Processando áudio: {caso_id}")

    # =========================
    # PASTAS
    # =========================
    pasta_audio = (pasta_processada / "audio")
    pasta_transcricao = (pasta_processada / "transcricao")
    pasta_audio.mkdir(parents=True, exist_ok=True)
    pasta_transcricao.mkdir(parents=True, exist_ok=True)
    audio_path = (pasta_audio / "extracted_audio.wav")
    chunks_dir = (pasta_audio / "chunks")

    try:
        extract_audio_from_video(video, audio_path)
        chunks = split_audio(audio_path, chunks_dir)
        
        texto = transcribe_audio_chunks(chunks)
        salvar_texto(texto, pasta_transcricao /"transcription.txt")

        print("✔ Transcrição salva")
        resumo = gerar_resumo(texto)
        salvar_json(resumo, pasta_transcricao /"resumo_ia.json")

        print("✔ Resumo salvo")

        analise = gerar_analise_emocional(texto)

        salvar_json(analise, pasta_transcricao / "analise_emocional.json")

        print("✔ Análise emocional salva")

    finally:
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
            print("🧹 Chunks removidos")