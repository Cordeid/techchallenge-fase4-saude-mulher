from pathlib import Path
import wave
import moviepy as mp
import json
from openai import OpenAI
import speech_recognition as sr
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

# =========================
# CONFIGURAÇÕES
# =========================

DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent
PROTOCOLO_PATH = BASE_DIR / "protocolos" / "protocolo_consulta_geral.md"

def carregar_protocolo():

    with open(PROTOCOLO_PATH, "r", encoding="utf-8") as f:
        return f.read()

# =========================
# EXTRAÇÃO DE ÁUDIO
# =========================

def extract_audio_from_video(video_path, audio_path):

    print("Extraindo áudio...")

    video = mp.VideoFileClip(str(video_path))

    try:
        video.audio.write_audiofile(
            str(audio_path),
            fps=16000,
            nbytes=2
        )

    finally:
        video.close()

    print("✔ Áudio extraído")



# =========================
# TRANSCRIÇÃO
# =========================

def split_audio(audio_path, chunks_dir, chunk_duration_sec=60):

    print("Dividindo áudio em chunks...")

    chunks_dir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(audio_path), "rb") as wav_file:

        params = wav_file.getparams()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        frames_per_chunk = frame_rate * chunk_duration_sec

        chunk_paths = []

        for i, start in enumerate(range(0, n_frames, frames_per_chunk)):

            wav_file.setpos(start)
            frames = wav_file.readframes(frames_per_chunk)

            chunk_path = chunks_dir / f"chunk_{i}.wav"

            with wave.open(str(chunk_path), "wb") as chunk_file:
                chunk_file.setparams(params)
                chunk_file.writeframes(frames)

            chunk_paths.append(chunk_path)

    print(f"✔ {len(chunk_paths)} chunks criados")

    return chunk_paths

def transcrever_audio_chunks(chunks):

    recognizer = sr.Recognizer()
    texto_completo = []

    print("Iniciando transcrição...")

    for i, chunk_path in enumerate(chunks):
        print(
            f"Transcrevendo chunk "
            f"{i+1}/{len(chunks)}"
        )

        with sr.AudioFile(str(chunk_path)) as source:

            audio_data = recognizer.record(source)

            try:
                texto = recognizer.recognize_google(audio_data, language="pt-BR")
                texto_completo.append(texto)

            except sr.UnknownValueError:
                print(
                    f"[ERRO] Não entendeu "
                    f"{chunk_path.name}"
                )

            except sr.RequestError as e:
                print(f"[ERRO API] {e}")

    print("✔ Transcrição concluída")

    return "\n".join(texto_completo)

# =========================
# RESUMO
# =========================

def gerar_resumo_caso(caso_info):
    
    pasta_processada = caso_info["processed_path"]

    transcricao_path = pasta_processada / "analise_comunicacao" / "transcription.txt"

    if not transcricao_path.exists():
        print("❌ Transcrição não encontrada para resumo")
        return
    
    with open(transcricao_path, "r", encoding="utf-8") as f:
        transcricao = f.read()

    prompt = f"""
    Você é um auditor clínico especializado em comunicação médica.
    Sua tarefa é gerar um resumo técnico da consulta com base na transcrição com, no máximo, 300 palavras.
    Transcrição:
    {transcricao}
    Resumo técnico:
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

    resumo = response.choices[0].message.content.strip()
    resumo_path = pasta_processada / "analise_comunicacao" / "resumo_caso.txt"
    
    with open(resumo_path, "w", encoding="utf-8") as f:
        f.write(resumo)

# =========================
# OPENAI
# =========================

def analisar_comunicacao(texto, protocolo):

    prompt = f"""
Você é um auditor clínico especializado em:

- comunicação médica;
- humanização;
- qualidade assistencial;
- saúde da mulher;
- escuta ativa;
- comunicação empática.

Sua tarefa é avaliar criticamente
a comunicação da consulta.

Você DEVE utilizar o protocolo abaixo
como referência oficial.

==================================
PROTOCOLO OFICIAL
==================================

{protocolo}

==================================
INSTRUÇÕES
==================================

Avalie:

- aderência ao protocolo;
- acolhimento;
- empatia;
- escuta ativa;
- validação emocional;
- clareza;
- linguagem acessível;
- excesso de tecnicismo;
- humanização;
- organização comunicacional;
- postura mecanizada;
- possíveis sinais de desconforto psicológico.

IMPORTANTE:

- Não faça diagnóstico clínico.
- Não confirme violência.
- Não invente informações.
- Baseie-se SOMENTE na transcrição.
- Seja técnico e crítico.

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
    "score_comunicacao": 0,
    "pontos_positivos": [],
    "pontos_negativos": [],
    "analise_comunicacao": "",
}}

==================================
TRANSCRIÇÃO
==================================

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
        ],
        response_format={"type": "json_object"}
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

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)

# =========================
# PROCESSAMENTO
# =========================

def processar_caso(caso_info):
    """
    Responsável pela análise comunicacional da consulta médica.
    O script extrai o áudio do vídeo, realiza transcrição automática da conversa e utiliza um protocolo de comunicação clínica humanizada para avaliar acolhimento, empatia, clareza, escuta ativa e sinais de desumanização.
    A análise é realizada por LLM, gerando score comunicacional, resumo técnico da consulta e auditoria crítica baseada exclusivamente na transcrição da interação médico-paciente.
    """

    caso_id = caso_info["caso_id"]
    video = caso_info["video"]
    pasta_processada = caso_info["processed_path"]

    print("\n======================")
    print(f"Processando áudio: {caso_id}")

    # =========================
    # PASTAS
    # =========================
    pasta_audio = pasta_processada / "data_audio"
    pasta_transcricao = pasta_processada / "analise_comunicacao"

    audio_path = pasta_audio / "extracted_audio.wav"
    chunks_dir = pasta_audio / "chunks"

    try:
        extract_audio_from_video(video, audio_path)       
        
        chunks = split_audio(audio_path, chunks_dir)
        texto = transcrever_audio_chunks(chunks)
        salvar_texto(texto, pasta_transcricao /"transcription.txt")
        print("✔ Transcrição salva")

        gerar_resumo_caso(caso_info)

        protocolo = carregar_protocolo()
        analise_comunicacao = analisar_comunicacao(texto, protocolo)
        salvar_json(analise_comunicacao, pasta_transcricao /"analise_comunicacao.json")

        print("✔ Análise de comunicação salva")

    finally:
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir)
            print("🧹 Chunks removidos")