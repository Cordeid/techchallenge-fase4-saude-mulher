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
Você é um auditor clínico especializado em:

- humanização do atendimento;
- comunicação médica;
- qualidade assistencial;
- empatia clínica;
- linguagem acessível;
- desconforto psicológico em consultas de saúde da mulher.

Sua tarefa é avaliar criticamente a comunicação da consulta.

==================================
CRITÉRIOS DE AVALIAÇÃO
==================================

Avalie:

- empatia;
- clareza das explicações;
- escuta ativa;
- humanização;
- adaptação ao perfil da paciente;
- excesso de termos técnicos;
- validação emocional;
- qualidade comunicacional geral.

Considere NEGATIVO quando houver:

- consulta fria ou mecanizada;
- excesso de tecnicismo;
- pouca adaptação ao nível da paciente;
- interrupções excessivas;
- falta de acolhimento;
- ausência de validação emocional;
- comunicação confusa;
- tom distante;
- pouca empatia;
- postura excessivamente protocolar.

==================================
SCORE
==================================

O campo "score_comunicacao" deve ser um número inteiro de 0 a 100.

Use esta escala:

- 90-100:
  Comunicação excelente, humanizada e empática.

- 75-89:
  Comunicação adequada, com pequenas falhas.

- 60-74:
  Comunicação razoável, mas com problemas perceptíveis.

- 40-59:
  Comunicação ruim, fria, técnica ou pouco empática.

- 0-39:
  Comunicação crítica, inadequada ou potencialmente danosa.

IMPORTANTE:
- Nunca utilize escala de 0 a 10.
- O score deve refletir criticamente a qualidade humana da consulta.
- Consultas frias ou excessivamente técnicas NÃO devem receber score alto.

==================================
RESTRIÇÕES
==================================

- Não faça diagnóstico clínico.
- Não confirme violência.
- Apenas identifique sinais de atenção.
- Seja técnico, crítico e cauteloso.

==================================
FORMATO DE SAÍDA
==================================

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
Você é um auditor especializado em:

- análise emocional;
- comunicação clínica;
- humanização do atendimento;
- comportamento humano em consultas médicas;
- desconforto psicológico em saúde da mulher.

Sua tarefa é analisar criticamente os aspectos emocionais presentes na consulta.

==================================
CRITÉRIOS DE ANÁLISE
==================================

Avalie sinais de:

- medo;
- ansiedade;
- insegurança;
- hesitação;
- desconforto psicológico;
- submissão;
- constrangimento;
- dificuldade de compreensão;
- silêncio excessivo;
- respostas curtas;
- intimidação;
- tensão emocional;
- sofrimento emocional;
- baixa empatia;
- excesso de tecnicismo;
- comunicação inadequada;
- ausência de acolhimento emocional.

Também avalie:

- postura comunicacional do profissional;
- validação emocional da paciente;
- capacidade de escuta;
- adaptação ao perfil da paciente;
- clareza das explicações.

Considere NEGATIVO quando houver:

- comunicação fria ou mecanizada;
- excesso de linguagem técnica;
- pouca empatia;
- pouca escuta ativa;
- minimização das emoções da paciente;
- ausência de validação emocional;
- paciente aparentando desconforto recorrente;
- respostas monossilábicas frequentes;
- tom excessivamente protocolar.

==================================
NÍVEL DE ATENÇÃO
==================================

Use:

- "baixo"
- "moderado"
- "elevado"

Considere "elevado" quando houver múltiplos sinais emocionais relevantes.

==================================
SCORE
==================================

O campo "score_emocional" deve ser um número inteiro de 0 a 100.

Use esta escala:

- 90-100:
  Ambiente emocional seguro, acolhedor e humanizado.

- 75-89:
  Boa condução emocional com pequenas falhas.

- 60-74:
  Condução emocional razoável, porém com desconfortos perceptíveis.

- 40-59:
  Consulta emocionalmente inadequada, fria ou pouco empática.

- 0-39:
  Ambiente emocional crítico, desconfortável ou potencialmente danoso.

IMPORTANTE:
- Nunca utilize escala de 0 a 10.
- Consultas frias ou excessivamente técnicas devem receber score reduzido.
- O score deve refletir criticamente o impacto emocional da interação.

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