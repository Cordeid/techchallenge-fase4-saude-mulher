from pathlib import Path
import json
import os
import fitz

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIG
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# EXTRAÇÃO PDF
# =========================

def extrair_texto_pdf(pdf_path):

    texto = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        texto += page.get_text()
    doc.close()
    return texto


# =========================
# SALVAR JSON
# =========================

def salvar_json(dados, caminho):

    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(
            dados,
            f,
            ensure_ascii=False,
            indent=4
        )


# =========================
# SALVAR TEXTO
# =========================

def salvar_texto(texto, caminho):

    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)


# =========================
# LLM
# =========================

def analisar_prontuario(texto):

    prompt = f"""
Você é um auditor clínico especializado em:

- qualidade documental;
- humanização;
- comunicação clínica;
- segurança assistencial;
- saúde da mulher.

Sua tarefa é avaliar criticamente a qualidade do prontuário médico.

Avalie:

- clareza do registro;
- completude documental;
- organização;
- humanização;
- empatia;
- escuta ativa;
- investigação emocional;
- linguagem excessivamente técnica;
- registro mecanizado;
- coerência clínica;
- qualidade assistencial;
- documentação de sofrimento emocional;
- qualidade da conduta descrita.

IMPORTANTE:
- Não faça diagnóstico clínico.
- Não confirme violência.
- Seja técnico e crítico.

Escala de score:
0-20 → muito ruim
21-40 → ruim
41-60 → regular
61-80 → adequado
81-100 → excelente

Retorne APENAS JSON válido.

Formato:

{{
    "qualidade_documental": "",
    "pontos_positivos": [],
    "pontos_negativos": [],
    "score_prontuario": 0,
    "analise_prontuario": ""
}}

PRONTUÁRIO:

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

    resposta = (
        response
        .choices[0]
        .message
        .content
    )

    resposta = (
        resposta
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(resposta)


# =========================
# PROCESSAMENTO
# =========================

def processar_caso(caso_info):

    caso_id = caso_info["caso_id"]

    pasta_raw = caso_info["raw_path"]
    pasta_processada = caso_info["processed_path"]

    print("\n======================")
    print(f"Analisando prontuário: {caso_id}")

    pdfs = list(
        pasta_raw.glob("*.pdf")
    )

    if not pdfs:
        print("❌ PDF não encontrado")
        return

    pdf_path = pdfs[0]

    texto = extrair_texto_pdf(pdf_path)
    pasta_saida = (pasta_processada / "analise_prontuario")
    salvar_texto(texto, pasta_saida / "prontuario_extraido.txt")
    resultado = analisar_prontuario(texto)

    salvar_json(resultado, pasta_saida / "analise_prontuario.json")

    print("✔ Prontuário analisado")
