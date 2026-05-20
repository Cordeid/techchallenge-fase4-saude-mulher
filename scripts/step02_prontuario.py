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

BASE_DIR = Path(__file__).resolve().parent.parent
PROTOCOLO_PATH = BASE_DIR / "protocolos" / "protocolo_prontuario.md"

# =========================
# CARREGAR PROTOCOLO
# =========================

def carregar_protocolo():

    with open(PROTOCOLO_PATH, "r", encoding="utf-8") as f:
        return f.read()

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

def analisar_prontuario(texto, protocolo):

    prompt = f"""
Você é um auditor clínico especializado em:

- qualidade documental;
- humanização;
- comunicação clínica;
- segurança assistencial;
- saúde da mulher.

Sua tarefa é realizar uma auditoria crítica de prontuário médico.

Você DEVE utilizar o protocolo abaixo como referência oficial
de avaliação.

==================================================
PROTOCOLO OFICIAL DE AVALIAÇÃO
==================================================

{protocolo}

==================================================
INSTRUÇÕES DE AUDITORIA
==================================================

Avalie criticamente:

- aderência ao protocolo;
- completude documental;
- clareza do registro;
- organização;
- coerência clínica;
- humanização;
- empatia;
- escuta ativa;
- investigação emocional;
- linguagem excessivamente técnica;
- registro mecanizado;
- qualidade assistencial;
- documentação de sofrimento emocional;
- qualidade da conduta descrita;
- segurança documental.

IMPORTANTE:

- Não faça diagnóstico clínico.
- Não confirme violência.
- Não invente informações.
- Baseie-se SOMENTE no conteúdo presente no prontuário.
- Seja técnico, crítico e objetivo.

Você deve identificar:

- critérios adequadamente atendidos;
- critérios parcialmente atendidos;
- critérios ausentes;
- possíveis riscos documentais;
- sinais de desumanização;
- falhas de comunicação clínica.

==================================================
ESCALA
==================================================

0-20 → muito ruim
21-40 → ruim
41-60 → regular
61-80 → adequado
81-100 → excelente

==================================================
FORMATO OBRIGATÓRIO
==================================================

Retorne APENAS JSON válido.

{{
    "score_prontuario": 0,

    "pontos_positivos": [],
    "pontos_negativos": [],

    "analise_prontuario": ""
}}

==================================================
PRONTUÁRIO
==================================================

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
    """
    Responsável pela auditoria documental do prontuário médico.
    O script realiza extração textual de PDFs clínicos e utiliza um protocolo especializado para avaliar qualidade documental, clareza, organização, humanização e aderência assistencial.
    A análise é conduzida por LLM, gerando score protocolar, pontos positivos, falhas identificadas e resumo crítico do prontuário.
    """

    caso_id = caso_info["caso_id"]

    pasta_raw = caso_info["raw_path"]
    pasta_processada = caso_info["processed_path"]

    print("\n======================")
    print(f"Analisando prontuário: {caso_id}")

    pdfs = list(pasta_raw.glob("*.pdf"))

    if not pdfs:
        print("❌ PDF não encontrado")
        return

    pdf_path = pdfs[0]
    texto = extrair_texto_pdf(pdf_path)
    protocolo = carregar_protocolo()

    pasta_saida = (pasta_processada / "analise_prontuario")

    salvar_texto(texto, pasta_saida / "prontuario_extraido.txt")

    resultado = analisar_prontuario(texto=texto, protocolo=protocolo)

    salvar_json(resultado, pasta_saida / "analise_prontuario.json")

    print("✔ Prontuário analisado")
