from pathlib import Path
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# CONFIGURAÇÕES
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não encontrada no .env")


client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# UTILITÁRIOS
# =========================

def ler_texto(caminho: Path) -> str:
    if not caminho.exists():
        return ""

    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()


def ler_json(caminho: Path):
    if not caminho.exists():
        return None

    with open(caminho, "r", encoding="utf-8") as f:
        try:
            return json.load(f)

        except Exception as e:
            print(f"❌ Erro JSON: {caminho}")
            print(e)
            return {}




def salvar_texto(texto: str, caminho: Path):
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)


def salvar_json(dados: dict, caminho: Path):
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)


# =========================
# CARREGAMENTO DOS AGENTES
# =========================

def carregar_resultados_caso(caso_path: Path) -> dict:
    analise_prontuario = ler_json(caso_path / "analise_prontuario" / "analise_prontuario.json")
    analise_comunicacao = ler_json(caso_path / "analise_comunicacao" / "analise_comunicacao.json")
    analise_vocal = ler_json(caso_path / "analise_vocal" / "analise_vocal.json")
    analise_video = ler_json(caso_path / "analise_video" / "analise_video.json")

    return {
        "caso_id": caso_path.name,
        "analise_prontuario": analise_prontuario,
        "analise_comunicacao": analise_comunicacao,
        "analise_vocal": analise_vocal,
        "analise_video": analise_video,
    }


# =========================
# FUSÃO MULTIMODAL
# =========================

def gerar_fusao_multimodal(dados: dict, resumo_caso: str) -> dict:

    score_data = calcular_score_final(dados)

    prompt = f"""
Você é um auditor clínico multimodal especializado em:

- humanização do atendimento;
- comunicação médica;
- qualidade assistencial;
- comportamento humano;
- desconforto psicológico em consultas de saúde da mulher.

Sua tarefa é produzir uma análise crítica integrada da consulta médica.

Você receberá:
1. análise de prontuário;
2. análise de comunicação;
3. análise vocal;
4. análise visual;
5. scores determinísticos calculados pelo sistema.

IMPORTANTE:
- Os scores já foram calculados pelo sistema.
- NÃO altere os scores.
- Utilize os scores como referência para sua análise crítica.
- Seu papel é interpretar os resultados multimodais.

Avalie:
- qualidade comunicacional;
- empatia;
- humanização;
- excesso de tecnicismo;
- adaptação ao perfil da paciente;
- clareza das explicações;
- validação emocional;
- sinais de desconforto;
- coerência entre os agentes;
- aderência protocolar;
- necessidade de revisão humana.

IMPORTANTE:
- Não faça diagnóstico clínico.
- Não confirme violência.
- Apenas indique sinais de atenção.
- Seja técnico, crítico e cauteloso.
- Consultas frias ou excessivamente técnicas devem ser avaliadas negativamente.
- Não considere cordialidade superficial como humanização adequada.

Retorne APENAS JSON válido.

Formato esperado:

{{
    "caso_id": "{dados['caso_id']}",
    "classificacao": "{score_data['classificacao']}",
    "score_final": {score_data['score_final']},
    "score_prontuario": {score_data['score_prontuario']},
    "score_comunicacao": {score_data['score_comunicacao']},
    "score_vocal": {score_data['score_vocal']},
    "score_video": {score_data['score_video']},
    "necessita_revisao_humana": {str(score_data['necessita_revisao_humana']).lower()},
    "resumo_executivo": ""
}}

========================
SCORES CALCULADOS
========================

{json.dumps(score_data, ensure_ascii=False, indent=2)}

========================
SOBRE O CASO
========================

{resumo_caso}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um auditor multimodal especializado "
                    "em qualidade assistencial e humanização "
                    "de consultas médicas."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    resposta = response.choices[0].message.content

    resposta = (
        resposta
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:

        resultado = json.loads(resposta)

        # =========================
        # GARANTE SCORES OFICIAIS
        # =========================

        resultado.update({

            "score_final": score_data["score_final"],

            "score_prontuario": score_data["score_prontuario"],
            "score_comunicacao": score_data["score_comunicacao"],
            "score_vocal": score_data["score_vocal"],
            "score_video": score_data["score_video"],

            "classificacao": score_data["classificacao"],

            "necessita_revisao_humana": (score_data["necessita_revisao_humana"])
        })

        return resultado


    except json.JSONDecodeError:

        return {
            "caso_id": dados["caso_id"],
            "erro": "Resposta do modelo não estava em JSON válido.",
            "resposta_bruta": resposta,
            "score_data": score_data
        }



# =========================
# RELATÓRIO EXECUTIVO
# =========================


def gerar_relatorio_markdown(resumo_caso: str, dados: dict, resultado: dict) -> str:

    caso_id = resultado.get("caso_id", "N/A")

    if "erro" in resultado:
        return f"""# Relatório Final — {caso_id}

    ## Erro

    {resultado.get("erro")}

    ## Resposta bruta

    ```txt
    {resultado.get("resposta_bruta")}
    ````

    """


    return f"""# Relatório Final de Auditoria Multimodal - Sistema Hórus
    
    ## Caso
    **{caso_id}**
    {resumo_caso}

    ---

    ## Resultado Geral

    **Score final:** {resultado.get("score_final")} de 100
    **Classificação:** {resultado.get("classificacao")}
    **Necessita revisão humana:** {"SIM" if resultado.get("necessita_revisao_humana") else "NÃO"}

    ---

    ## Scores por Dimensão

    | Dimensão            |                               Score |
    | ------------------- | ----------------------------------: |
    | Prontuário          |      {resultado.get("score_prontuario")} |
    | Comunicação         |     {resultado.get("score_comunicacao")} |
    | Emoções (vocal)     |  {resultado.get("score_vocal")} |
    | Vídeo (comportamento/postura) |      {resultado.get("score_video")} |

    ---

    ## Resumo Executivo
    {resultado.get("resumo_executivo")}

    ---

    ## Análise Detalhada

    ### Análise de Prontuário
    {dados["analise_prontuario"].get("analise_prontuario", "Não informado.")}

    ### Análise de Comunicação/Áudio
    {dados["analise_comunicacao"].get("analise_comunicacao", "Não informado.")}

    ### Análise Emocional (vocal)
    {dados["analise_vocal"].get("analise_vocal", "Não informado.")}

    ### Análise de Vídeo (comportamento/postura)
    {dados["analise_video"].get("analise_video", "Não informado.")}

    ## Observação
    Este relatório possui caráter assistivo e preventivo.
    O sistema não realiza diagnóstico clínico e não confirma situações de violência.
    A análise possui finalidade de apoio à auditoria de qualidade assistencial e humanização do atendimento.
    Casos com risco moderado ou elevado devem ser revisados por equipe humana especializada.
    """


# =========================
# SCORE FINAL
# =========================

def calcular_score_final(dados: dict):
    # =========================
    # SCORES BASE
    # =========================
    analise_prontuario = (dados.get("analise_prontuario", {}) or {})
    analise_comunicacao = (dados.get("analise_comunicacao", {}) or {})
    analise_vocal = (dados.get("analise_vocal", {}) or {})
    analise_video = (dados.get("analise_video", {}) or {})

    score_prontuario = analise_prontuario.get("score_prontuario", 70)
    score_comunicacao = analise_comunicacao.get("score_comunicacao", 70)
    score_vocal = analise_vocal.get("score_vocal", 70)
    score_video = analise_video.get("score_video", 70)

    # =========================
    # SCORE BASE PONDERADO
    # =========================
    score_base = (
        (score_prontuario * 0.20) +
        (score_comunicacao * 0.25) +
        (score_vocal * 0.30) +
        (score_video * 0.25)
    )

    score_final = max(0, min(100, round(score_base)))

    # =========================
    # CLASSIFICAÇÃO
    # =========================

    if score_final >= 90:
        classificacao = "EXCELENTE"
    elif score_final >= 75:
        classificacao = "ADEQUADA"
    elif score_final >= 60:
        classificacao = "ATENÇÃO"
    elif score_final >= 40:
        classificacao = "CRÍTICA"
    else:
        classificacao = "GRAVE"


    # =========================
    # REVISÃO HUMANA
    # =========================
    necessita_revisao_humana = (score_final < 70) or (score_vocal < 60) or (score_video < 60)

    return {
        "score_base": round(score_base, 2),
        "score_final": score_final,
        "score_prontuario": score_prontuario,
        "score_comunicacao": score_comunicacao,
        "score_vocal": score_vocal,
        "score_video": score_video,
        "classificacao": classificacao,
        "necessita_revisao_humana": (necessita_revisao_humana),
    }


# =========================
# PROCESSAMENTO
# =========================

def processar_caso(caso_info):

    caso_id = caso_info["caso_id"]
    pasta_processada = (caso_info["processed_path"])

    print("\n======================")
    print(f"Fusionando caso: {caso_id}")

    resumo_caso_path = pasta_processada / "analise_comunicacao" / "resumo_caso.txt"
    resumo_caso = ler_texto(resumo_caso_path)
    dados = carregar_resultados_caso(pasta_processada)

    avaliacao_final = gerar_fusao_multimodal(dados, resumo_caso)
    pasta_saida = (pasta_processada /"relatorio_final")

    salvar_json(avaliacao_final, pasta_saida /"resultado_final.json")

    relatorio_md = gerar_relatorio_markdown(resumo_caso, dados, avaliacao_final)

    salvar_texto(relatorio_md, pasta_saida /"relatorio_final.md")

    print("✔ Resultado final salvo")