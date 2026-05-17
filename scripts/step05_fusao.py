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
    transcricao = ler_texto(caso_path / "transcricao" / "transcription.txt")
    resumo_audio = ler_json(caso_path / "transcricao" / "resumo_ia.json")
    analise_emocional = ler_json(caso_path / "transcricao" / "analise_emocional.json")
    analise_video = ler_json(caso_path / "analise_video" / "resumo_video_ia.json")
    deteccoes_video = ler_json(caso_path / "analise_video" / "deteccoes.json")
    analise_protocolar = ler_json(caso_path / "analise_protocolar" / "score_aderencia.json")

    return {
        "caso_id": caso_path.name,
        "transcricao": transcricao,
        "resumo_audio": resumo_audio,
        "analise_emocional": analise_emocional,
        "analise_video": analise_video,
        "deteccoes_video": deteccoes_video,
        "analise_protocolar": analise_protocolar,
    }


# =========================
# FUSÃO MULTIMODAL
# =========================

def gerar_fusao_multimodal(dados: dict) -> dict:

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
1. análises de áudio;
2. análises emocionais;
3. análises visuais;
4. auditoria protocolar;
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
    "risco_geral": "{score_data['risco_geral']}",
    "classificacao": "{score_data['classificacao']}",
    "score_final": {score_data['score_final']},
    "score_audio": {score_data['score_audio']},
    "score_emocional": {score_data['score_emocional']},
    "score_video": {score_data['score_video']},
    "score_protocolar": {score_data['score_protocolar']},
    "necessita_revisao_humana": {str(score_data['necessita_revisao_humana']).lower()},
    "penalidades_aplicadas": {json.dumps(score_data['penalidades_aplicadas'], ensure_ascii=False)},
    "avaliacao_comunicacao": "",
    "avaliacao_empatia": "",
    "avaliacao_humanizacao": "",
    "linguagem_tecnica_excessiva": false,
    "adaptacao_paciente": "",
    "principais_alertas": [],
    "evidencias_audio": [],
    "evidencias_video": [],
    "evidencias_protocolares": [],
    "fatores_desconforto": [],
    "pontos_positivos": [],
    "pontos_negativos": [],
    "recomendacao": "",
    "resumo_executivo": ""
}}

========================
SCORES CALCULADOS
========================

{json.dumps(score_data, ensure_ascii=False, indent=2)}

========================
DADOS DO CASO
========================

{json.dumps(dados, ensure_ascii=False, indent=2)}
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

            "score_base": score_data["score_base"],
            "penalidade_total": score_data["penalidade_total"],

            "score_final": score_data["score_final"],

            "score_audio": score_data["score_audio"],
            "score_emocional": score_data["score_emocional"],
            "score_video": score_data["score_video"],
            "score_protocolar": score_data["score_protocolar"],

            "classificacao": score_data["classificacao"],
            "risco_geral": score_data["risco_geral"],

            "necessita_revisao_humana": (
                score_data["necessita_revisao_humana"]
            ),

            "penalidades_aplicadas": (
                score_data["penalidades_aplicadas"]
            )

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


def gerar_relatorio_markdown(resultado: dict) -> str:

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

    alertas = resultado.get("principais_alertas", [])
    evidencias_audio = resultado.get("evidencias_audio", [])
    evidencias_video = resultado.get("evidencias_video", [])
    evidencias_protocolares = resultado.get("evidencias_protocolares", [])
    pontos_positivos = resultado.get("pontos_positivos", [])
    pontos_negativos = resultado.get("pontos_negativos", [])
    fatores_desconforto = resultado.get("fatores_desconforto", [])
    penalidades = resultado.get("penalidades_aplicadas", [])

    def lista_md(itens):
        if not itens:
            return "- Nenhum item identificado."

        return "\n".join([
            f"- {item}"
            for item in itens
        ])

    return f"""# Relatório Final de Auditoria Multimodal - Sistema Hórus
    
    ## Caso

    **{caso_id}**

    ---

    ## Resultado Geral

    **Risco geral:** {resultado.get("risco_geral")}

    **Classificação:** {resultado.get("classificacao")}

    **Score final:** {resultado.get("score_final")} de 100

    **Necessita revisão humana:** {"SIM" if resultado.get("necessita_revisao_humana") else "NÃO"}

    ---

    ## Composição do Score

    | Item             |                               Valor |
    | ---------------- | ----------------------------------: |
    | Score Base       |       {resultado.get("score_base")} |
    | Penalidade Total | {resultado.get("penalidade_total")} |
    | Score Final      |      {resultado.get("score_final")} |

    ---

    ## Scores por Dimensão

    | Dimensão            |                               Score |
    | ------------------- | ----------------------------------: |
    | Comunicação/Áudio   |      {resultado.get("score_audio")} |
    | Emoções             |  {resultado.get("score_emocional")} |
    | Vídeo/Comportamento |      {resultado.get("score_video")} |
    | Protocolo           | {resultado.get("score_protocolar")} |

    ---

    ## Resumo Executivo

    {resultado.get("resumo_executivo")}

    ---

    ## Avaliação de Comunicação

    {resultado.get("avaliacao_comunicacao", "Não informado.")}

    ---

    ## Avaliação de Empatia

    {resultado.get("avaliacao_empatia", "Não informado.")}

    ---

    ## Avaliação de Humanização

    {resultado.get("avaliacao_humanizacao", "Não informado.")}

    ---

    ## Principais Alertas

    {lista_md(alertas)}

    ---

    ## Penalidades Aplicadas

    {lista_md(penalidades)}

    ---

    ## Fatores de Desconforto

    {lista_md(fatores_desconforto)}

    ---

    ## Evidências de Áudio

    {lista_md(evidencias_audio)}

    ---

    ## Evidências de Vídeo

    {lista_md(evidencias_video)}

    ---

    ## Evidências Protocolares

    {lista_md(evidencias_protocolares)}

    ---

    ## Pontos Positivos

    {lista_md(pontos_positivos)}

    ---

    ## Pontos Negativos

    {lista_md(pontos_negativos)}

    ---

    ## Recomendação

    {resultado.get("recomendacao")}

    ---

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
    resumo_audio = (dados.get("resumo_audio", {}) or {})
    analise_emocional = (dados.get("analise_emocional", {}) or {})
    analise_video = (dados.get("analise_video", {}) or {})
    analise_protocolar = (dados.get("analise_protocolar", {}) or {})

    score_audio = resumo_audio.get("score_comunicacao", 70)
    score_emocional = analise_emocional.get("score_emocional", 70)
    score_video = analise_video.get("score_visual", 70)
    score_protocolar = analise_protocolar.get("score_aderencia", 70)    

    # =========================
    # SCORE BASE PONDERADO
    # =========================
    score_base = (
        (score_audio * 0.25) +
        (score_emocional * 0.20) +
        (score_video * 0.20) +
        (score_protocolar * 0.35)
    )

    score_final = score_base

    penalidades = []
    penalidade_total = 0

    # =========================
    # Linguagem técnica excessiva
    # =========================
    comunicacao = analise_protocolar.get("comunicacao", {})
    if comunicacao.get("linguagem_tecnica_excessiva", False):
        penalidade_total += 15
        penalidades.append("Linguagem técnica excessiva")

    # =========================
    # Comunicação inadequada
    # =========================
    avaliacao_comunicacao = (analise_emocional.get("avaliacao_comunicacao", "").lower())

    if (
        "ruim" in avaliacao_comunicacao or
        "inadequada" in avaliacao_comunicacao or
        "baixa" in avaliacao_comunicacao
    ):

        penalidade_total += 7
        penalidades.append("Comunicação inadequada")  

    score_final = round(
        score_final - penalidade_total,
        2
    )


    # =========================
    # NORMALIZAÇÃO
    # =========================

    score_final = max(0, min(100, round(score_final)))

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
    # RISCO GERAL
    # =========================
    if score_final >= 75:
        risco = "BAIXO"
    elif score_final >= 50:
        risco = "MODERADO"
    else:
        risco = "ELEVADO"

    # =========================
    # REVISÃO HUMANA
    # =========================
    necessita_revisao_humana = (risco != "BAIXO")

    return {
        "score_base": round(score_base, 2),
        "penalidade_total": penalidade_total,
        "score_final": score_final,
        "score_audio": score_audio,
        "score_emocional": score_emocional,
        "score_video": score_video,
        "score_protocolar": score_protocolar,
        "classificacao": classificacao,
        "risco_geral": risco,
        "necessita_revisao_humana": (
            necessita_revisao_humana
        ),
        "penalidades_aplicadas": penalidades
    }

# =========================
# PROCESSAMENTO
# =========================

def processar_caso(caso_info):

    caso_id = caso_info["caso_id"]
    pasta_processada = (caso_info["processed_path"])

    print("\n======================")
    print(f"Fusionando caso: {caso_id}")

    dados = carregar_resultados_caso(pasta_processada)
    resultado = gerar_fusao_multimodal(dados)
    pasta_saida = (pasta_processada /"relatorio_final")

    salvar_json(resultado,pasta_saida /"resultado_final.json")

    relatorio_md = gerar_relatorio_markdown(resultado)

    salvar_texto(relatorio_md,pasta_saida /"relatorio_final.md")

    print("✔ Resultado final salvo")
    print("✔ Relatório final salvo")