from pathlib import Path
import json
import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
from typer import prompt

# =========================
# CONFIGURAÇÕES
# =========================

load_dotenv()

PROTOCOLS_PATH = Path(os.getenv("PROTOCOLS_PATH"))
CHROMA_PATH = os.getenv("CHROMA_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# LLM
# =========================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# =========================
# CARREGAR PROTOCOLOS
# =========================

def carregar_protocolos():
    documentos = []
    arquivos = list(PROTOCOLS_PATH.glob("*.md"))

    for arquivo in arquivos:
        loader = TextLoader(str(arquivo), encoding="utf-8")
        docs = loader.load()
        documentos.extend(docs)

    print(f"✔ Protocolos carregados: {len(documentos)}")

    return documentos


# =========================
# CHUNKS
# =========================

def gerar_chunks(documentos):

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)

    print(f"✔ Chunks gerados: {len(chunks)}")

    return chunks


# =========================
# VECTOR DB
# =========================

def inicializar_rag():

    documentos = carregar_protocolos()
    chunks = gerar_chunks(documentos)
    chroma_path = Path(CHROMA_PATH)

    if (chroma_path.exists() and any(chroma_path.iterdir())):
        print("✔ Carregando Vector DB existente")

        vector_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name="protocolos_saude_mulher"
        )

        return vector_db

    print("⚠ Vector DB não encontrado")
    print("Criando novo Vector DB...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="protocolos_saude_mulher"
    )

    print("✔ Vector DB criado")

    return vector_db




# =========================
# CARREGAR TRANSCRIÇÃO
# =========================

def carregar_transcricao(caso_path):

    transcricao_path = (caso_path / "transcricao" / "transcription.txt")

    with open(transcricao_path, "r", encoding="utf-8") as f:
        texto = f.read()

    return texto


# =========================
# RECUPERAÇÃO RAG
# =========================

def recuperar_contexto(
    vector_db,
    texto_consulta
):

    docs = vector_db.similarity_search(texto_consulta, k=5)

    contexto = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    return contexto


# =========================
# ANÁLISE PROTOCOLAR
# =========================
def analisar_aderencia(
    transcricao,
    contexto
):
    prompt = f"""
Você é um auditor clínico especializado em:

- humanização do atendimento;
- comunicação médica;
- qualidade assistencial;
- empatia clínica;
- segurança emocional da paciente;
- auditoria protocolar em saúde da mulher.

Sua tarefa é realizar uma auditoria crítica da consulta médica.

Você deve utilizar:

1. a transcrição da consulta;
2. os protocolos clínicos recuperados.

==================================
CRITÉRIOS DE AUDITORIA
==================================

Avalie criticamente:

- acolhimento inicial;
- escuta ativa;
- investigação clínica;
- clareza da comunicação;
- excesso de linguagem técnica;
- adaptação ao perfil da paciente;
- empatia;
- humanização;
- acolhimento emocional;
- investigação emocional;
- validação emocional;
- orientação final;
- confirmação de entendimento;
- sinais de desconforto psicológico;
- possíveis falhas comunicacionais.

Considere NEGATIVO quando houver:

- consulta fria ou mecanizada;
- excesso de tecnicismo;
- linguagem pouco acessível;
- pouca adaptação ao nível da paciente;
- ausência de escuta ativa;
- interrupções excessivas;
- baixa empatia;
- comunicação distante;
- ausência de validação emocional;
- ausência de investigação emocional;
- orientações pouco claras;
- ausência de confirmação de entendimento;
- acolhimento superficial.

Não considere cordialidade superficial como humanização adequada.

==================================
SCORE
==================================

O campo "score_aderencia" deve ser um número inteiro de 0 a 100.

Use esta escala:

- 90-100:
  Consulta altamente aderente, humanizada e acolhedora.

- 75-89:
  Boa aderência protocolar com pequenas falhas.

- 60-74:
  Aderência razoável, mas com problemas relevantes.

- 40-59:
  Consulta pouco humanizada, fria ou inadequada.

- 0-39:
  Consulta crítica, inadequada ou potencialmente danosa.

IMPORTANTE:
- Nunca utilize escala de 0 a 10.
- Consultas frias ou excessivamente técnicas devem receber score reduzido.
- O score deve refletir criticamente a qualidade humana da consulta.

==================================
NÍVEL DE RISCO
==================================

Use:

- "baixo"
- "moderado"
- "elevado"

Considere "elevado" quando houver:
- múltiplas falhas humanas;
- ausência de acolhimento emocional;
- comunicação inadequada;
- sinais importantes de desconforto psicológico.

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

Formato esperado:

{{
    "acolhimento_inicial": {{
        "presente": true,
        "qualidade": "",
        "evidencias": []
    }},

    "escuta_ativa": {{
        "presente": true,
        "qualidade": "",
        "evidencias": []
    }},

    "investigacao_clinica": {{
        "presente": true,
        "qualidade": "",
        "evidencias": []
    }},

    "comunicacao": {{
        "clareza": "",
        "linguagem_tecnica_excessiva": false,
        "adaptacao_paciente": "",
        "evidencias": []
    }},

    "humanizacao": {{
        "nivel": "",
        "empatia": "",
        "acolhimento_emocional": "",
        "evidencias": []
    }},

    "avaliacao_emocional": {{
        "realizada": false,
        "qualidade": "",
        "evidencias": []
    }},

    "orientacao_final": {{
        "presente": true,
        "clareza": "",
        "confirmacao_entendimento": false
    }},

    "alertas": [],

    "pontos_positivos": [],

    "pontos_negativos": [],

    "score_aderencia": 0,

    "nivel_risco": "",

    "resumo_critico": ""
}}

==================================
PROTOCOLOS
==================================

{contexto}

==================================
TRANSCRIÇÃO
==================================

{transcricao}
"""
    
    response = llm.invoke(prompt)
    resposta = response.content
    resposta = re.sub(
        r"```json|```",
        "",
        resposta
    ).strip()

    try:

        return json.loads(resposta)

    except json.JSONDecodeError:

        return {
            "erro": "JSON inválido",
            "resposta_bruta": resposta
        }



# =========================
# SALVAR
# =========================

def salvar_resultados(pasta_saida, resultado_json):

    pasta_saida.mkdir(parents=True, exist_ok=True)

    json_path = (pasta_saida / "score_aderencia.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
        resultado_json,
        f,
        ensure_ascii=False,
        indent=4
    )

    print("✔ Resultado salvo")


# =========================
# PROCESSAMENTO
# =========================

def processar_caso(caso_info, vector_db):

    caso_id = caso_info["caso_id"]
    pasta_processada = (caso_info["processed_path"])

    print("\n======================")
    print(f"RAG Protocolar: {caso_id}")

    try:
        transcricao = carregar_transcricao(pasta_processada)
        contexto = recuperar_contexto(vector_db, transcricao)
        resultado = analisar_aderencia(transcricao, contexto)

        pasta_saida = (pasta_processada / "analise_protocolar")

        salvar_resultados(pasta_saida, resultado)

    except Exception as e:
        print(f"❌ Erro: {e}")