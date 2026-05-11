# Tech Challenge Fase 4 — Sistema Multimodal de Monitoramento Clínico para Saúde da Mulher

**Autor:** Diego Valentim Cordeiro (RM369026)  
**Curso:** Pós-Tech IADT (IA Para Devs) — FIAP  
**Fase:** 4 — Tech Challenge Final  

---

## Descrição

Sistema multimodal que processa vídeo, áudio e documentos de uma consulta médica para detectar sinais precoces de risco psicológico ou violência doméstica. Utiliza visão computacional, transcrição de fala e NLP, orquestrados por um grafo LangGraph com decisão final por GPT-4.

---

## Instalação

### Pré-requisitos

- Python 3.11+
- NVIDIA GPU com CUDA 12.8+ (testado em RTX 5070 / Blackwell sm_120)
- Credenciais AWS (Textract + Comprehend) e chave OpenAI

### 1. Clonar e configurar ambiente

```bash
git clone <url-do-repo>
cd techchallenge-fase4-saude-mulher
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2. Instalar PyTorch (OBRIGATÓRIO antes das demais dependências)

```bash
# Para GPU Blackwell (RTX 4xxx/5xxx, sm_120) — nightly cu128
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. Instalar demais dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar credenciais

```bash
cp .env.example .env
# Editar .env com suas chaves OpenAI e AWS
```

---

## Uso

```bash
# Pipeline completo
python scripts/run_pipeline.py --video data/input/video_consulta.mp4

# Com laudo PDF opcional
python scripts/run_pipeline.py --video data/input/video_consulta.mp4 --pdf data/input/laudo_exemplo.pdf

# Testar nós individualmente
python scripts/01_test_video_node.py
python scripts/02_test_audio_node.py
python scripts/03_test_nlp_node.py
python scripts/04_test_doc_node.py
python scripts/05_test_fusion_node.py
```

---

## Arquitetura

Detalhes em `PROJECT_SPEC.md` e `RELATORIO_TECNICO.md`.

---

## Demonstração

> Link do vídeo no YouTube: *(será adicionado após gravação)*

---

## Aviso ético

Este sistema é uma **ferramenta de apoio à decisão clínica** e **não substitui avaliação médica profissional**. Os resultados devem ser interpretados por equipe especializada.
