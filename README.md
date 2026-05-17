# Sistema Multiagente de Auditoria Protocolar para Consultas de Saúde da Mulher

## Pós-Tech FIAP — IA para Devs

### Tech Challenge — Fase 4

---

# 1. Visão Geral do Projeto

## Nome do Projeto

**FemAI Guardian — Sistema Multiagente de Auditoria Protocolar para Consultas de Saúde da Mulher**

---

## Objetivo

Desenvolver um sistema multimodal baseado em Inteligência Artificial capaz de analisar consultas médicas voltadas à saúde da mulher, utilizando vídeo, áudio e texto para identificar sinais de desconforto psicológico, medo, ansiedade e possíveis indícios de violência psicológica ou doméstica.

O sistema atuará como ferramenta de apoio à coordenação técnica hospitalar, auxiliando equipes de auditoria clínica e qualidade hospitalar na identificação precoce de atendimentos atípicos e situações de risco.

---

# 2. Problema

Hospitais e clínicas frequentemente possuem dificuldades para monitorar qualitativamente todas as consultas realizadas diariamente.

Além disso:

- sinais de violência psicológica ou doméstica podem passar despercebidos;
- sinais não verbais de medo ou desconforto podem não ser identificados;
- auditorias humanas são demoradas e caras;
- protocolos clínicos podem não ser seguidos integralmente.

O projeto propõe uma solução multimodal capaz de automatizar parte desse processo de auditoria.

---

# 3. Objetivos Técnicos

O sistema deverá:

- Processar vídeos de consultas médicas;
- Extrair e transcrever áudio;
- Analisar expressões faciais e linguagem corporal;
- Avaliar padrões vocais associados a desconforto;
- Comparar consultas com protocolos clínicos;
- Gerar alertas automáticos;
- Produzir relatórios analíticos automatizados.

---

# 4. Relação com o Tech Challenge

O projeto atende diretamente aos requisitos da Fase 4:

## ✔ Análise de Vídeo

- Análise de consultas;
- Identificação de sinais não verbais;
- Detecção de desconforto psicológico.

## ✔ Análise de Áudio

- Transcrição automática;
- Avaliação emocional da fala;
- Identificação de hesitação e ansiedade.

## ✔ Processamento Multimodal

- Vídeo;
- Áudio;
- Texto;
- Documentos.

## ✔ Detecção de Anomalias

- Casos atípicos;
- Comportamentos incomuns;
- Não aderência ao protocolo.

## ✔ Uso de IA em Nuvem

- OpenAI API;
- Possível integração futura com Azure Cognitive Services.

---

# 5. Arquitetura Geral

```txt
                +----------------------+
                |   Upload do Caso     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Agente de Ingestão  |
                +----------+-----------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
+-------------------+         +-------------------------+
| Agente de Áudio   |         | Agente de Vídeo         |
+-------------------+         +-------------------------+
| - extração áudio  |         | - extração frames       |
| - transcrição     |         | - análise facial        |
| - análise vocal   |         | - postura corporal      |
| - emoções         |         | - comportamento         |
+-------------------+         +-------------------------+
          |                                 |
          +----------------+----------------+
                           |
                           v
               +------------------------+
               | Agente Protocolar RAG |
               +------------------------+
               | - consulta protocolos |
               | - verifica aderência  |
               | - detecta desvios     |
               +------------------------+
                           |
                           v
               +------------------------+
               | Agente Fusionador      |
               +------------------------+
               | - consolida análises   |
               | - gera alertas         |
               | - score de risco       |
               +------------------------+
                           |
                           v
               +------------------------+
               | Dashboard / Relatório |
               +------------------------+
```

---

# 6. Estrutura de Pastas

```txt
/project

│
├── /data_raw
│   ├── caso_001
│   │   ├── consulta.mp4
│   │   ├── prontuario.pdf
│   │   └── observacoes.txt
│   │
│   └── caso_002
│
├── /data_processed
│   ├── audios
│   ├── transcricoes
│   ├── frames
│   ├── embeddings
│   └── relatorios
│
├── /protocolos
│   ├── protocolo_consulta.md
│   ├── protocolo_triagem.md
│   └── protocolo_violencia.md
│
├── /scripts
│   ├── step01_ingestao.py
│   ├── step02_audio.py
│   ├── step03_video.py
│   ├── step04_rag.py
│   ├── step05_fusao.py
│   └── utils.py
│
├── /models
│
├── /dashboard
│
├── requirements.txt
│
├── README.md
│
└── main.py
```

---

# 7. Fluxo dos Agentes

---

# 7.1 Agente de Ingestão

## Responsabilidades

- validar arquivos;
- verificar formatos;
- criar estrutura do caso;
- gerar resumo inicial;
- organizar diretórios.

## Tecnologias

- Python;
- pathlib;
- ffmpeg;
- OpenCV.

---

# 7.2 Agente de Áudio

## Responsabilidades

### Extração de áudio

- ffmpeg.

### Transcrição

- Whisper/OpenAI.

### Análise emocional textual

- hesitação;
- ansiedade;
- medo;
- silêncio excessivo;
- pausas longas.

### Geração de resumo clínico

- OpenAI GPT.

---

## Possíveis análises

- Paciente apresenta medo?
- Há hesitação ao responder?
- Há interrupções frequentes?
- O médico utilizou abordagem acolhedora?

---

## Tecnologias

- OpenAI Whisper;
- ffmpeg;
- transformers;
- OpenAI API.

---

# 7.3 Agente de Vídeo

## Responsabilidades

### Processamento de vídeo

- extração de frames;
- detecção de pessoas;
- análise facial.

### Avaliação comportamental

- postura corporal;
- expressões faciais;
- evasão visual;
- inquietação;
- desconforto.

---

## Técnicas utilizadas

### YOLOv8

- detecção de pessoas;
- detecção de objetos.

### MediaPipe

- landmarks faciais;
- postura corporal.

### DeepFace

- emoções faciais.

---

## Possíveis alertas

- sinais visuais de ansiedade;
- comportamento retraído;
- desconforto contínuo;
- baixa interação visual.

---

# 7.4 Agente Protocolar (RAG)

## Objetivo

Comparar a consulta realizada com protocolos clínicos hospitalares.

---

## Fluxo

1. Transcrição é convertida em embeddings;
2. Protocolos são indexados;
3. Sistema recupera trechos relevantes;
4. GPT compara consulta com protocolo.

---

## Verificações

- médico realizou acolhimento?
- perguntas obrigatórias foram feitas?
- houve investigação de sintomas?
- houve explicação adequada?

---

## Tecnologias

- LangChain;
- ChromaDB;
- OpenAI Embeddings;
- GPT-4o-mini / GPT-5-mini.

---

# 7.5 Agente Fusionador

## Objetivo

Consolidar todas as análises em um relatório final.

---

## Entrada

- score emocional;
- score visual;
- aderência protocolar;
- inconsistências;
- alertas.

---

## Saída

```json
{
  "caso_id": "001",
  "risco_geral": "MODERADO",
  "aderencia_protocolar": 78,
  "alertas": [
    "Paciente apresentou hesitação frequente",
    "Expressões faciais indicaram desconforto",
    "Etapa obrigatória não identificada"
  ]
}
```

---

# 8. Pipeline de Processamento

```txt
Vídeo MP4
   ↓
Extração de áudio
   ↓
Transcrição Whisper
   ↓
Análise emocional textual
   ↓
Extração de frames
   ↓
Análise facial e corporal
   ↓
RAG protocolar
   ↓
Fusão multimodal
   ↓
Relatório final
```

---

# 9. Tecnologias Utilizadas

| Tecnologia | Objetivo            |
| ---------- | ------------------- |
| Python     | Linguagem principal |
| OpenAI API | LLM                 |
| Whisper    | Transcrição         |
| YOLOv8     | Visão computacional |
| MediaPipe  | Face e pose         |
| DeepFace   | Emoções             |
| OpenCV     | Processamento vídeo |
| LangChain  | Orquestração        |
| ChromaDB   | Banco vetorial      |
| ffmpeg     | Extração áudio      |
| Streamlit  | Dashboard           |
| Docker     | Containerização     |

---

# 10. Dashboard

## Funcionalidades

- upload de casos;
- visualização da transcrição;
- alertas automáticos;
- score de aderência;
- resumo analítico;
- visualização de riscos.

---

## Exemplo

```txt
CASO #001
RISCO: MODERADO

✔ Consulta processada
✔ Áudio analisado
✔ Vídeo analisado

ALERTAS:
- Hesitação vocal detectada
- Linguagem corporal retraída
- Falha em etapa protocolar
```

---

# 11. Dados Utilizados

## Estratégia adotada

Serão utilizados dados sintéticos para fins acadêmicos.

---

## Tipos de dados

- vídeos simulados;
- vozes sintéticas;
- documentos fictícios;
- consultas encenadas.

---

## Justificativa

- evitar exposição de dados reais;
- evitar problemas éticos;
- evitar violações da LGPD;
- facilitar criação de cenários.

---

# 12. Limitações

O sistema:

- não substitui profissionais de saúde;
- não realiza diagnóstico clínico;
- não confirma violência doméstica;
- atua apenas como apoio técnico.

---

# 13. Possíveis Evoluções Futuras

- integração com Azure Cognitive Services;
- monitoramento em tempo real;
- análise de múltiplos participantes;
- modelos especializados treinados;
- integração hospitalar;
- autenticação e prontuário eletrônico.

---

# 14. Resultados Esperados

Espera-se que o sistema:

- reduza tempo de auditoria;
- identifique consultas atípicas;
- aumente aderência protocolar;
- ajude equipes hospitalares;
- demonstre o potencial da IA multimodal na saúde.

---

# 15. Conclusão

O projeto propõe um sistema multimodal baseado em Inteligência Artificial para auditoria protocolar de consultas voltadas à saúde da mulher.

A solução integra:

- visão computacional;
- processamento de áudio;
- análise textual;
- modelos generativos;
- recuperação semântica de protocolos clínicos.

O sistema busca apoiar equipes hospitalares na identificação precoce de situações de risco, desconforto psicológico e não conformidades clínicas, demonstrando o potencial da IA aplicada à saúde digital.

# 16. Fontes do vídeos

- Caso 001 - Vídeo 4 - 23º TEMFC
  Link: https://www.youtube.com/watch?v=9aAqDc8YOyQ

- Caso 002 - Vídeo 5 - TEMFC 25
  Link: https://www.youtube.com/watch?v=aa7oRbmmdvE

- Caso 003 - Vídeo 6 - TEMFC 25
  Link: https://www.youtube.com/watch?v=8gzBMd2pmQA

- Caso 004 - 20° TEMFC Vídeo 1
  Link: https://www.youtube.com/watch?v=ti2-wH3msrM

- Caso 005 - Letramento em Saúde: exemplo do que NÃO fazer em uma consulta médica!
  Link: https://www.youtube.com/watch?v=UGP6Gwb7oLk

- Caso 006 - 20° TEMFC Vídeo 4
  Linl: https://www.youtube.com/watch?v=RG8Y28FBqFc
