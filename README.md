# Sistema Hórus

## Sistema Multiagente de Auditoria Protocolar Multimodal para Consultas de Saúde da Mulher

## Pós-Tech FIAP — IA para Devs

### Tech Challenge — Fase 4

---

# 1. Visão Geral do Projeto

## Nome do Projeto

**Sistema Hórus - Sistema Multiagente de Auditoria Protocolar Multimodal para Consultas de Saúde da Mulher **

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

O Sistema Hórus utiliza uma arquitetura multiagente protocolar multimodal.

A solução combina:

- modelos tradicionais de IA;
- modelos generativos;
- protocolos estruturados;
- fusão determinística de scores;
- auditoria multimodal.

A arquitetura foi organizada em agentes independentes especializados, permitindo separação entre:

- percepção multimodal;
- interpretação semântica;
- avaliação protocolar;
- fusão de resultados.

---

## Arquitetura

```
                         +----------------------+
                         |   Upload do Caso     |
                         +----------+-----------+
                                    |
                                    v

                  +-----------------------------------+
                  | Agente de Ingestão                |
                  +-----------------------------------+
                  | - valida arquivos                 |
                  | - organiza diretórios             |
                  | - estrutura o caso                |
                  +----------------+------------------+
                                   |
         +-------------------------+--------------------------+--------------------------+
         |                         |                          |                          |
         v                         v                          v                          v

+-------------------+   +-------------------+   +-------------------+   +-------------------+
| Agente            |   | Agente            |   | Agente            |   | Agente            |
| Prontuário        |   | Comunicação       |   | Vocal/Emocional   |   | Vídeo             |
+-------------------+   +-------------------+   +-------------------+   +-------------------+
| - leitura PDF     |   | - extração áudio  |   | - pitch vocal     |   | - extração frames |
| - auditoria doc   |   | - transcrição     |   | - energia         |   | - DeepFace        |
| - protocolo doc   |   | - auditoria texto |   | - ritmo           |   | - YOLOv8          |
| - score clínico   |   | - protocolo com.  |   | - tensão vocal    |   | - postura corporal|
+-------------------+   +-------------------+   +-------------------+   +-------------------+

                                    |
                                    v

                     +-----------------------------+
                     | Agente Fusionador           |
                     +-----------------------------+
                     | - consolida scores          |
                     | - integra análises          |
                     | - classificação final       |
                     | - revisão humana            |
                     +-----------------------------+

                                    |
                                    v

                     +-----------------------------+
                     | Relatório Final             |
                     +-----------------------------+
                     | - markdown                  |
                     | - PDF                       |
                     | - score final               |
                     | - auditoria multimodal      |
                     +-----------------------------+
```

---

# 6. Estrutura de Pastas

```
/project
│
├── /data_raw
│   ├── caso_001
│   │   ├── consulta001.mp4
│   │   ├── caso001.pdf
│   └── caso_002
│   │   ├── consulta002.mp4
│   │   ├── caso002.pdf
│   └── caso_003
│   │   ├── consulta003.mp4
│   │   ├── caso003.pdf
│   └── caso_004
│   │   ├── consulta004.mp4
│   │   ├── caso004.pdf
│   └── caso_005
│   │   ├── consulta005.mp4
│   │   ├── caso005.pdf
│   └── caso_006
│       ├── consulta006.mp4
│       ├── caso006.pdf
│
├── /data_processed
│
├── /protocolos
│   ├── protocolo_comunicacao_vocal.md
│   ├── protocolo_consulta_geral.md
│   └── protocolo_postura_comunicacao_naoverbal.md
│   └── protocolo_prontuario.md
│
├── /scripts
│   ├── step01_ingestao.py
│   ├── step02_prontuario.py
│   ├── step03_comunicacao.py
│   ├── step04_vocal.py
│   ├── step05_video.py
│   ├── step06_fusao.py
│   └── step07_pdf.py
│
├── requirements.txt
│
├── README.md
│
└── main.py
```

---

# 7. Tecnologias Utilizadas

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

# 8. Limitações

O sistema:

- não substitui profissionais de saúde;
- não realiza diagnóstico clínico;
- não confirma violência doméstica;
- atua apenas como apoio técnico.

---

# 9. Possíveis Evoluções Futuras

- integração com Azure Cognitive Services;
- monitoramento em tempo real;
- análise de múltiplos participantes;
- modelos especializados treinados;
- integração hospitalar;
- autenticação e prontuário eletrônico.

---

# 10. Resultados Esperados

Espera-se que o sistema:

- reduza tempo de auditoria;
- identifique consultas atípicas;
- aumente aderência protocolar;
- ajude equipes hospitalares;
- demonstre o potencial da IA multimodal na saúde.

---

# 11. Conclusão

O projeto propõe um sistema multimodal baseado em Inteligência Artificial para auditoria protocolar de consultas voltadas à saúde da mulher.

A solução integra:

- visão computacional;
- processamento de áudio;
- análise textual;
- modelos generativos;
- recuperação semântica de protocolos clínicos.

O sistema busca apoiar equipes hospitalares na identificação precoce de situações de risco, desconforto psicológico e não conformidades clínicas, demonstrando o potencial da IA aplicada à saúde digital.

# 12. Fontes do vídeos

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
  Link: https://www.youtube.com/watch?v=RG8Y28FBqFc
