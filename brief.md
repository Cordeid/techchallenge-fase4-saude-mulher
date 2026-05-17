# Sistema HORUS

## Auditoria Multimodal Inteligente para Consultas de Saúde da Mulher

---

# 1. Visão Geral do Projeto

O Sistema HORUS é uma plataforma experimental de inteligência artificial multimodal desenvolvida para apoiar auditorias técnicas de consultas médicas em contextos de saúde da mulher.

O projeto foi concebido como resposta ao desafio crescente de garantir:

- qualidade assistencial;
- aderência protocolar;
- humanização do atendimento;
- identificação precoce de sinais de desconforto emocional ou violência psicológica durante consultas clínicas.

A solução utiliza uma arquitetura baseada em múltiplos agentes especializados capazes de analisar simultaneamente:

- vídeo;
- áudio;
- transcrição textual;
- emoções;
- protocolos clínicos;
- evidências multimodais.

O objetivo principal não é substituir profissionais humanos, mas atuar como um sistema de apoio à auditoria clínica, permitindo que equipes técnicas identifiquem casos prioritários para revisão especializada.

---

# 2. Problema

Hospitais e clínicas frequentemente enfrentam dificuldades para realizar auditorias qualitativas em consultas médicas devido a fatores como:

- alto volume de atendimentos;
- limitação de equipes técnicas;
- subjetividade das avaliações;
- dificuldade em revisar vídeos e áudios manualmente;
- ausência de mecanismos automatizados de aderência protocolar.

Além disso, sinais sutis de:

- desconforto emocional;
- medo;
- hesitação;
- ansiedade;
- comunicação inadequada;
- possível violência psicológica;

podem passar despercebidos em auditorias tradicionais.

O desafio proposto foi desenvolver uma solução de IA capaz de:

- interpretar diferentes modalidades de dados;
- consolidar evidências;
- comparar consultas com protocolos clínicos;
- gerar análises estruturadas e explicáveis.

---

# 3. Objetivo da Solução

O Sistema HORUS foi desenvolvido para:

- analisar consultas médicas multimodais;
- transcrever atendimentos automaticamente;
- detectar emoções e comportamentos;
- realizar auditoria protocolar via RAG;
- identificar possíveis sinais de atenção emocional;
- consolidar resultados de múltiplos agentes;
- gerar relatórios executivos automatizados.

---

# 4. Arquitetura Geral

O projeto utiliza uma arquitetura baseada em pipeline multiagente.

Cada agente possui responsabilidade específica dentro do fluxo de processamento.

```txt
                 ┌────────────────────┐
                 │   Vídeo da Consulta │
                 └─────────┬──────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Agente 1 - Ingestão    │
              └─────────┬──────────────┘
                        │
        ┌───────────────┴────────────────┐
        ▼                                 ▼
┌─────────────────┐             ┌─────────────────┐
│ Agente 2 Áudio  │             │ Agente 3 Vídeo │
└────────┬────────┘             └────────┬────────┘
         ▼                                ▼
  Transcrição IA                  YOLO + Emoções
         │                                │
         └──────────────┬─────────────────┘
                        ▼
              ┌────────────────────┐
              │ Agente 4 - RAG     │
              │ Protocolar         │
              └─────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ Agente 5 - Fusão   │
              │ Multimodal         │
              └─────────┬──────────┘
                        ▼
              Relatório Executivo Final
```

---

# 5. Agentes do Sistema

---

## Agente 1 — Ingestão de Casos

Responsável por:

- identificar casos;
- validar estrutura de arquivos;
- criar diretórios de processamento;
- gerar metadados iniciais;
- preparar o pipeline.

### Entrada

- vídeos `.mp4`
- documentos `.pdf`

### Saída

- estrutura organizada em `data_processed`

---

## Agente 2 — Análise de Áudio

Responsável por:

- extrair áudio do vídeo;
- dividir áudio em chunks;
- realizar transcrição automática;
- gerar resumo da consulta;
- analisar sinais emocionais no discurso.

### Tecnologias

- MoviePy
- SpeechRecognition
- OpenAI GPT-4o-mini

### Análises realizadas

- medo;
- ansiedade;
- hesitação;
- insegurança;
- desconforto psicológico.

---

## Agente 3 — Análise de Vídeo

Responsável por:

- extrair frames;
- detectar pessoas;
- analisar emoções faciais;
- identificar comportamentos visuais relevantes.

### Tecnologias

- YOLOv8
- OpenCV
- DeepFace

### Análises realizadas

- emoções predominantes;
- presença de múltiplas pessoas;
- possíveis sinais de tensão;
- linguagem corporal básica.

---

## Agente 4 — RAG Protocolar

Responsável por:

- carregar protocolos clínicos;
- gerar embeddings;
- criar base vetorial;
- recuperar contexto relevante;
- comparar consulta com protocolos.

### Tecnologias

- LangChain
- ChromaDB
- OpenAI Embeddings

### Objetivos

- medir aderência protocolar;
- detectar ausência de etapas obrigatórias;
- identificar inconsistências clínicas;
- gerar score de conformidade.

---

## Agente 5 — Fusionador Multimodal

Responsável por:

- consolidar resultados dos demais agentes;
- integrar evidências multimodais;
- gerar score final;
- produzir relatório executivo.

### Resultado final

- classificação de risco;
- alertas principais;
- evidências;
- recomendações;
- resumo executivo.

---

# 6. Tecnologias Utilizadas

| Categoria              | Tecnologia         |
| ---------------------- | ------------------ |
| Linguagem              | Python             |
| IA Generativa          | OpenAI GPT-4o-mini |
| Visão Computacional    | YOLOv8             |
| Emoções Faciais        | DeepFace           |
| Transcrição            | SpeechRecognition  |
| Processamento de Vídeo | OpenCV             |
| Processamento de Áudio | MoviePy            |
| RAG                    | LangChain          |
| Banco Vetorial         | ChromaDB           |
| Embeddings             | OpenAI Embeddings  |

---

# 7. Estrutura do Projeto

```txt
/data_raw
    caso001/
        caso001.mp4
        caso001.pdf

/data_processed
    caso001/
        audio/
        frames/
        transcricao/
        analise_video/
        analise_protocolar/
        relatorio_final/

/protocolos
    protocolo_saude_mulher.md

/scripts
    step01_ingestao.py
    step02_audio.py
    step03_video.py
    step04_rag.py
    step05_fusao.py

main.py
requirements.txt
README.md
.env
```

---

# 8. Pipeline de Execução

O sistema é executado de forma centralizada pelo arquivo:

```bash
python main.py
```

Fluxo:

1. ingestão;
2. análise de áudio;
3. análise de vídeo;
4. auditoria protocolar;
5. fusão multimodal;
6. geração de relatório final.

---

# 9. Exemplo de Resultado

O sistema produz relatórios contendo:

- score geral;
- score protocolar;
- score emocional;
- alertas prioritários;
- evidências identificadas;
- recomendação de revisão humana.

Exemplo:

- risco emocional elevado;
- ausência de avaliação emocional;
- sinais de ansiedade;
- hesitação verbal;
- predominância de emoções tristes.

---

# 10. Diferenciais do Projeto

O Sistema HORUS se diferencia por:

- abordagem multimodal;
- arquitetura multiagente;
- uso de RAG clínico;
- integração entre vídeo, áudio e protocolos;
- geração automatizada de relatórios executivos;
- foco em auditoria assistiva e não substitutiva.

---

# 11. Limitações

O projeto possui limitações importantes:

- não realiza diagnóstico clínico;
- não confirma violência doméstica;
- depende da qualidade do áudio e vídeo;
- utiliza modelos generalistas;
- ainda não possui validação clínica formal.

O sistema deve ser entendido como ferramenta de apoio técnico.

---

# 12. Possíveis Evoluções Futuras

- reconhecimento de fala local;
- análise temporal de emoções;
- detecção avançada de linguagem corporal;
- dashboard web;
- autenticação hospitalar;
- integração com prontuários eletrônicos;
- uso de LangGraph/CrewAI;
- modelos especializados em saúde.

---

# 13. Conclusão

O Sistema HORUS demonstra como arquiteturas multiagentes e IA multimodal podem apoiar auditorias clínicas modernas.

A solução integra:

- visão computacional;
- processamento de linguagem natural;
- embeddings;
- RAG;
- análise emocional;
- fusão multimodal;

para produzir análises estruturadas, explicáveis e assistivas.

O projeto evidencia o potencial da inteligência artificial como ferramenta de apoio à qualidade assistencial e à humanização do cuidado em saúde.
