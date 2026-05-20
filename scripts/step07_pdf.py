from pathlib import Path
import os
import re

from dotenv import load_dotenv

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4

load_dotenv()

DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED"))

# =========================
# ESTILOS
# =========================

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "CustomTitle",
    parent=styles["Title"],
    fontSize=24,
    leading=30,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#1F3A5F"),
    spaceAfter=30,
)

h2_style = ParagraphStyle(
    "CustomH2",
    parent=styles["Heading2"],
    fontSize=16,
    leading=22,
    textColor=colors.HexColor("#1F3A5F"),
    spaceBefore=12,
    spaceAfter=12,
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontSize=11,
    leading=18,
    alignment=TA_JUSTIFY,
)

bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    alignment=TA_JUSTIFY,
    leftIndent=15,
    bulletIndent=5,
)

# =========================
# LEITURA
# =========================

def ler_markdown(caminho):

    with open(
        caminho,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()

# =========================
# TABELA
# =========================

def criar_tabela(dados):

    tabela = Table(
        dados,
        colWidths=[240, 180]
    )

    tabela.setStyle(TableStyle([

        # HEADER
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#1F3A5F")),

        ("TEXTCOLOR", (0, 0), (-1, 0),
         colors.white),

        ("FONTNAME", (0, 0), (-1, 0),
         "Helvetica-Bold"),

        ("FONTSIZE", (0, 0), (-1, 0),
         11),

        ("BOTTOMPADDING", (0, 0), (-1, 0),
         10),

        ("TOPPADDING", (0, 0), (-1, 0),
         10),

        # BODY
        ("BACKGROUND", (0, 1), (-1, -1),
         colors.whitesmoke),

        ("GRID", (0, 0), (-1, -1),
         1, colors.HexColor("#CCCCCC")),

        ("FONTNAME", (0, 1), (-1, -1),
         "Helvetica"),

        ("FONTSIZE", (0, 1), (-1, -1),
         10),

        ("BOTTOMPADDING", (0, 1), (-1, -1),
         8),

        ("TOPPADDING", (0, 1), (-1, -1),
         8),

    ]))

    return tabela

# =========================
# MARKDOWN -> PDF
# =========================

def markdown_para_pdf(
    markdown_text,
    output_pdf
):

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    elementos = []
    linhas = markdown_text.split("\n")
    tabela_buffer = []

    for linha in linhas:
        linha = linha.strip()

        if not linha:
            if tabela_buffer:
                elementos.append(criar_tabela(tabela_buffer))
                tabela_buffer = []
                elementos.append(Spacer(1, 20))
            elementos.append(Spacer(1, 8))
            continue

        if linha.startswith("|"):
            partes = [
                p.strip()
                for p in linha.split("|")[1:-1]
            ]

            # IGNORA LINHA ---
            if all(
                set(p) <= {"-", ":"}
                for p in partes
            ):
                continue

            tabela_buffer.append(partes)
            continue

        else:
            if tabela_buffer:
                elementos.append(criar_tabela(tabela_buffer))
                tabela_buffer = []
                elementos.append(Spacer(1, 20))

        if linha.startswith("# "):
            texto = linha.replace("# ", "")
            elementos.append(
                Paragraph(
                    texto,
                    title_style
                )
            )
            continue

        if linha.startswith("## "):
            texto = linha.replace("## ", "")
            elementos.append(
                Paragraph(
                    texto,
                    h2_style
                )
            )
            continue

        if linha.startswith("---"):
            elementos.append(Spacer(1, 10))
            elementos.append(
                HRFlowable(
                    width="100%",
                    color=colors.HexColor("#CCCCCC")
                )
            )
            elementos.append(
                Spacer(1, 10)
            )
            continue

        if linha.startswith("- "):
            texto = linha.replace("- ", "• ")
            texto = re.sub(
                r"\*\*(.*?)\*\*",
                r"<b>\1</b>",
                texto
            )
            elementos.append(
                Paragraph(
                    texto,
                    bullet_style
                )
            )
            continue

        linha = re.sub(
            r"\*\*(.*?)\*\*",
            r"<b>\1</b>",
            linha
        )
        elementos.append(
            Paragraph(
                linha,
                body_style
            )
        )

    if tabela_buffer:
        elementos.append(criar_tabela(tabela_buffer))

    doc.build(elementos)
    print(f"✔ PDF gerado: {output_pdf}")

# =========================
# PROCESSAMENTO
# =========================

def processar_caso(caso_info):
    """
    Responsável pela geração do relatório final em PDF.
    O script converte o relatório multimodal em markdown para um documento PDF estruturado e formatado institucionalmente, utilizando ReportLab.
    Implementa estilos tipográficos, tabelas, títulos, listas e organização visual do conteúdo, produzindo um relatório executivo consolidado da auditoria multimodal realizada pelo Sistema Hórus.
    """

    caso_id = caso_info["caso_id"]
    pasta_processada = (caso_info["processed_path"])

    print("\n======================")
    print(f"Gerando PDF: {caso_id}")

    markdown_path = (pasta_processada/ "relatorio_final"/ "relatorio_final.md")

    pdf_path = (pasta_processada / "relatorio_final" / "relatorio_final.pdf")

    if not markdown_path.exists():
        print("❌ Markdown não encontrado")
        return

    markdown_text = ler_markdown(markdown_path)

    markdown_para_pdf(markdown_text,pdf_path)