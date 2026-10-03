"""
=============================================================================
NITI DRISHTI: Executive Policy Report PDF Generator
=============================================================================
Generates official municipal policy brief PDFs using ReportLab.
=============================================================================
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


def generate_policy_pdf(report_markdown: str, filename: str = "NitiDrishti_Executive_Policy_Directive.pdf") -> io.BytesIO:
    """
    Converts markdown policy brief text into an executive PDF document buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom styling
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#06B6D4'),
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=14,
        spaceAfter=6
    )

    h3_style = ParagraphStyle(
        'SectionH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    # Title & Header
    story.append(Paragraph("NITI DRISHTI | MUNICIPAL POLICY DIRECTIVE", title_style))
    story.append(Paragraph("CHHATRAPATI SAMBHAJINAGAR DISTRICT • URBAN RESOURCE CENTRES URC-1 & URC-2", subtitle_style))
    
    timestamp_str = datetime.now().strftime("%B %d, %Y • %H:%M IST")
    meta_text = f"<b>Document Classification:</b> OFFICIAL DECISION SUPPORT BRIEF &nbsp;|&nbsp; <b>Generated:</b> {timestamp_str} &nbsp;|&nbsp; <b>Statutory Baseline:</b> NEP 2020 / RTE 2009"
    story.append(Paragraph(meta_text, meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#06B6D4'), spaceAfter=14))

    # Parse markdown lines into story elements
    lines = report_markdown.split("\n")
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            story.append(Spacer(1, 4))
            continue

        if line_clean.startswith("# "):
            story.append(Paragraph(line_clean[2:], title_style))
        elif line_clean.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(line_clean[3:], h2_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
        elif line_clean.startswith("### "):
            story.append(Paragraph(line_clean[4:], h3_style))
        elif line_clean.startswith("- ") or line_clean.startswith("* "):
            formatted_bullet = "• " + line_clean[2:].replace("**", "<b>").replace("**", "</b>")
            story.append(Paragraph(formatted_bullet, bullet_style))
        elif line_clean.startswith("1. ") or line_clean.startswith("2. ") or line_clean.startswith("3. "):
            formatted_num = line_clean.replace("**", "<b>").replace("**", "</b>")
            story.append(Paragraph(formatted_num, bullet_style))
        elif line_clean.startswith("|") and line_clean.endswith("|"):
            # Table lines can be rendered or simplified
            continue
        elif line_clean.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=8))
        else:
            formatted_body = line_clean.replace("**", "<b>").replace("**", "</b>").replace("*", "<i>").replace("*", "</i>")
            story.append(Paragraph(formatted_body, body_style))

    # Build Document
    doc.build(story)
    buffer.seek(0)
    return buffer
