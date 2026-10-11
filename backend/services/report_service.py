"""
=============================================================================
NITI DRISHTI: Policy Report & PDF Export Service
=============================================================================
Converts policy briefs and district assessments into publication-grade PDFs.
=============================================================================
"""

import io
from backend.rag.pdf_exporter import generate_policy_pdf


class ReportService:
    @staticmethod
    def generate_pdf_buffer(report_markdown: str, filename: str = "NitiDrishti_Official_Policy_Directive.pdf") -> io.BytesIO:
        if not report_markdown or not str(report_markdown).strip():
            report_markdown = "# NITI DRISHTI: District Infrastructure Assessment Brief\nNo content provided."
        return generate_policy_pdf(report_markdown, filename=filename)


# Global singleton instance
report_service = ReportService()
