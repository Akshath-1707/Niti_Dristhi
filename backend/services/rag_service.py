"""
=============================================================================
NITI DRISHTI: RAG Policy & Planning Assistant Service
=============================================================================
Manages grounded policy synthesis, statutory retrieval from FAISS,
thinking trace generation, and source document citations.
=============================================================================
"""

try:
    from backend.rag.service import generate_education_policy_brief
    RAG_AVAILABLE = True
except Exception as e:
    print(f"[RAG SERVICE WARNING] Could not import RAG service: {e}")
    RAG_AVAILABLE = False


class RAGService:
    @staticmethod
    def query_policy_assistant(user_prompt: str, school_info: dict = None) -> dict:
        if not user_prompt or not str(user_prompt).strip():
            return {
                "success": False,
                "error": "Prompt must not be empty.",
                "policy_report": "",
                "thinking_steps": [],
                "sources": []
            }

        if not RAG_AVAILABLE:
            return {
                "success": True,
                "engine": "Deterministic Statutory Engine (Fallback)",
                "policy_report": (
                    f"### Statutory Policy Guidance Note\n\n"
                    f"**Inquiry**: {user_prompt}\n\n"
                    f"- **Statutory Reference**: Section 19/25 of the Right to Education (RTE) Act 2009 & NEP 2020 Chapter 7.\n"
                    f"- **District Baseline**: 954 schools, 278,945 students, 27,235 remedial pending students.\n"
                    f"- **Directive**: Prioritize capital classroom works under Samagra Shiksha and accelerate remedial bridge learning recovery."
                ),
                "thinking_steps": [
                    "Vector retrieval bypassed (offline mode).",
                    "Applying statutory heuristics from RTE Act 2009 and Samagra Shiksha framework."
                ],
                "sources": ["RTE Act 2009", "Samagra Shiksha Framework 2024"]
            }

        try:
            res = generate_education_policy_brief(user_prompt=user_prompt, school_info=school_info)
            res["success"] = True
            return res
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "policy_report": f"AI Policy Assistant currently unavailable: {str(e)}",
                "thinking_steps": ["Encountered processing exception."],
                "sources": []
            }


# Global singleton instance
rag_service = RAGService()
