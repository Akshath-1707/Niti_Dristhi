# 🏛️ NITI DRISHTI: AI-Powered Infrastructure Decision Support System

> **B.Tech Final Year Project in Data Science**  
> **Target District:** Chhatrapati Sambhajinagar (URC-1 & URC-2 Urban Core)  
> **Horizon Target:** Vision 2037 • **Empirical Data Window:** 2021–2025  

---

## 📌 Executive Summary

**NITI DRISHTI** is an AI-driven, municipal decision-support system (DSS) designed to evaluate urban infrastructure adequacy, calculate a multi-pillar **Infrastructure Readiness Index (0–100 scale)**, forecast demographic capacity deficits through **2037**, and synthesize grounded statutory policy directives via a **Retrieval-Augmented Generation (RAG)** copilot over government planning documents (NEP 2020, RTE Act 2009, Samagra Shiksha Framework 2024).

---

## 📐 Mathematical Formulation: Education Readiness Index (IRI)

The composite district readiness score evaluates **954 educational institutions** across four key statutory pillars:

$$\text{IRI}_{\text{District}} = \sum_{k=1}^{4} w_k \cdot \text{Pillar}_k$$

Where the statutory weight allocation is configured as:

| Pillar | Weight ($w_k$) | Key Indicators Measured | Target Benchmark |
| :--- | :---: | :--- | :--- |
| **1. Equity & Inclusivity** | **$25\%$** | Gender Parity Index ($GPI = \text{Girls}/\text{Boys}$), Female Enrolment Ratio | Optimal $GPI = 1.00$ |
| **2. Promotion & Remedial** | **$30\%$** | Empirical URC Pass Rate vs. Pending Remedial Backlog ($27,235$ students) | Pass Rate $\ge 90\%$, Zero Backlog |
| **3. Cohort Retention** | **$25\%$** | Upper Primary (Classes 6–8) $\to$ Secondary (Classes 9–10) Transition Ratio | Transition Ratio $\ge 0.67$ |
| **4. Operational & Capacity** | **$20\%$** | Institutional operational status and RTE Pupil-Teacher Ratio compliance | PTR $\le 30:1$, Active Facilities |

### Priority Tier Classification
- 🔴 **Tier 1 (Critical Intervention)**: Readiness Score $< 55.0$ *(Immediate administrative focus & funding)*
- 🟡 **Tier 2 (Developing Needs)**: Readiness Score $55.0 - 74.9$ *(Targeted capacity expansion)*
- 🟢 **Tier 3 (Optimal / Resilient)**: Readiness Score $\ge 75.0$ *(High operational resilience)*

---

## 📈 Horizon 2037 Predictive Forecasting Engine

Utilizes time-series demographic trend modeling (Scikit-Learn Huber / Prophet models) across the 5-year historical trajectory (2021–2025) to project:
- Total student population growth up to **2037** with $85\%$ confidence bounds.
- Class-wise cohort volumes (Primary, Upper Primary, Secondary, Higher Secondary).
- Statutory classroom requirements and cumulative capacity deficits based on RTE standards ($30:1$ ratio).

---

## 🤖 Dual-Engine RAG Policy Copilot

Grounds large language models on empirical district data and official statutory PDF documents (`backend/rag/docs/`):
- **Retrieval Engine**: FAISS vector store with `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- **LLM Synthesis**: Google Gemini (`gemini-3.6-flash`) with automatic fallback to local **Ollama (`llama3.1:8b`)** or deterministic statutory rule synthesis.
- **Reasoning Trace Animation**: Renders step-by-step thinking logs, statutory source citations, and structured implementation roadmaps (Immediate 0–6M, Medium 6–24M, Vision 2037).

---

## 🗂️ Project Directory Structure

```text
Niti_Dristhi/
├── backend/
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── readiness_index.py      # Multi-pillar 0-100 Readiness Index Engine
│   │   └── forecasting.py          # 2021-2037 Time-Series Demographic Forecast
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── service.py              # Dual-Engine Grounded RAG Copilot
│   │   ├── build_index.py          # Vectorstore indexing pipeline
│   │   ├── docs/                   # Statutory Policy PDFs (NEP 2020, RTE Act, PM SHRI)
│   │   └── vectorstore/           # FAISS Vector Index
│   └── requirements.txt
├── data/
│   ├── raw/                        # Original Promotion PDFs & Raw Excel spreadsheets
│   ├── processed/                  # Clean school master & feature datasets
│   ├── timeseries/                 # Back-casted 2021-2025 & Forecast 2037 CSVs
│   └── transport/                  # Pilot transport baseline dataset
├── scripts/
│   ├── process_schools.py          # PDF table extractor & feature generation
│   ├── data_loader.py              # Safe path resolution utilities
│   └── test_rag_pipeline.py        # Independent RAG verification test
├── reports/
│   └── verified_policy_report.md   # Grounded statutory policy deliverable
├── app.py                          # Flagship Obsidian & Deep Teal Executive Dashboard
├── requirements.txt                # Root environment dependencies
├── .env                            # API Keys configuration
└── README.md                       # Master project documentation
```

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Launch the Flagship Dashboard
Run the executive dashboard:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8050/
```

### 3. Run RAG Copilot Verification (Optional)
To test the standalone RAG engine:
```powershell
python backend/rag/service.py
```
