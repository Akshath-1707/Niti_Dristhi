# NITI DRISHTI: National Infrastructure Readiness Index
### A Multi-Persona Government Decision-Support System for District Education Infrastructure Planning

---

## 1. Project Overview & Executive Summary

**NITI DRISHTI** is an AI-powered, multi-persona **Government Decision Support System (DSS)** engineered for infrastructure capacity planning, statutory compliance verification, and longitudinal demographic forecasting in India's public education sector.

Centering on empirical district data from **Chhatrapati Sambhajinagar (Aurangabad), Maharashtra**, the platform analyzes **954 educational institutions** across Urban Resource Centers **URC-1 and URC-2** spanning historical actuals (**2021–2025**) and projects infrastructural stress up to **Horizon 2037–2040**.

The system translates raw UDISE+ metrics and legal mandates—specifically the **Right of Children to Free and Compulsory Education (RTE) Act 2009**, the **National Education Policy (NEP) 2020**, and the **Samagra Shiksha Framework**—into role-tailored decision intelligence across three isolated stakeholders:
1. **Public Citizens**: Community transparency, plain-language insights, and district GIS visualization.
2. **Educational Institutions (Principals / Admins)**: Confidential institutional diagnostic audits, 30:1 PTR compliance, and 3-step internal remedial roadmaps.
3. **Government Authorities (District Planning Officers / Collectors)**: "What-If" macro-simulation sandbox, capital infrastructure allocation, official school inspection console, and statutory policy generation.

---

## 2. Technical Architecture & System Design

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    FRONTEND LAYER                      │
                               │  Plotly Dash 4.x, HTML5, Custom CSS3 Design System     │
                               │  Interactive Choropleths, KaTeX, ReportLab PDF Export  │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                    BACKEND SERVICES                    │
                               │   FastAPI / Flask ASGI Engine + Modular Services       │
                               └──────────┬──────────────────┬───────────────────┬──────┘
                                          │                  │                   │
                                          ▼                  ▼                   ▼
    ┌───────────────────────────────────────────┐  ┌───────────────────┐  ┌───────────────────────┐
    │          RAG AI POLICY ENGINE             │  │ FORECAST SERVICE  │  │   SCENARIO SERVICE    │
    │  • FAISS / TF-IDF Vector Search           │  │ • 2021–2040 Cohort│  │ • Multi-Criteria DSS  │
    │  • Google Gemini API integration          │  │   Demographic     │  │ • Dynamic Weights     │
    │  • Statutory Corpus (NEP/RTE/Samagra)     │  │   Envelope        │  │ • RTE 30:1 Deficit    │
    │  • Persona-Conditioned Synthesis          │  │ • Net Deficits    │  │ • CapEx Simulation    │
    └───────────────────────────────────────────┘  └───────────────────┘  └───────────────────────┘
                                          │                  │                   │
                                          └──────────────────┼───────────────────┘
                                                             ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                      DATA LAYER                        │
                               │  • Master Dataset: master_school_dataset.csv (954 inst)│
                               │  • Forecast Dataset: student_forecast_2021_2040.csv    │
                               │  • Spatial Geometries: URC-1 & URC-2 GeoJSON           │
                               │  • Statutory Policy Markdown Documents                 │
                               └────────────────────────────────────────────────────────┘
```

### Technology Stack Specifications
- **Application Framework**: Plotly Dash 4.4.1 (Flask/ASGI-backed).
- **Programming Language**: Python 3.11+.
- **Design System**: Strict compliance with Indian Government Web Guidelines (`assets/style.css`), featuring Deep Navy (`#123B63`), Government Blue (`#1E5A92`), Pine, Ochre, and accessible typography.
- **Geospatial & Visualization**: Plotly Graph Objects (`Scattermapbox` / `Scattergeo` GIS choropleth, radar spider charts, funnel cohort diagrams, radial gauges).
- **Data Engineering**: Pandas, NumPy, Scikit-Learn.
- **Generative AI & RAG**: Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-pro`) integrated with a local vector retriever indexing official policy acts.
- **Document Generation**: ReportLab PDF Generation engine for official stamped directive documents.

---

## 3. Four-Pillar Infrastructure Readiness Framework

NITI DRISHTI scores educational quality and infrastructural adequacy on a composite **0–100 scale** using four statutory pillars:

$$\text{Readiness Score} = w_1 \cdot P_{\text{Equity}} + w_2 \cdot P_{\text{Promotion}} + w_3 \cdot P_{\text{Retention}} + w_4 \cdot P_{\text{Capacity}}$$

Where default statutory weights are:
- $w_1 = 0.25$ (Gender Equity & Inclusion)
- $w_2 = 0.30$ (Pass Rate & Academic Quality)
- $w_3 = 0.25$ (Cohort Retention & Transition)
- $w_4 = 0.20$ (Classroom & Faculty Capacity)

### Pillar Descriptions
1. **Pillar 1: Gender Equity ($P_{\text{Equity}}$)**:
   - Evaluates the Gender Parity Index ($\text{GPI} = \frac{\text{Total Girls}}{\text{Total Boys}}$) against the target of 1.00.
   - Audits dedicated, functional female sanitation infrastructure and safety compliance.
2. **Pillar 2: Academic Promotion & Remedial Learning ($P_{\text{Promotion}}$)**:
   - Measures annual promotion pass rates.
   - Identifies non-promoted student volumes requiring targeted remedial bridge courses.
3. **Pillar 3: Cohort Retention & Secondary Transition ($P_{\text{Retention}}$)**:
   - Tracks cohort survival from Primary (Classes 1–5) to Higher Secondary (Classes 11–12).
   - Diagnoses the critical **38.4% drop-off** that occurs between Upper Primary (Class 8) and Secondary (Class 9).
4. **Pillar 4: Physical Infrastructure & PTR Capacity ($P_{\text{Capacity}}$)**:
   - Evaluates functional classrooms against total student enrollment under the **RTE Act statutory ceiling of 30:1 Student-Teacher / Student-Classroom Ratio**.
   - Highlights severe overcrowding ($\text{PTR} > 35:1$) versus optimal capacity.

---

## 4. Multi-Persona Portal Structure & Isolation

NITI DRISHTI enforces **Strict Persona Isolation** to maintain institutional privacy and prevent unauthorized access:

### Portal 1: Public Citizen Portal (Open Access)
- **Clearance Badge**: `PUBLIC CITIZEN VIEW` (Green Status Badge).
- **Key Modules**:
  - **4 Live KPI Indicators**: Operational institutions count, total enrolled student volume (with girls' census and GPI), pending remedial learner backlog (~27k district-wide), and cumulative RTE classroom gap (+2,388 rooms).
  - **Readiness Radial Gauge**: Responsive 0–100 score dial displaying operational health.
  - **Cohort Progression Funnel**: Visual tracking of enrollment drop-offs from Class 1 to Class 12.
  - **Longitudinal Forecast Envelope (2021–2040)**: Historical actuals with an 85% confidence band.
  - **Pillar Spider Radar**: 4-axis statutory compliance radar.
  - **Interactive Ward GIS Map**: Plots all 954 institutions across URC-1 and URC-2 with priority color codes.
  - **Citizen RAG Assistant**: Plain-language explanations of district educational metrics and government support programs.

---

### Portal 2: Institution Assessment Portal (Principals & School Admins)
- **Clearance Badge**: `INSTITUTION ASSESSMENT PORTAL` (Cyan Status Badge).
- **Authentication**: Institutional clearance required (Default credentials: `school_admin` / `school123`).
- **Key Modules**:
  - **Locked Institution Identity**: Authenticated school name and 11-digit UDISE code locked to the session.
  - **Target Milestone Selector**: Allows the school to simulate capacity stress for future years (2026, 2028, 2030, etc.).
  - **Private Diagnostic Form**: Fields for active enrollment, boy/girl split, functional classrooms, teaching staff, pass rate, and remedial backlog.
  - **Diagnostic Quality Audit**: Instantly calculates PTR, classroom deficit, faculty gap, priority tier, and formulates a **3-Step Institutional Action Plan** (Immediate 0–90 days, Medium 6–24 months, Strategic long-term).
  - **School Action Copilot (RAG)**: Generates internal school policies, sanitation compliance roadmaps, and remedial bridge course schedules.

---

### Portal 3: Government Authority Portal (District Collectors & Planners)
- **Clearance Badge**: `GOVERNMENT SCENARIO ANALYSIS` (Amber Status Badge).
- **Authentication**: Administrative clearance required (Default credentials: `govt_admin` / `govt123`).
- **Key Modules**:
  - **"What-If" Simulation Sandbox**:
    - **Planning Horizon Slider**: Adjusts target milestone between 2026 and 2040.
    - **Planned New Classrooms Slider**: Simulates building 0 to 1,000 classrooms to test deficit resolution percentages.
    - **Multi-Criteria Weight Recalibration**: Dynamically shifts weights across the 4 pillars to simulate policy reprioritizations.
  - **Executive Infrastructure Intelligence Dossier**: Summarizes district population, RTE classroom gaps, planned room resolutions, and faculty hiring demands.
  - **Official School Search & Inspection Console**: Allows officers to search any institution across the district to inspect its complete profile, PTR compliance, causal factors, and action plan.
  - **Macro Policy RAG**: Drafts administrative master plans, composite secondary complex directives (NEP 2020 §3.1), and downloadable signed PDFs.

---

## 5. Persona-Conditioned RAG (Retrieval-Augmented Generation)

```
User Inquiry / Executive Directive Preset
                   │
                   ▼
       [Persona Classification]
       ├── Citizen: Public, jargon-free, supportive explanation
       ├── School: Internal PTR compliance & remedial action plan
       └── Govt: Macro-resource master plan & statutory mandate
                   │
                   ▼
     [Statutory Corpus Retrieval]
     ├── RTE Act 2009 (Schedule §19/25: 30:1 PTR, sanitation norms)
     ├── NEP 2020 (§3.1: Secondary drop-off, composite school complexes)
     └── Samagra Shiksha (Vocational integration & remedial grants)
                   │
                   ▼
       [Numerical Grounding]
       Injects empirical district data, school gaps, or simulation values
                   │
                   ▼
       [Gemini Generative Engine]
       Synthesizes formal directive with reasoning trace & cited legal acts
                   │
                   ▼
       [ReportLab PDF Engine]
       Exports stamped, formatted official government directive PDF
```

---

## 6. Directory Structure

```
Niti_Dristhi/
├── app.py                             # Main Dash Application (Layouts, Routing, Callbacks)
├── assets/
│   ├── style.css                      # Official Indian Government Portal Design System
│   └── favicon.ico                    # Platform Emblem / Favicon
├── backend/
│   ├── rag/
│   │   ├── service.py                 # RAG Orchestration, Prompt Engineering, Gemini Bridge
│   │   └── corpus/                    # Statutory Policy Corpus (RTE, NEP, Samagra Shiksha)
│   └── services/
│       ├── forecast_service.py        # 2021–2040 Cohort Demographic Forecasting
│       ├── institution_service.py     # Institutional Profile, Gap Analysis & Action Plans
│       └── scenario_service.py        # "What-If" Multi-Criteria Simulation Engine
├── data/
│   ├── master_school_dataset.csv      # Empirical District Master Dataset (954 institutions)
│   └── student_forecast_2021_2040.csv # Longitudinal Demographic Projections
└── requirements.txt                   # Production Dependency Specifications
```

---

## 7. Setup & Execution Guide

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv` or `conda`)
- Active Google Gemini API Key (optional; platform includes heuristic fallback)

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/your-username/Niti_Dristhi.git
cd Niti_Dristhi

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set Gemini API Key (Optional)
export GEMINI_API_KEY="your_api_key_here"  # Linux / macOS
$env:GEMINI_API_KEY="your_api_key_here"    # Windows PowerShell

# 5. Launch the application
python app.py
```
Access the application at `http://127.0.0.1:8050/`.

---

## 8. Summary for LLM Analysis (Copy-Paste Prompts)

When discussing this project with ChatGPT or Gemini, use the following prompts:
1. *"Explain how NITI DRISHTI models the Right to Education (RTE) Act 2009 30:1 Student-Teacher Ratio to forecast district-wide classroom shortfalls."*
2. *"How does the multi-persona architecture in NITI DRISHTI prevent unauthorized data access between citizens, school principals, and government authorities?"*
3. *"Analyze the 4-pillar readiness score formula in NITI DRISHTI and explain how dynamic weight recalibration helps district planners balance competing educational priorities."*
