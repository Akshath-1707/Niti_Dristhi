# NITI DRISHTI: Project Changes & Tracking Log

> This document tracks all user-requested changes, modifications, feature updates, and verification status.
> **Status:** Planning & Review phase. Implementation will start ONLY upon explicit user approval.

---

## 🏛️ 3-Tier Multi-Persona System Architecture (Approved Blueprint)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          TIER 1: PUBLIC CITIZEN DASHBOARD                              │
│                          (Accessible by Default — No Login)                            │
│  • Public district transparency overview (Enrollment, Schools, Dropout Funnel, Radar) │
│  • School Finder & Public Facility Inspection                                          │
│  • Citizen-Friendly Explanatory RAG: Plain-English explanations of data & charts      │
│  • Top Navigation: [Login for School / Government Administrator]                       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Authenticate
                                            ▼
                        ┌───────────────────────────────────────┐
                        │       PORTAL AUTHENTICATION MODAL     │
                        │   [School Admin]   |   [Govt Admin]   │
                        └───────────────────┬───────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│     TIER 2: SCHOOL ADMINISTRATOR      │ │       TIER 3: GOVERNMENT ADMINISTRATOR        │
│          (Principal / HM)             │ │       (District Collector / DEO / NITI)       │
│  • School Data Entry & Update Form    │ │  • Complete District Macro Analytics & KPIs   │
│  • Add / Edit / Remove School Records │ │  • "What-If" Policy Simulation Sandbox        │
│  • Institutional Diagnostic Passport  │ │  • Dynamic Live 2037 Forecasting Engine       │
│  • Single-School PDF Inspection Card  │ │  • Statutory RAG Policy Copilot (NEP/RTE)     │
│  • Instant Impact Preview on Deficit  │ │  • Executive Municipal Directive PDF Export   │
└───────────────────────────────────────┘ └───────────────────────────────────────────────┘
```

---

## 📋 Comprehensive Change Requests & Specifications

| # | Persona / Component | Feature Description | Status |
| :-: | :--- | :--- | :-: |
| **CR-01** | **Tier 1: Public Citizen View** | • Open-access landing dashboard (no login barrier).<br>• Transparent view of district schools, drop-out funnel, and radar.<br>• School Search & Facility Passport.<br>• **Citizen-Friendly RAG Guide**: Explains charts and data in simple, jargon-free language. | ✅ **Implemented & Verified** |
| **CR-02** | **Portal Navigation Switcher** | • Clean, prominent "ACTIVE PORTAL VIEW" top navigation bar.<br>• 1-click seamless switching between Public Citizen, School Admin, and Government Admin views. | ✅ **Implemented & Verified** |
| **CR-03** | **Tier 2: School Administrator** | • **Data Management Form**: Pre-fill or add custom school metrics (enrolment, boys/girls, functional classrooms, teachers, pass rates, pending students).<br>• **Data Quality & Completeness Auditor**: Automatic completeness score (0–100%), confidence degradation labels, and domain validity checks.<br>• **School Diagnostic Scorecard**: Exact Teacher Gap ($\Delta_T$), Classroom Gap ($\Delta_C$), and Tri-Phased Action Plan. | ✅ **Implemented & Verified** |
| **CR-04** | **Tier 3: Government Administrator** | • **Deep Macro Analytics**: URC-1/URC-2 comparisons, resource allocation matrix.<br>• **"What-If" Simulation Sandbox**: Sliders to test capital classroom additions and recalibrate statutory weights with live recalculation.<br>• **Statutory RAG Copilot**: Generates binding administrative directives citing NEP 2020 §3.1, RTE Act §19, and Samagra Shiksha.<br>• **Executive Policy PDF**: Full municipal brief export via ReportLab. | ✅ **Implemented & Verified** |
| **CR-05** | **Cognitive Gap & XAI Reasoning Engine** | • Exact physical unit deficits ($\Delta_T, \Delta_C$).<br>• Dominant Primary Bottleneck Isolation.<br>• 8 Declarative Inspectable Statutory Rules.<br>• Causal "WHY" Factor Checklist.<br>• Plain-English Citizen Explanations. | ✅ **Implemented & Verified** |
| **CR-06** | **Dual-Mode Context-Aware RAG** | • **Citizen Mode**: Explains what the data means simply without complex administrative jargon.<br>• **Administrator Mode**: Drafts rigorous legal/statutory policy directives with budget and timeline breakdowns. | ✅ **Implemented & Verified** |
| **CR-07** | **Interactive Infrastructure GIS Map** | • **Full Geospatial Ward Mapping**: Plots all 954 institutions across authentic municipal centroids of Chhatrapati Sambhajinagar (URC-1 & URC-2).<br>• **Priority Pin Sizing & Colors**: Red (Critical), Amber (Developing), Green (Optimal) with marker radius scaling by student volume.<br>• **Click-to-Dossier Integration**: Clicking any pin on the map instantly loads that school's full Explainable Decision Dossier. | ✅ **Implemented & Verified** |

---

## 🧪 Verification & Academic Benchmark Test Results

| Test Scenario | Module Tested | Input | Observed Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **1. Largest School Diagnostic** | `gaps.py`, `explanation.py` | Deogiri Mahavidyalya (UDISE: 27191004128) | Identified `Teacher Gap: +25`, `Classroom Gap: 0`, Dominant Bottleneck: `Severe Teacher Deficit`. Outputted Causal WHY factors & Citizen plain-English text. | ✅ Passed |
| **2. Data Quality & Domain Validator** | `data_quality.py` | `Boys (150) + Girls (150) > Enrolment (200)` and `Promo Rate = 120%` | Flagged domain violations, assigned `Reduced Confidence`, and set `is_valid_for_submission = False`. | ✅ Passed |
| **3. What-If Capital Simulation** | `app.py` Callback | Sliders: `+500 Classrooms`, weights recalibrated | Dynamically reduced 2037 classroom deficit and live-recalculated District School Score. | ✅ Passed |
| **4. App Integrity Check** | `app.py` | Full import test | Zero syntax, callback, or layout errors. Clean initialization on port 8050. | ✅ Passed |

