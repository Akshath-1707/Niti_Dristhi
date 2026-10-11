"""
=============================================================================
NITI DRISHTI: FastAPI Institutional Intelligence Router
=============================================================================
Exposes clean RESTful endpoints for district summaries, institution audits,
forecasting, GIS geospatial records, scenario simulations, and RAG.
=============================================================================
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from backend.services.readiness_service import readiness_service
from backend.services.forecast_service import forecast_service
from backend.services.gis_service import gis_service
from backend.services.scenario_service import scenario_service
from backend.services.institution_service import institution_service
from backend.services.rag_service import rag_service
from backend.services.report_service import report_service
from backend.core.config import DEFAULT_WEIGHTS, BENCHMARKS, PRIORITY_THRESHOLDS

api_router = APIRouter()


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ScenarioRequest(BaseModel):
    classroom_infusion: int = Field(0, description="Additional capital classrooms to infuse")
    w_equity: float = Field(0.25, ge=0.0, le=1.0, description="Statutory weight for Equity Pillar")
    w_promo: float = Field(0.30, ge=0.0, le=1.0, description="Statutory weight for Promotion Pillar")
    w_retention: float = Field(0.25, ge=0.0, le=1.0, description="Statutory weight for Retention Pillar")
    w_capacity: float = Field(0.20, ge=0.0, le=1.0, description="Statutory weight for Capacity Pillar")
    target_year: int = Field(2030, description="Target planning milestone year (e.g. 2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040)")


class AssessmentRequest(BaseModel):
    school_name: str = Field(..., description="Official school name")
    udise_code: Optional[int] = Field(None, description="11-digit UDISE identifier")
    total_enrolment: float = Field(..., ge=0, description="Total active student census")
    total_boys: float = Field(0, ge=0, description="Male student enrollment")
    total_girls: float = Field(0, ge=0, description="Female student enrollment")
    functional_classrooms: Optional[float] = Field(None, description="Operational classrooms")
    total_teachers: Optional[float] = Field(None, description="Total instructional teaching staff")
    promotion_rate: float = Field(85.0, ge=0.0, le=100.0, description="Annual pass/promotion percentage")
    pending_students: float = Field(0, ge=0, description="Remedial non-promoted backlog")
    target_year: int = Field(2030, description="Target planning horizon milestone")


class RAGQueryRequest(BaseModel):
    prompt: str = Field(..., description="Policy directive inquiry or statutory prompt")


class ReportGenerateRequest(BaseModel):
    markdown_content: str = Field(..., description="Markdown policy brief text")
    filename: Optional[str] = Field("NitiDrishti_Official_Policy_Directive.pdf", description="Target PDF filename")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api_router.get("/health", tags=["System"])
def get_health_status():
    """Returns platform operational readiness status and system metadata."""
    return {
        "status": "HEALTHY",
        "system": "NITI DRISHTI (National Infrastructure Readiness Index)",
        "district": "Chhatrapati Sambhajinagar",
        "focus_sector": "Education Infrastructure",
        "horizon_year": 2037,
        "database_status": "ONLINE (UDISE+ 2021-2025)",
        "institutions_cataloged": len(readiness_service.get_master_df())
    }


@api_router.get("/district/summary", tags=["Analytics"])
def get_district_summary(
    block: Optional[str] = Query("ALL", description="Filter by administrative Block / Area"),
    tier: Optional[str] = Query("ALL", description="Filter by Priority Status Tier")
):
    """Returns composite district scores, pupil demographics, and 4-pillar averages."""
    filtered_df = readiness_service.filter_schools(block=block, tier=tier)
    summary = readiness_service.get_district_summary(filtered_df)
    return summary


@api_router.get("/institutions", tags=["Institutions"])
def list_institutions(
    search: Optional[str] = Query(None, description="Search by name or UDISE"),
    block: Optional[str] = Query("ALL", description="Filter by Block / Area"),
    management: Optional[str] = Query("ALL", description="Filter by school management type"),
    category: Optional[str] = Query("ALL", description="Filter by school category level"),
    tier: Optional[str] = Query("ALL", description="Filter by status tier"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Returns paginated, searchable directory of registered district institutions."""
    filtered_df = readiness_service.filter_schools(
        block=block, management=management, category=category, tier=tier, search_query=search
    )
    cols = [
        "udise_code", "school_name", "block_name", "school_management",
        "total_enrolment", "promotion_rate", "pending_students", "readiness_score", "priority_tier"
    ]
    avail_cols = [c for c in cols if c in filtered_df.columns]
    total_matches = len(filtered_df)
    page_records = filtered_df[avail_cols].iloc[offset : offset + limit].to_dict(orient="records")

    return {
        "total_matches": total_matches,
        "limit": limit,
        "offset": offset,
        "institutions": page_records
    }


@api_router.get("/institutions/{id}", tags=["Institutions"])
def get_institution_detail(id: str):
    """Returns complete CDSS diagnostic dossier for a specific school (by UDISE code)."""
    school_record = institution_service.get_institution_by_udise(id)
    if not school_record:
        raise HTTPException(status_code=404, detail=f"Institution with UDISE '{id}' not found.")
    
    evaluation = institution_service.evaluate_institution(school_record)
    return evaluation


@api_router.get("/readiness", tags=["Analytics"])
def get_readiness_framework():
    """Returns statutory 4-pillar weights, benchmark standards, and district readiness score."""
    master_df = readiness_service.get_master_df()
    summary = readiness_service.get_district_summary(master_df)

    return {
        "framework_name": "National Infrastructure Readiness Index (IRI)",
        "district_score": summary.get("district_readiness_score", 82.5),
        "pillars": {
            "equity": {"weight": DEFAULT_WEIGHTS["equity"], "target_gpi": 1.00, "average_score": summary["sub_pillar_averages"]["equity"]},
            "promotion": {"weight": DEFAULT_WEIGHTS["promotion"], "target_pass_rate": 90.0, "average_score": summary["sub_pillar_averages"]["promotion"]},
            "retention": {"weight": DEFAULT_WEIGHTS["retention"], "target_transition": 0.67, "average_score": summary["sub_pillar_averages"]["retention"]},
            "capacity": {"weight": DEFAULT_WEIGHTS["capacity"], "target_ptr": BENCHMARKS["ptr_target_elementary"], "average_score": summary["sub_pillar_averages"]["capacity"]},
        },
        "thresholds": PRIORITY_THRESHOLDS
    }


@api_router.get("/forecast", tags=["Forecasting"])
def get_enrollment_forecast(sim_classrooms: int = Query(0, ge=0)):
    """Returns historical actuals (2021-2025) and predictive 2037 forecasts with confidence bounds."""
    summary = forecast_service.get_forecast_summary(sim_classrooms=sim_classrooms)
    df_fc = forecast_service.get_forecast_df()
    return {
        "summary": summary,
        "timeseries": df_fc.to_dict(orient="records")
    }


@api_router.get("/gis", tags=["Geospatial"])
def get_gis_records(
    tier: Optional[str] = Query("ALL", description="Filter by Priority Status Tier")
):
    """Returns geocoded institution coordinates for map visualization."""
    filtered_df = readiness_service.filter_schools(tier=tier)
    records = gis_service.get_spatial_records(filtered_df)
    return {
        "district": "Chhatrapati Sambhajinagar",
        "total_institutions_mapped": len(records),
        "institutions": records
    }


@api_router.post("/scenario", tags=["Simulation"])
def simulate_scenario(req: ScenarioRequest):
    """Executes dynamic multi-criteria weight recalibration and capital classroom infusion simulation."""
    res = scenario_service.run_simulation(
        sim_classrooms=req.classroom_infusion,
        w_equity=req.w_equity,
        w_promo=req.w_promo,
        w_retention=req.w_retention,
        w_capacity=req.w_capacity,
        target_year=req.target_year
    )
    return res


@api_router.post("/assessment", tags=["Institutions"])
def audit_and_assess_institution(req: AssessmentRequest):
    """Runs data completeness audit, physical gap analysis, and phased action plan on input data."""
    data_dict = req.model_dump()
    target_yr = data_dict.pop("target_year", 2030)
    res = institution_service.evaluate_institution(data_dict, target_year=target_yr)
    return res


@api_router.post("/rag/query", tags=["AI Copilot"])
def query_policy_assistant(req: RAGQueryRequest):
    """Executes vector retrieval over statutory policy corpus and synthesizes grounded directives."""
    res = rag_service.query_policy_assistant(req.prompt)
    return res


@api_router.post("/reports/generate", tags=["Reports"])
def generate_pdf_report(req: ReportGenerateRequest):
    """Generates an executive PDF report buffer from markdown text."""
    pdf_buf = report_service.generate_pdf_buffer(req.markdown_content, filename=req.filename)
    return Response(
        content=pdf_buf.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={req.filename}"}
    )
