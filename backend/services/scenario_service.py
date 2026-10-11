"""
=============================================================================
NITI DRISHTI: Infrastructure Scenario Analysis Service
=============================================================================
Computes what-if simulations for capital classroom infusions and dynamic
statutory pillar weight recalibrations (Equity, Promotion, Retention, Capacity).
Calibrated to variable demographic planning horizons (2026-2040).
=============================================================================
"""

import pandas as pd
from backend.services.readiness_service import readiness_service
from backend.services.forecast_service import forecast_service


class ScenarioService:
    @staticmethod
    def run_simulation(
        sim_classrooms: int = 0,
        w_equity: float = 0.25,
        w_promo: float = 0.30,
        w_retention: float = 0.25,
        w_capacity: float = 0.20,
        target_year: int = 2030,
        filtered_df: pd.DataFrame = None
    ) -> dict:
        target_year = int(target_year or 2030)
        df = (filtered_df if filtered_df is not None else readiness_service.get_master_df()).copy()
        active_subset = df[df["is_operational"] == True] if "is_operational" in df.columns else df

        # Normalize weights
        tot_w = max(0.001, float(w_equity) + float(w_promo) + float(w_retention) + float(w_capacity))
        w_eq = float(w_equity) / tot_w
        w_pr = float(w_promo) / tot_w
        w_ret = float(w_retention) / tot_w
        w_cap = float(w_capacity) / tot_w

        # Compute dynamic composite readiness series
        if not active_subset.empty and "score_equity" in active_subset.columns:
            composite_series = (
                active_subset["score_equity"] * w_eq +
                active_subset["score_promotion"] * w_pr +
                active_subset["score_retention"] * w_ret +
                active_subset["score_capacity"] * w_cap
            )
            sim_score = round(float(composite_series.mean()), 1)
        else:
            sim_score = round(float(active_subset["readiness_score"].mean()), 1) if not active_subset.empty else 0.0

        # Baseline comparison
        baseline_score = round(float(active_subset["readiness_score"].mean()), 1) if not active_subset.empty else 0.0
        score_delta = round(sim_score - baseline_score, 1)

        # Dynamic Variable Horizon Deficit impact
        total_students = int(active_subset["total_enrolment"].sum()) if not active_subset.empty else 278945
        forecast_metrics = forecast_service.get_forecast_summary(
            baseline_students=total_students,
            sim_classrooms=sim_classrooms,
            target_year=target_year
        )

        baseline_deficit = forecast_metrics["net_classroom_deficit"] + int(sim_classrooms)
        simulated_deficit = forecast_metrics["net_classroom_deficit"]
        deficit_resolved_pct = round((min(baseline_deficit, int(sim_classrooms)) / max(1, baseline_deficit)) * 100.0, 1) if baseline_deficit > 0 else 100.0

        # Affected critical schools
        critical_count = int((active_subset["readiness_score"] < 55.0).sum()) if "readiness_score" in active_subset.columns else 0

        # Work-oriented intervention narrative (focused on physical infrastructure & staffing)
        if sim_classrooms >= 500:
            intervention = f"Substantial classroom expansion (+{sim_classrooms} classrooms) meets {deficit_resolved_pct}% of the space needed for {target_year}. Prioritize teacher recruitment and specialized learning spaces for remaining demand."
        elif sim_classrooms > 0:
            intervention = f"Adding +{sim_classrooms} planned classrooms reduces the {target_year} shortfall to {simulated_deficit} classrooms (a {deficit_resolved_pct}% improvement). Continued classroom additions and teacher recruitment will meet full standards."
        else:
            intervention = f"Status Quo Baseline (Year {target_year}): Projected shortfall of +{simulated_deficit} classrooms and +{forecast_metrics['required_teachers']:,} teachers to meet standard 30 students per classroom."

        return {
            "target_year": target_year,
            "sim_classrooms": int(sim_classrooms),
            "normalized_weights": {
                "equity": round(w_eq, 3),
                "promotion": round(w_pr, 3),
                "retention": round(w_ret, 3),
                "capacity": round(w_cap, 3)
            },
            "district_population": forecast_metrics["district_population"],
            "school_age_population": forecast_metrics["school_age_population"],
            "projected_students": forecast_metrics["projected_students_target"],
            "baseline_readiness_score": baseline_score,
            "simulated_readiness_score": sim_score,
            "readiness_score_delta": score_delta,
            "baseline_classroom_deficit": baseline_deficit,
            "simulated_classroom_deficit": simulated_deficit,
            "deficit_resolved_pct": deficit_resolved_pct,
            "infusion_capex_crores": forecast_metrics["infusion_capex_crores"],
            "remaining_capex_crores": forecast_metrics["estimated_capex_crores"],
            "required_teachers": forecast_metrics["required_teachers"],
            "critical_institutions_count": critical_count,
            "recommended_intervention": intervention
        }


# Global singleton instance
scenario_service = ScenarioService()
