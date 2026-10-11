"""
=============================================================================
NITI DRISHTI: High-Precision Demographic & Infrastructure Forecasting Service
=============================================================================
Manages district population dynamics, school-age cohort progressions (2021-2040),
confidence envelope bounds, and statutory RTE 30:1 classroom deficit modeling.
Supports variable planning horizons (2026, 2028, 2030, 2032, 2034, 2036, 2038, 2040).
=============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from backend.analytics.forecasting import EducationForecaster
from backend.services.readiness_service import find_dataset

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class ForecastService:
    _instance = None
    _df_forecast = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ForecastService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        forecast_file = find_dataset([
            "data/timeseries/yearly_enrollment_forecast_2021_2040.csv",
            "data/timeseries/yearly_enrollment_forecast_2021_2037.csv",
            "data/enrollment_forecast.csv"
        ])

        if forecast_file:
            self._df_forecast = pd.read_csv(forecast_file)
        else:
            forecaster = EducationForecaster()
            forecaster.save_forecast()
            self._df_forecast = pd.read_csv(BASE_DIR / "data" / "timeseries" / "yearly_enrollment_forecast_2021_2040.csv")

    def get_forecast_df(self) -> pd.DataFrame:
        return self._df_forecast.copy()

    def get_forecast_summary(self, baseline_students: int = 278945, sim_classrooms: int = 0, target_year: int = 2030) -> dict:
        df = self._df_forecast
        target_year = int(target_year or 2030)

        row_2025 = df[df["year"] == 2025].iloc[0] if (df["year"] == 2025).any() else df.iloc[4]

        # Match target milestone year or closest available
        if (df["year"] == target_year).any():
            row_target = df[df["year"] == target_year].iloc[0]
        else:
            row_target = df.iloc[-1]
            target_year = int(row_target.get("year", 2040))

        district_2025 = int(row_2025.get("total_students", 278945))
        district_target = int(row_target.get("total_students", 306469))
        district_lower = int(row_target.get("yhat_lower", district_target * 0.95))
        district_upper = int(row_target.get("yhat_upper", district_target * 1.05))

        # Demographic & population metrics for target year
        district_pop = int(row_target.get("district_population", 4888000))
        school_age_pop = int(row_target.get("school_age_population", 920166))

        # Subset scaling factor (e.g., if user filtered to a specific block or management category)
        scale = max(0.001, float(baseline_students) / max(district_2025, 1))

        scaled_target = int(round(district_target * scale)) if abs(scale - 1.0) > 0.01 else district_target
        scaled_lower = int(round(district_lower * scale)) if abs(scale - 1.0) > 0.01 else district_lower
        scaled_upper = int(round(district_upper * scale)) if abs(scale - 1.0) > 0.01 else district_upper

        # Classrooms needed at statutory RTE 30:1 ratio
        baseline_cr = int(np.ceil(baseline_students / 30.0))
        projected_cr = int(np.ceil(scaled_target / 30.0))
        deficit = max(0, projected_cr - (baseline_cr + int(sim_classrooms)))

        # CapEx budget requirement (Samagra Shiksha PWD schedule: ₹12.50 Lakhs per classroom unit)
        capex_crores = round(deficit * 0.125, 2)
        total_infusion_capex = round(int(sim_classrooms) * 0.125, 2)

        # Gross Enrollment Ratio (GER) of public/aided system
        effective_sch_age = int(round(school_age_pop * scale))
        ger_pct = round((scaled_target / max(1, effective_sch_age)) * 100.0, 1)

        return {
            "baseline_year": 2025,
            "baseline_students": int(baseline_students),
            "horizon_year": target_year,
            "district_population": district_pop,
            "school_age_population": effective_sch_age,
            "projected_students_target": scaled_target,
            "confidence_lower_target": scaled_lower,
            "confidence_upper_target": scaled_upper,
            "projected_students_2037": scaled_target,
            "confidence_lower_2037": scaled_lower,
            "confidence_upper_2037": scaled_upper,
            "growth_rate_pct": round(((scaled_target - baseline_students) / max(baseline_students, 1)) * 100, 2),
            "baseline_classrooms": baseline_cr,
            "projected_classrooms_target": projected_cr,
            "capital_infusion_applied": int(sim_classrooms),
            "net_classroom_deficit": deficit,
            "required_teachers": projected_cr,
            "estimated_capex_crores": capex_crores,
            "infusion_capex_crores": total_infusion_capex,
            "ger_pct": ger_pct,
            "statutory_standard": "RTE Act 2009 (30:1 Pupil-Teacher/Classroom Ratio)"
        }


# Global singleton instance
forecast_service = ForecastService()
