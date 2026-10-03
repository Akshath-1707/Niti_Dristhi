"""
=============================================================================
NITI DRISHTI: Education Infrastructure Readiness Index (IRI) Engine
=============================================================================
Computes a multi-criteria quantitative Readiness Index (0–100 scale)
for municipal & district educational infrastructure in Chhatrapati Sambhajinagar.

Mathematical Formulation:
    IRI_District = Sum(w_k * Pillar_k)
    where:
      1. Equity & Inclusivity Pillar (25%): GPI proximity to 1.0 & female enrolment ratio
      2. Promotion & Remedial Efficacy Pillar (30%): Pass rate & backlog remediation
      3. Cohort Retention Pillar (25%): Upper Primary -> Secondary transition ratio
      4. Operational & Capacity Pillar (20%): Operational integrity & size balance
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


class EducationReadinessCalculator:
    """
    Computes school-level and district-level Infrastructure Readiness Scores.
    """

    def __init__(self, weights=None):
        # Default statutory weights aligned with NITI Aayog / NEP 2020 priorities
        self.weights = weights or {
            "equity": 0.25,
            "promotion": 0.30,
            "retention": 0.25,
            "capacity": 0.20,
        }
        # Normalize weights to ensure sum == 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute_equity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Pillar 1: Equity & Inclusivity (0-100)
        - Target GPI = 1.00 (penalizes deviations below and above).
        - Rewards healthy female enrolment ratio (45%-55%).
        """
        gpi = df.get("gender_parity_index", pd.Series(1.0, index=df.index)).fillna(1.0)
        # Symmetrical penalty for GPI deviation from ideal 1.0 (clamped 0 to 100)
        gpi_score = np.clip(100 - (np.abs(gpi - 1.0) * 100), 0, 100)

        female_ratio = df.get("female_enrolment_ratio", pd.Series(50.0, index=df.index)).fillna(50.0)
        # Optimal ratio centered around 50%
        female_score = np.clip(100 - (np.abs(female_ratio - 50.0) * 2), 0, 100)

        return (0.60 * gpi_score + 0.40 * female_score).round(2)

    def compute_promotion_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Pillar 2: Promotion & Remedial Efficacy (0-100)
        - Uses empirical promotion rates from URC reports.
        - Penalizes schools with high pending non-promoted student backlog.
        """
        promo_rate = df.get("promotion_rate", pd.Series(85.0, index=df.index)).fillna(85.0)
        base_promo = np.clip(promo_rate, 0, 100)

        # Pending backlog penalty
        pending = df.get("pending_students", pd.Series(0, index=df.index)).fillna(0)
        total = df.get("total_enrolment", df.get("total_students", pd.Series(100, index=df.index))).replace(0, 1)
        backlog_ratio = (pending / total) * 100
        backlog_penalty = np.clip(backlog_ratio * 0.5, 0, 30)

        return np.clip(base_promo - backlog_penalty, 0, 100).round(2)

    def compute_retention_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Pillar 3: Cohort Retention & Progression (0-100)
        - Evaluates the critical transition from Upper Primary (Classes 6-8) to Secondary (Classes 9-10).
        - If a school does not offer secondary grades, evaluate against standard primary completion.
        """
        upper_pri = df.get("enrolment_upper_primary", pd.Series(0, index=df.index)).fillna(0)
        sec = df.get("enrolment_secondary", pd.Series(0, index=df.index)).fillna(0)
        primary = df.get("enrolment_primary", pd.Series(0, index=df.index)).fillna(0)

        # For composite schools offering both Upper Primary & Secondary:
        composite_mask = (upper_pri > 0) & (sec > 0)
        # Expected ratio roughly 2 grades (9-10) vs 3 grades (6-8) is ~0.67
        sec_ratio = (sec / upper_pri.replace(0, 1)) / 0.67
        sec_score = np.clip(sec_ratio * 100, 0, 100)

        # For primary-only schools, evaluate Primary cohort health
        pri_score = np.where(primary > 0, 85.0, 75.0)

        retention_score = np.where(composite_mask, sec_score, pri_score)
        return pd.Series(retention_score, index=df.index).round(2)

    def compute_capacity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Pillar 4: Operational Integrity & Infrastructure Capacity (0-100)
        - Operational status: 100 if operational, 0 if closed/inactive.
        - School scale efficiency: penalizes extreme overcrowding (>1500) or underutilization (<30).
        """
        is_op = df.get("is_operational", pd.Series(True, index=df.index))
        # Handle boolean or string mapping
        if is_op.dtype == object:
            is_op = is_op.astype(str).str.strip().str.lower().map({"true": True, "1": True, "yes": True}).fillna(False)

        total_std = df.get("total_enrolment", df.get("total_students", pd.Series(200, index=df.index))).fillna(0)
        
        # Scale score
        scale_score = np.where(
            total_std < 20, 50.0,
            np.where(total_std > 2000, 60.0, 95.0)
        )

        cap_score = np.where(is_op, scale_score, 0.0)
        return pd.Series(cap_score, index=df.index).round(2)

    def evaluate_schools(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes composite readiness scores and priority tiers for all schools in the dataframe.
        """
        res = df.copy()
        res["score_equity"] = self.compute_equity_score(res)
        res["score_promotion"] = self.compute_promotion_score(res)
        res["score_retention"] = self.compute_retention_score(res)
        res["score_capacity"] = self.compute_capacity_score(res)

        # Weighted composite score
        res["readiness_score"] = (
            self.weights["equity"] * res["score_equity"]
            + self.weights["promotion"] * res["score_promotion"]
            + self.weights["retention"] * res["score_retention"]
            + self.weights["capacity"] * res["score_capacity"]
        ).round(2)

        # Deficit / Priority Score (Inversion: 100 - Readiness)
        # High deficit score = High intervention urgency
        res["priority_deficit_score"] = (100.0 - res["readiness_score"]).round(2)

        # Priority Tiers
        conditions = [
            res["readiness_score"] < 55.0,
            (res["readiness_score"] >= 55.0) & (res["readiness_score"] < 75.0),
            res["readiness_score"] >= 75.0,
        ]
        choices = ["CRITICAL_INTERVENTION", "DEVELOPING_NEEDS", "OPTIMAL_RESILIENT"]
        res["priority_tier"] = np.select(conditions, choices, default="DEVELOPING_NEEDS")

        return res

    def get_district_summary(self, df: pd.DataFrame) -> dict:
        """
        Returns high-level executive diagnostic metrics for the district.
        """
        evaluated = self.evaluate_schools(df)
        active_schools = evaluated[evaluated["is_operational"] == True] if "is_operational" in evaluated.columns else evaluated

        total_schools = len(evaluated)
        active_count = len(active_schools)
        total_students = int(evaluated.get("total_enrolment", evaluated.get("total_students", pd.Series(0))).sum())
        total_boys = int(evaluated.get("total_boys", pd.Series(0)).sum())
        total_girls = int(evaluated.get("total_girls", pd.Series(0)).sum())
        gpi_district = round(total_girls / max(total_boys, 1), 2)

        avg_readiness = round(float(active_schools["readiness_score"].mean()), 1)
        tier_counts = evaluated["priority_tier"].value_counts().to_dict()

        return {
            "district_readiness_score": avg_readiness,
            "total_schools": total_schools,
            "operational_schools": active_count,
            "total_students": total_students,
            "total_boys": total_boys,
            "total_girls": total_girls,
            "district_gpi": gpi_district,
            "tier_distribution": {
                "critical": tier_counts.get("CRITICAL_INTERVENTION", 0),
                "developing": tier_counts.get("DEVELOPING_NEEDS", 0),
                "optimal": tier_counts.get("OPTIMAL_RESILIENT", 0),
            },
            "sub_pillar_averages": {
                "equity": round(float(active_schools["score_equity"].mean()), 1),
                "promotion": round(float(active_schools["score_promotion"].mean()), 1),
                "retention": round(float(active_schools["score_retention"].mean()), 1),
                "capacity": round(float(active_schools["score_capacity"].mean()), 1),
            },
        }


if __name__ == "__main__":
    # Test script execution
    base_dir = Path(__file__).resolve().parent.parent.parent
    sample_file = base_dir / "data" / "processed" / "database_ready_schools_urc.csv"
    if not sample_file.exists():
        sample_file = base_dir / "data" / "database_ready_schools_urc.csv"

    if sample_file.exists():
        df_test = pd.read_csv(sample_file)
        calc = EducationReadinessCalculator()
        summary = calc.get_district_summary(df_test)
        print("\n" + "=" * 60)
        print("NITI DRISHTI: District Education Readiness Summary")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("=" * 60)
