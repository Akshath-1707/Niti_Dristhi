"""
=============================================================================
NITI DRISHTI: Data Quality & Completeness Auditor
=============================================================================
Evaluates data completeness and flags domain-validity violations before
data updates touch the predictive modeling or forecasting pipelines.
=============================================================================
"""

import numpy as np
import pandas as pd


CRITICAL_FIELDS = [
    "school_name",
    "udise_code",
    "total_enrolment",
    "total_boys",
    "total_girls",
    "enrolment_primary",
    "enrolment_upper_primary",
    "promotion_rate",
    "pending_students"
]


def audit_school_record(record: dict) -> dict:
    """
    Audits a single school data record for completeness and domain validity.
    Returns:
      - 'completeness_score': float (0-100)
      - 'missing_fields': list[str]
      - 'domain_violations': list[str]
      - 'confidence_status': 'Sufficient' | 'Moderate Confidence' | 'Reduced Confidence'
      - 'confidence_message': str
    """
    missing_fields = []
    total_critical = len(CRITICAL_FIELDS)
    present_critical = 0

    for field in CRITICAL_FIELDS:
        val = record.get(field)
        if val is None or (isinstance(val, (int, float)) and np.isnan(val)) or str(val).strip() == "":
            missing_fields.append(field)
        else:
            present_critical += 1

    completeness_score = round((present_critical / total_critical) * 100.0, 1)

    domain_violations = []

    # 1. Total student count validation
    total_enrol = record.get("total_enrolment", record.get("total_students", 0))
    try:
        total_enrol = float(total_enrol) if total_enrol is not None else 0.0
    except (ValueError, TypeError):
        total_enrol = 0.0

    if total_enrol < 0:
        domain_violations.append("Total enrollment cannot be negative.")

    # 2. Gender sub-total consistency
    boys = float(record.get("total_boys", 0) or 0)
    girls = float(record.get("total_girls", 0) or 0)
    if boys < 0 or girls < 0:
        domain_violations.append("Gender enrollment counts cannot be negative.")
    elif total_enrol > 0 and (boys + girls) > (total_enrol * 1.05):
        domain_violations.append(f"Sum of boys ({int(boys)}) and girls ({int(girls)}) exceeds total enrollment ({int(total_enrol)}).")

    # 3. Promotion and pending counts
    pending = float(record.get("pending_students", 0) or 0)
    if pending < 0:
        domain_violations.append("Pending remedial students cannot be negative.")
    elif total_enrol > 0 and pending > (total_enrol * 1.5):
        domain_violations.append("Pending remedial count is disproportionately higher than total school enrollment.")

    promo_rate = record.get("promotion_rate")
    if promo_rate is not None:
        try:
            pr = float(promo_rate)
            if pr < 0 or pr > 100:
                domain_violations.append("Promotion rate must be a percentage between 0% and 100%.")
        except (ValueError, TypeError):
            domain_violations.append("Promotion rate must be a numerical value.")

    # 4. Classroom consistency if provided
    func_cr = record.get("functional_classrooms")
    tot_cr = record.get("total_classrooms")
    if func_cr is not None and tot_cr is not None:
        try:
            if float(func_cr) > float(tot_cr):
                domain_violations.append("Functional classrooms cannot exceed total available classrooms.")
        except (ValueError, TypeError):
            pass

    # Determine confidence status
    if len(domain_violations) > 0 or completeness_score < 70.0:
        confidence_status = "Reduced Confidence"
        confidence_message = "Recommendation confidence reduced due to incomplete or contradictory school data."
    elif completeness_score < 90.0:
        confidence_status = "Moderate Confidence"
        confidence_message = "Adequate baseline data present; minor secondary attributes missing."
    else:
        confidence_status = "Sufficient"
        confidence_message = "High-confidence audit: complete institutional record verified against UDISE+ standards."

    return {
        "completeness_score": completeness_score,
        "missing_fields": missing_fields,
        "domain_violations": domain_violations,
        "confidence_status": confidence_status,
        "confidence_message": confidence_message,
        "is_valid_for_submission": len(domain_violations) == 0 and completeness_score >= 60.0
    }
