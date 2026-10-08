"""
=============================================================================
NITI DRISHTI: Infrastructure Gap Analyzer & Dominant Bottleneck Engine
=============================================================================
Computes school-level physical unit gaps against statutory RTE Act 2009 &
UDISE+ benchmarks and isolates the dominant primary constraint.
=============================================================================
"""

import math
from backend.core.config import BENCHMARKS


def analyze_school_infrastructure_gaps(school_row: dict) -> dict:
    """
    Computes exact physical unit deficits for a school:
      - Teacher Gap (Delta_T)
      - Classroom Gap (Delta_C)
      - Future Capacity Deficit (Delta_Future)
      - Sanitary & Facility Gaps
      - Dominant Bottleneck Isolation
    """
    students = float(school_row.get("total_enrolment", school_row.get("total_students", 0)) or 0)
    boys = float(school_row.get("total_boys", 0) or 0)
    girls = float(school_row.get("total_girls", 0) or 0)

    # If explicit teacher count is missing, use empirical PTR estimate
    teachers = school_row.get("total_teachers")
    if teachers is None or math.isnan(float(teachers)):
        # Default empirical baseline roughly ~34:1
        teachers = max(1, round(students / 34.0)) if students > 0 else 1
    else:
        teachers = max(1, int(teachers))

    # If functional classrooms is missing, use empirical PCR estimate
    classrooms = school_row.get("functional_classrooms", school_row.get("total_classrooms"))
    if classrooms is None or math.isnan(float(classrooms)):
        # Default empirical baseline roughly ~38:1
        classrooms = max(1, round(students / 38.0)) if students > 0 else 1
    else:
        classrooms = max(1, int(classrooms))

    # 1. Statutory RTE Benchmarks
    ptr_target = BENCHMARKS["ptr_target_elementary"]  # 30:1
    pcr_target = BENCHMARKS["pcr_target"]             # 40:1

    actual_ptr = round(students / max(teachers, 1), 1)
    actual_pcr = round(students / max(classrooms, 1), 1)

    required_teachers = math.ceil(students / ptr_target) if students > 0 else 1
    required_classrooms = math.ceil(students / pcr_target) if students > 0 else 1

    teacher_gap = max(0, required_teachers - teachers)
    classroom_gap = max(0, required_classrooms - classrooms)

    # 2. Built Physical Capacity vs Projected Surge
    built_capacity = classrooms * int(pcr_target)
    # Estimate 3-year projected enrollment with 2.2% annual growth
    projected_3yr_enrol = round(students * ((1.0 + 0.022) ** 3))
    future_capacity_gap = max(0, projected_3yr_enrol - built_capacity)
    is_capacity_saturated = projected_3yr_enrol > built_capacity

    # 3. Sanitation Gaps (RTE 40:1 for girls, 50:1 for boys)
    req_girls_toilets = math.ceil(girls / BENCHMARKS["girls_per_toilet_target"]) if girls > 0 else 1
    req_boys_toilets = math.ceil(boys / BENCHMARKS["boys_per_toilet_target"]) if boys > 0 else 1

    actual_girls_toilets = int(school_row.get("girls_toilets", max(1, round(girls / 45.0))))
    actual_boys_toilets = int(school_row.get("boys_toilets", max(1, round(boys / 55.0))))

    girls_toilet_gap = max(0, req_girls_toilets - actual_girls_toilets)
    boys_toilet_gap = max(0, req_boys_toilets - actual_boys_toilets)

    # 4. Remedial Backlog
    pending_students = int(school_row.get("pending_students", 0) or 0)
    backlog_ratio = round((pending_students / max(students, 1)) * 100, 1)

    # 5. Dominant Bottleneck Isolation
    dominant_bottleneck = "Compliant / Resilient Baseline"
    dominant_severity = "LOW"
    dominant_reason = "School meets statutory RTE standards with adequate classroom and faculty capacity."

    if actual_ptr >= BENCHMARKS["ptr_critical_threshold"] or teacher_gap >= 4:
        dominant_bottleneck = "Severe Teacher Deficit"
        dominant_severity = "CRITICAL"
        dominant_reason = f"Pupil-Teacher Ratio is {actual_ptr}:1 (RTE limit is 30:1). Institutional deficit of {teacher_gap} instructional staff."
    elif actual_pcr >= BENCHMARKS["pcr_critical_threshold"] or classroom_gap >= 3:
        dominant_bottleneck = "Classroom Overcrowding"
        dominant_severity = "CRITICAL"
        dominant_reason = f"Pupil-Classroom Ratio is {actual_pcr}:1 (UDISE norm is 40:1). Shortage of {classroom_gap} functional classrooms."
    elif future_capacity_gap >= 40:
        dominant_bottleneck = "Imminent Capacity Saturation"
        dominant_severity = "HIGH"
        dominant_reason = f"Projected enrollment ({projected_3yr_enrol}) exceeds physical built capacity ({built_capacity} seats) by {future_capacity_gap} students."
    elif backlog_ratio >= 25.0:
        dominant_bottleneck = "Remedial Retention Backlog"
        dominant_severity = "HIGH"
        dominant_reason = f"{pending_students} students ({backlog_ratio}% of enrollment) held back or pending academic promotion."
    elif girls_toilet_gap >= 2:
        dominant_bottleneck = "Gender Sanitation Barrier"
        dominant_severity = "MODERATE"
        dominant_reason = f"Deficit of {girls_toilet_gap} dedicated sanitation units for girl students under RTE standards."
    elif teacher_gap > 0 or classroom_gap > 0:
        dominant_bottleneck = "Developing Capacity Needs"
        dominant_severity = "MODERATE"
        dominant_reason = f"Minor operational deficit: requires {classroom_gap} classrooms and {teacher_gap} teachers."

    return {
        "total_students": int(students),
        "total_teachers": teachers,
        "functional_classrooms": classrooms,
        "actual_ptr": actual_ptr,
        "actual_pcr": actual_pcr,
        "teacher_gap": teacher_gap,
        "classroom_gap": classroom_gap,
        "built_capacity": built_capacity,
        "projected_3yr_enrol": projected_3yr_enrol,
        "future_capacity_gap": future_capacity_gap,
        "is_capacity_saturated": is_capacity_saturated,
        "girls_toilet_gap": girls_toilet_gap,
        "boys_toilet_gap": boys_toilet_gap,
        "pending_students": pending_students,
        "backlog_ratio": backlog_ratio,
        "dominant_bottleneck": dominant_bottleneck,
        "dominant_severity": dominant_severity,
        "dominant_reason": dominant_reason
    }
