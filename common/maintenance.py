"""
Step 12 - Maintenance scheduling insights.

These categories and recommendation strings are ILLUSTRATIVE PROJECT OUTPUT
for an academic predictive-maintenance demo. They are not a validated
industrial maintenance standard, not derived from real maintenance-cost
data, and should not be used to make real scheduling or budget decisions
without domain validation.
"""

LOW_RISK_MAX = None   # set at runtime from the validated threshold (see below)
MEDIUM_RISK_MAX = None


def risk_category(probability, low_max=0.30, high_min=0.60):
    """Map a failure probability to a Low/Medium/High label.

    `low_max`/`high_min` default to the same project-defined demonstration
    thresholds used throughout the app; when a validated operating
    threshold has been selected (Step 10), callers should pass it in as
    `high_min` so the "High risk" boundary lines up with the model's tuned
    decision point rather than an arbitrary number.
    """
    if probability < low_max:
        return "LOW"
    elif probability < high_min:
        return "MEDIUM"
    else:
        return "HIGH"


RECOMMENDATIONS = {
    "LOW": (
        "Continue routine monitoring. No immediate action indicated by the "
        "model; keep following the standard inspection schedule."
    ),
    "MEDIUM": (
        "Consider closer monitoring and a scheduled inspection in the near "
        "term. The model sees elevated risk factors but not an acute one - "
        "treat this as a prompt to check in sooner, not an emergency."
    ),
    "HIGH": (
        "Prioritize inspection and a maintenance assessment. The model's "
        "risk estimate is high enough that unplanned failure is a real "
        "possibility if the machine keeps running under these conditions."
    ),
}


def recommendation_for(probability, low_max=0.30, high_min=0.60):
    category = risk_category(probability, low_max=low_max, high_min=high_min)
    return category, RECOMMENDATIONS[category]
