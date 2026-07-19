from datetime import datetime
from app.schemas.api import RecommendationItem

# Priority bands (win_prob thresholds — fully contiguous, no gaps):
#   < 0.40  → CRITICAL  (at-risk deal, immediate intervention)
#   < 0.65  → HIGH      (needs active nurturing to cross the line)
#   < 0.80  → MEDIUM    (on track, reinforce preference)
#   ≥ 0.80  → LOW       (strong deal, focus on accelerating close)

def generate_recommendations(prediction, opportunity, agent_avg_deal_value: float) -> list[dict]:
    """Generates recommendations based on decision rules."""
    win_prob = prediction.win_probability

    # Calculate days since engage
    try:
        engage_dt = datetime.strptime(opportunity.engage_date, "%Y-%m-%d")
        days_since_engage = max(1, (datetime.now() - engage_dt).days)
    except Exception:
        days_since_engage = 1

    # Urgency score: value-weighted probability normalised by deal age
    urgency_score = (win_prob * agent_avg_deal_value) / days_since_engage

    recommendations = []

    if win_prob < 0.40:
        recommendations.append({
            "action": "Schedule exec sponsor call; validate budget authority immediately",
            "rationale": prediction.nl_explanation,
            "expected_impact": "Immediate intervention required — deal is at high risk of loss.",
            "priority": "CRITICAL",
            "urgency_score": urgency_score,
        })
    elif win_prob < 0.65:
        recommendations.append({
            "action": "Arrange C-level intro; send tailored ROI case study",
            "rationale": prediction.nl_explanation,
            "expected_impact": "Build trust and demonstrate value to improve win probability.",
            "priority": "HIGH",
            "urgency_score": urgency_score,
        })
    elif win_prob < 0.80:
        recommendations.append({
            "action": "Share competitive differentiators; confirm stakeholder alignment",
            "rationale": prediction.nl_explanation,
            "expected_impact": "Solidify preference and remove remaining blockers.",
            "priority": "MEDIUM",
            "urgency_score": urgency_score,
        })
    else:
        recommendations.append({
            "action": "Send closing checklist; propose contract review date",
            "rationale": prediction.nl_explanation,
            "expected_impact": "Accelerate time to close on a strong opportunity.",
            "priority": "LOW",
            "urgency_score": urgency_score,
        })

    return recommendations
