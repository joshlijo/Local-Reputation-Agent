"""
Urgency detection for restaurant reviews.

Flags reviews that require immediate management attention based on
keyword pattern matching with severity scoring.

Design choice — why keyword matching instead of ML?
- Urgency triggers are concrete and enumerable (food poisoning, FSSAI, etc.).
- False negatives are expensive (missing a poisoning report), so explicit
  keyword lists give us high recall for known patterns.
- The severity scoring system lets us rank urgency — a food poisoning
  hospitalisation (severity 10) is more urgent than a rude-staff complaint
  (severity 7).

CRITICAL SAFETY RULE (added in refactor):
- Health/safety categories (food_poisoning, hygiene_severe, authority_escalation,
  safety_concern) must NEVER have their severity reduced by a high star rating.
  A 5-star review mentioning "food poisoning" in positive context (e.g. "No food
  poisoning!") is handled by the sentiment layer, not by suppressing urgency.
  Suppressing urgency here caused silent failures where sickness reports were
  downgraded and missed by alerting.
"""

import config


# Categories where severity must never be reduced, regardless of rating.
# Rationale: a food poisoning mention in ANY context warrants investigation.
# False positives (e.g. "No stomach issues!") are cheaper than false negatives
# (missing an actual poisoning report).
_HEALTH_SAFETY_CATEGORIES = frozenset({
    "food_poisoning",
    "hygiene_severe",
    "authority_escalation",
    "safety_concern",
})


def detect_urgency(review_text: str, rating: int) -> dict:
    """
    Detect whether a review is urgent and why.

    Returns:
      urgent          – True / False
      urgency_reason  – category label or "none"
      severity_score  – 0-10 integer
      matched_patterns – list of keywords that triggered detection
    """
    if not review_text or not review_text.strip():
        return {
            "urgent": False,
            "urgency_reason": "none",
            "severity_score": 0,
            "matched_patterns": [],
        }

    text_lower = review_text.lower()

    max_severity = 0
    best_category = "none"
    all_matches = []
    # Track whether the highest-severity category is health/safety.
    best_is_health_safety = False

    for category, cfg in config.URGENCY_PATTERNS.items():
        matches = [kw for kw in cfg["keywords"] if kw in text_lower]
        if matches:
            all_matches.extend(matches)
            if cfg["severity"] > max_severity:
                max_severity = cfg["severity"]
                best_category = category
                best_is_health_safety = category in _HEALTH_SAFETY_CATEGORIES

    # Rating modifiers — ONLY for non-health/safety categories.
    #
    # OLD BUG: rating >= 4 reduced severity by 3 for ALL categories.
    # This caused food_poisoning (10) to drop to 7, and hygiene_severe (9)
    # to drop to 6 — barely urgent.  Contract requires:
    #   "hygiene severe → urgency true even if rating ≥ 4"
    #   "food poisoning / hospitalization → urgent = true, severity = 10"
    #
    # FIX: Only apply rating modifier to non-health/safety categories
    # (currently just rude_staff).  For health/safety, severity is never
    # reduced — false positives are acceptable, false negatives are not.
    if all_matches and not best_is_health_safety:
        if rating == 1:
            max_severity = min(10, max_severity + 1)
        elif rating >= 4:
            # Positive-context mentions of rude_staff keywords with high
            # rating are likely not genuine complaints — reduce severity.
            max_severity = max(0, max_severity - 3)

    urgent = max_severity >= config.URGENCY_THRESHOLD

    return {
        "urgent": urgent,
        "urgency_reason": best_category if urgent else "none",
        "severity_score": max_severity,
        "matched_patterns": list(set(all_matches))[:5],
    }
