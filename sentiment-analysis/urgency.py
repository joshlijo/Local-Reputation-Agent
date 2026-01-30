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
  (severity 6).

Edge case handling:
- Positive context with urgency keywords (e.g. "No stomach issues!") is
  partially mitigated by checking the star rating: a 5-star review with
  a urgency keyword gets its severity reduced because the reviewer is
  clearly not complaining.
"""

import config


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

    for category, cfg in config.URGENCY_PATTERNS.items():
        matches = [kw for kw in cfg["keywords"] if kw in text_lower]
        if matches:
            all_matches.extend(matches)
            if cfg["severity"] > max_severity:
                max_severity = cfg["severity"]
                best_category = category

    # Rating modifiers:
    # 1-star + urgency keyword → boost severity (reviewer is angry AND reporting)
    # 4-5 stars + urgency keyword → reduce severity (likely positive context,
    #   e.g. "No stomach issues afterwards!" from a 5-star review)
    if all_matches:
        if rating == 1:
            max_severity = min(10, max_severity + 1)
        elif rating >= 4:
            max_severity = max(0, max_severity - 3)

    urgent = max_severity >= config.URGENCY_THRESHOLD

    return {
        "urgent": urgent,
        "urgency_reason": best_category if urgent else "none",
        "severity_score": max_severity,
        "matched_patterns": list(set(all_matches))[:5],
    }
