"""
Post-processing enforcement layer for the sentiment analysis pipeline.

This module runs AFTER sentiment, aspect, and urgency analysis are complete.
It enforces hard invariants from the sentiment contract that span multiple
modules.  No other module should need to know about cross-module rules —
they are all enforced here.

WHY THIS FILE EXISTS:
  The original pipeline computed sentiment, aspects, and urgency independently
  and never reconciled them.  This caused states like:
    - urgent=true AND sentiment="Positive" (contract violation)
    - food poisoning report with confidence="HIGH" (dangerous)
    - severity_score >= 7 AND sentiment="Positive" (contract violation)
    - rating 1-2 AND sentiment="Positive" (should be impossible)

  These invariants MUST hold after this module runs.  If any downstream
  consumer of this data sees a violation, it is a bug in THIS file.

INVARIANTS ENFORCED:
  1. urgent == true  → sentiment MUST be "Negative"
  2. severity >= 7   → sentiment MUST be "Negative"
  3. rating <= 2     → sentiment CANNOT be "Positive"
  4. rating == 3     → sentiment CANNOT be "Positive"
  5. urgent == true AND sentiment == "Positive" → IMPOSSIBLE
  6. food_poisoning AND confidence == "HIGH"   → IMPOSSIBLE
  7. Aspect sentiments for hygiene/safety/service complaints → NEGATIVE ONLY
"""


def enforce_contract(record: dict) -> dict:
    """
    Apply hard invariant rules to a fully-computed analysis record.

    Mutates and returns the record.  Every rule is commented with its
    justification.

    This function is idempotent — calling it twice produces the same result.
    """

    # --- Schema / type invariants ---
    assert record.get("confidence") in {"LOW", "MEDIUM", "HIGH"}, (
        f"Invalid confidence value: {record.get('confidence')}"
    )

    # --- Rule 1: Urgency forces Negative sentiment ---
    # Rationale: an urgent review (food poisoning, rude staff, FSSAI threat)
    # is by definition a crisis.  Classifying it as anything other than
    # Negative would be dangerous for alerting and summary systems.
    if record.get("urgent") and record["overall_sentiment"] != "Negative":
        record["overall_sentiment"] = "Negative"
        record["rating_override"] = True

    # --- Rule 2: High severity forces Negative sentiment ---
    # Rationale: severity >= 7 means rude staff, hygiene failure, or worse.
    # Even if VADER somehow scored the text as positive (sarcasm, negation
    # failure), the severity score is a stronger signal.
    if record.get("severity_score", 0) >= 7 and record["overall_sentiment"] != "Negative":
        record["overall_sentiment"] = "Negative"
        record["rating_override"] = True

    # --- Rule 3: Rating 1-2 cannot be Positive ---
    # Rationale: a reviewer who gave 1-2 stars is unhappy.  Even if VADER
    # found positive words in the text, the overall intent is negative.
    # This is a ceiling, not a floor: Negative stays Negative.
    if record.get("rating", 5) <= 2 and record["overall_sentiment"] == "Positive":
        record["overall_sentiment"] = "Negative"
        record["rating_override"] = True

    # --- Rule 4: Rating 3 cannot be Positive ---
    # Rationale: 3 stars is mediocre at best.  Scoring it Positive would
    # inflate aggregate sentiment metrics.
    if record.get("rating", 5) == 3 and record["overall_sentiment"] == "Positive":
        record["overall_sentiment"] = "Neutral"
        record["rating_override"] = True

    # --- Rule 5: Assertion — no urgent + positive (defense in depth) ---
    # If rules 1-2 are correct, this should never trigger.  But we check
    # anyway because silent failure is unacceptable.
    assert not (record.get("urgent") and record["overall_sentiment"] == "Positive"), (
        f"CONTRACT VIOLATION: urgent=true AND sentiment=Positive "
        f"for review_id={record.get('review_id')}"
    )

    # --- Rule 6: Food poisoning cannot have HIGH confidence ---
    # Rationale: food poisoning reviews are inherently high-conflict
    # (strong text + low rating).  Reporting "HIGH" confidence is
    # misleading — it suggests certainty we don't have about the
    # sentiment polarity.
    if record.get("urgency_reason") == "food_poisoning":
        if record.get("confidence") == "HIGH":
            record["confidence"] = "MEDIUM"

    # --- Rule 7: Aspect sentiment enforcement ---
    _enforce_aspect_sentiments(record)

    return record


def _enforce_aspect_sentiments(record: dict) -> None:
    """
    Post-process aspect sentiments to enforce forbidden-outcome rules.

    Forbidden:
    - Hygiene complaint sentences scored positive or neutral
    - Safety issue sentences scored positive or neutral
    - Service complaint sentences scored positive or neutral

    Detection: if any mention contains complaint language OR the rating
    is very low (1-2 stars), the aspect sentiment must be negative.
    """
    aspect_sentiments = record.get("aspect_sentiments", {})
    rating = record.get("rating", 5)

    _COMPLAINT_INDICATORS = {
        "poor", "bad", "worst", "terrible", "horrible", "dirty", "filthy",
        "rude", "slow", "concern", "issue", "issues", "problem", "unsafe",
        "improvement", "improve", "nonexistent", "stained", "flies",
        "cockroach", "unhygienic", "careless", "negligent", "shouting",
        "disrespectful", "mannerless", "pathetic", "disgusting",
        "not clean", "no railing", "negative",
    }

    for aspect in ("hygiene", "safety", "service"):
        if aspect not in aspect_sentiments:
            continue

        data = aspect_sentiments[aspect]
        mentions = data.get("mentions", [])

        mentions_text = " ".join(mentions).lower()
        has_complaint = any(ind in mentions_text for ind in _COMPLAINT_INDICATORS)
        low_rating = rating <= 2

        if data["sentiment"] in ("positive", "neutral") and (has_complaint or low_rating):
            data["sentiment"] = "negative"
            if data.get("score", 0) > 0:
                data["score"] = -abs(data["score"])