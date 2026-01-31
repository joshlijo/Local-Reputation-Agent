"""
Gemini LLM agent for semantic review analysis.

This module wraps Google's Generative AI SDK to produce structured
sentiment, aspect, and urgency opinions for restaurant reviews.

The LLM acts as a "senior reputation analyst" — it provides semantic
interpretations that the deterministic pipeline may miss (sarcasm,
Hinglish idioms, implicit complaints).  Its output is an *opinion*,
not the final answer.  The caller in run_analysis.py fuses it with
deterministic signals, and enforce_contract() has final authority.

Failure mode: if the API call fails or returns invalid JSON, this
module returns None.  The caller must treat None as "use deterministic
result only."
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_ALLOWED_ASPECTS = {"food", "service", "hygiene", "price", "ambience", "safety"}
_ALLOWED_SENTIMENTS = {"Positive", "Neutral", "Negative"}
_ALLOWED_ASPECT_SENTIMENTS = {"positive", "neutral", "negative"}
_ALLOWED_URGENCY_REASONS = {
    "food_poisoning", "hygiene_severe", "rude_staff",
    "safety_concern", "authority_escalation", None,
}

_SYSTEM_PROMPT = (
    "You are a senior reputation analyst for Indian restaurants. "
    "You analyze Google Business reviews to extract sentiment, aspects, and urgency. "
    "You understand Indian English, Hinglish, and regional food terminology. "
    "Be precise. Do not infer what is not stated. "
    "Do not hallucinate aspects that the reviewer did not mention."
)

_USER_PROMPT_TEMPLATE = """Analyze this restaurant review.

Reviewer: {reviewer_name}
Rating: {rating}/5
Review: {review_text}

Return a JSON object with exactly these fields:
- "overall_sentiment": one of "Positive", "Neutral", "Negative"
- "aspects": array of objects, each with:
  - "aspect": one of "food", "service", "hygiene", "price", "ambience", "safety"
  - "sentiment": one of "positive", "neutral", "negative"
  - "evidence": short quoted snippet from the review (max 15 words)
- "urgent": true if the review describes something requiring immediate business attention (food poisoning, hygiene crisis, safety hazard, abusive staff, legal threat), false otherwise
- "urgency_reason": one of "food_poisoning", "hygiene_severe", "rude_staff", "safety_concern", "authority_escalation", or null if not urgent
- "reasoning": one sentence explaining your overall assessment

Only include aspects the reviewer actually mentioned. Return valid JSON only."""

# Lazy-initialized client.  _model_unavailable is a sentinel to avoid
# repeating the same warning on every call after the first failure.
_model = None
_model_unavailable = False


def _get_model():
    """Lazy-initialize the Gemini model. Returns None if unavailable."""
    global _model, _model_unavailable
    if _model is not None:
        return _model
    if _model_unavailable:
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai package not installed; LLM disabled")
        _model_unavailable = True
        return None

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set; LLM disabled")
        _model_unavailable = True
        return None

    genai.configure(api_key=api_key)
    _model = genai.GenerativeModel(
        model_name=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        system_instruction=_SYSTEM_PROMPT,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )
    return _model


def _validate_response(data: dict) -> dict | None:
    """Validate and sanitize LLM response. Returns None if invalid."""
    if not isinstance(data, dict):
        return None

    sentiment = data.get("overall_sentiment")
    if sentiment not in _ALLOWED_SENTIMENTS:
        return None

    # Filter aspects to only allowed values
    raw_aspects = data.get("aspects", [])
    if not isinstance(raw_aspects, list):
        raw_aspects = []
    clean_aspects = []
    for a in raw_aspects:
        if not isinstance(a, dict):
            continue
        if a.get("aspect") not in _ALLOWED_ASPECTS:
            continue
        if a.get("sentiment") not in _ALLOWED_ASPECT_SENTIMENTS:
            continue
        clean_aspects.append({
            "aspect": a["aspect"],
            "sentiment": a["sentiment"],
            "evidence": str(a.get("evidence", ""))[:200],
        })

    urgent = bool(data.get("urgent", False))
    urgency_reason = data.get("urgency_reason")
    if urgency_reason not in _ALLOWED_URGENCY_REASONS:
        urgency_reason = None
    if not urgent:
        urgency_reason = None

    reasoning = str(data.get("reasoning", ""))[:500]

    return {
        "overall_sentiment": sentiment,
        "aspects": clean_aspects,
        "urgent": urgent,
        "urgency_reason": urgency_reason,
        "reasoning": reasoning,
    }


def analyze_review(review_text: str, rating: int, reviewer_name: str) -> dict | None:
    """
    Send a review to Gemini for semantic analysis.

    Returns a validated dict on success, or None on any failure.
    The caller should treat None as "fall back to deterministic only."
    """
    model = _get_model()
    if model is None:
        return None

    prompt = _USER_PROMPT_TEMPLATE.format(
        reviewer_name=reviewer_name,
        rating=rating,
        review_text=review_text,
    )

    try:
        response = model.generate_content(prompt)
        raw = response.text
    except Exception as e:
        logger.warning("Gemini API call failed: %s", e)
        return None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Gemini returned invalid JSON: %.200s", raw)
        return None

    result = _validate_response(data)
    if result is None:
        logger.warning("Gemini response failed validation: %.200s", raw)
    return result
