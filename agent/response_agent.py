"""
Response Genie — LLM-powered response drafting agent.

Uses Google Gemini API to generate professional responses to negative reviews.
Structured worker with persona, constraints, and validation.
"""

import logging
import os
import time

import google.generativeai as genai

logger = logging.getLogger("agent.response")

# Module-level model instance (lazy initialization)
_model = None


def _get_model(api_key: str, model_name: str = "gemini-flash-lite-latest"):
    """
    Lazy-initialize the Gemini model.
    """
    global _model

    if _model is not None:
        return _model

    if not api_key:
        logger.warning("GEMINI_API_KEY not set — Gemini disabled")
        return None

    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 300,
            },
        )
        logger.info("Gemini model initialized: %s", model_name)
        return _model
    except Exception as e:
        logger.error("Failed to initialize Gemini: %s", e)
        return None


def _build_prompt(
    review_text: str,
    rating: int,
    reviewer_name: str,
    business_name: str,
) -> str:
    """
    Build the prompt for Gemini with persona and constraints.
    """
    return f"""You are a senior reputation manager for {business_name}.

CONSTRAINTS:
- Respond with empathy and professionalism
- NEVER admit legal liability or fault
- NEVER argue with the reviewer
- Always thank the reviewer for their feedback
- Invite them to continue the conversation privately via email or phone
- Do NOT offer refunds, discounts, or free items
- Do NOT make specific promises
- Write in plain text only, no markdown, no emojis
- Response must be 50-100 words

REVIEW TO RESPOND TO:
Reviewer: {reviewer_name}
Rating: {rating}/5 stars
Review: {review_text}

Write a single professional response paragraph:"""


def draft_response(
    review_text: str,
    rating: int,
    reviewer_name: str,
    business_name: str,
    api_key: str,
    model_name: str = "gemini-flash-lite-latest",
    max_retries: int = 3,
) -> tuple[str, bool, bool]:
    """
    Generate a draft response to a negative review using Google Gemini.

    Returns (text, success, quota_exhausted):
      - success=True: text is a real AI draft
      - success=False, quota_exhausted=True: API quota hit, caller should stop
      - success=False, quota_exhausted=False: transient failure for this review
    Never raises — failures are handled gracefully.
    """
    if not api_key:
        logger.warning("No GEMINI_API_KEY set — returning placeholder")
        return "[GENERATION FAILED - no API key configured. Draft manually.]", False, False

    model = _get_model(api_key, model_name)
    if model is None:
        return "[GENERATION FAILED - Gemini unavailable. Draft manually.]", False, False

    prompt = _build_prompt(
        review_text=review_text,
        rating=rating,
        reviewer_name=reviewer_name,
        business_name=business_name,
    )

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            result = model.generate_content(prompt)

            # Check for blocked content
            if not result.candidates:
                logger.warning("Gemini returned no candidates (possibly blocked)")
                return "[GENERATION FAILED - content blocked. Draft manually.]", False, False

            text = result.text.strip()

            word_count = len(text.split())
            if word_count < 20:
                logger.warning(
                    "Response too short (%d words), using placeholder",
                    word_count,
                )
                return "[GENERATION FAILED - response too short. Draft manually.]", False, False

            if word_count > 200:
                text = " ".join(text.split()[:150])

            return text, True, False

        except Exception as e:
            error_str = str(e).lower()

            # Check for quota exhaustion — signal caller to stop
            if "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str:
                logger.warning("Gemini API quota exhausted: %s", e)
                return "[GENERATION FAILED - API quota exhausted. Draft manually.]", False, True

            # Retry for transient errors
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    "Gemini API error (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                logger.error("Gemini API error after %d attempts: %s", max_retries, e)
                return f"[GENERATION FAILED - {type(e).__name__}. Draft manually.]", False, False

    return "[GENERATION FAILED - max retries exceeded. Draft manually.]", False, False
