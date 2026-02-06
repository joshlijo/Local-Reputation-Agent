"""
Response Genie — LLM-powered response drafting agent.

Uses Google ADK (Agent Development Kit) with Gemini to generate
professional responses to negative reviews.
Structured worker with persona, constraints, and validation.
"""

import logging
import os
import time

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

logger = logging.getLogger("agent.response")

# Module-level ADK components (lazy initialization)
_runner = None
_session_service = None

PERSONA = """You are a senior reputation manager for {business_name}.

VOICE:
- Write like a real person, not a corporate template
- Reference specific details the reviewer mentioned (a dish, a date, a staff member, their situation)
- Vary your opening — NEVER start with "Dear [Name], thank you for taking the time..."
- Match your emotional register to the complaint severity

HARD RULES:
- NEVER admit legal liability or fault
- NEVER offer refunds, discounts, or free items
- NEVER argue with the reviewer
- Always invite them to continue privately (email or phone)
- Plain text only, no markdown, no emojis
- 50-100 words

You will receive a COMPLAINT ANALYSIS block with the detected complaint type, \
affected aspects, and specific concerns. Use this to tailor your response — \
a food poisoning complaint demands a different tone than a slow-service complaint."""

# Complaint-type-specific guidance injected into the user message
COMPLAINT_GUIDANCE = {
    "food_poisoning": (
        "- Express genuine concern for their health and recovery\n"
        "- Mention that food safety standards are being reviewed internally\n"
        "- Do NOT minimize or question their experience"
    ),
    "hygiene_severe": (
        "- Acknowledge the specific hygiene issue they described\n"
        "- Show that cleanliness standards matter to you\n"
        "- Do NOT make generic 'we take hygiene seriously' claims without specificity"
    ),
    "rude_staff": (
        "- Acknowledge that the staff conduct they described is unacceptable\n"
        "- Reference the specific interaction they mentioned\n"
        "- Do NOT make excuses for the staff"
    ),
    "authority_escalation": (
        "- Take the regulatory concern seriously — do not dismiss it\n"
        "- Show willingness to address the specific compliance issue raised\n"
        "- This reviewer is considering legal/regulatory action — urgency matters"
    ),
    "safety_concern": (
        "- Acknowledge the specific safety hazard they reported\n"
        "- Show that patron safety is being addressed\n"
        "- Do NOT dismiss or minimize the physical risk they described"
    ),
    "none": (
        "- Address the specific complaint they raised\n"
        "- Reference a concrete detail from their review"
    ),
}


def _build_context(review_text, rating, reviewer_name, aspects, urgency_info):
    """Build structured context block for the LLM instead of raw text dump."""
    lines = []

    reason = urgency_info.get("reason", "none") if urgency_info else "none"
    severity = urgency_info.get("severity", 0) if urgency_info else 0
    patterns = urgency_info.get("patterns", []) if urgency_info else []

    if reason != "none":
        label = reason.replace("_", " ").title()
        lines.append(f"COMPLAINT TYPE: {label} (Severity: {severity}/10)")

    if aspects:
        neg_aspects = [a for a, s in aspects.items() if s == "negative"]
        if neg_aspects:
            lines.append(f"NEGATIVE ASPECTS: {', '.join(neg_aspects)}")

    if patterns:
        lines.append(f"KEY CONCERNS: {', '.join(patterns)}")

    lines.append("")
    lines.append(f"Reviewer: {reviewer_name} | Rating: {rating}/5")
    lines.append(f'Review: "{review_text}"')

    lines.append("")
    guidance = COMPLAINT_GUIDANCE.get(reason, COMPLAINT_GUIDANCE["none"])
    lines.append(f"RESPONSE GUIDANCE:\n{guidance}")

    return "\n".join(lines)


def _get_runner(api_key: str, model_name: str, business_name: str):
    """Lazy-initialize the ADK runner and agent."""
    global _runner, _session_service

    if _runner is not None:
        return _runner

    if not api_key:
        logger.warning("GEMINI_API_KEY not set — Gemini disabled")
        return None

    try:
        # ADK reads API key from GEMINI_API_KEY env var
        os.environ["GEMINI_API_KEY"] = api_key

        agent = LlmAgent(
            model=model_name,
            name="response_agent",
            description="Drafts professional responses to negative reviews",
            instruction=PERSONA.format(business_name=business_name),
            generate_content_config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=300,
            ),
        )

        _session_service = InMemorySessionService()
        _runner = Runner(
            agent=agent,
            app_name="reputation-agent",
            session_service=_session_service,
            auto_create_session=True,
        )
        logger.info("ADK agent initialized: %s", model_name)
        return _runner
    except Exception as e:
        logger.error("Failed to initialize ADK agent: %s", e)
        return None


def draft_response(
    review_text: str,
    rating: int,
    reviewer_name: str,
    business_name: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    aspects: dict = None,
    urgency_info: dict = None,
    max_retries: int = 3,
) -> tuple[str, bool, bool]:
    """
    Generate a draft response to a negative review using Google ADK.

    Args:
      aspects: detected aspect sentiments, e.g. {"food": "negative", "hygiene": "negative"}
      urgency_info: urgency details, e.g. {"reason": "food_poisoning", "severity": 10, "patterns": [...]}

    Returns (text, success, quota_exhausted):
      - success=True: text is a real AI draft
      - success=False, quota_exhausted=True: API quota hit, caller should stop
      - success=False, quota_exhausted=False: transient failure for this review
    Never raises — failures are handled gracefully.
    """
    if not api_key:
        logger.warning("No GEMINI_API_KEY set — returning placeholder")
        return "[GENERATION FAILED - no API key configured. Draft manually.]", False, False

    runner = _get_runner(api_key, model_name, business_name)
    if runner is None:
        return "[GENERATION FAILED - ADK agent unavailable. Draft manually.]", False, False

    user_message = _build_context(review_text, rating, reviewer_name, aspects, urgency_info)

    for attempt in range(max_retries):
        try:
            # Each review gets its own session (no conversation history needed)
            session_id = f"review-{hash(review_text) & 0xFFFFFFFF}"

            events = runner.run(
                user_id="scheduler",
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=user_message)],
                ),
            )

            # Extract the final response from events
            response_text = ""
            for event in events:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            response_text += part.text

                # Check for errors in the event
                if event.error_message:
                    error_str = event.error_message.lower()
                    if "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str:
                        logger.warning("Gemini API quota exhausted: %s", event.error_message)
                        return "[AI drafting paused, daily Gemini quota exhausted. Resume automatically on next run. Draft manually.]", False, True
                    raise RuntimeError(event.error_message)

            text = response_text.strip()

            # Strip ADK agent name prefix (e.g. "response_agent:\n\n")
            if text.lower().startswith("response_agent:"):
                text = text[len("response_agent:"):].strip()

            if not text:
                # ADK swallows 429 errors in background threads, resulting in
                # empty responses instead of proper error propagation. Treat
                # empty responses as likely quota exhaustion so the caller stops.
                logger.warning("ADK agent returned empty response (likely 429 quota error)")
                return "[GENERATION FAILED - empty response. Draft manually.]", False, True

            word_count = len(text.split())
            if word_count < 20:
                logger.warning("Response too short (%d words), using placeholder", word_count)
                return "[GENERATION FAILED - response too short. Draft manually.]", False, False

            if word_count > 200:
                text = " ".join(text.split()[:150])

            return text, True, False

        except Exception as e:
            error_str = str(e).lower()

            if "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str:
                logger.warning("Gemini API quota exhausted: %s", e)
                return "[GENERATION FAILED - API quota exhausted. Draft manually.]", False, True

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    "ADK error (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    max_retries,
                    e,
                    wait_time,
                )
                time.sleep(wait_time)
            else:
                logger.error("ADK error after %d attempts: %s", max_retries, e)
                return f"[GENERATION FAILED - {type(e).__name__}. Draft manually.]", False, False

    return "[GENERATION FAILED - max retries exceeded. Draft manually.]", False, False
