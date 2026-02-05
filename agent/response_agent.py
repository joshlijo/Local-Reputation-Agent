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

Write a single professional response paragraph."""


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
    max_retries: int = 3,
) -> tuple[str, bool, bool]:
    """
    Generate a draft response to a negative review using Google ADK.

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

    user_message = (
        f"Reviewer: {reviewer_name}\n"
        f"Rating: {rating}/5 stars\n"
        f"Review: {review_text}"
    )

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
                        return "[GENERATION FAILED - API quota exhausted. Draft manually.]", False, True
                    raise RuntimeError(event.error_message)

            text = response_text.strip()
            if not text:
                logger.warning("ADK agent returned empty response")
                return "[GENERATION FAILED - empty response. Draft manually.]", False, False

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
