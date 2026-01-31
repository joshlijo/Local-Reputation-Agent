"""
Response Genie — LLM-powered response drafting agent.

Uses HuggingFace Inference API (chat/completions) to generate
professional responses to negative reviews. Structured worker
with persona, constraints, and validation.
"""

import logging
from huggingface_hub import InferenceClient

logger = logging.getLogger("agent.response")


def _build_messages(
    review_text: str,
    rating: int,
    reviewer_name: str,
    business_name: str,
):
    """
    Build structured chat messages for an instruction-tuned model.
    """
    system_message = (
        f"You are a senior reputation manager for {business_name}. "
        "You respond with empathy and professionalism. "
        "You never admit legal liability or fault. "
        "You never argue with the reviewer. "
        "You always thank the reviewer for their feedback and invite them "
        "to continue the conversation privately via email or phone. "
        "You do not offer refunds, discounts, or free items. "
        "You do not make specific promises. "
        "You write in plain text only, no markdown, no emojis. "
        "Your response must be under 100 words."
    )

    user_message = (
        f"Review by {reviewer_name} (rating: {rating}/5):\n\n"
        f"{review_text}\n\n"
        "Write a single professional response paragraph."
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def draft_response(
    review_text: str,
    rating: int,
    reviewer_name: str,
    business_name: str,
    hf_token: str,
    model: str = "mistralai/Mistral-7B-Instruct-v0.3",
) -> str:
    """
    Generate a draft response to a negative review.

    Returns the draft text, or a failure placeholder if the LLM call fails.
    Never raises — failures are handled gracefully.
    """
    if not hf_token:
        logger.warning("No HF_API_TOKEN set — returning placeholder")
        return "[GENERATION FAILED - no API token configured. Draft manually.]"

    messages = _build_messages(
        review_text=review_text,
        rating=rating,
        reviewer_name=reviewer_name,
        business_name=business_name,
    )

    try:
        client = InferenceClient(token=hf_token)

        result = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )

        text = result.choices[0].message.content.strip()

        # Validation: must be 20–200 words
        word_count = len(text.split())
        if word_count < 20:
            logger.warning(
                "Response too short (%d words), using placeholder",
                word_count,
            )
            return "[GENERATION FAILED - response too short. Draft manually.]"

        if word_count > 200:
            text = " ".join(text.split()[:150])

        return text

    except Exception as e:
        logger.error("HuggingFace API error: %s", e)
        return f"[GENERATION FAILED - {type(e).__name__}. Draft manually.]"
