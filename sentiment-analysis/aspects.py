"""
Aspect-Based Sentiment Analysis (ABSA).

Approach:
1. Split the review into sentences (regex-based, with fallback for
   single-sentence reviews).
2. For each sentence, check if it contains keywords for any aspect.
3. Run VADER on the matched sentence to get per-aspect sentiment.
4. If an aspect is mentioned in multiple sentences, average the scores.

Why keyword-based detection instead of an NER/ML model?
- The aspect set is small and domain-specific (restaurant reviews).
- Keyword lists can be tuned quickly when the restaurant changes
  (e.g. adds a new menu category).
- Fully interpretable — you can trace exactly which keyword triggered
  which aspect, which matters for an internal tool.
- No model download / GPU required.

Limitation: cannot detect implicit aspects ("too pricey" → price) unless
the keyword is present.  We mitigate this by including synonyms and
colloquial terms in the keyword lists.
"""

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing import clean_text
from hinglish_lexicon import normalize_hinglish
import config


_vader = SentimentIntensityAnalyzer()

# Pre-compile word-boundary patterns for each aspect's keywords.
# Sorted longest-first to avoid partial matches ("self-service" before "service").
_ASPECT_PATTERNS: dict[str, re.Pattern] = {}
for _aspect, _keywords in config.ASPECT_KEYWORDS.items():
    sorted_kw = sorted(_keywords, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted_kw) + r")\b",
        re.IGNORECASE,
    )
    _ASPECT_PATTERNS[_aspect] = pattern


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    Handles:
    - Standard sentence-ending punctuation (. ! ?)
    - Numbered lists ("1. Food is good  2. Service is bad")
    - Newlines as sentence boundaries (common in Google reviews)

    Returns at least one element (the full text) even if no boundaries found.
    """
    # Split on period/exclamation/question followed by space or end,
    # or on newlines, or on numbered list markers.
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?=\d+\.\s)", text)
    sentences = [s.strip() for s in parts if s and s.strip()]
    return sentences if sentences else [text]


def detect_aspects(review_text: str) -> dict:
    """
    Detect aspects and their sentiment in a review.

    Returns:
      aspects_detected   – list of aspect names found
      aspect_sentiments  – dict mapping each aspect to:
          sentiment  – "positive" / "neutral" / "negative"
          score      – average VADER compound across matched sentences
          mentions   – up to 3 sentence snippets for explainability
    """
    if not review_text or not review_text.strip():
        return {"aspects_detected": [], "aspect_sentiments": {}}

    cleaned = clean_text(review_text)
    sentences = _split_sentences(cleaned)

    # Collect sentences per aspect
    aspect_sentences: dict[str, list[str]] = {}
    for sentence in sentences:
        for aspect, pattern in _ASPECT_PATTERNS.items():
            if pattern.search(sentence):
                aspect_sentences.setdefault(aspect, []).append(sentence)

    # Score each aspect
    aspect_sentiments = {}
    for aspect, matched in aspect_sentences.items():
        scores = []
        for sentence in matched:
            normalised = normalize_hinglish(sentence)
            compound = _vader.polarity_scores(normalised)["compound"]
            scores.append(compound)

        avg_score = sum(scores) / len(scores)

        if avg_score > config.POSITIVE_THRESHOLD:
            label = "positive"
        elif avg_score < config.NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        aspect_sentiments[aspect] = {
            "sentiment": label,
            "score": round(avg_score, 4),
            # Keep up to 3 mention snippets for transparency
            "mentions": matched[:3],
        }

    return {
        "aspects_detected": list(aspect_sentiments.keys()),
        "aspect_sentiments": aspect_sentiments,
    }
