"""
Aspect-Based Sentiment Analysis (ABSA).

Approach:
1. Split the review into sentences (regex-based).
2. For each sentence, check if it contains keywords for any aspect.
3. Run VADER on the matched sentence to get per-aspect sentiment.
4. Apply negation-awareness: if the sentence contains negation words near
   the aspect keyword, flip or clamp the VADER score.
5. Enforce forbidden outcomes: hygiene complaints, safety issues, and
   service complaints must NEVER be scored as positive.

REFACTOR NOTES (why each change was made):

BUG 1 — Negation blindness:
  VADER handles some negation ("not good" → negative) but misses many
  patterns common in Indian English reviews:
    "don't expect food, hygiene, or service to actually exist"
    "the level of cleanliness is negative at best"
    "no responsible staff are available"
  Fix: explicit negation word detection at sentence level (broad by design) of the
  aspect keyword.  If negation is found AND VADER scored positive,
  we flip the score negative.

BUG 2 — Forbidden positive outcomes:
  A sentence like "Hygiene is a big concern" was scored positive by VADER
  (the word "big" has mild positive valence in VADER's lexicon).
  Contract forbids: hygiene complaints positive, safety issues positive,
  service complaints positive.
  Fix: after scoring, check for negative-indicator words in the sentence.
  If found AND the aspect is hygiene/safety/service, clamp to negative.

BUG 3 — Crowd management mapped to ambience:
  "crowd management" complaints are about service failure, not venue
  ambience.  Fixed in config.py: "crowd", "packed", "crowded" moved
  to service keywords.
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


# --- Negation detection ---
# These words, when appearing within a short window of an aspect keyword,
# indicate the sentiment is being negated.  VADER catches some of these
# but not reliably in complex Indian English sentence structures.
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "lack", "without",
    "absent", "nonexistent", "none", "cannot",
    # Expanded forms from preprocessing (contractions are expanded)
    "do not", "does not", "did not", "will not", "would not",
    "could not", "should not", "is not", "are not", "was not",
    "were not", "has not", "have not", "had not",
})

# Single-word negators for fast word-level scanning
_NEGATION_SINGLES = frozenset({
    "not", "no", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "lack", "without",
    "absent", "nonexistent", "none", "cannot",
})


# --- Negative indicator words ---
# If these appear in a sentence about hygiene/safety/service, the sentence
# is expressing a complaint, regardless of what VADER scores.
# This catches patterns like "hygiene is a big concern" where VADER sees
# "big" as positive.
_NEGATIVE_INDICATORS = frozenset({
    "average", "okay", "ok", "decent", "delay", "delayed", "waiting",
"poor", "bad", "worst", "terrible", "horrible", "awful", "pathetic",
    "disgusting", "concern", "issue", "issues", "problem", "problems",
    "complaint", "dirty", "filthy", "stained", "flies", "cockroach",
    "rude", "shouting", "shout", "disrespectful", "mannerless",
    "slow", "careless", "negligent", "unsafe", "hazard", "danger",
    "improvement", "improve", "needs", "lacking", "nonexistent",
    "negative", "unhygienic", "unclean", "declined", "decreased",
    "overpriced", "expensive",
})

# Aspects where positive sentiment on a complaint sentence is forbidden.
# If the sentence contains negative indicators AND matches these aspects,
# the sentiment MUST be negative, never positive.
_COMPLAINT_SENSITIVE_ASPECTS = frozenset({"hygiene", "safety", "service"})


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    Handles:
    - Standard sentence-ending punctuation (. ! ?)
    - Numbered lists ("1. Food is good  2. Service is bad")
    - Newlines as sentence boundaries (common in Google reviews)

    Returns at least one element (the full text) even if no boundaries found.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?=\d+\.\s)", text)
    sentences = [s.strip() for s in parts if s and s.strip()]
    return sentences if sentences else [text]


def _sentence_has_negation(sentence: str) -> bool:
    """
    Check if a sentence contains negation words.

    Uses simple word-level scanning.  This is intentionally broad — it's
    better to flag a sentence as negated (and potentially flip a false-positive
    VADER score) than to miss negation and let a complaint score as positive.
    """
    words = sentence.lower().split()
    return bool(_NEGATION_SINGLES.intersection(words))


def _sentence_has_negative_indicators(sentence: str) -> bool:
    """Check if a sentence contains words that indicate a complaint."""
    words_in_sentence = set(sentence.lower().split())
    return bool(_NEGATIVE_INDICATORS.intersection(words_in_sentence))


def _score_sentence(sentence: str, aspect: str) -> float:
    """
    Score a single sentence's sentiment for a given aspect.

    Applies three layers of correction on top of raw VADER:
    1. Negation flip: if VADER scored positive but negation words present,
       flip the score.
    2. Forbidden positive: if the aspect is complaint-sensitive and the
       sentence contains negative indicators, clamp to negative.
    3. Score floor: ensure that obvious complaint language never produces
       a positive score for sensitive aspects.
    """
    normalised = normalize_hinglish(sentence)
    compound = _vader.polarity_scores(normalised)["compound"]

    has_negation = _sentence_has_negation(sentence)
    has_neg_indicators = _sentence_has_negative_indicators(sentence)

    # Layer 1: Negation flip.
    # If VADER scored positive (> 0) but the sentence has negation,
    # the true sentiment is likely negative.  Flip the sign.
    # Example: "do not expect hygiene" → VADER might see "expect" as positive.
    if compound > 0 and has_negation:
        compound = -abs(compound)

    # Layer 2: Forbidden positive for complaint-sensitive aspects.
    # If the sentence contains explicit negative indicators AND the aspect
    # is one where positive-on-complaint is forbidden, force negative.
    if aspect in _COMPLAINT_SENSITIVE_ASPECTS and has_neg_indicators and compound > 0:
        compound = -0.3  # Mild negative — we know it's a complaint but not how severe.

    # Layer 3: General negative indicator override.
    # Even for non-sensitive aspects, if the sentence has strong negative
    # indicators and VADER scored it positive, clamp to at most neutral.
    if has_neg_indicators and compound > 0:
        compound = min(compound, 0.0)
    
    # Final floor: complaint sentences must not end up neutral
    if has_neg_indicators and compound >= 0:
        compound = -0.1

    return compound


def detect_aspects(review_text: str) -> dict:
    """
    Detect aspects and their sentiment in a review.

    Returns:
      aspects_detected   – list of aspect names found
      aspect_sentiments  – dict mapping each aspect to:
          sentiment  – "positive" / "neutral" / "negative"
          score      – average corrected VADER compound across matched sentences
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

    # Score each aspect with negation-aware, forbidden-outcome-safe scoring
    aspect_sentiments = {}
    for aspect, matched in aspect_sentences.items():
        scores = [_score_sentence(sentence, aspect) for sentence in matched]

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
            "mentions": matched[:3],
        }

    return {
        "aspects_detected": list(aspect_sentiments.keys()),
        "aspect_sentiments": aspect_sentiments,
    }
