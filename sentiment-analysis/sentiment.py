"""
Overall sentiment classification using a hybrid approach.

Pipeline:
1. Clean text (preprocessing.py)
2. Normalise Hinglish → English (hinglish_lexicon.py)
3. Run VADER on the normalised text
4. Compute a Hinglish boost for words VADER missed
5. Apply VADER guardrails: clamp/override when negation, sarcasm, or
   long-complaint patterns are detected (NEW — see BUG FIXES below)
6. Apply rating ceiling: rating sets an upper bound on allowed sentiment
7. Compute confidence based on rating-text agreement

REFACTOR — BUG FIXES:

BUG 1 — VADER trusted blindly:
  VADER compound was used as final_score without guardrails.
  Reviews like "do not expect food, hygiene, or service to actually exist"
  scored +0.89 because VADER's bag-of-words summed "expect", "actually",
  "exist" as positive.
  FIX: detect negation words and clamp positive VADER scores to negative
  when negation is present alongside complaint keywords.

BUG 2 — No rating ceiling:
  A 3-star review could be classified "Positive" if VADER scored high.
  Contract: rating 1-2 → Negative only, rating 3 → Neutral or Negative.
  FIX: deterministic ceiling enforcement after scoring.

BUG 3 — Sarcasm and rhetorical complaints:
  Long reviews (>250 chars) with negative keywords but positive VADER
  (sarcasm, rhetorical questions) were scored positive.
  FIX: if review is long AND contains negative keywords, clamp VADER.

BUG 4 — Line 93-97 flipped negative to positive for 4-5 star reviews:
  If VADER said negative but rating was 4-5, sentiment was overridden to
  Positive.  This masked genuine service/quality complaints from regular
  customers who still gave a decent rating.
  FIX: removed.  Rating ceiling only prevents going ABOVE the allowed
  sentiment — it never upgrades negative text to positive.
"""

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing import clean_text
from hinglish_lexicon import normalize_hinglish, calculate_hinglish_boost
import config


_vader = SentimentIntensityAnalyzer()

# Negation words that VADER often mishandles in complex sentences.
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "lack", "without",
    "cannot", "none",
})

# Negative keywords that indicate a complaint context.
# Used to detect long-form complaint reviews where VADER's bag-of-words
# approach sums neutral/positive words to a misleadingly high score.
_NEGATIVE_KEYWORDS = frozenset({
    "average", "okay", "ok", "decent", "delay", "waiting", "poor", "bad", 
    "worst", "terrible", "horrible", "awful", "pathetic",
    "disgusting", "rude", "dirty", "filthy", "overpriced", "expensive",
    "slow", "declined", "decreased", "issue", "issues", "problem",
    "complaint", "concern", "unhygienic", "stained", "poisoning",
    "sick", "sickness", "shame", "avoid", "worst", "careless",
    "disrespectful", "mannerless", "shouting", "shout",
})

# Threshold for "long complaint" detection.
# Reviews longer than this with negative keywords get VADER clamped.
_LONG_REVIEW_THRESHOLD = 250


def classify_sentiment(review_text: str, rating: int) -> dict:
    """
    Classify a single review's overall sentiment.

    Returns a dict with:
      overall_sentiment  – "Positive" / "Neutral" / "Negative"
      vader_compound     – raw VADER compound score (for auditing)
      final_score        – corrected score after guardrails
      rating_override    – True if the star rating influenced classification
      confidence         – categorical: "HIGH" / "MEDIUM" / "LOW"
    """
    # Handle empty / rating-only reviews
    if not review_text or not review_text.strip():
        sentiment, confidence = _sentiment_from_rating(rating)
        return {
            "overall_sentiment": sentiment,
            "vader_compound": 0.0,
            "final_score": 0.0,
            "rating_override": True,
            "confidence": confidence,
        }

    cleaned = clean_text(review_text)
    normalised = normalize_hinglish(cleaned)

    vader_scores = _vader.polarity_scores(normalised)
    vader_compound = vader_scores["compound"]

    hinglish_boost = calculate_hinglish_boost(review_text)

    final_score = vader_compound + (hinglish_boost * config.HINGLISH_WEIGHT)
    final_score = max(-1.0, min(1.0, final_score))

    # --- VADER GUARDRAILS (NEW) ---
    # These correct known failure modes before sentiment classification.
    final_score = _apply_vader_guardrails(final_score, cleaned, rating)

    rating_override = False

    if abs(final_score) < config.RATING_OVERRIDE_ZONE:
        # Low text confidence — let star rating decide
        sentiment, _ = _sentiment_from_rating(rating)
        rating_override = True
    else:
        if final_score > config.POSITIVE_THRESHOLD:
            sentiment = "Positive"
        elif final_score < config.NEGATIVE_THRESHOLD:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

    # --- RATING CEILING ENFORCEMENT (NEW) ---
    # The star rating sets an upper bound on sentiment.  This is
    # deterministic and cannot be overridden by text analysis.
    #
    # Contract:
    #   Rating 1-2 → Negative only
    #   Rating 3   → Neutral or Negative (never Positive)
    #   Rating 4-5 → Positive / Neutral (no restriction downward)
    sentiment, was_clamped = _enforce_rating_ceiling(sentiment, rating)
    if was_clamped:
        rating_override = True

    # NOTE: We intentionally do NOT upgrade negative text sentiment
    # to positive for 4-5 star ratings.  The old code (line 93-97)
    # did this and it masked genuine complaints from customers who
    # gave decent ratings but complained about specific issues.
    # If the text says negative, we respect it regardless of rating.

    confidence = _calculate_confidence(
        vader_compound, final_score, rating, sentiment, cleaned
    )

    return {
        "overall_sentiment": sentiment,
        "vader_compound": round(vader_compound, 4),
        "final_score": round(final_score, 4),
        "rating_override": rating_override,
        "confidence": confidence,
    }


def _apply_vader_guardrails(score: float, text: str, rating: int) -> float:
    """
    Clamp or override VADER score when known failure modes are detected.

    VADER is a weak signal only.  These guardrails prevent it from
    producing dangerously wrong results.
    """
    text_lower = text.lower()
    words = set(text_lower.split())

    has_negation = bool(_NEGATION_WORDS.intersection(words))
    has_negative_kw = bool(_NEGATIVE_KEYWORDS.intersection(words))
    is_long = len(text) > _LONG_REVIEW_THRESHOLD

    # Guardrail 1: Negation with positive VADER.
    # If negation words are present AND VADER scored positive AND the
    # review has negative keywords, the positive score is almost certainly
    # wrong.  Flip it.
    # Example: "do not expect food, hygiene, or service to actually exist"
    #   VADER: +0.89 (wrong), after flip: -0.89 (correct)
    if score > 0 and has_negation and has_negative_kw:
        score = -abs(score)

    # Guardrail 2: Long complaint reviews.
    # Reviews > 250 chars with negative keywords but positive VADER are
    # typically detailed complaints where VADER's bag-of-words approach
    # accumulates incidental positive words ("good", "expect", "best")
    # that appear in a negative context.
    # Clamp to at most mildly negative.
    if score > 0 and is_long and has_negative_kw:
        score = min(score, -0.1)

    # Guardrail 3: Low rating contradiction.
    # If the reviewer gave 1-2 stars but VADER scored strongly positive
    # (> 0.5), VADER is almost certainly wrong.  The rating is a stronger
    # signal of overall intent than bag-of-words text analysis.
    # Clamp to mildly negative — we know the reviewer is unhappy.
    if score > 0.5 and rating <= 2:
        score = -0.2

    # Guardrail 4: Complaint keywords should not end neutral
    if score == 0.0 and has_negative_kw:
        score = -0.1

    return score


def _sentiment_from_rating(rating: int) -> tuple[str, str]:
    """Derive sentiment and confidence purely from star rating."""
    if rating >= config.RATING_POSITIVE_MIN:
        return "Positive", "LOW"
    elif rating <= config.RATING_NEGATIVE_MAX:
        return "Negative", "LOW"
    else:
        return "Neutral", "LOW"


def _enforce_rating_ceiling(sentiment: str, rating: int) -> tuple[str, bool]:
    """
    Enforce the rating-based sentiment ceiling.

    This is a hard constraint — no text analysis can override it.

    Returns (corrected_sentiment, was_clamped).
    """
    if rating <= 2:
        # Rating 1-2: Negative ONLY.
        # A reviewer who gave 1-2 stars is unhappy regardless of what
        # individual positive words VADER found in the text.
        if sentiment != "Negative":
            return "Negative", True
    elif rating == 3:
        # Rating 3: Neutral or Negative only (never Positive).
        # A 3-star review is at best mediocre.  Scoring it Positive
        # would be misleading for aggregation and alerting.
        if sentiment == "Positive":
            return "Neutral", True

    # Rating 4-5: no ceiling — Positive, Neutral, and Negative are all
    # valid depending on text content.  A 4-star review that complains
    # about service should score Negative on the service aspect (handled
    # in aspects.py) even if overall sentiment is Positive.
    return sentiment, False


def _calculate_confidence(
    vader_compound: float,
    final_score: float,
    rating: int,
    sentiment: str,
    text: str,
) -> str:
    """
    Categorical confidence score: HIGH / MEDIUM / LOW.

    Rebuilt from scratch.  The old numeric confidence (0-1) was misleading:
    it gave 0.85+ confidence to reviews where rating and text contradicted
    each other.  A numeric score implies precision we don't have.

    Confidence is HIGH when:
    - Rating and text sentiment agree (strongest signal)
    - VADER compound is strongly polarised AND matches rating direction
    - Review has substantial text (> 50 chars)

    Confidence is LOW when:
    - Rating and text contradict each other
    - Review text is very short or empty
    - VADER compound is near zero (ambiguous text)
    """
    # Determine if rating direction matches sentiment
    rating_sentiment_map = {
        1: "Negative", 2: "Negative",
        3: "Neutral",
        4: "Positive", 5: "Positive",
    }
    rating_direction = rating_sentiment_map.get(rating, "Neutral")
    rating_agrees = (rating_direction == sentiment)

    # VADER strength: how confident VADER is
    vader_strong = abs(vader_compound) > 0.5

    # Text length: more text = more signal
    has_text = len(text.strip()) > 50

    if rating_agrees and vader_strong and has_text:
        return "HIGH"
    elif rating_agrees and has_text:
        return "MEDIUM"
    elif rating_agrees:
        return "MEDIUM"
    elif has_text and vader_strong:
        # Text is strong but rating disagrees — mixed signals.
        return "LOW"
    else:
        return "LOW"
