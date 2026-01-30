"""
Overall sentiment classification using a hybrid approach.

Pipeline:
1. Clean text (preprocessing.py)
2. Normalise Hinglish → English (hinglish_lexicon.py)
3. Run VADER on the normalised text
4. Compute a Hinglish boost for words VADER missed
5. Combine: final_score = vader_compound + (hinglish_boost × weight)
6. When final_score is in the low-confidence zone, use star rating as tiebreaker

Why VADER?
- Free, no API calls, fast (~1ms per review)
- Good at handling punctuation, caps, emojis — all common in Google reviews
- Its main weakness (non-English words) is patched by the Hinglish lexicon

Why rating calibration?
- A 5-star review that says "Good" has low VADER signal but is clearly positive.
- A 1-star review with sarcasm ("Great, just great") may fool VADER but the
  rating reveals true sentiment.
- We only use the override in the low-confidence zone to avoid masking genuine
  text signals.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from preprocessing import clean_text
from hinglish_lexicon import normalize_hinglish, calculate_hinglish_boost
import config


_vader = SentimentIntensityAnalyzer()


def classify_sentiment(review_text: str, rating: int) -> dict:
    """
    Classify a single review's overall sentiment.

    Returns a dict with:
      overall_sentiment  – "Positive" / "Neutral" / "Negative"
      vader_compound     – raw VADER compound score
      hinglish_boost     – Hinglish lexicon contribution
      final_score        – combined score used for classification
      rating_override    – True if the star rating overrode text analysis
      confidence         – 0-1 heuristic confidence in the label
    """
    # Handle empty / rating-only reviews
    if not review_text or not review_text.strip():
        sentiment, confidence = _sentiment_from_rating(rating)
        return {
            "overall_sentiment": sentiment,
            "vader_compound": 0.0,
            "hinglish_boost": 0.0,
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
    # Clamp to [-1, 1] to stay in VADER's expected range
    final_score = max(-1.0, min(1.0, final_score))

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

    # Strong disagreement override: when VADER says positive but the
    # reviewer gave 1-2 stars (or vice versa), the rating is more reliable.
    # This catches mixed reviews like "Food is good but service is terrible"
    # where VADER's bag-of-words approach sums to positive, but the overall
    # intent is negative as revealed by the low star rating.
    if sentiment == "Positive" and rating <= config.RATING_NEGATIVE_MAX:
        sentiment = "Negative"
        rating_override = True
    elif sentiment == "Negative" and rating >= config.RATING_POSITIVE_MIN:
        # Rare but possible: sarcastic text that VADER reads as negative
        # but 4-5 star rating signals genuine satisfaction.
        sentiment = "Positive"
        rating_override = True

    confidence = _calculate_confidence(vader_compound, hinglish_boost, rating, sentiment)

    return {
        "overall_sentiment": sentiment,
        "vader_compound": round(vader_compound, 4),
        "hinglish_boost": round(hinglish_boost, 4),
        "final_score": round(final_score, 4),
        "rating_override": rating_override,
        "confidence": round(confidence, 2),
    }


def _sentiment_from_rating(rating: int) -> tuple[str, float]:
    """Derive sentiment and confidence purely from star rating."""
    if rating >= config.RATING_POSITIVE_MIN:
        return "Positive", 0.6
    elif rating <= config.RATING_NEGATIVE_MAX:
        return "Negative", 0.6
    else:
        return "Neutral", 0.5


def _calculate_confidence(
    vader_compound: float,
    hinglish_boost: float,
    rating: int,
    sentiment: str,
) -> float:
    """
    Heuristic confidence score (0-1).

    Higher when:
    - VADER compound is strongly polarised (far from 0)
    - Star rating agrees with the text-derived sentiment
    - Hinglish boost reinforces rather than contradicts VADER
    """
    # Base confidence from VADER strength (0.3 - 0.7 range)
    base = 0.3 + 0.4 * min(abs(vader_compound), 1.0)

    # Rating agreement bonus (+0.2 if rating direction matches sentiment)
    rating_agrees = (
        (sentiment == "Positive" and rating >= 4)
        or (sentiment == "Negative" and rating <= 2)
        or (sentiment == "Neutral" and rating == 3)
    )
    if rating_agrees:
        base += 0.2

    # Hinglish reinforcement bonus (+0.1 if boost direction matches VADER)
    if vader_compound * hinglish_boost > 0:
        base += 0.1

    return min(base, 1.0)
