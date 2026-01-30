"""
Hinglish language support for sentiment analysis.

VADER was trained on English text and does not recognise Hindi or Hinglish
words.  This module provides:

1. A sentiment lexicon mapping common Hinglish words to VADER-scale scores
   (-4 to +4, matching VADER's internal lexicon range).
2. A normalisation function that replaces Hinglish words with English
   equivalents so that VADER can process them.

The lexicon was curated from patterns observed in Cafe Amudham reviews and
common Indian-English restaurant review vocabulary.  It is intentionally
small and conservative — only words with unambiguous sentiment polarity are
included to avoid false positives.
"""

import re

# VADER-scale sentiment scores for Hinglish words.
# Positive values = positive sentiment, negative = negative.
# Magnitudes follow VADER conventions: 1-2 mild, 2-3 moderate, 3-4 strong.
HINGLISH_SENTIMENT = {
    # --- Negative ---
    "bakwas": -2.5,       # rubbish / nonsense
    "bekar": -2.0,        # useless
    "bekaar": -2.0,       # useless (variant spelling)
    "ganda": -2.5,        # dirty
    "gandagi": -3.0,      # filth
    "kharab": -2.0,       # spoiled / bad
    "faltu": -1.5,        # waste / pointless
    "wahiyat": -3.0,      # terrible
    "ghatiya": -3.0,      # low-quality / disgusting
    "bura": -2.0,         # bad
    "tatti": -3.5,        # vulgar: terrible
    "mehnga": -1.0,       # expensive (mild negative in value context)
    "mehenga": -1.0,      # variant
    # --- Positive ---
    "achha": 2.0,         # good
    "accha": 2.0,         # good (variant)
    "acha": 2.0,          # good (variant)
    "badhiya": 2.5,       # excellent
    "zabardast": 3.0,     # awesome
    "mast": 2.0,          # great / fun
    "sahi": 1.5,          # right / good
    "shaandar": 3.0,      # splendid
    "lajawab": 3.0,       # matchless / outstanding
    "kamaal": 2.5,        # wonderful
    "shandaar": 3.0,      # variant of shaandar
    "tagda": 2.0,         # strong / solid (positive in food context)
    "sasta": 1.0,         # cheap / affordable (mild positive for value)
    # --- Neutral / mild ---
    "thik": 0.5,          # okay / fine
    "theek": 0.5,         # okay (variant)
    "chalta": 0.3,        # passable
}

# Map Hinglish words to English equivalents so VADER can score the
# surrounding context correctly.  Only content words are mapped;
# stop words and particles are left alone.
_HINGLISH_TO_ENGLISH = {
    "khana": "food",
    "khaana": "food",
    "khane": "food",
    "seva": "service",
    "saaf": "clean",
    "safai": "cleanliness",
    "ganda": "dirty",
    "gandagi": "filth",
    "sasta": "cheap",
    "mehnga": "expensive",
    "mehenga": "expensive",
    "daam": "price",
    "paisa": "money",
    "mahaul": "ambience",
    "jagah": "place",
    "bahut": "very",
    "bohot": "very",
    "ekdum": "totally",
    "bilkul": "completely",
    "zyada": "too much",
    "kam": "less",
    "achha": "good",
    "accha": "good",
    "acha": "good",
    "bura": "bad",
    "bakwas": "rubbish",
    "bekar": "useless",
    "bekaar": "useless",
    "badhiya": "excellent",
    "zabardast": "awesome",
    "mast": "great",
    "lajawab": "outstanding",
    "wahiyat": "terrible",
    "ghatiya": "disgusting",
}

# Pre-compile a regex that matches any Hinglish word at word boundaries.
# Sorted longest-first so "mehenga" matches before "mehng" (defensive).
_sorted_words = sorted(_HINGLISH_TO_ENGLISH.keys(), key=len, reverse=True)
_HINGLISH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _sorted_words) + r")\b",
    re.IGNORECASE,
)


def normalize_hinglish(text: str) -> str:
    """Replace Hinglish words with English equivalents for VADER processing."""
    def _replace(match):
        word = match.group(0).lower()
        return _HINGLISH_TO_ENGLISH.get(word, word)
    return _HINGLISH_PATTERN.sub(_replace, text)


def calculate_hinglish_boost(text: str) -> float:
    """
    Scan text for Hinglish sentiment words and return an aggregate score.

    The score is the mean of all matched word polarities, or 0.0 if no
    Hinglish sentiment words are found.  Using the mean (not sum) prevents
    long reviews with many Hinglish words from getting disproportionate boosts.
    """
    text_lower = text.lower()
    scores = []
    for word, score in HINGLISH_SENTIMENT.items():
        # Word-boundary check to avoid partial matches
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            scores.append(score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
