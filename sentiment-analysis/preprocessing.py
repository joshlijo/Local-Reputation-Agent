"""
Text preprocessing for sentiment analysis.

Design decisions:
- We do NOT strip punctuation because VADER uses "!" and "?" to boost intensity.
- We do NOT remove stop words because VADER needs context ("not good" ≠ "good").
- We do NOT lowercase aggressively because VADER treats ALL-CAPS as emphasis.
- We DO normalise whitespace, expand common contractions, and strip URLs,
  which are noise for sentiment scoring.
"""

import re
import unicodedata


# Common contractions in Indian English reviews.
_CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "can't": "cannot",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "let's": "let us",
    # Smart-quote variants (common when copy-pasting from phones)
    "don\u2019t": "do not",
    "doesn\u2019t": "does not",
    "didn\u2019t": "did not",
    "won\u2019t": "will not",
    "can\u2019t": "cannot",
    "it\u2019s": "it is",
    "that\u2019s": "that is",
}

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Clean review text while preserving sentiment-bearing signals.

    Steps:
    1. Unicode normalise (NFC) — ensures consistent representation of
       characters like ₹ and accented letters.
    2. Strip URLs — they carry no sentiment.
    3. Expand contractions — "don't" → "do not" helps VADER detect negation.
    4. Collapse whitespace — multi-line reviews from Google often have
       double newlines that confuse sentence splitting.
    """
    if not text:
        return ""

    # NFC normalisation keeps composed characters (é stays é, not e + ´)
    text = unicodedata.normalize("NFC", text)

    # Remove URLs
    text = _URL_PATTERN.sub("", text)

    # Expand contractions (case-insensitive match, preserve surrounding case)
    for contraction, expansion in _CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)

    # Collapse runs of whitespace (including newlines) into single spaces,
    # but keep sentence-ending punctuation intact.
    text = re.sub(r"\s+", " ", text).strip()

    return text
