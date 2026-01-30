"""
Central configuration for the sentiment analysis pipeline.

All thresholds, keyword dictionaries, and paths live here so that
tuning the system means editing one file, not hunting through modules.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "..", "tap-google-reviews", "reviews.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# VADER sentiment thresholds
# ---------------------------------------------------------------------------
# VADER compound score ranges from -1 to +1.
# These thresholds were chosen after inspecting the distribution of scores
# on the Cafe Amudham dataset: most clearly positive reviews score > 0.05,
# and clearly negative ones score < -0.05.  The narrow neutral band reflects
# the fact that genuinely neutral restaurant reviews are rare — most people
# either praise or complain.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# When the VADER compound score falls in a low-confidence zone (close to 0),
# the star rating can override the text-based classification.
# This helps with very short reviews ("Ok.", "Good") where VADER has little
# signal, and with sarcastic or Hinglish-heavy text where VADER under-performs.
RATING_OVERRIDE_ZONE = 0.15  # |score| below this → rating may override
RATING_POSITIVE_MIN = 4      # 4-5 stars → positive when in override zone
RATING_NEGATIVE_MAX = 2      # 1-2 stars → negative when in override zone

# Weight applied to the Hinglish lexicon boost before adding to VADER score.
# Kept low (0.3) because most reviews are primarily English and VADER handles
# those well; the boost is a correction for the minority of Hindi/Hinglish words.
HINGLISH_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# Aspect keyword mappings
# ---------------------------------------------------------------------------
# Each aspect maps to a list of keywords (English + Hinglish).
# Word-boundary matching is used to avoid false positives like "priceless"
# matching "price".  Food keywords intentionally include specific dish names
# common at South Indian restaurants because reviewers rarely write "the food";
# they name the dish.
ASPECT_KEYWORDS = {
    "food": [
        # Generic English
        "food", "taste", "tasty", "dish", "dishes", "meal", "meals",
        "breakfast", "lunch", "dinner", "snack", "menu", "delicious",
        "flavour", "flavor", "spicy", "bland", "oily", "crispy", "fresh",
        "stale", "undercooked", "overcooked", "portion", "quantity",
        # South Indian specifics (Cafe Amudham menu items)
        "dosa", "idli", "idly", "vada", "sambar", "chutney", "pongal",
        "upma", "rice", "bisibele", "bath", "roti", "rotti", "bajji",
        "coffee", "tea", "filter coffee", "ghee", "butter", "puri",
        "sagu", "poori", "kesari", "khara",
        # Hinglish
        "khana", "khaana", "khane", "swad", "swaad",
    ],
    "service": [
        "service", "staff", "waiter", "server", "manager", "wait",
        "waiting", "slow", "fast", "quick", "rude", "polite", "friendly",
        "helpful", "attentive", "cashier", "billing", "order", "delivery",
        "self-service", "counter",
        # Hinglish
        "seva", "kaam", "behave", "behavior", "behaviour",
    ],
    "hygiene": [
        "hygiene", "hygienic", "clean", "cleanliness", "dirty", "filthy",
        "wash", "unwashed", "sanitize", "sanitise", "flies", "cockroach",
        "insect", "stain", "stained", "plates", "utensils", "stomach",
        "food poisoning", "poisoning", "sick", "vomit", "diarrhea",
        "diarrhoea", "infection",
        # Hinglish
        "saaf", "safai", "ganda", "gandagi",
    ],
    "price": [
        "price", "pricing", "expensive", "cheap", "cost", "costly",
        "value", "money", "worth", "overpriced", "affordable", "budget",
        "rupee", "rupees", "rs", "inr",
        # Hinglish
        "mehnga", "mehenga", "sasta", "daam", "paisa",
    ],
    "ambience": [
        "ambience", "ambiance", "atmosphere", "decor", "decoration",
        "seating", "seat", "space", "spacious", "crowded", "crowd",
        "packed", "location", "parking", "interior", "exterior", "vibe",
        "music", "noise", "noisy", "lake", "view",
        # Hinglish
        "mahaul", "jagah",
    ],
    "safety": [
        "safety", "safe", "unsafe", "railing", "stairs", "staircase",
        "accident", "injury", "hazard", "fire", "emergency", "security",
        "guard",
    ],
}

# ---------------------------------------------------------------------------
# Urgency detection patterns
# ---------------------------------------------------------------------------
# Severity scale 0-10.  The threshold for flagging a review as "urgent"
# is deliberately set at 6 — high enough to filter noise from casual
# complaints but low enough to catch rude-staff incidents that erode trust.
URGENCY_PATTERNS = {
    "food_poisoning": {
        "keywords": [
            "food poisoning", "poisoning", "hospitalized", "hospitalised",
            "hospital", "fell sick", "severe stomach", "gut issues",
            "vomiting", "diarrhea", "diarrhoea", "food borne",
            "foodborne", "toxic",
        ],
        "severity": 10,
    },
    "hygiene_severe": {
        "keywords": [
            "dirty kitchen", "filthy", "flies in", "cockroach",
            "unwashed", "stained plates", "insect", "unhygienic",
            "not clean",
        ],
        "severity": 9,
    },
    "authority_escalation": {
        "keywords": [
            "fssai", "health department", "health inquiry",
            "health inspector", "legal action", "legal notice",
            "complaint to", "report to", "consumer court",
            "food safety", "inspection", "authority",
        ],
        "severity": 8,
    },
    "safety_concern": {
        "keywords": [
            "unsafe", "no railing", "broken stairs", "accident",
            "injury", "hazard", "fire safety", "emergency exit",
        ],
        "severity": 7,
    },
    "rude_staff": {
        "keywords": [
            "rude", "shouting", "shout", "disrespectful", "mannerless",
            "misbehave", "misbehaved", "arguing", "abusive", "arrogant",
            "insult", "insulted", "humiliate", "humiliated",
        ],
        "severity": 6,
    },
}

URGENCY_THRESHOLD = 6  # minimum severity to flag as urgent
