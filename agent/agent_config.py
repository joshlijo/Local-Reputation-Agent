"""
Configuration for the agentic reputation management system.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")

# Database
DB_PATH = os.path.join(BASE_DIR, "reputation.db")

# Input data
INPUT_CSV = os.path.join(PROJECT_ROOT, "tap-google-reviews", "reviews.csv")

# Business identity (used in response agent persona)
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "Cafe Amudham")

# Google Gemini API (for response drafting)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

# Scheduler
SCHEDULE_HOURS = int(os.getenv("SCHEDULE_HOURS", "6"))

# Thresholds
NEGATIVE_RATING_MAX = 3  # rating <= this triggers response drafting

# Sentiment analysis module path
SENTIMENT_DIR = os.path.join(PROJECT_ROOT, "sentiment-analysis")

# Scraper configuration
SCRAPER_EXTRACT_SHARE_LINKS = os.getenv("SCRAPER_EXTRACT_SHARE_LINKS", "true").lower() == "true"
