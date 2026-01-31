"""
Background scheduler — the agentic heartbeat.

Runs the review processing pipeline on a schedule:
1. Load reviews from CSV
2. Detect new reviews (not yet in DB)
3. Run sentiment analysis on each new review
4. Draft responses for SOME negative reviews (rating <= 3), capped per run
5. Save everything to SQLite for the Streamlit UI

Usage:
    python scheduler.py              # run once + start schedule loop
    python scheduler.py --once       # run once and exit
"""

import argparse
import logging
import os
import sys
import time

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(AGENT_DIR, "..")

sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sentiment-analysis"))

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import schedule

import agent_config
from db import init_db, get_seen_ids, insert_review, insert_response
from response_agent import draft_response

# Sentiment pipeline
from sentiment import classify_sentiment
from aspects import detect_aspects
from urgency import detect_urgency
from sentiment_rules import enforce_contract
from utils import load_reviews

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MAX_DRAFTS_PER_RUN = 5  # cost / rate-limit guardrail

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("agent.scheduler")


def run_cycle():
    """Execute one full processing cycle."""
    logger.info("Starting processing cycle")

    # -------------------------------------------------------------------------
    # 1. Load reviews
    # -------------------------------------------------------------------------
    try:
        reviews = load_reviews(agent_config.INPUT_CSV)
    except FileNotFoundError:
        logger.error("Input CSV not found: %s", agent_config.INPUT_CSV)
        return

    logger.info("Loaded %d reviews from CSV", len(reviews))

    # -------------------------------------------------------------------------
    # 2. Detect new reviews
    # -------------------------------------------------------------------------
    seen = get_seen_ids()
    new_reviews = [r for r in reviews if r["review_id"] not in seen]

    logger.info(
        "Found %d new reviews (%d already processed)",
        len(new_reviews),
        len(seen),
    )

    if not new_reviews:
        logger.info("No new reviews — cycle complete")
        return

    # -------------------------------------------------------------------------
    # 3. Process each new review
    # -------------------------------------------------------------------------
    drafted = 0

    for i, review in enumerate(new_reviews):
        text = review["review_text"]
        rating = review["rating"]

        # --- Sentiment pipeline ---
        sentiment_result = classify_sentiment(text, rating)
        aspect_result = detect_aspects(text)
        urgency_result = detect_urgency(text, rating)

        record = {
            "review_id": review["review_id"],
            "reviewer_name": review["reviewer_name"],
            "review_text": text,
            "rating": rating,
            "review_date": review.get("review_date", ""),
            "overall_sentiment": sentiment_result["overall_sentiment"],
            "vader_compound": sentiment_result["vader_compound"],
            "final_score": sentiment_result["final_score"],
            "confidence": sentiment_result["confidence"],
            "rating_override": sentiment_result["rating_override"],
            "aspects_detected": aspect_result["aspects_detected"],
            "aspect_sentiments": aspect_result["aspect_sentiments"],
            "urgent": urgency_result["urgent"],
            "urgency_reason": urgency_result["urgency_reason"],
            "severity_score": urgency_result["severity_score"],
            "matched_patterns": urgency_result["matched_patterns"],
        }

        # Enforce sentiment contract invariants
        record = enforce_contract(record)

        # Save review to DB
        insert_review(record)

        # ---------------------------------------------------------------------
        # 4. Draft response for SOME negative reviews (rate-limited)
        # ---------------------------------------------------------------------
        if rating <= agent_config.NEGATIVE_RATING_MAX:
            if drafted < MAX_DRAFTS_PER_RUN:
                draft = draft_response(
                    text,
                    rating,
                    review["reviewer_name"],
                    agent_config.BUSINESS_NAME,
                    agent_config.HF_TOKEN,
                    agent_config.HF_MODEL,
                )
                insert_response(review["review_id"], draft)
                drafted += 1
            else:
                # LLM budget exhausted — still enqueue without AI draft
                insert_response(
                    review["review_id"],
                    "[AI draft skipped due to rate limit — please draft manually.]",
                )

        if (i + 1) % 25 == 0:
            logger.info(
                "Processed %d / %d new reviews",
                i + 1,
                len(new_reviews),
            )

    logger.info(
        "Cycle complete: %d reviews processed, %d AI responses drafted (cap=%d)",
        len(new_reviews),
        drafted,
        MAX_DRAFTS_PER_RUN,
    )


def main():
    parser = argparse.ArgumentParser(description="Reputation agent scheduler")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    init_db()
    run_cycle()

    if args.once:
        return

    logger.info(
        "Starting schedule loop (every %d hours)",
        agent_config.SCHEDULE_HOURS,
    )
    schedule.every(agent_config.SCHEDULE_HOURS).hours.do(run_cycle)

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()