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
import subprocess
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
MAX_DRAFTS_PER_RUN = int(os.getenv("MAX_DRAFTS_PER_RUN", "5"))  # cost / rate-limit guardrail

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("agent.scheduler")


def run_tap():
    """Run the Google Reviews tap to fetch fresh review data."""
    tap_dir = os.path.join(PROJECT_ROOT, "tap-google-reviews")
    jsonl_path = os.path.join(tap_dir, "output.jsonl")

    logger.info("Running tap-google-reviews scraper...")
    with open(jsonl_path, "w") as out:
        result = subprocess.run(
            [sys.executable, "-m", "tap_google_reviews.tap", "--config", "config.json"],
            cwd=tap_dir,
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        logger.error("Tap failed: %s", result.stderr)
        return False

    logger.info("Converting JSONL to CSV...")
    result = subprocess.run(
        [sys.executable, "convert_jsonl_to_csv.py"],
        cwd=tap_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("JSONL-to-CSV conversion failed: %s", result.stderr)
        return False

    logger.info("Tap pipeline complete")
    return True


def run_cycle():
    """Execute one full processing cycle."""
    logger.info("Starting processing cycle")

    # -------------------------------------------------------------------------
    # 0. Scrape fresh reviews
    # -------------------------------------------------------------------------
    if not run_tap():
        logger.warning("Tap failed — falling back to existing CSV (if any)")

    # -------------------------------------------------------------------------
    # 1. Load reviews
    # -------------------------------------------------------------------------
    try:
        reviews = load_reviews(agent_config.INPUT_CSV)
    except FileNotFoundError:
        logger.error("Input CSV not found: %s", agent_config.INPUT_CSV)
        logger.info("The tap may have failed to produce output. Check logs above.")
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
    # 3. Process each new review (collect first, draft later)
    # -------------------------------------------------------------------------
    urgent_queue = []
    normal_negative_queue = []

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

        record = enforce_contract(record)
        insert_review(record)

        # Queue for drafting later (keep record for structured context)
        if rating <= agent_config.NEGATIVE_RATING_MAX:
            if record["urgent"]:
                urgent_queue.append((review, record))
            else:
                normal_negative_queue.append((review, record))

        if (i + 1) % 25 == 0:
            logger.info(
                "Processed %d / %d new reviews",
                i + 1,
                len(new_reviews),
            )

    # -------------------------------------------------------------------------
    # 4. Draft responses (URGENT first, capped)
    # -------------------------------------------------------------------------
    all_negatives = urgent_queue + normal_negative_queue
    drafted = 0
    failed = 0
    attempted = 0
    quota_hit = False

    for review, record in all_negatives:
        if drafted >= MAX_DRAFTS_PER_RUN or quota_hit:
            break

        # Extract aspect sentiments as simple dict for the response agent
        aspects = {}
        if record.get("aspect_sentiments"):
            for aspect, info in record["aspect_sentiments"].items():
                if isinstance(info, dict):
                    aspects[aspect] = info.get("sentiment", "neutral")
                else:
                    aspects[aspect] = str(info)

        urgency_info = {
            "reason": record.get("urgency_reason", "none"),
            "severity": record.get("severity_score", 0),
            "patterns": record.get("matched_patterns", []),
        }

        text, success, quota_exhausted = draft_response(
            review["review_text"],
            review["rating"],
            review["reviewer_name"],
            agent_config.BUSINESS_NAME,
            agent_config.GEMINI_API_KEY,
            agent_config.GEMINI_MODEL,
            aspects=aspects,
            urgency_info=urgency_info,
        )

        insert_response(review["review_id"], text)
        attempted += 1

        if success:
            drafted += 1
            # Rate-limit delay: free-tier Gemini caps at ~15 RPM
            time.sleep(4)
        else:
            failed += 1
            if quota_exhausted:
                quota_hit = True
                logger.warning("API quota exhausted — stopping draft generation")

    # Add placeholders for remaining negatives
    for review, record in all_negatives[attempted:]:
        insert_response(
            review["review_id"],
            "[AI draft skipped due to rate limit — please draft manually.]",
        )

    logger.info(
        "Cycle complete: %d reviews processed, %d AI responses drafted, %d failed (cap=%d)",
        len(new_reviews),
        drafted,
        failed,
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