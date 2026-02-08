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
import csv
import json
import logging
import os
import random
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
from db import (
    init_db, get_seen_ids, insert_review, insert_response,
    get_failed_draft_ids, update_response, get_review_by_id,
    get_undrafted_negative_ids, get_all_reviews,
)
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
    """Run the Google Reviews tap to fetch fresh review data.

    Tries Meltano ELT pipeline first (``meltano run``), which provides
    automatic state management and Singer-spec orchestration.  Falls back
    to direct tap invocation if Meltano is not installed or not initialized.
    """
    tap_dir = os.path.join(PROJECT_ROOT, "tap-google-reviews")

    # --- Attempt 1: Meltano ELT pipeline ---
    # Only attempt if plugins have been installed (extractors dir exists inside .meltano)
    meltano_extractors = os.path.join(PROJECT_ROOT, ".meltano", "extractors")
    if os.path.isdir(meltano_extractors):
        try:
            logger.info("Running tap via Meltano ELT pipeline...")
            result = subprocess.run(
                ["meltano", "run", "tap-google-reviews", "target-jsonl"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                # target-jsonl writes raw records to {destination_path}/{stream_name}.jsonl
                meltano_output = os.path.join(tap_dir, "reviews.jsonl")
                if os.path.exists(meltano_output):
                    logger.info("Converting Meltano output to CSV...")
                    conv = subprocess.run(
                        [sys.executable, "convert_jsonl_to_csv.py", "reviews.jsonl"],
                        cwd=tap_dir,
                        capture_output=True,
                        text=True,
                    )
                    if conv.returncode == 0:
                        logger.info("Meltano ELT pipeline complete")
                        return True
                    logger.warning("CSV conversion failed: %s", conv.stderr)
            else:
                logger.warning("Meltano run failed: %s", result.stderr[:500])
        except FileNotFoundError:
            logger.info("Meltano not installed — using direct tap invocation")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            logger.warning("Meltano error (%s) — falling back to direct invocation", exc)
    else:
        logger.info("Meltano not initialized — using direct tap invocation")

    # --- Attempt 2: Direct tap invocation (fallback) ---
    jsonl_path = os.path.join(tap_dir, "output.jsonl")

    logger.info("Running tap-google-reviews scraper...")
    with open(jsonl_path, "w", encoding="utf-8") as out:
        result = subprocess.run(
            [sys.executable, "-m", "tap_google_reviews.tap", "--config", "config.json"],
            cwd=tap_dir,
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
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

    # -------------------------------------------------------------------------
    # 3. Process each new review (collect first, draft later)
    # -------------------------------------------------------------------------
    urgent_queue = []
    normal_negative_queue = []

    if not new_reviews:
        logger.info("No new reviews to process")


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
    # 4. Re-queue previously failed drafts + undrafted negatives
    # -------------------------------------------------------------------------
    failed_ids = get_failed_draft_ids()
    undrafted_ids = get_undrafted_negative_ids(agent_config.NEGATIVE_RATING_MAX)
    backlog_ids = failed_ids | undrafted_ids
    retry_queue = []
    if backlog_ids:
        logger.info(
            "Found %d reviews needing drafts (%d failed, %d undrafted)",
            len(backlog_ids), len(failed_ids), len(undrafted_ids),
        )
        for rid in backlog_ids:
            review_row = get_review_by_id(rid)
            if review_row:
                aspects_raw = {}
                try:
                    aspects_raw = json.loads(review_row.get("aspects", "{}") or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass
                record = {
                    "aspect_sentiments": aspects_raw,
                    "urgency_reason": "none",
                    "severity_score": review_row.get("severity_score", 0),
                    "matched_patterns": [],
                    "urgent": bool(review_row.get("urgent", 0)),
                }
                retry_queue.append((review_row, record))

    # -------------------------------------------------------------------------
    # 5. Draft responses (URGENT first, then normal, then retries — capped)
    # -------------------------------------------------------------------------
    all_negatives = urgent_queue + normal_negative_queue + retry_queue
    drafted = 0
    failed = 0
    attempted = 0
    quota_hit = False

    for review, record in all_negatives:
        if drafted >= MAX_DRAFTS_PER_RUN or quota_hit:
            break

        # Pre-call delay to stay under free-tier RPM limit (~5 RPM for 2.0-flash)
        if attempted > 0:
            delay = random.uniform(10, 15)
            logger.info("Rate-limit delay: %.1fs before next API call", delay)
            time.sleep(delay)

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

        is_retry = review["review_id"] in failed_ids
        if is_retry:
            update_response(review["review_id"], text)
        else:
            insert_response(review["review_id"], text)
        attempted += 1

        if success:
            drafted += 1
        else:
            failed += 1
            if quota_exhausted:
                # Wait and retry once before giving up entirely
                logger.warning("Quota signal received — waiting 60s before retry...")
                time.sleep(60)
                text2, success2, still_exhausted = draft_response(
                    review["review_text"],
                    review["rating"],
                    review["reviewer_name"],
                    agent_config.BUSINESS_NAME,
                    agent_config.GEMINI_API_KEY,
                    agent_config.GEMINI_MODEL,
                    aspects=aspects,
                    urgency_info=urgency_info,
                )
                if success2:
                    if is_retry:
                        update_response(review["review_id"], text2)
                    else:
                        update_response(review["review_id"], text2)
                    drafted += 1
                    failed -= 1  # undo the earlier increment
                else:
                    quota_hit = True
                    logger.warning("API quota confirmed exhausted — stopping draft generation")

    # Note: un-attempted reviews are left without entries — next run picks them up

    logger.info(
        "Cycle complete: %d reviews processed, %d AI responses drafted, %d failed (cap=%d)",
        len(new_reviews),
        drafted,
        failed,
        MAX_DRAFTS_PER_RUN,
    )

    export_sentiment_csv()


def export_sentiment_csv():
    """Export all reviews with sentiment analysis results to CSV."""
    out_path = os.path.join(PROJECT_ROOT, "sentiment_results.csv")
    reviews = get_all_reviews()
    if not reviews:
        logger.info("No reviews to export")
        return

    fieldnames = [
        "reviewer_name", "rating", "review_date", "overall_sentiment",
        "urgent", "severity_score", "aspects", "review_text",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in reviews:
            # Parse aspects JSON into readable string
            aspects_str = ""
            try:
                aspects = json.loads(r.get("aspects") or "{}")
                aspects_str = ", ".join(
                    f"{k}: {v.get('sentiment', '?')}" if isinstance(v, dict) else f"{k}: {v}"
                    for k, v in aspects.items()
                )
            except (json.JSONDecodeError, TypeError):
                pass
            r["aspects"] = aspects_str
            r["urgent"] = "Yes" if r.get("urgent") else "No"
            writer.writerow(r)

    logger.info("Sentiment results exported to %s (%d reviews)", out_path, len(reviews))


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