"""
Entrypoint for the sentiment analysis pipeline.

Usage:
    python run_analysis.py
    python run_analysis.py --input ../tap-google-reviews/reviews.csv
    python run_analysis.py --log-level DEBUG

Pipeline order:
    1. Load reviews from CSV
    2. Deduplicate by review_id (MANDATORY — duplicates skew aggregates)
    3. For each review: sentiment → aspects → urgency
    4. Post-process: enforce contract invariants (sentiment_rules.py)
    5. Save outputs (CSV + JSON)
"""

import argparse
import os
import sys

# Ensure the sentiment-analysis package directory is on sys.path so that
# sibling module imports work regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import setup_logging, load_reviews, ensure_output_dir, save_csv, save_json
from sentiment import classify_sentiment
from aspects import detect_aspects
from urgency import detect_urgency
from sentiment_rules import enforce_contract


def _deduplicate_reviews(reviews: list[dict], logger) -> list[dict]:
    """
    Deduplicate reviews by review_id, keeping only the first occurrence.

    DEDUPLICATION STRATEGY (documented per contract requirement):
    - Uniqueness key: review_id (SHA-256 hash of review content)
    - Policy: first-seen wins.  If two reviews share a review_id, the
      first one in the input CSV is kept and duplicates are discarded.
    - Rationale: duplicates can arise from re-scraping the same review,
      or from the same reviewer posting identical text.  Counting them
      twice inflates sentiment aggregates and skews urgency counts.
    - Logging: duplicate count is logged at WARNING level so operators
      can investigate the data source if duplicates are frequent.
    """
    seen = set()
    unique = []
    dupe_count = 0

    for review in reviews:
        rid = review.get("review_id")
        if rid in seen:
            dupe_count += 1
            continue
        seen.add(rid)
        unique.append(review)

    if dupe_count > 0:
        logger.warning(
            "Deduplicated %d duplicate review_id(s) — %d unique reviews remain",
            dupe_count, len(unique),
        )

    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Analyse restaurant reviews: sentiment, aspects, urgency."
    )
    parser.add_argument(
        "--input",
        default=config.INPUT_CSV,
        help="Path to the input reviews CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help="Directory for output files (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("Starting sentiment analysis pipeline")

    # Load reviews
    reviews = load_reviews(args.input)
    logger.info("Loaded %d reviews from %s", len(reviews), args.input)

    # Deduplicate BEFORE any scoring (contract requirement)
    reviews = _deduplicate_reviews(reviews, logger)

    # Process each review
    results = []
    for i, review in enumerate(reviews):
        text = review["review_text"]
        rating = review["rating"]

        sentiment_result = classify_sentiment(text, rating)
        aspect_result = detect_aspects(text)
        urgency_result = detect_urgency(text, rating)

        record = {
            "review_id": review["review_id"],
            "reviewer_name": review["reviewer_name"],
            "review_text": text,
            "rating": rating,
            "review_date": review["review_date"],
            # Sentiment
            "overall_sentiment": sentiment_result["overall_sentiment"],
            "vader_compound": sentiment_result["vader_compound"],
            "final_score": sentiment_result["final_score"],
            "confidence": sentiment_result["confidence"],
            "rating_override": sentiment_result["rating_override"],
            # Aspects
            "aspects_detected": aspect_result["aspects_detected"],
            "aspect_sentiments": aspect_result["aspect_sentiments"],
            # Urgency
            "urgent": urgency_result["urgent"],
            "urgency_reason": urgency_result["urgency_reason"],
            "severity_score": urgency_result["severity_score"],
            "matched_patterns": urgency_result["matched_patterns"],
        }

        # --- CONTRACT ENFORCEMENT (NEW) ---
        # This is the final safety net.  After all three modules have
        # produced their results, enforce cross-module invariants:
        #   - urgency overrides sentiment (always)
        #   - rating ceiling is respected
        #   - forbidden aspect sentiment states are corrected
        # See sentiment_rules.py for the full list of invariants.
        record = enforce_contract(record)

        results.append(record)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d / %d reviews", i + 1, len(reviews))

    # Save outputs
    ensure_output_dir(args.output_dir)
    csv_path = os.path.join(args.output_dir, "analysis_results.csv")
    json_path = os.path.join(args.output_dir, "analysis_results.json")

    assert len(results) == len({r["review_id"] for r in results}), (
        "Duplicate review_id detected in output results"
    )

    save_csv(results, csv_path)
    save_json(results, json_path)

    logger.info("Results saved to %s", args.output_dir)

    # Print summary
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a quick summary of the analysis to stdout."""
    total = len(results)
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
    urgent_count = 0
    aspect_counts: dict[str, int] = {}

    for r in results:
        sentiments[r["overall_sentiment"]] += 1
        if r["urgent"]:
            urgent_count += 1
        for a in r["aspects_detected"]:
            aspect_counts[a] = aspect_counts.get(a, 0) + 1

    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total reviews analysed: {total}")
    print()
    print("Sentiment distribution:")
    for label, count in sentiments.items():
        pct = count / total * 100 if total else 0
        print(f"  {label:10s}: {count:4d}  ({pct:.1f}%)")
    print()
    print(f"Urgent reviews: {urgent_count}")
    print()
    if aspect_counts:
        print("Aspect mentions:")
        for aspect, count in sorted(aspect_counts.items(), key=lambda x: -x[1]):
            print(f"  {aspect:12s}: {count:4d}")
    print("=" * 50)


if __name__ == "__main__":
    main()
