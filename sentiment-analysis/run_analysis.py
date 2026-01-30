"""
Entrypoint for the sentiment analysis pipeline.

Usage:
    python run_analysis.py
    python run_analysis.py --input ../tap-google-reviews/reviews.csv
    python run_analysis.py --log-level DEBUG
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
        results.append(record)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d / %d reviews", i + 1, len(reviews))

    # Save outputs
    ensure_output_dir(args.output_dir)
    csv_path = os.path.join(args.output_dir, "analysis_results.csv")
    json_path = os.path.join(args.output_dir, "analysis_results.json")

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
