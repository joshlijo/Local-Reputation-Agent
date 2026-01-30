"""
Shared utilities for file I/O and logging.
"""

import csv
import json
import logging
import os
from typing import Any


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the pipeline logger."""
    logger = logging.getLogger("sentiment_analysis")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def load_reviews(filepath: str) -> list[dict]:
    """
    Load reviews.csv into a list of dicts.

    Uses the csv module instead of pandas to keep dependencies minimal.
    Handles UTF-8-BOM that Excel sometimes adds, and treats empty
    review_text fields as empty strings rather than None.
    """
    reviews = []
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["rating"] = int(row["rating"])
            row["review_text"] = row.get("review_text") or ""
            reviews.append(row)
    return reviews


def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(output_dir, exist_ok=True)


def save_csv(results: list[dict], filepath: str) -> None:
    """
    Save analysis results to CSV with UTF-8 encoding.

    Complex fields (lists, dicts) are JSON-serialised into their cells
    so the CSV remains parseable by downstream tools.
    """
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            serialised = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    serialised[k] = json.dumps(v, ensure_ascii=False)
                else:
                    serialised[k] = v
            writer.writerow(serialised)


def save_json(results: list[dict], filepath: str) -> None:
    """Save analysis results to JSON with UTF-8 encoding and pretty formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
