"""Stream definitions for tap-google-reviews."""

from __future__ import annotations

import time
from datetime import datetime

from singer_sdk import typing as th
from singer_sdk.streams import Stream

from tap_google_reviews.scraper import GoogleReviewsScraper


class ReviewsStream(Stream):
    """Stream for Google Maps reviews."""

    name = "reviews"
    primary_keys = ["review_id"]
    replication_key = "review_date"
    is_sorted = False

    schema = th.PropertiesList(
        th.Property("review_id", th.StringType, description="Unique review identifier"),
        th.Property("reviewer_name", th.StringType),
        th.Property("rating", th.IntegerType, description="Star rating 1-5"),
        th.Property(
            "review_text",
            th.StringType,
            description="Review body text (nullable)",
        ),
        th.Property("review_date", th.StringType, description="ISO 8601 date"),
        th.Property("review_link", th.StringType, description="Business review page URL"),
    ).to_dict()

    def get_records(self, context: dict | None) -> iter:
        """Fetch and yield review records with retry logic and incremental filtering."""
        google_maps_url = self.config.get("google_maps_url")
        headless = self.config.get("headless", True)
        max_pages = self.config.get("max_pages", 100)
        rate_limit_seconds = self.config.get("rate_limit_seconds", 1.0)
        initial_full_scrape = self.config.get("initial_full_scrape", True)
        max_retries = 3
        retry_delay = 2.0

        # Read bookmark from Singer state
        last_seen = self.get_starting_replication_key_value(context)
        if last_seen:
            self.logger.info("Incremental sync from bookmark: %s", last_seen)
        elif initial_full_scrape:
            self.logger.info("No bookmark found — running full initial scrape")
        else:
            self.logger.info("No bookmark and initial_full_scrape=False — emitting nothing")
            return

        scraper = GoogleReviewsScraper(
            google_maps_url=google_maps_url,
            headless=headless,
            max_pages=max_pages,
            rate_limit_seconds=rate_limit_seconds,
        )

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                reviews = scraper.scrape()

                # Filter to only new reviews when we have a bookmark
                if last_seen:
                    reviews = [r for r in reviews if r["review_date"] > last_seen]

                self.logger.info("Emitting %d review records", len(reviews))

                for review in reviews:
                    yield review

                return

            except Exception as e:
                self.logger.error(
                    "Scrape attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    str(e),
                )
                if attempt >= max_retries:
                    raise
                time.sleep(retry_delay)