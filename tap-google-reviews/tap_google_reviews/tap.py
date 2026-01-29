"""TapGoogleReviews - Meltano SDK tap for Google Maps reviews."""

from __future__ import annotations

import logging

from singer_sdk import Stream, Tap
from singer_sdk.typing import (
    BooleanType,
    IntegerType,
    NumberType,
    PropertiesList,
    Property,
    StringType,
)

from tap_google_reviews.streams import ReviewsStream

logger = logging.getLogger(__name__)


class TapGoogleReviews(Tap):
    """Google Maps reviews tap built with the Meltano Singer SDK."""

    name = "tap-google-reviews"

    config_jsonschema = PropertiesList(
        Property(
            "google_maps_url",
            StringType,
            required=True,
            description="Google Maps short link or full URL for the business.",
        ),
        Property(
            "headless",
            BooleanType,
            default=True,
            description="Run Playwright browser in headless mode.",
        ),
        Property(
            "max_pages",
            IntegerType,
            default=100,
            description="Maximum number of scroll actions (safety limit).",
        ),
        Property(
            "rate_limit_seconds",
            NumberType,
            default=1.0,
            description="Seconds to wait between scroll actions.",
        ),
        Property(
            "initial_full_scrape",
            BooleanType,
            default=True,
            description="On first run (no state), scrape all reviews.",
        ),
    ).to_dict()

    def discover_streams(self) -> list[Stream]:
        """Return a list of discovered streams."""
        return [ReviewsStream(tap=self)]


if __name__ == "__main__":
    TapGoogleReviews.cli()