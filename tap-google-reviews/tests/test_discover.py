"""Tests for tap discovery (schema validation)."""

from tap_google_reviews.schema import REVIEWS_SCHEMA
from tap_google_reviews.tap import TapGoogleReviews


MINIMAL_CONFIG = {
    "google_maps_url": "https://maps.app.goo.gl/cGamxLRZ6VqMfCXB7",
}


def test_schema_has_required_properties():
    """Schema must contain all required fields."""
    props = REVIEWS_SCHEMA["properties"]
    assert "review_id" in props
    assert "reviewer_name" in props
    assert "rating" in props
    assert "review_text" in props
    assert "review_date" in props
    assert "review_link" in props


def test_schema_required_fields():
    """Required fields must be marked as such."""
    required = REVIEWS_SCHEMA.get("required", [])
    for field in ["review_id", "reviewer_name", "rating", "review_date", "review_link"]:
        assert field in required, f"{field} should be required"


def test_review_text_is_nullable():
    """review_text should not be in required list (it's nullable)."""
    required = REVIEWS_SCHEMA.get("required", [])
    assert "review_text" not in required


def test_discover_returns_reviews_stream():
    """Tap.discover_streams() should return a list containing the reviews stream."""
    tap = TapGoogleReviews(config=MINIMAL_CONFIG, parse_env_config=False)
    streams = tap.discover_streams()
    assert len(streams) == 1
    assert streams[0].name == "reviews"


def test_discover_stream_primary_key():
    """The reviews stream primary key should be review_id."""
    tap = TapGoogleReviews(config=MINIMAL_CONFIG, parse_env_config=False)
    stream = tap.discover_streams()[0]
    assert stream.primary_keys == ["review_id"]
