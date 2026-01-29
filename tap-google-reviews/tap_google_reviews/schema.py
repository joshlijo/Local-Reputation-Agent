"""JSON schema definitions for the reviews stream."""

from singer_sdk.typing import (
    IntegerType,
    PropertiesList,
    Property,
    StringType,
)

REVIEWS_SCHEMA = PropertiesList(
    Property("review_id", StringType, required=True, description="SHA256 hash ID"),
    Property("reviewer_name", StringType, required=True),
    Property("rating", IntegerType, required=True, description="Star rating 1-5"),
    Property("review_text", StringType, description="Review body text (nullable)"),
    Property("review_date", StringType, required=True, description="ISO 8601 date"),
    Property("review_link", StringType, required=True, description="Business review page URL"),
).to_dict()
