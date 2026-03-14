from datetime import datetime

import pytest
from pydantic import ValidationError

from sentistream.shared.schemas import ProcessedReview, ReviewRaw


def test_review_in_schema():
    """Test that the ReviewRaw schema correctly parses valid data."""
    valid_data = {
        "id": "12345",
        "text": "This is a great product!",
        "metadata": {"source": "web"},
    }

    review = ReviewRaw(**valid_data)

    assert review.id == "12345"
    assert review.text == "This is a great product!"
    assert review.metadata["source"] == "web"
    assert isinstance(review.timestamp, datetime)


def test_review_in_schema_missing_fields():
    """Test that the ReviewRaw schema fails when missing necessary data."""
    invalid_data = {"text": "Missing ID!"}

    with pytest.raises(ValidationError):
        ReviewRaw(**invalid_data)


def test_processed_review_schema():
    """Test that the ProcessedReview properly enforces the extended requirements."""
    valid_processed = {
        "id": "abc",
        "text": "Processing works perfectly",
        "embedding": [0.1, 0.2, 0.3],  # Pretend this is our 384d embed
        "reduced_coords": [0.1, 0.2, 0.3, 0.4, 0.5],  # 5d UMAP
        "cluster_id": 1,
        "cluster_name": "Positive Experience",
    }

    processed = ProcessedReview(**valid_processed)

    assert processed.cluster_id == 1
    assert len(processed.reduced_coords) == 5
    assert processed.cluster_name == "Positive Experience"
