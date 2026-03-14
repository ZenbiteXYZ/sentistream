from datetime import datetime

from pydantic import BaseModel, Field


class ReviewRaw(BaseModel):
    id: str
    text: str
    timestamp: datetime = Field(default_factory=datetime.now())
    metadata: dict | None = None


class ReviewEmbedded(ReviewRaw):
    embedding: list[float]
    reduced_coords: list[float]  # length 5 from UMAP


class ReviewClustered(ReviewEmbedded):
    cluster_id: int


class ProcessedReview(ReviewClustered):
    cluster_name: str | None = None
