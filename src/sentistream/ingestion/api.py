import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from sentistream.shared.config import settings
from sentistream.shared.kafka_client import get_kafka_producer
from sentistream.shared.schemas import ReviewRaw


logger = logging.getLogger(settings.app.name)

# Global producer instance
producer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app to securely open/close Kafka connections."""
    global producer
    logger.info("Starting up Ingestion API...")
    try:
        producer = await get_kafka_producer()
        yield
    finally:
        logger.info("Shutting down Ingestion API...")
        if producer:
            await producer.stop()


# Initialize FastAPI application
app = FastAPI(
    title="SentiStream Ingestion API",
    description="High-throughput API for ingesting customer reviews.",
    lifespan=lifespan,
)


class ReviewCreate(BaseModel):
    """Schema for incoming HTTP requests. ID is optional and auto-generated if omitted."""

    text: str
    id: str | None = None
    metadata: dict | None = None


@app.post("/reviews", status_code=status.HTTP_202_ACCEPTED)
async def ingest_review(review: ReviewCreate):
    """
    Receives a raw text review, normalizes it, and publishes it to the Kafka
    topic to enter the event-driven ML pipeline.
    """
    if not producer:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Kafka producer not initialized",
        )

    # Auto-generate a unique ID if the client didn't supply one
    review_id = review.id or str(uuid.uuid4())

    # Construct the strictly validated pipeline schema
    raw_review = ReviewRaw(id=review_id, text=review.text, metadata=review.metadata)

    topic = settings.kafka.topics.get("reviews_raw", "reviews_raw")

    try:
        # Publish asynchronously to Kafka
        await producer.send(
            topic,
            value=raw_review.model_dump(mode="json"),
            key=review_id.encode("utf-8"),
        )
        return {"status": "accepted", "id": review_id}
    except Exception as e:
        logger.error(f"Failed to publish review {review_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to enqueue review"
        ) from e


# Added a basic healthcheck for container orchestration
@app.get("/health")
async def health_check():
    return {"status": "healthy", "kafka_connected": producer is not None}
