import asyncio
import json
import logging
import os
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from sentistream.shared.config import settings
from sentistream.shared.db import AsyncSessionLocal, redis_client
from sentistream.shared.kafka_client import get_kafka_consumer, get_kafka_producer
from sentistream.shared.models import ReviewRecord
from sentistream.shared.schemas import ReviewClustered, ReviewEmbedded
from sentistream.worker.clusterer import StreamClusterer


logger = logging.getLogger(settings.app.name)


async def load_clusterer_state(clusterer: StreamClusterer, state_path: str) -> None:
    if clusterer.load_state(state_path):
        logger.info(
            "Loaded DBSTREAM state from %s (replayed %s points)",
            state_path,
            clusterer.get_last_replay_count(),
        )
    else:
        logger.info("No DBSTREAM state found. Starting fresh.")


async def get_kafka_consumer_with_retry(
    input_topic: str, max_retries: int, retry_delay: float
) -> AIOKafkaConsumer | None:
    consumer: AIOKafkaConsumer | None = None
    for attempt in range(1, max_retries + 1):
        try:
            consumer = await get_kafka_consumer(input_topic, group_id="clusterer-group")
            return consumer
        except Exception as e:
            logger.warning(
                f"Kafka consumer connection failed (attempt {attempt}/{max_retries}): {e}"
            )
            if attempt == max_retries:
                raise
            await asyncio.sleep(retry_delay)
    return consumer


async def process_message(
    msg: Any,
    clusterer: StreamClusterer,
    save_every: int,
    state_path: str,
    output_topic: str,
    producer: AIOKafkaProducer,
) -> None:
    try:
        # 1. Parse incoming embedding
        data = msg.value
        if not isinstance(data, dict):
            logger.error(f"Kafka message value is not a dict: {data}")
            return
        try:
            embedded_review = ReviewEmbedded(**data)
        except Exception as e:
            logger.error(f"Failed to construct ReviewEmbedded: {e}; data: {data}")
            return

        # 2. Stateful Clustering (Fast enough to run synchronously in loop)
        cluster_id = clusterer.get_cluster(embedded_review.reduced_coords)

        if save_every > 0 and clusterer.records_processed % save_every == 0:
            try:
                clusterer.save_state(state_path)
                logger.debug(
                    "DBSTREAM state saved (%s points).",
                    clusterer.get_recent_points_count(),
                )
            except Exception as e:
                logger.warning(f"Failed to save DBSTREAM state: {e}")

        # 3. Create next-stage payload
        clustered_review = ReviewClustered(
            **embedded_review.model_dump(), cluster_id=cluster_id
        )
        clustered_dict = clustered_review.model_dump(mode="json")

        # 4. Forward to next topic (Namer Service)
        await producer.send(
            output_topic,
            value=clustered_dict,
            key=clustered_review.id.encode("utf-8"),
        )

        # 5. Save to PostgreSQL Database persistently
        async with AsyncSessionLocal() as session:
            db_record = ReviewRecord(
                id=clustered_review.id,
                text=clustered_review.text,
                timestamp=clustered_review.timestamp.replace(tzinfo=None),
                metadata_col=clustered_review.metadata,
                cluster_id=clustered_review.cluster_id,
                cluster_name=None,  # Handled asynchronously by Namer if enabled
                reduced_coords=clustered_review.reduced_coords,
                full_embedding=clustered_review.embedding,
            )
            session.add(db_record)
            await session.commit()

        # 6. Push real-time event to Dash board via Redis Pub/Sub
        # Strip the massive 384D embedding from frontend stream to save bandwidth
        dash_payload = {
            "id": clustered_review.id,
            "text": clustered_review.text,
            "coords": clustered_review.reduced_coords,
            "cluster_id": cluster_id,
        }
        await redis_client.publish("dash_stream", json.dumps(dash_payload))

        logger.debug(f"Clustered review: {embedded_review.id} -> Cluster {cluster_id}")

    except Exception as e:
        logger.error(f"Clusterer failed on message: {e}", exc_info=True)


async def run_clusterer_service():
    """
    Consumes embedded reviews, runs River DBStream statefully,
    and pushes cluster assignments to `reviews_clustered`.
    Must run as a single global instance (or appropriately partitioned).
    """
    input_topic = settings.kafka.topics.get("reviews_embedded", "reviews_embedded")
    output_topic = settings.kafka.topics.get("reviews_clustered", "reviews_clustered")

    logger.info("Initializing River DBStream Clusterer...")
    clusterer = StreamClusterer()
    state_path = os.getenv("SENTISTREAM_DBSTREAM_STATE", "data/dbstream_state.json")
    save_every = int(os.getenv("SENTISTREAM_DBSTREAM_SAVE_EVERY", "200"))
    await load_clusterer_state(clusterer, state_path)

    max_retries = int(os.getenv("SENTISTREAM_KAFKA_RETRIES", "30"))
    retry_delay = float(os.getenv("SENTISTREAM_KAFKA_RETRY_DELAY", "2.0"))
    consumer = await get_kafka_consumer_with_retry(
        input_topic, max_retries, retry_delay
    )
    producer = await get_kafka_producer()

    logger.info(f"Clusterer Service listening on {input_topic}...")
    try:
        if consumer is not None:
            async for msg in consumer:
                await process_message(
                    msg, clusterer, save_every, state_path, output_topic, producer
                )
        else:
            logger.error("Kafka consumer is not initialized; cannot process messages.")
    finally:
        try:
            clusterer.save_state(state_path)
            logger.info(
                "DBSTREAM state saved on shutdown (%s points).",
                clusterer.get_recent_points_count(),
            )
        except Exception as e:
            logger.warning(f"Failed to save DBSTREAM state on shutdown: {e}")
        if consumer is not None:
            await consumer.stop()
        if producer is not None:
            await producer.stop()
        await redis_client.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=settings.app.log_level)
    asyncio.run(run_clusterer_service())
