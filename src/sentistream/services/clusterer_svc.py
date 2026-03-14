import asyncio
import json
import logging

from sentistream.shared.config import settings
from sentistream.shared.db import redis_client
from sentistream.shared.kafka_client import get_kafka_consumer, get_kafka_producer
from sentistream.shared.schemas import ReviewClustered, ReviewEmbedded
from sentistream.worker.clusterer import StreamClusterer


logger = logging.getLogger(settings.app.name)


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

    consumer = await get_kafka_consumer(input_topic, group_id="clusterer-group")
    producer = await get_kafka_producer()

    logger.info(f"Clusterer Service listening on {input_topic}...")
    try:
        async for msg in consumer:
            try:
                # 1. Parse incoming embedding
                data = msg.value
                embedded_review = ReviewEmbedded(**data)

                # 2. Stateful Clustering (Fast enough to run synchronously in loop)
                cluster_id = clusterer.get_cluster(embedded_review.reduced_coords)

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

                # 5. Push real-time event to Dash board via Redis Pub/Sub
                # Strip the massive 384D embedding from frontend stream to save bandwidth
                dash_payload = {
                    "id": clustered_review.id,
                    "text": clustered_review.text,
                    "coords": clustered_review.reduced_coords,
                    "cluster_id": cluster_id,
                }
                await redis_client.publish("dash_stream", json.dumps(dash_payload))

                logger.debug(
                    f"Clustered review: {embedded_review.id} -> Cluster {cluster_id}"
                )

            except Exception as e:
                logger.error(f"Clusterer failed on message: {e}", exc_info=True)

    finally:
        await consumer.stop()
        await producer.stop()
        await redis_client.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=settings.app.log_level)
    asyncio.run(run_clusterer_service())
