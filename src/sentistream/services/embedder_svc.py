import asyncio
import logging

from sentistream.shared.config import settings
from sentistream.shared.kafka_client import get_kafka_consumer, get_kafka_producer
from sentistream.shared.schemas import ReviewEmbedded, ReviewRaw
from sentistream.worker.embedder import PipelineEmbedder


logger = logging.getLogger(settings.app.name)


async def run_embedder_service():
    """
    Consumes raw reviews, maps them to embeddings/reduced coords,
    and pushes them to `reviews_embedded`. This service is completely stateless
    and can process messages concurrently or be horizontally scaled without limits.
    """
    input_topic = settings.kafka.topics.get("reviews_raw", "reviews_raw")
    output_topic = settings.kafka.topics.get("reviews_embedded", "reviews_embedded")

    logger.info("Initializing Embedder ONNX model...")
    embedder = PipelineEmbedder()

    # "embedder-group" tracks offsets so scaling out instances balances the queue dynamically
    consumer = await get_kafka_consumer(input_topic, group_id="embedder-group")
    producer = await get_kafka_producer()

    logger.info(f"Embedder Service listening on {input_topic}...")
    try:
        async for msg in consumer:
            try:
                # 1. Parse raw message
                raw_data = msg.value
                review = ReviewRaw(**raw_data)

                # 2. Run inference in a thread to keep async loop fast
                full_embed, reduced_coords = await asyncio.to_thread(
                    embedder.embed_and_reduce, review.text
                )

                # 3. Create next-stage payload
                embedded_review = ReviewEmbedded(
                    **review.model_dump(),
                    embedding=full_embed,
                    reduced_coords=reduced_coords,
                )

                # 4. Forward
                await producer.send(
                    output_topic,
                    value=embedded_review.model_dump(mode="json"),
                    key=review.id.encode("utf-8"),  # Helps Kafka partition balancing
                )
                logger.debug(f"Embedded review: {review.id}")

            except Exception as e:
                logger.error(f"Embedder failed on message: {e}", exc_info=True)

    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=settings.app.log_level)
    asyncio.run(run_embedder_service())
