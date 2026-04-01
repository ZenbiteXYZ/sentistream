import asyncio
import json
import logging

from sqlalchemy import update

from sentistream.shared.config import settings
from sentistream.shared.db import AsyncSessionLocal, redis_client
from sentistream.shared.kafka_client import get_kafka_consumer
from sentistream.shared.models import ReviewRecord
from sentistream.shared.schemas import ProcessedReview, ReviewClustered
from sentistream.worker.naming import ClusterNamer


logger = logging.getLogger(settings.app.name)


async def run_namer_service():
    """
    Consumes clustered reviews, manages threshold logic, and triggers LLM
    naming API dynamically. Pushes final topics to DB/Redis.
    """
    input_topic = settings.kafka.topics.get("reviews_clustered", "reviews_clustered")

    logger.info("Initializing LLM Cluster Namer...")
    namer = ClusterNamer()

    # State maps to track when to trigger names
    cluster_names: dict[int, str] = {}
    cluster_samples: dict[int, list[str]] = {}
    naming_threshold = 5

    consumer = await get_kafka_consumer(input_topic, group_id="namer-group")

    logger.info(f"Namer Service listening on {input_topic}...")
    try:
        async for msg in consumer:
            try:
                # 1. Parse incoming data
                data = msg.value
                clustered_review = ReviewClustered(**data)
                cid = clustered_review.cluster_id

                # Assign default / ongoing logic
                if cid == -1:
                    active_name = "Noise / Unclustered"
                else:
                    if cid not in cluster_samples:
                        cluster_samples[cid] = []
                        cluster_names[cid] = "Evaluating Topic..."

                    if len(cluster_samples[cid]) < naming_threshold:
                        cluster_samples[cid].append(clustered_review.text)

                        if len(cluster_samples[cid]) == naming_threshold:
                            logger.info(f"Triggering LLM for Cluster {cid}...")
                            samples = cluster_samples[cid]

                            new_name = await asyncio.to_thread(
                                namer.generate_cluster_name, samples
                            )
                            cluster_names[cid] = new_name

                            # Push update event to dash clients explicitly when named
                            await redis_client.publish(
                                "dash_names_update",
                                json.dumps(
                                    {"cluster_id": cid, "cluster_name": new_name}
                                ),
                            )
                            logger.info(f"Cluster {cid} named: {new_name}")

                    active_name = cluster_names[cid]

                # 2. Finalize
                final_review = ProcessedReview(
                    **clustered_review.model_dump(), cluster_name=active_name
                )

                # 3. Update PostgreSQL Database persistently
                async with AsyncSessionLocal() as session:
                    stmt = (
                        update(ReviewRecord)
                        .where(ReviewRecord.id == final_review.id)
                        .values(cluster_name=final_review.cluster_name)
                    )
                    await session.execute(stmt)
                    await session.commit()

                logger.info(
                    f"Finalized {final_review.id} -> {active_name} (Saved to DB)"
                )

            except Exception as e:
                logger.error(f"Namer failed on message: {e}", exc_info=True)

    finally:
        await consumer.stop()
        await redis_client.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=settings.app.log_level)
    asyncio.run(run_namer_service())
