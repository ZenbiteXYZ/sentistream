import json
import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from sentistream.shared.config import settings


logger = logging.getLogger(settings.app.name)


async def get_kafka_producer() -> AIOKafkaProducer:
    """
    Initializes and starts an async Kafka Producer.
    Serializes outgoing messages to JSON via utf-8 bytes.
    """
    try:
        producer = AIOKafkaProducer(
            bootstrap_servers=",".join(settings.kafka.bootstrap_servers),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await producer.start()
        logger.info("Kafka Producer connected.")
        return producer
    except Exception as e:
        logger.error(f"Failed to initialize Kafka Producer: {e}")
        raise


async def get_kafka_consumer(
    topic: str, group_id: str = "sentistream-worker"
) -> AIOKafkaConsumer:
    """
    Initializes and starts an async Kafka Consumer for a specific topic.
    Deserializes incoming JSON utf-8 bytes to Python dicts.
    """
    try:
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=",".join(settings.kafka.bootstrap_servers),
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",  # Start reading from the beginning if no offset is found
        )
        await consumer.start()
        logger.info(f"Kafka Consumer connected to topic: {topic}")
        return consumer
    except Exception as e:
        logger.error(f"Failed to initialize Kafka Consumer for topic {topic}: {e}")
        raise
