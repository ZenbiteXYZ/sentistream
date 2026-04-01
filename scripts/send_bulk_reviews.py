import asyncio
import json
import random
import uuid

from aiokafka import AIOKafkaProducer


KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "reviews_raw"

CLUSTERS = {
    "Shipping Delays": [
        "Package showed up 8 days late and tracking never updated.",
        "My order was stuck in transit for a week.",
        "Delivery window was missed twice.",
        "Shipping took way longer than promised.",
        "Carrier lost my package and it arrived late.",
        "I waited 12 days for a 2-day shipment.",
    ],
    "Damaged Packaging": [
        "Box arrived crushed and the product was dented.",
        "Packaging was torn and items were loose.",
        "The seal was broken and the box looked opened.",
        "Corners were smashed, not giftable.",
        "Item scratched because there was no padding.",
        "The package looked like it was dropped hard.",
    ],
    "Great Customer Support": [
        "Support replied in minutes and fixed my issue.",
        "Agent was polite and replaced the unit immediately.",
        "They walked me through setup step-by-step.",
        "Customer service actually listened and solved it.",
        "Refund was processed the same day I asked.",
        "Chat support was fast and super helpful.",
    ],
    "App/UI Bugs": [
        "Checkout button freezes every time I tap it.",
        "App crashes on the login screen.",
        "Search results won't load on Wi-Fi.",
        "The cart empties itself randomly.",
        "Notifications don't open the right page.",
        "Dark mode makes text unreadable in the app.",
    ],
    "Price / Value Concerns": [
        "Too expensive for what you get.",
        "Feels cheap for the price.",
        "Not worth the cost compared to competitors.",
        "Overpriced for the quality.",
        "I expected more features at this price point.",
        "Way too pricey for a basic product.",
    ],
    "Battery / Performance Issues": [
        "Battery drains in two hours after the update.",
        "Device overheats during normal use.",
        "Performance is laggy and slow.",
        "Battery health dropped fast in a month.",
        "It shuts down at 30% battery.",
        "Charging takes forever now.",
    ],
}

ALL_REVIEWS = []
for cluster, reviews in CLUSTERS.items():
    for r in reviews:
        ALL_REVIEWS.append((r, cluster))

# Add some noise
NOISE = [
    "I love dogs",
    "Best purchase ever!",
    "Not what I expected.",
    "Would buy again.",
    "Terrible experience.",
    "Five stars!",
    "One star.",
    "No comment.",
    "Quick delivery.",
    "Unusable after update.",
]


async def main():
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()
    try:
        reviews = []
        # 25 from each cluster (6 clusters * 25 = 150)
        for _ in range(25):
            for text, cluster in ALL_REVIEWS:
                reviews.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "metadata": {"sim_cluster": cluster},
                    }
                )
        # Add 10 random noise reviews
        for _ in range(10):
            reviews.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": random.choice(NOISE),
                    "metadata": {"sim_cluster": "Noise"},
                }
            )
        random.shuffle(reviews)
        for review in reviews[:150]:
            await producer.send(TOPIC, value=review, key=review["id"].encode("utf-8"))
        print(f"Sent {len(reviews[:150])} reviews to {TOPIC}")
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
