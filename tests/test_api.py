from fastapi.testclient import TestClient

from sentistream.ingestion.api import app


client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "kafka_connected": False}


def test_ingest_review_missing_kafka():
    # Since producer is not initialized in TestClient without lifespan, we can test the 500 error
    response = client.post("/reviews", json=dict(text="This product is good"))
    assert response.status_code == 500
    assert response.json() == {"detail": "Kafka producer not initialized"}
