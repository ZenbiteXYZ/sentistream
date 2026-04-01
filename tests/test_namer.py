import os

import pytest

from sentistream.shared.config import settings
from sentistream.worker.naming import ClusterNamer


def test_namer_fallback_when_no_key(monkeypatch):
    monkeypatch.setattr(settings.llm, "api_key", "your_api_key_here")
    namer = ClusterNamer()

    name = namer.generate_cluster_name(["Great UI", "Love the support team"])

    assert name == "Generated Topic (No LLM)"


def test_namer_empty_reviews(monkeypatch):
    monkeypatch.setattr(settings.llm, "api_key", "test-key")
    namer = ClusterNamer()

    name = namer.generate_cluster_name([])

    assert name == "Uncategorized"


@pytest.mark.skipif(
    not os.getenv("SENTISTREAM_LLM_TESTS") or not os.getenv("LITELLM_API_KEY"),
    reason="Enable with SENTISTREAM_LLM_TESTS=1 and LITELLM_API_KEY",
)
def test_namer_live_llm_call(monkeypatch):
    monkeypatch.setattr(settings.llm, "api_key", os.getenv("LITELLM_API_KEY"))
    model_override = os.getenv("SENTISTREAM_LLM_MODEL")
    if model_override:
        monkeypatch.setattr(settings.llm, "model", model_override)

    namer = ClusterNamer()
    name = namer.generate_cluster_name(
        ["Fast delivery", "Shipping was late", "Package arrived damaged"]
    )

    assert isinstance(name, str)
    assert len(name.strip()) > 0
