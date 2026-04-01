import numpy as np

from sentistream.worker.embedder import PipelineEmbedder


class FakeEncoded:
    def __init__(self):
        self.ids = [101, 102, 103]
        self.attention_mask = [1, 1, 1]


class FakeTokenizer:
    def encode(self, text: str) -> FakeEncoded:
        return FakeEncoded()

    def enable_truncation(self, max_length: int) -> None:
        return None

    def token_to_id(self, token: str):
        return None

    def enable_padding(self, pad_id: int) -> None:
        return None


class FakeInput:
    def __init__(self, name: str):
        self.name = name


class FakeSession:
    def __init__(self, output, input_name: str = "input_ids"):
        self._output = output
        self._input_name = input_name

    def get_inputs(self):
        return [FakeInput("input_ids"), FakeInput("attention_mask")]

    def run(self, _, inputs):
        return [self._output]


class FakeUmapSession:
    def __init__(self, output, input_name: str = "input"):
        self._output = output
        self._input_name = input_name

    def get_inputs(self):
        return [FakeInput(self._input_name)]

    def run(self, _, inputs):
        return [self._output]


def _make_embedder(monkeypatch: object) -> PipelineEmbedder:
    monkeypatch.setattr(
        PipelineEmbedder, "_ensure_models_downloaded", lambda self: None
    )
    monkeypatch.setattr(PipelineEmbedder, "_load_models", lambda self: None)
    embedder = PipelineEmbedder()
    embedder.tokenizer = FakeTokenizer()
    return embedder


def test_embed_and_reduce_with_umap(monkeypatch):
    embedder = _make_embedder(monkeypatch)
    bge_output = np.ones((1, 2, 384), dtype=np.float32)
    umap_output = np.ones((1, 5), dtype=np.float32)

    embedder.bge_session = FakeSession(bge_output)
    embedder.umap_session = FakeUmapSession(umap_output)
    embedder.scaler_mean = np.zeros(384, dtype=np.float32)
    embedder.scaler_scale = np.ones(384, dtype=np.float32)

    full_embed, reduced = embedder.embed_and_reduce("sample text")

    assert len(full_embed) == 384
    assert len(reduced) == 5


def test_embed_and_reduce_without_umap(monkeypatch):
    embedder = _make_embedder(monkeypatch)
    bge_output = np.ones((1, 2, 384), dtype=np.float32)

    embedder.bge_session = FakeSession(bge_output)
    embedder.umap_session = None

    full_embed, reduced = embedder.embed_and_reduce("sample text")

    assert len(full_embed) == 384
    assert len(reduced) == 5
    assert reduced == full_embed[:5]
