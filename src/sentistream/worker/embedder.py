import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from filelock import FileLock
from tokenizers import Tokenizer

from sentistream.shared.config import settings


logger = logging.getLogger(settings.app.name)


class PipelineEmbedder:
    def __init__(self):
        self.embedder_dir = settings.ml.embedder_onnx_dir
        self.umap_path = settings.ml.umap_onnx_path

        self.bge_session = None
        self.umap_session = None
        self.tokenizer = None

        self._ensure_models_downloaded()
        self._load_models()

    def _ensure_models_downloaded(self):
        """Checks if models exist locally in a multi-process safe way using file locks.
        If missing, downloads them from HuggingFace."""
        bge_model = os.path.join(self.embedder_dir, "model.onnx")
        bge_tokenizer = os.path.join(self.embedder_dir, "tokenizer.json")
        umap_model = self.umap_path

        # Derive the base 'models' folder to stick the lockfile in
        base_dir = Path(self.umap_path).parent
        if str(base_dir) == ".":
            base_dir = Path("models")

        os.makedirs(base_dir, exist_ok=True)
        lock_path = base_dir / ".download.lock"

        # We acquire a lock on the folder. If 3 processes start, only 1 gets the lock,
        # and the other 2 will block right here until process 1 finishes.
        with FileLock(lock_path):
            if not (
                os.path.exists(bge_model)
                and os.path.exists(bge_tokenizer)
                and os.path.exists(umap_model)
            ):
                repo_id = settings.ml.hf_repo_id
                if not repo_id or repo_id == "your_username/sentistream-models":
                    msg = "ML Models missing locally and no valid 'hf_repo_id' configured in config.yaml!"
                    logger.error(msg)
                    raise FileNotFoundError(msg)

                logger.info(
                    f"Models missing locally. Downloading from HuggingFace repo: '{repo_id}'..."
                )
                try:
                    from huggingface_hub import snapshot_download

                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(base_dir),
                        local_dir_use_symlinks=False,
                    )
                    logger.info("Successfully downloaded ML models from HuggingFace!")
                except Exception as e:
                    logger.error(f"Failed to download models from HuggingFace: {e}")
                    raise
            else:
                logger.debug("Models found locally. Proceeding to load.")

    def _load_models(self):
        """Loads the ONNX sessions and the Tokenizer."""
        # 1. Load Tokenizer
        tokenizer_path = os.path.join(self.embedder_dir, "tokenizer.json")
        try:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_truncation(max_length=512)
            if self.tokenizer.token_to_id("[PAD]") is not None:
                self.tokenizer.enable_padding(
                    pad_id=self.tokenizer.token_to_id("[PAD]")
                )
            logger.info(f"Tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")

        # 2. Load Embedder ONNX Model
        bge_model_path = os.path.join(self.embedder_dir, "model.onnx")
        try:
            self.bge_session = ort.InferenceSession(
                bge_model_path, providers=["CPUExecutionProvider"]
            )
            logger.info(f"BGE ONNX model loaded from {bge_model_path}")
        except Exception as e:
            logger.warning(f"Could not load BGE ONNX model: {e}")

        # 3. Load Parametric UMAP ONNX Model
        try:
            self.umap_session = ort.InferenceSession(
                self.umap_path, providers=["CPUExecutionProvider"]
            )
            logger.info(f"UMAP ONNX model loaded from {self.umap_path}")
        except Exception as e:
            logger.warning(f"Could not load UMAP ONNX model: {e}")

    def embed_and_reduce(self, text: str) -> tuple[list[float], list[float]]:
        """
        Takes raw text, generates a 384D embedding, and reduces it to 5D.
        Returns (original_384d_embedding, reduced_5d_embedding).
        """
        if not self.tokenizer or not self.bge_session or not self.umap_session:
            raise RuntimeError("Models are not loaded correctly.")

        # -- A. Tokenization --
        encoded = self.tokenizer.encode(text)

        # ONNX models typically expect int64 inputs
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        # Some HF exports require token_type_ids as well
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Check if model expects token_type_ids
        model_inputs = [i.name for i in self.bge_session.get_inputs()]
        if "token_type_ids" in model_inputs:
            inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        # -- B. BGE Embedding (384D) --
        outputs = self.bge_session.run(None, inputs)

        # Usually last_hidden_state is the first output
        last_hidden_state = outputs[0]

        # Use [CLS] token representation for sentence embedding
        sentence_embedding = last_hidden_state[:, 0, :]

        # Normalize the embedding (standard practice for BGE models)
        norm = np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
        sentence_embedding = sentence_embedding / np.maximum(norm, 1e-9)

        # Convert to float32 for the UMAP session
        sentence_embedding_f32 = sentence_embedding.astype(np.float32)

        # -- C. Parametric UMAP Reduction (5D) --
        umap_input_name = self.umap_session.get_inputs()[0].name
        reduced_embedding = self.umap_session.run(
            None, {umap_input_name: sentence_embedding_f32}
        )[0]

        # Return as flat Python lists for JSON serialization / Pydantic later
        return (
            sentence_embedding_f32.flatten().tolist(),
            reduced_embedding.flatten().tolist(),
        )
