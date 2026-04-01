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

        os.makedirs(self.embedder_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)
        lock_path = base_dir / ".download.lock"

        with FileLock(lock_path):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                logger.error("huggingface_hub missing! Cannot download models.")
                return

            self._download_bge_models(hf_hub_download, bge_model, bge_tokenizer)
            self._download_umap_models(hf_hub_download, umap_model, base_dir)

    def _download_bge_models(
        self, hf_hub_download, bge_model: str, bge_tokenizer: str
    ) -> None:
        if os.path.exists(bge_model) and os.path.exists(bge_tokenizer):
            return

        logger.info("Downloading BGE ONNX model from Xenova/bge-small-en-v1.5...")
        try:
            hf_hub_download(
                repo_id="Xenova/bge-small-en-v1.5",
                filename="tokenizer.json",
                local_dir=self.embedder_dir,
                local_dir_use_symlinks=False,
            )
            onnx_file = hf_hub_download(
                repo_id="Xenova/bge-small-en-v1.5",
                filename="onnx/model.onnx",
                local_dir=self.embedder_dir,
                local_dir_use_symlinks=False,
            )

            import shutil

            if os.path.exists(onnx_file) and onnx_file != bge_model:
                shutil.move(onnx_file, bge_model)

            logger.info("Successfully downloaded BGE models!")
        except Exception as e:
            logger.error(f"Failed to download BGE models: {e}")
            raise

    def _download_umap_models(
        self, hf_hub_download, umap_model: str, base_dir: Path
    ) -> None:
        if os.path.exists(umap_model):
            return

        repo_id = settings.ml.hf_repo_id
        if not repo_id or repo_id == "your_username/sentistream-models":
            logger.warning("No valid HF repo provided for UMAP. Using naive reduction.")
            return

        try:
            logger.info(f"Downloading UMAP model and components from {repo_id}...")
            files_to_dl = [
                "model.onnx",
                "sentistream_encoder_v4.onnx.data",
                "scaler_config.json",
            ]
            for fname in files_to_dl:
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=fname,
                        local_dir=str(base_dir),
                        local_dir_use_symlinks=False,
                    )
                except Exception as dl_e:
                    logger.warning(f"Could not download {fname}: {dl_e}")
        except Exception as e:
            logger.warning(f"Failed download UMAP from {repo_id}: {e}")

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

        # 3. Load Parametric UMAP ONNX Model & Scaler
        self.scaler_mean = None
        self.scaler_scale = None
        scaler_path = os.path.join(Path(self.umap_path).parent, "scaler_config.json")
        if os.path.exists(scaler_path):
            import json

            try:
                with open(scaler_path) as f:
                    scaler_data = json.load(f)
                self.scaler_mean = np.array(scaler_data["mean"], dtype=np.float32)
                self.scaler_scale = np.array(scaler_data["scale"], dtype=np.float32)
                logger.info(f"Loaded StandardScaler config from {scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load StandardScaler config: {e}")

        if os.path.exists(self.umap_path):
            try:
                self.umap_session = ort.InferenceSession(
                    self.umap_path, providers=["CPUExecutionProvider"]
                )
                logger.info(f"UMAP ONNX model loaded from {self.umap_path}")
            except Exception as e:
                logger.warning(f"Could not load UMAP ONNX model: {e}")
        else:
            logger.info(
                "UMAP model file not found locally. Skipping UMAP ONNX session load in favor of naive reduction."
            )

    def embed_and_reduce(self, text: str) -> tuple[list[float], list[float]]:
        """
        Takes raw text, generates a 384D embedding, and reduces it to 5D.
        Returns (original_384d_embedding, reduced_5d_embedding).
        """
        if not self.tokenizer or not self.bge_session:
            raise RuntimeError("Embedder models are not loaded correctly.")

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

        # -- C. Parametric UMAP Reduction & Scaling (5D) --
        if self.umap_session:
            # Apply StandardScaler Transformation first if we have loaded it
            if self.scaler_mean is not None and self.scaler_scale is not None:
                # (X - mean) / scale
                umap_input_data = (
                    sentence_embedding_f32 - self.scaler_mean
                ) / self.scaler_scale
            else:
                umap_input_data = sentence_embedding_f32

            umap_input_name = self.umap_session.get_inputs()[0].name
            reduced_embedding = self.umap_session.run(
                None, {umap_input_name: umap_input_data}
            )[0]
        else:
            # Fallback naive reduction
            reduced_embedding = sentence_embedding_f32[:, :5]

        # Return as flat Python lists for JSON serialization / Pydantic later
        return (
            sentence_embedding_f32.flatten().tolist(),
            reduced_embedding.flatten().tolist(),
        )
