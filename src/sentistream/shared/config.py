import os
from logging import getLogger

import yaml
from pydantic import BaseModel


logger = getLogger(__name__)


class AppConfig(BaseModel):
    name: str = "SentiStream Worker"
    env: str = "development"
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    model: str
    api_key: str


class KafkaConfig(BaseModel):
    bootstrap_servers: list[str]
    topics: dict[str, str]


class DatabaseConfig(BaseModel):
    postgres_dsn: str
    redis_url: str


class MLConfig(BaseModel):
    hf_repo_id: str | None = None
    embedder_onnx_dir: str
    umap_onnx_path: str


class Settings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    kafka: KafkaConfig
    database: DatabaseConfig
    ml: MLConfig


def _apply_env_overrides(cfg: dict) -> dict:
    """Overrides config values with environment variables when present."""
    env = os.environ.get

    if v := env("SENTISTREAM_LLM_MODEL"):
        cfg.setdefault("llm", {})["model"] = v
    if v := env("SENTISTREAM_LLM_API_KEY"):
        cfg.setdefault("llm", {})["api_key"] = v

    if v := env("SENTISTREAM_KAFKA_BOOTSTRAP_SERVERS"):
        cfg.setdefault("kafka", {})["bootstrap_servers"] = v.split(",")

    if v := env("SENTISTREAM_POSTGRES_DSN"):
        cfg.setdefault("database", {})["postgres_dsn"] = v
    if v := env("SENTISTREAM_REDIS_URL"):
        cfg.setdefault("database", {})["redis_url"] = v

    if v := env("SENTISTREAM_HF_REPO_ID"):
        cfg.setdefault("ml", {})["hf_repo_id"] = v
    if v := env("SENTISTREAM_EMBEDDER_DIR"):
        cfg.setdefault("ml", {})["embedder_onnx_dir"] = v
    if v := env("SENTISTREAM_UMAP_PATH"):
        cfg.setdefault("ml", {})["umap_onnx_path"] = v

    return cfg


def load_config(config_path: str = "config.yaml") -> Settings:
    """Loads config from YAML, then applies environment variable overrides."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict = _apply_env_overrides(config_dict)
    return Settings(**config_dict)


# Global singleton — import `settings` from this module to access config anywhere
settings = load_config()
