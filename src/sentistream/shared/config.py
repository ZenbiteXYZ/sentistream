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


def load_config(config_path: str = "config.yaml") -> Settings:
    """
    Loads config from a YAML file.
    Falls back to config.example.yaml if config.yaml does not exist.
    """
    if not os.path.exists(config_path):
        example_path = "config.example.yaml"
        logger.warning(
            f"{config_path} not found. Loading example config from {example_path}"
        )

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return Settings(**config_dict)


# Global singleton instance of settings
# Just import `settings` from this module to access configuration anywhere
settings = load_config()
