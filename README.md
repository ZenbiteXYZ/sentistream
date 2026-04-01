# SentiStream
AI-driven real-time review clustering system for streaming customer feedback.

## What it does
- Ingests reviews via HTTP.
- Embeds text with BGE small (384d).
- Reduces to 5d with Parametric UMAP.
- Clusters online with River DBSTREAM.
- Names clusters with an optional LLM.
- Streams results to Postgres, Redis, and a live Dash dashboard.

## Architecture
`FastAPI -> Kafka -> Embedder -> Clusterer -> Namer -> Postgres + Redis -> Dash`

## Quickstart (uv)
1. Create a virtual env and install deps:
	```bash
	uv venv
	source .venv/bin/activate
	uv pip install -e ".[dev,test]"
	```
2. Configure `config.yaml` (or use `config.docker.yaml` with Docker).
3. Start infrastructure (recommended):
	```bash
	docker compose up --build
	```
4. Ingest a review:
	```bash
	curl -X POST http://localhost:8000/reviews \
	  -H "Content-Type: application/json" \
	  -d '{"text": "Love the new UI"}'
	```
5. Open the dashboard at http://localhost:8050.

## Services (local run)
```bash
uv run python -m sentistream.services.embedder_svc
uv run python -m sentistream.services.clusterer_svc
uv run python -m sentistream.services.namer_svc
uv run python -m sentistream.dashboard.app
uv run uvicorn sentistream.ingestion.api:app --host 0.0.0.0 --port 8000
```

## Testing
```bash
uv run pytest
```

### Optional LLM naming test
Enable live LLM test only when you have a valid key:
```bash
export SENTISTREAM_LLM_TESTS=1
export LITELLM_API_KEY=your_key_here
export SENTISTREAM_LLM_MODEL=gpt-4o-mini  # optional
uv run pytest tests/test_namer.py
```

## Configuration
Key settings live in `config.yaml`:
- `llm.model`, `llm.api_key`
- `kafka.bootstrap_servers`, `kafka.topics`
- `database.postgres_dsn`, `database.redis_url`
- `ml.embedder_onnx_dir`, `ml.umap_onnx_path`, `ml.hf_repo_id`

## Notes
- Models are stored under `models/` and can be auto-downloaded from HF when configured.
- The naming service is optional; when disabled it falls back to a placeholder name.
