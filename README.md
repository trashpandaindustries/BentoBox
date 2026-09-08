# BentoBox

CLIP embedding and image/text similarity service for [Ratatoskr](../). Built on [BentoML](https://github.com/bentoml/BentoML) and Hugging Face `transformers`, serving a CLIP model (`openai/clip-vit-base-patch32` by default) for generating image embeddings, text embeddings, and image-to-text similarity ranking.

## Running

```bash
docker compose up -d bentobox
```

Configuration is driven by environment variables (see below), loaded via `.env`. The service listens on `CLIP_PORT` (default `3000`).

## Endpoints

All endpoints are served at `http://<host>:<CLIP_PORT>`.

### `POST /encode_image`

Generates a CLIP embedding for one or more images.

**Request:** `multipart/form-data`, repeated `items` fields (one per image).

```bash
curl -X POST http://localhost:3000/encode_image \
  -F 'items=@photo1.jpg;type=image/jpeg' \
  -F 'items=@photo2.jpg;type=image/jpeg'
```

**Response:** `200 OK`, one embedding vector per image.

```json
[
  [0.0123, -0.0456, "... 512 floats"],
  [0.0891, -0.0234, "... 512 floats"]
]
```

### `POST /encode_text`

Generates a CLIP embedding for one or more text strings, in the same vector space as image embeddings.

**Request:** `application/json`

```bash
curl -X POST http://localhost:3000/encode_text \
  -H 'Content-Type: application/json' \
  -d '{"items": ["a photograph of an old car", "a photograph of a house"]}'
```

**Response:** `200 OK`, one embedding vector per text item — same shape as `/encode_image`.

### `POST /rank`

Compares one or more images against a list of candidate text descriptions and returns similarity scores.

**Request:** `multipart/form-data` — `queries` (repeated, images) and `candidates` (JSON array of strings).

```bash
curl -X POST http://localhost:3000/rank \
  -F 'queries=@photo.jpg;type=image/jpeg' \
  -F 'candidates=["picture of a dog","picture of a cat"]'
```

**Response:** `200 OK`

```json
{
  "probabilities": [[0.92, 0.08]],
  "cosine_similarities": [[0.31, 0.11]]
}
```

`probabilities` are softmax-normalized scores across candidates for each query image; `cosine_similarities` are the raw pre-softmax values.

### `GET /model`

Returns metadata about the currently loaded model.

```bash
curl http://localhost:3000/model
```

**Response:** `200 OK`

```json
{
  "model_id": "openai/clip-vit-base-patch32",
  "embedding_dimension": 512,
  "device": "cuda",
  "model_revision": "unknown"
}
```

### `GET /health`

Runs a lightweight inference pass to confirm the model is loaded and usable — not just that the process is alive.

```bash
curl http://localhost:3000/health
```

**Response:** `200 OK` when healthy:

```json
{
  "status": "ok",
  "device": "cuda"
}
```

or, on failure:

```json
{
  "status": "error",
  "detail": "..."
}
```

## Configuration

All settings are read from environment variables at container startup (set via `.env`, loaded through `env_file` in `docker-compose.yml`). None require a rebuild — a container restart is enough.

| Variable | Default | Description |
|---|---|---|
| `CLIP_MODEL_ID` | `openai/clip-vit-base-patch32` | Hugging Face model ID to load. Changing this triggers a fresh download from Hugging Face on next startup. |
| `CLIP_MEMORY` | `4Gi` | Memory resource limit for the service. |
| `CLIP_PORT` | `3000` | Port the service listens on. Must match the `docker-compose.yml` port mapping. |
| `CLIP_CORS_ORIGINS` | `http://localhost:8080` | Comma-separated list of allowed CORS origins (e.g. for a Swagger UI served on a different port). |
| `CLIP_FORCE_DEVICE` | _(unset — auto-detect)_ | Set to `cpu` or `cuda` to override automatic device selection. |

### Example `.env`

```
CLIP_MODEL_ID=openai/clip-vit-base-patch32
CLIP_MEMORY=4Gi
CLIP_PORT=3000
CLIP_CORS_ORIGINS=http://localhost:8080
```

### `docker-compose.yml` port mapping

```yaml
ports:
  - ${CLIP_PORT:-3000}:${CLIP_PORT:-3000}
```

The `:-3000` fallback matters — without it, an unset `CLIP_PORT` resolves to an empty string and the port mapping silently fails rather than erroring loudly.

## Notes

- Image resizing/cropping is handled entirely by `CLIPProcessor`, using whatever dimensions the loaded model's own preprocessor config specifies. Nothing hardcodes a target resolution, so swapping `CLIP_MODEL_ID` to a variant with a different native input size works without code changes.
- `/encode_image` and `/encode_text` are batchable — send multiple items in a single request rather than one call per item where possible.
