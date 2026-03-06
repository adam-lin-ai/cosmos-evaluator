# Service Framework

The Cosmos Evaluator service framework provides building blocks for wrapping evaluation checks as REST API microservices. Each service follows a consistent pattern: receive a request, validate it, run the check, and return formatted results.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Application                 │
│                                                     │
│   GET /health     POST /process     GET /config     │
└────────┬──────────────┬──────────────────┬──────────┘
         │              │                  │
         │    ┌─────────▼──────────┐       │
         │    │  RequestHandler    │       │
         │    │  (parse & validate)│       │
         │    └─────────┬──────────┘       │
         │              │                  │
         │    ┌─────────▼──────────┐       │
         │    │    ServiceBase     │       │
         │    │(validate & process)│       │
         │    └─────────┬──────────┘       │
         │              │                  │
         │    ┌─────────▼──────────┐       │
         │    │ ResponseFormatter  │       │
         │    │  (format response) │       │
         │    └────────────────────┘       │
         │                                 │
    ┌────▼─────────────────────────────────▼──┐
    │           StorageProvider                │
    │    (upload results, download inputs)     │
    └─────────────────────────────────────────┘
```

## Core Components

### ServiceBase

The abstract base class that all checker services inherit from. Defines the contract every service must implement.

```python
from services.framework.service_base import ServiceBase

class MyService(ServiceBase[MyRequest, MyResult]):

    async def validate_input(self, request: MyRequest) -> bool:
        """Validate that the request data is processable.
        Return True if valid. Raise an exception with details if not."""
        ...

    async def process(self, request: MyRequest) -> MyResult:
        """Run the evaluation logic. Returns the result."""
        ...
```

`ServiceBase` is generic over `RequestType` and `ResponseType` — both should be Pydantic `BaseModel` subclasses.

It also provides:
- A pre-configured `self.logger` for structured logging
- `cleanup(output_dir)` to remove temporary directories after processing

### Protocols

The framework defines three protocol interfaces. These are runtime-checkable Python protocols, so you can implement them however you like as long as the method signatures match.

#### RequestHandler

Parses and validates incoming HTTP requests into your domain model.

```python
class RequestHandler(Protocol[RequestType]):
    async def parse(self, raw_request: Any) -> RequestType: ...
    async def validate(self, request_data: RequestType) -> bool: ...
    def get_content_type(self) -> str: ...
```

**Built-in implementation:** `JsonRequestHandler` — Parses JSON request bodies into Pydantic models.

#### ResponseFormatter

Formats service responses into a consistent envelope.

```python
class ResponseFormatter(Protocol[ResponseType]):
    async def format_success(self, data: Any, metadata: dict | None = None) -> ResponseType: ...
    async def format_error(self, error: Exception, status_code: int = 500) -> ResponseType: ...
    async def format_progress(self, progress: float, message: str = "") -> ResponseType: ...
    def get_content_type(self) -> str: ...
```

**Built-in implementation:** `JsonResponseFormatter` — Wraps responses in a standard JSON envelope:

**Success response:**

```json
{
  "success": true,
  "data": { ... },
  "metadata": { ... },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

**Error response:**

```json
{
  "success": false,
  "error": {
    "type": "ValueError",
    "message": "Invalid input",
    "code": null,
    "details": null
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

Additional methods on `JsonResponseFormatter`:
- `format_validation_error(validation_errors, status_code=422)` — For Pydantic validation failures
- `format_paginated_response(items, total, page, page_size)` — For paginated list endpoints

#### StorageProvider

Cloud-agnostic interface for storing and retrieving files and data.

```python
class StorageProvider(Protocol):
    async def store(self, data: Any, key: str, metadata: dict | None = None) -> str: ...
    async def retrieve(self, key: str) -> Any: ...
    async def delete(self, key: str) -> bool: ...
    async def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str = "") -> AsyncIterator[str]: ...
    async def store_file(self, file_path: Path, key: str, ...) -> StorageUrls: ...
    async def generate_presigned_url(self, key: str, ...) -> str: ...
    async def download_from_url(self, url: str, destination: Path) -> Path: ...
```

**Built-in implementations:**
- `LocalStorageProvider` — File system storage (good for development)
- `S3StorageProvider` — AWS S3 with presigned URL support
- `StorageProviderFactory` — Creates the right provider based on settings

`StorageUrls` is a dataclass with `raw` (permanent) and `presigned` (temporary access) URL fields.

### SettingsBase

Base class for service configuration using Pydantic Settings:

```python
from services.settings_base import SettingsBase

class Settings(SettingsBase):
    my_custom_setting: str = Field(default="value")

    model_config = SettingsConfigDict(
        env_prefix="MYCHECKER_",
        ...
    )
```

Common settings inherited from `SettingsBase`:

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `env` | `COSMOS_EVALUATOR_ENV` | `local` | Environment: `local` for development |
| `log_level` | `COSMOS_EVALUATOR_LOG_LEVEL` | `INFO` | Logging level |
| `storage_type` | `COSMOS_EVALUATOR_STORAGE_TYPE` | `s3` | Storage backend: `s3` or `local` |
| `storage_bucket` | `COSMOS_EVALUATOR_STORAGE_BUCKET` | — | S3 bucket name |
| `storage_region` | `COSMOS_EVALUATOR_STORAGE_REGION` | — | AWS region |
| `storage_access_key` | `COSMOS_EVALUATOR_STORAGE_ACCESS_KEY` | — | AWS access key |
| `storage_secret_key` | `COSMOS_EVALUATOR_STORAGE_SECRET_KEY` | — | AWS secret key |

In `local` environment, credentials are automatically loaded from `~/.cosmos_evaluator/.env`.

## Standard Endpoints

Every Cosmos Evaluator service exposes three endpoints:

### `GET /health`

Returns service status for monitoring and service discovery.

```json
{
  "success": true,
  "data": {
    "service": "obstacle_correspondence",
    "status": "healthy",
    "version": "1.15.0",
    "environment": "local"
  }
}
```

### `POST /process`

The main processing endpoint. Accepts a request, runs the check, and returns results.

The request and response models are service-specific (see individual check documentation). The response is always wrapped in the standard envelope.

### `GET /config`

Returns the default configuration for the checker, useful for clients to discover available options.

```json
{
  "success": true,
  "data": {
    "default_config": {
      "av.obstacle": { ... }
    }
  }
}
```

## Running Services

```bash
# All services together
dazel run //services:rest_apis

# Individual services
dazel run //services/obstacle_correspondence:rest_api
dazel run //services/vlm:rest_api

# With custom port
dazel run //services/obstacle_correspondence:rest_api -- --port 9000
```

Default ports (local development):

| Service | Port |
|---------|------|
| Obstacle Correspondence | 8082 |
| VLM | 8083 |

## Example: Calling the Obstacle Service

```bash
# Check health
curl http://localhost:8082/health

# Get default config
curl http://localhost:8082/config

# Process a request
curl -X POST http://localhost:8082/process \
  -H "Content-Type: application/json" \
  -d '{
    "rds_hq_url": "https://storage.example.com/rds_hq.zip",
    "augmented_video_url": "https://storage.example.com/video.mp4",
    "camera_name": "camera_front_wide_120fov"
  }'
```

## Building a Custom Service

See the [Custom Checker Tutorial](../tutorials/custom-checker.md) for a complete walkthrough of building a new checker service from scratch.

The general steps are:

1. **Define models** — Pydantic request/response models
2. **Implement service** — Subclass `ServiceBase` with your check logic
3. **Configure settings** — Subclass `SettingsBase` for service-specific config
4. **Build REST API** — FastAPI app with `/health`, `/process`, `/config` endpoints
5. **Create BUILD file** — Bazel targets for library, binary, and tests

The Obstacle and VLM services in `services/obstacle_correspondence/` and `services/vlm/` serve as reference implementations.
