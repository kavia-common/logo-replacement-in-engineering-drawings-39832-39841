import json
from pathlib import Path

# Import the FastAPI app; this will register all routes so the schema includes
# - POST /jobs
# - POST /jobs/{job_id}/upload
# - POST /jobs/{job_id}/start
# - GET  /jobs/{job_id}/status
# - GET  /jobs/{job_id}/download
# - DELETE /jobs/{job_id}
from src.api.main import app

# Import models so Pydantic schemas are registered and included in OpenAPI
# Note: These imports are not used directly here but ensure components/schemas
# include JobCreated, JobStatus, JobState, etc.
from src.models import schemas as _schemas  # noqa: F401


def main() -> None:
    """
    Generate and write the OpenAPI schema to interfaces/openapi.json.

    Usage:
        - Run via module:     python -m src.api.generate_openapi
        - Or run the file:    python src/api/generate_openapi.py

    Notes:
        - The output file is used by the frontend and for API docs.
        - Ensure you run this after modifying routes or models so the schema stays up to date.
    """
    # Build schema after all routes are registered
    openapi_schema = app.openapi()

    # Validate basic structure
    if not isinstance(openapi_schema, dict) or "openapi" not in openapi_schema or "paths" not in openapi_schema:
        raise RuntimeError("Failed to build OpenAPI schema from FastAPI app")

    # Ensure output directory exists
    output_dir = Path("interfaces")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "openapi.json"

    # Write schema (pretty-printed)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

    print(f"OpenAPI schema generated at: {output_path.resolve()}")


if __name__ == "__main__":
    # Allow running as a script
    main()
