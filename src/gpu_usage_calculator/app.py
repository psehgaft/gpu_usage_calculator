import json
import os
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .benchmark import run_benchmark
from .calculator import estimate_capacity
from .models import BenchmarkRequest, BenchmarkResult, SizingRequest, SizingResult


APP_VERSION = "0.1.0"
REQUESTS = Counter(
    "gpu_calculator_requests_total", "Calculator API requests", ["operation", "status"]
)
ESTIMATE_LATENCY = Histogram(
    "gpu_calculator_estimate_duration_seconds", "Capacity estimate duration"
)

app = FastAPI(
    title="GPU Usage Calculator",
    version=APP_VERSION,
    description="Estimate and validate GPU capacity for token-based inference workloads.",
)


def _config_path() -> Path:
    return Path(os.getenv("GPU_PROFILES_PATH", "/app/config/gpu-profiles.yaml"))


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    template = Path(__file__).with_name("templates") / "index.html"
    return template.read_text(encoding="utf-8")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/v1/gpu-profiles")
def gpu_profiles() -> dict:
    path = _config_path()
    if not path.exists():
        fallback = Path(__file__).parents[2] / "config" / "gpu-profiles.yaml"
        path = fallback if fallback.exists() else path
    if not path.exists():
        raise HTTPException(status_code=404, detail="GPU profile catalog is not mounted")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@app.post("/api/v1/estimate", response_model=SizingResult)
def estimate(request: SizingRequest) -> SizingResult:
    with ESTIMATE_LATENCY.time():
        try:
            result = estimate_capacity(request)
            REQUESTS.labels("estimate", "success").inc()
            return result
        except Exception as exc:
            REQUESTS.labels("estimate", "error").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/benchmark", response_model=BenchmarkResult)
async def benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    result = await run_benchmark(request)
    REQUESTS.labels("benchmark", "success" if not result.failed_requests else "partial").inc()
    return result


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/example")
def example() -> dict:
    sample = SizingRequest()
    return {"request": json.loads(sample.model_dump_json()), "result": estimate_capacity(sample)}
