import asyncio
import math
import time

import httpx

from .models import BenchmarkRequest, BenchmarkResult
from .telemetry import sample_nvidia_smi


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    semaphore = asyncio.Semaphore(request.concurrency)
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    errors: list[str] = []
    counter_lock = asyncio.Lock()

    headers = {"Content-Type": "application/json"}
    if request.api_key:
        headers["Authorization"] = f"Bearer {request.api_key}"
    url = str(request.endpoint).rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "stream": False,
    }

    telemetry_stop = asyncio.Event()
    telemetry_task = (
        asyncio.create_task(sample_nvidia_smi(telemetry_stop))
        if request.collect_local_gpu_telemetry
        else None
    )

    async with httpx.AsyncClient(
        timeout=request.timeout_seconds, verify=request.verify_tls, headers=headers
    ) as client:

        async def execute_one() -> None:
            nonlocal prompt_tokens, completion_tokens
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    usage = body.get("usage", {})
                    async with counter_lock:
                        prompt_tokens += int(usage.get("prompt_tokens", 0))
                        completion_tokens += int(usage.get("completion_tokens", 0))
                        latencies.append(time.perf_counter() - started)
                except Exception as exc:  # errors belong in the benchmark result
                    async with counter_lock:
                        if len(errors) < 20:
                            errors.append(f"{type(exc).__name__}: {exc}")

        started_all = time.perf_counter()
        await asyncio.gather(*(execute_one() for _ in range(request.requests)))
        elapsed = time.perf_counter() - started_all

    telemetry_stop.set()
    gpu_samples = await telemetry_task if telemetry_task else []

    successful = len(latencies)
    return BenchmarkResult(
        attempted_requests=request.requests,
        successful_requests=successful,
        failed_requests=request.requests - successful,
        elapsed_seconds=round(elapsed, 4),
        requests_per_second=round(successful / elapsed, 4) if elapsed else 0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        output_tokens_per_second=round(completion_tokens / elapsed, 4) if elapsed else 0,
        p50_latency_seconds=round(_percentile(latencies, 0.50), 4),
        p95_latency_seconds=round(_percentile(latencies, 0.95), 4),
        gpu_samples=len(gpu_samples),
        average_gpu_utilization_percent=(
            round(sum(sample[0] for sample in gpu_samples) / len(gpu_samples), 2)
            if gpu_samples
            else None
        ),
        peak_gpu_memory_used_mib=(
            round(max(sample[1] for sample in gpu_samples), 2) if gpu_samples else None
        ),
        errors=errors,
    )
