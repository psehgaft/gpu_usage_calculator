from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class SizingRequest(BaseModel):
    model_name: str = "example-70b"
    model_parameters_billions: float = Field(70, gt=0)
    weight_precision_bits: Literal[4, 8, 16, 32] = 16
    weight_overhead_fraction: float = Field(0.10, ge=0, le=1)

    gpu_model: str = "H100-80GB"
    gpu_memory_gib: float = Field(80, gt=0)
    tensor_parallel_size: int = Field(2, ge=1)
    usable_memory_fraction: float = Field(0.90, gt=0, le=1)
    runtime_reserve_fraction: float = Field(0.10, ge=0, lt=1)

    layers: int = Field(80, ge=1)
    kv_heads: int = Field(8, ge=1)
    head_dimension: int = Field(128, ge=1)
    kv_cache_precision_bytes: Literal[1, 2, 4] = 2

    average_input_tokens: int = Field(1000, ge=1)
    average_output_tokens: int = Field(250, ge=1)
    peak_requests_per_second: float = Field(2, gt=0)
    peak_concurrent_requests: int = Field(16, ge=1)
    measured_output_tokens_per_second_per_replica: float = Field(500, gt=0)
    capacity_headroom_fraction: float = Field(0.25, ge=0, le=2)

    @model_validator(mode="after")
    def validate_memory_policy(self):
        if self.runtime_reserve_fraction >= self.usable_memory_fraction:
            raise ValueError("runtime_reserve_fraction must be smaller than usable_memory_fraction")
        return self


class SizingResult(BaseModel):
    model_name: str
    gpu_model: str
    weights_gib: float
    weights_fit_in_replica: bool
    kv_cache_mib_per_request: float
    kv_cache_capacity_per_replica: int
    workload_input_tokens_per_second: float
    workload_output_tokens_per_second: float
    workload_total_tokens_per_second: float
    workload_total_tokens_per_day: float
    replicas_for_memory: int
    replicas_for_throughput: int
    recommended_replicas: int
    recommended_gpus: int
    estimated_gpu_utilization_fraction: float
    bottleneck: Literal["memory", "throughput", "balanced"]
    warnings: list[str]
    assumptions: list[str]


class BenchmarkRequest(BaseModel):
    endpoint: HttpUrl
    model: str
    api_key: str | None = None
    prompt: str = "Explain Dynamic Resource Allocation in Kubernetes in two sentences."
    max_tokens: int = Field(128, ge=1, le=8192)
    requests: int = Field(10, ge=1, le=10000)
    concurrency: int = Field(2, ge=1, le=1000)
    timeout_seconds: float = Field(120, gt=0, le=1800)
    verify_tls: bool = True
    collect_local_gpu_telemetry: bool = True


class BenchmarkResult(BaseModel):
    attempted_requests: int
    successful_requests: int
    failed_requests: int
    elapsed_seconds: float
    requests_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output_tokens_per_second: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    gpu_samples: int = 0
    average_gpu_utilization_percent: float | None = None
    peak_gpu_memory_used_mib: float | None = None
    errors: list[str]
