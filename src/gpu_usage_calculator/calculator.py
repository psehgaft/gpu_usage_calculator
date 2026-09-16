import math

from .models import SizingRequest, SizingResult


GIB = 1024**3
MIB = 1024**2


def estimate_capacity(request: SizingRequest) -> SizingResult:
    weight_bytes = (
        request.model_parameters_billions
        * 1_000_000_000
        * request.weight_precision_bits
        / 8
        * (1 + request.weight_overhead_fraction)
    )
    weights_gib = weight_bytes / GIB

    replica_raw_gib = request.gpu_memory_gib * request.tensor_parallel_size
    allocatable_gib = replica_raw_gib * request.usable_memory_fraction
    runtime_reserve_gib = replica_raw_gib * request.runtime_reserve_fraction
    kv_available_gib = max(0.0, allocatable_gib - runtime_reserve_gib - weights_gib)

    kv_bytes_per_token = (
        2
        * request.layers
        * request.kv_heads
        * request.head_dimension
        * request.kv_cache_precision_bytes
    )
    request_tokens = request.average_input_tokens + request.average_output_tokens
    kv_bytes_per_request = kv_bytes_per_token * request_tokens
    kv_capacity = math.floor((kv_available_gib * GIB) / kv_bytes_per_request)
    weights_fit = weights_gib + runtime_reserve_gib < allocatable_gib

    input_tps = request.peak_requests_per_second * request.average_input_tokens
    output_tps = request.peak_requests_per_second * request.average_output_tokens
    total_tps = input_tps + output_tps
    headroom_multiplier = 1 + request.capacity_headroom_fraction

    if weights_fit and kv_capacity > 0:
        replicas_for_memory = math.ceil(
            request.peak_concurrent_requests * headroom_multiplier / kv_capacity
        )
    else:
        replicas_for_memory = 0
    replicas_for_throughput = math.ceil(
        output_tps
        * headroom_multiplier
        / request.measured_output_tokens_per_second_per_replica
    )
    recommended_replicas = (
        max(replicas_for_memory, replicas_for_throughput, 1) if weights_fit else 0
    )
    recommended_gpus = recommended_replicas * request.tensor_parallel_size

    if not weights_fit:
        bottleneck = "memory"
    elif replicas_for_memory > replicas_for_throughput:
        bottleneck = "memory"
    elif replicas_for_throughput > replicas_for_memory:
        bottleneck = "throughput"
    else:
        bottleneck = "balanced"

    capacity = (
        recommended_replicas * request.measured_output_tokens_per_second_per_replica
    )
    utilization = output_tps / capacity if capacity else 0.0
    warnings: list[str] = []
    if not weights_fit:
        warnings.append(
            "Model weights and runtime reserve do not fit in the selected tensor-parallel replica."
        )
    if kv_capacity < request.peak_concurrent_requests and weights_fit:
        warnings.append("One replica cannot hold the requested peak concurrency in KV cache.")
    if request.capacity_headroom_fraction < 0.15:
        warnings.append("Headroom below 15% can make latency unstable during bursts or failures.")
    if request.measured_output_tokens_per_second_per_replica == 500:
        warnings.append(
            "The throughput value is the example default. Replace it with a benchmark from your model and serving stack."
        )

    return SizingResult(
        model_name=request.model_name,
        gpu_model=request.gpu_model,
        weights_gib=round(weights_gib, 2),
        weights_fit_in_replica=weights_fit,
        kv_cache_mib_per_request=round(kv_bytes_per_request / MIB, 2),
        kv_cache_capacity_per_replica=max(kv_capacity, 0),
        workload_input_tokens_per_second=round(input_tps, 2),
        workload_output_tokens_per_second=round(output_tps, 2),
        workload_total_tokens_per_second=round(total_tps, 2),
        workload_total_tokens_per_day=round(total_tps * 86400, 2),
        replicas_for_memory=replicas_for_memory,
        replicas_for_throughput=replicas_for_throughput,
        recommended_replicas=recommended_replicas,
        recommended_gpus=recommended_gpus,
        estimated_gpu_utilization_fraction=round(utilization, 4),
        bottleneck=bottleneck,
        warnings=warnings,
        assumptions=[
            "Inference workload with a persistent model replica.",
            "KV cache estimate uses K and V tensors for every layer and request token.",
            "Throughput comes from a representative benchmark, not the GPU name alone.",
            "Recommendation covers peak demand plus configured headroom, before failure-domain reserve.",
        ],
    )
