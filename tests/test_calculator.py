from gpu_usage_calculator.calculator import estimate_capacity
from gpu_usage_calculator.models import SizingRequest


def test_recommends_capacity_for_valid_model() -> None:
    result = estimate_capacity(
        SizingRequest(
            model_parameters_billions=70,
            weight_precision_bits=8,
            tensor_parallel_size=2,
            measured_output_tokens_per_second_per_replica=850,
        )
    )
    assert result.weights_fit_in_replica
    assert result.recommended_gpus >= 2
    assert result.workload_total_tokens_per_second == 2500


def test_detects_model_that_does_not_fit() -> None:
    result = estimate_capacity(
        SizingRequest(
            model_parameters_billions=405,
            weight_precision_bits=16,
            gpu_memory_gib=24,
            tensor_parallel_size=1,
        )
    )
    assert not result.weights_fit_in_replica
    assert result.recommended_gpus == 0
    assert result.warnings


def test_throughput_drives_replica_count() -> None:
    result = estimate_capacity(
        SizingRequest(
            model_parameters_billions=7,
            weight_precision_bits=8,
            tensor_parallel_size=1,
            gpu_memory_gib=80,
            peak_requests_per_second=20,
            average_output_tokens=500,
            peak_concurrent_requests=2,
            measured_output_tokens_per_second_per_replica=1000,
            capacity_headroom_fraction=0.2,
        )
    )
    assert result.replicas_for_throughput == 12
    assert result.bottleneck == "throughput"
