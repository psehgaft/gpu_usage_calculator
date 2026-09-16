from fastapi.testclient import TestClient

from gpu_usage_calculator.app import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_estimate_api() -> None:
    response = client.post(
        "/api/v1/estimate",
        json={
            "model_parameters_billions": 7,
            "weight_precision_bits": 8,
            "gpu_memory_gib": 24,
            "tensor_parallel_size": 1,
            "measured_output_tokens_per_second_per_replica": 200,
        },
    )
    assert response.status_code == 200
    assert response.json()["weights_fit_in_replica"] is True

