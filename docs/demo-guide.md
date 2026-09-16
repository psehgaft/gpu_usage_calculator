# Demo guide

## Demo objective

Show how the same daily token volume can require different GPU capacity when concurrency, context length, measured throughput, or headroom changes.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn gpu_usage_calculator.app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` or run:

```bash
gpu-usage-calculator estimate --file examples/sizing-request.json
```

## Kubernetes or OpenShift

Build and push the image, replace the example image in `deployment.yaml`, then deploy:

```bash
kubectl apply -k deploy/kubernetes/base
kubectl -n gpu-sizing port-forward service/gpu-usage-calculator 8080:8080
```

On OpenShift:

```bash
oc apply -k deploy/kubernetes/base
oc apply -f deploy/openshift/route.yaml
oc -n gpu-sizing get route gpu-usage-calculator
```

## Live demonstration sequence

1. Start with a 70B model at 16-bit precision and one 80 GiB GPU. Show that the model does not fit.
2. Increase tensor parallelism to two GPUs or change precision to 8 bits. Explain the memory tradeoff.
3. Increase average input tokens and concurrency. Show KV cache becoming the constraint.
4. Enter measured throughput from the benchmark endpoint. Show throughput becoming the constraint as peak RPS rises.
5. Add 25% headroom and discuss node failure reserve, which remains an explicit platform decision.
6. Compare the result with namespace quota and node capacity. Explain why a valid quota does not guarantee schedulable capacity.

## Optional benchmark

Point the benchmark at an OpenAI-compatible endpoint:

```bash
gpu-usage-calculator benchmark --file examples/benchmark-request.json
```

Do not place a production API key in the JSON file. Supply secrets through your platform's secret-management mechanism when automating the demo.

If the calculator runs with the GPU overlay and `nvidia-smi` is visible, the result includes local utilization and peak memory. For a remote inference endpoint, correlate the test interval with DCGM Exporter metrics in Prometheus.
