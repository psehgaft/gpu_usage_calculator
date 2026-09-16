# GPU Usage Calculator

`gpu_usage_calculator` turns token demand and model assumptions into an explainable GPU capacity estimate for Kubernetes and OpenShift. It also benchmarks an OpenAI-compatible inference endpoint so that measured throughput can replace generic GPU assumptions.

The project supports the DevConf session **“GPUs, DRA, and Smarter Scheduling for AI: Preparing Kubernetes for Accelerated Workloads.”**

## What it calculates

- Approximate model-weight memory after precision and overhead
- KV cache memory per active request
- Concurrency that fits in one tensor-parallel replica
- Input, output, total tokens per second, and total tokens per day
- Replicas required by memory and measured output throughput
- Recommended GPU count with configurable headroom
- The dominant constraint and warnings about unsafe assumptions

## What it measures

The optional benchmark calls `/v1/chat/completions` on an OpenAI-compatible endpoint and reports:

- successful and failed requests
- requests per second
- prompt, completion, and total tokens
- output tokens per second
- p50 and p95 request latency
- local GPU utilization and peak memory when `nvidia-smi` is available in the pod

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
gpu-usage-calculator estimate --file examples/sizing-request.json
uvicorn gpu_usage_calculator.app:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080/docs` for the API or `http://localhost:8080` for the simple calculator.

## Container

```bash
podman build -t gpu-usage-calculator:latest -f Containerfile .
podman run --rm -p 8080:8080 gpu-usage-calculator:latest
```

The image uses Red Hat UBI and runs as a non-root user. The default deployment does not require a GPU because planning and remote benchmarking should not consume accelerator capacity.

## Kubernetes

Replace `quay.io/example/gpu-usage-calculator:latest` with the published image, then run:

```bash
kubectl apply -k deploy/kubernetes/base
kubectl -n gpu-sizing port-forward service/gpu-usage-calculator 8080:8080
```

The optional overlay requests one NVIDIA GPU for demonstrations that require direct device access:

```bash
kubectl apply -k deploy/kubernetes/overlays/gpu
```

When the benchmark targets a remote inference service, GPU telemetry must come from that service's monitoring stack, such as DCGM Exporter and Prometheus. Local `nvidia-smi` sampling only represents devices visible to the calculator pod.

## OpenShift

```bash
oc apply -k deploy/kubernetes/base
oc apply -f deploy/openshift/route.yaml
oc -n gpu-sizing get route gpu-usage-calculator
```

The workload follows the restricted security profile: non-root, no privilege escalation, dropped Linux capabilities, and a read-only root filesystem.

## API example

```bash
curl -s http://localhost:8080/api/v1/estimate \
  -H 'Content-Type: application/json' \
  --data @examples/sizing-request.json
```

## Calculation model

The estimator uses four linked calculations:

1. **Weight memory:** parameters multiplied by bytes per weight and an overhead factor.
2. **KV cache:** two tensors per layer, multiplied by KV heads, head dimension, element size, and active tokens.
3. **Throughput:** peak requests per second multiplied by average output tokens, divided by measured output tokens per second for one replica.
4. **Capacity:** the larger of memory-driven and throughput-driven replica counts, including configured headroom.

See [GPU platform sizing guide](docs/sizing-guide.md) for interpretation and operational considerations.

## Repository map

```text
src/gpu_usage_calculator/  API, CLI, estimator, and benchmark runner
config/                    GPU memory profiles
deploy/                    Kubernetes, OpenShift, quota, priority, and DRA examples
docs/                      Architecture, sizing, demo, and talk structure
examples/                  Reproducible input files
tests/                     Unit and API tests
```

## Documentation

- [Architecture](docs/architecture.md)
- [GPU platform sizing guide](docs/sizing-guide.md)
- [Demo guide](docs/demo-guide.md)
- [DevConf talk structure](docs/talk-outline.md)
- [Technical references](docs/references.md)

## DRA and compatibility

The DRA example targets the stable `resource.k8s.io/v1` API in Kubernetes 1.35 or later. A compatible DRA driver must publish the referenced `DeviceClass` and device attributes. Keep the extended-resource deployment as the portable baseline for clusters that do not yet provide the required DRA driver and API version.

## Limits of the estimate

This tool supports architecture decisions but does not replace model-specific benchmarking. Performance changes with the serving runtime, model architecture, quantization method, prompt distribution, batching, cache policy, GPU topology, interconnect, drivers, and software versions. Validate production capacity under peak load and failure conditions.

## License

Apache License 2.0.
