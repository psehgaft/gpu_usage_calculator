# GPU platform sizing guide

## 1. Describe demand as a distribution

Capture average and p95 input tokens, output tokens, requests per second, concurrent requests, burst duration, and the latency objective. A daily token total alone hides the peak that drives capacity.

## 2. Establish model memory fit

The first approximation for weight memory is:

`parameters × bits per weight ÷ 8`

Add overhead for quantization metadata and runtime structures. Split the model across the GPUs in a tensor-parallel replica when one device cannot hold the model. Preserve memory for the runtime and KV cache.

## 3. Estimate KV cache pressure

For a transformer using grouped-query attention, this project approximates KV bytes per token as:

`2 × layers × KV heads × head dimension × KV element bytes`

Multiply by active sequence length and concurrency. Prefix caching, paged attention, speculative decoding, sliding windows, and cache quantization can change the effective capacity.

## 4. Measure token throughput

Benchmark the exact combination of model, quantization, serving runtime, GPU, tensor parallelism, prompt distribution, output length, and concurrency. Record output tokens per second, time to first token, inter-token latency, p50 latency, p95 latency, errors, GPU memory, GPU utilization, and power.

## 5. Calculate replicas

Calculate replicas independently for memory and throughput. Use the larger value, then add capacity headroom and failure-domain reserve. A platform that needs to survive one node failure must keep enough capacity outside the failed node.

## 6. Map capacity to scheduling

Treat the GPU as a device with topology and health, not just an integer. Consider device class, memory, interconnect, NUMA locality, MIG profile, time-slicing policy, node labels, taints, affinity, topology spread, priority, preemption, and gang scheduling for distributed jobs.

## 7. Design fair quotas

GPU count quotas control allocation but do not express token demand or service importance. Combine namespace quotas with admission policy, per-team budgets, workload priority, queueing, and observability. Keep interactive inference separate from long-running training when their service objectives conflict.

## 8. Validate the production envelope

Test node loss, cold model loading, image and model download time, autoscaling delay, fragmented MIG capacity, network bottlenecks, storage throughput, and degraded devices. Recalculate after changing the model or serving stack.

## Capacity worksheet

| Dimension | Required input | Platform consequence |
|---|---|---|
| Demand | Peak RPS, tokens/request, concurrency | Throughput and queue depth |
| Latency | TTFT and end-to-end SLO | Headroom and batch limits |
| Model | Parameters, precision, architecture | Weight and KV memory |
| Runtime | Engine, batching, cache policy | Measured tokens/second |
| Device | Memory, topology, partitioning | Feasible replica shape |
| Resilience | Node and zone failure target | Spare capacity |
| Tenancy | Teams, priorities, budgets | Quota and queue policy |
| Operations | Drivers, telemetry, upgrades | Lifecycle and support model |

## Important interpretation

The output is a planning estimate. Procurement and production commitments require representative benchmarks and failure testing.
