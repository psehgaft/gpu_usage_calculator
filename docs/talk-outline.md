# DevConf talk structure

**Title:** GPUs, DRA, and Smarter Scheduling for AI: Preparing Kubernetes for Accelerated Workloads

**Core question:** What does Kubernetes need in order to become truly ready for large-scale AI and accelerated workloads?

**Audience:** Platform engineers, SREs, and architects with basic Kubernetes knowledge. GPU experience is useful but optional.

**Recommended duration:** 45 minutes, including a 7-minute demonstration.

## Narrative and timing

| Time | Section | Audience outcome |
|---:|---|---|
| 0–4 min | The AI workload changes the scheduling problem | Recognize why CPU and memory assumptions do not describe accelerators |
| 4–9 min | GPU capacity has several dimensions | Separate device count, memory fit, token throughput, latency, and concurrency |
| 9–14 min | Device plugins and extended resources | Understand the current allocation model and its limits |
| 14–20 min | Dynamic Resource Allocation | Understand DeviceClass, ResourceClaim, selection, sharing, and per-workload configuration |
| 20–27 min | Smarter scheduling | Connect topology, partitions, affinity, spread, queues, priorities, and gang scheduling |
| 27–33 min | Fairness and quota | Design tenancy controls around scarce, heterogeneous devices |
| 33–40 min | GPU Usage Calculator demo | Convert token demand and model assumptions into a testable capacity estimate |
| 40–44 min | Production capacity checklist | Apply resilience, observability, lifecycle, and cost controls |
| 44–45 min | Closing answer | Summarize the architecture needed for an AI-ready Kubernetes platform |

## Proposed slide sequence

1. Title
2. The question Kubernetes must answer
3. Why the default scheduling model breaks down
4. Five capacity signals: memory, tokens, latency, concurrency, and topology
5. GPU allocation with device plugins
6. Limits of integer extended resources
7. DRA resource model
8. DeviceClass and ResourceClaim example
9. Scheduling policy for heterogeneous accelerators
10. Partitioning and sharing: full GPU, MIG, and time slicing
11. Quota does not equal capacity
12. Capacity model used by `gpu_usage_calculator`
13. Demo assumptions
14. Demo: memory fit and KV cache
15. Demo: throughput and headroom
16. Production failure scenarios
17. Reference platform architecture
18. AI-ready platform checklist
19. Answer to the session question
20. Q&A and repository

## Closing answer

Kubernetes becomes ready for accelerated AI when the platform combines device-aware allocation, topology-aware scheduling, workload measurements, fair admission and quota policy, failure reserve, and continuous GPU telemetry. GPU count by itself is insufficient.

## Template mapping

- Use the existing DevConf title layout for slide 1.
- Use section dividers before DRA, the calculator demo, and the production checklist.
- Use the comparison layout for device plugins versus DRA and full GPU versus partitioned or shared GPU.
- Use the feature layout for terminal screenshots and calculator results.
- Reserve bold statement layouts for the key conclusions: “Quota does not equal capacity” and “GPU count is only one input.”
