# Measured results

One ASUS DGX Spark, August 2026, box otherwise idle for every figure.

**Compute** GB10, 20 cores, 128 GB unified, 916 GB NVMe. Ubuntu 24.04,
kernel 6.17.0-1031-nvidia, aarch64. Driver 580.173.02, CUDA 13.0.
Docker 29.2.1, nvidia-container-toolkit 1.20 via CDI.

**Serving.** SGLang + `RadixArk/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`
(8 draft tokens), `--kv-cache-dtype fp8_e4m3`, 262144 context,
`--mem-fraction-static 0.82`, CPU pinned to `5-9,15-19` (the Cortex-X5 cores).

Throughput is code generation (LRUCache prompt) at temperature 0, counting
`completion_tokens` over wall time.

## Latency and single-stream

| Metric | Result |
|---|---:|
| Time to first token | 190 ms (184–193, n=5) |
| Decode, thinking off | 60.0 tok/s (n=5; 59.9–60.1) |
| Decode, before concurrency tuning | 63.1 tok/s |
| Decode, thinking on | 46.7 tok/s |
| Prose decode | 26.0 tok/s |
| Gateway overhead (LiteLLM vs direct) | 62.3 vs 62.5 — none |

## Concurrency

Aggregate tok/s at each `max_running_requests` cap.

| Streams | cap 4 | cap 12 | cap 16 | TTFT p50 (16) | TTFT max (16) |
|---:|---:|---:|---:|---:|---:|
| 1 | 58.3 | 53.4 | 52.9 | 0.20 s | 0.20 s |
| 2 | 93.1 | 92.6 | 87.8 | 0.75 s | 1.20 s |
| 4 | 187.1 | 191.5 | 193.1 | 0.41 s | 0.41 s |
| 8 | 190.1 | 296.0 | 315.9 | 0.37 s | 0.37 s |
| 16 | — | 307.8 | **480.7** | 0.68 s | 0.68 s |
| 24 | — | 319.6 | 416.5 | 0.47 s | 9.98 s |
| 32 | — | 368.9 | 469.8 | 5.23 s | 11.09 s |

At the shipped cap of 4, going 4 → 8 gains 2% — that flatline is config, not
hardware. At cap 16, TTFT p50 and max are identical at 16 streams (zero
queueing, streams equal the cap); 24 and 32 are *slower* in aggregate with
worst-case TTFT past 10 s.

## Prefill / long context

| Prompt | TTFT | Prefill rate |
|---:|---:|---:|
| 2,448 tok | 1.54 s | 1,591 tok/s |
| 9,698 tok | 4.46 s | **2,173 tok/s** |
| 38,698 tok | 20.42 s | 1,895 tok/s |
| 120,865 tok | **94.89 s** | 1,274 tok/s |

Prefill peaks near 10K tokens then degrades. The 262K context is real, but a
~120K prompt costs ~95 s before the first token.

## Quality — HumanEval

164 problems, temperature 0, each executed against its real unit tests in a
`--network none` container. Not self-judged.

| Mode | pass@1 | Tokens/problem | Wall |
|---|---:|---:|---:|
| Thinking off | 93.9% (154/164) | ~200 | 3 min |
| Thinking on | **97.0%** (159/164) | ~945 | 19 min |

Thinking fixed 7 genuine failures (77, 91, 93, 103, 116, 127, 160).

Of the 5 remaining thinking-mode failures, **only one is a wrong answer**
(HumanEval/115). The other four burned a full 16,384-token budget without
emitting code. Excluding runaways: 99.4% (159/160).

**Three measurement traps**, all hit here:

- **Truncation reads as a quality regression.** A first thinking run at a 4,096
  cap scored 90.9%; 14 of its 15 failures had `finish_reason == "length"`.
  Always check it before quoting a thinking-mode score.
- **Greedy is not bitwise deterministic.** Temperature 0 still flips 2–3
  problems between runs, because dynamic batching plus speculative decoding
  changes reduction order. Treat pass@1 as ±2 problems.
- **Harness bugs masquerade as model errors.** A naive run scored 92.7% where 5
  of 12 "failures" were mine: the fence regex required a *closing* fence, and
  HumanEval/38 and /50 define a helper above the target that the tests need.

## Thermals

At 96% GPU utilisation, sustained:

| | |
|---|---:|
| GPU temperature | 59 °C (idle 36 °C) |
| SM clock | 2405 MHz |
| Throttle reasons active | `0x0` — none |
| ACPI thermal zones | 59–70 °C |

Full load costs ~23 °C over idle with clocks pinned at maximum and no
throttling. SparkStation suspends at 80 °C, so ~21 °C of headroom.

Reported ~32 W board power under load is implausibly low for GB10 — that rail
appears to cover only part of the SoC. Don't use it for power budgeting.

## Memory

| `max-mamba-cache-size` | GDN pool | Free GPU after load | Peak aggregate |
|---:|---:|---:|---:|
| 20 | 3.8 GB | 38.1 GB | 190 tok/s |
| 64 | 12.2 GB | 24.1 GB | 369 tok/s |
| 80 | 15.6 GB | 20.6 GB | **480.7 tok/s** |

At 80 slots: `ssm_state` 5.70 GB + `intermediate_ssm_state_cache` 9.56 GB +
conv caches 0.38 GB. On this hybrid architecture concurrency is bought with
mamba state, not KV cache — worth remembering when sizing a second model
alongside it.

## Reference comparison

| Configuration | Reported | Source |
|---|---:|---|
| FP8 + vLLM | ~32 tok/s | widely-shared Reddit recipe |
| NVFP4 + vLLM | +29–34% over FP8 | NVIDIA developer forums |
| NVFP4 + SGLang + DFlash2 | 50–51 tok/s | MiaAI-Lab, hasso5703 |
| **This build** | **60.0 single / 480.7 aggregate** | measured here |

Prompt shapes differ across sources, so absolute comparison is approximate. The
FP8-vs-NVFP4 and vLLM-vs-SGLang ordering is the durable part.
