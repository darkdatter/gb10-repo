# Measured results

One ASUS DGX Spark, August 2026. Box otherwise idle for every figure — competing
GPU work skews all of them.

**Rig.** GB10, 20 cores, 128 GB unified, 916 GB NVMe. Ubuntu 24.04,
kernel 6.17.0-1031-nvidia, aarch64. Driver 580.173.02, CUDA 13.0.
Docker 29.2.1, nvidia-container-toolkit 1.20 via CDI.

**Serving.** SGLang + `RadixArk/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`
(8 draft tokens), `--kv-cache-dtype fp8_e4m3`, 262144 context,
`--mem-fraction-static 0.82`, CPU pinned to `5-9,15-19` (the Cortex-X5 cores).

Throughput is code generation (LRUCache prompt) at temperature 0, counting
`completion_tokens` over wall time.

---

## Latency and single-stream

| Metric | Result |
|---|---:|
| Time to first token | **190 ms** (184–193, n=5) |
| Single-stream decode, thinking off | **60.0 tok/s** (n=5; 59.9–60.1) |
| Single-stream decode, before concurrency tuning | 63.1 tok/s |
| Single-stream decode, thinking on | 46.7 tok/s |
| Prose decode | 26.0 tok/s |
| Gateway overhead (LiteLLM vs direct) | 62.3 vs 62.5 tok/s — none |

## Concurrency

Aggregate tok/s at each `max_running_requests` cap.

| Streams | cap 4 | cap 12 | cap 16 | TTFT p50 (cap 16) | TTFT max (cap 16) |
|---:|---:|---:|---:|---:|---:|
| 1 | 58.3 | 53.4 | 52.9 | 0.20 s | 0.20 s |
| 2 | 93.1 | 92.6 | 87.8 | 0.75 s | 1.20 s |
| 4 | 187.1 | 191.5 | 193.1 | 0.41 s | 0.41 s |
| 8 | 190.1 | 296.0 | 315.9 | 0.37 s | 0.37 s |
| 16 | — | 307.8 | **480.7** | 0.68 s | 0.68 s |
| 24 | — | 319.6 | 416.5 | 0.47 s | 9.98 s |
| 32 | — | 368.9 | 469.8 | 5.23 s | 11.09 s |

**16 streams is a true optimum, not a plateau.** TTFT p50 and max are identical
there — zero queueing, because stream count equals the cap. At 24 and 32 the
aggregate is *lower* and worst-case TTFT exceeds 10 s.

At the shipped cap of 4, going 4 → 8 streams gains 2%. That flatline is config,
not hardware.

## Prefill / long context

| Prompt | TTFT | Prefill rate |
|---:|---:|---:|
| 2,448 tok | 1.54 s | 1,591 tok/s |
| 9,698 tok | 4.46 s | **2,173 tok/s** |
| 38,698 tok | 20.42 s | 1,895 tok/s |
| 120,865 tok | **94.89 s** | 1,274 tok/s |

Prefill peaks near 10K tokens then degrades. The 262K context is real, but a
~120K-token prompt costs ~95 seconds before the first token — fine for batch,
painful interactively.

## Quality — HumanEval

164 problems, temperature 0, every candidate executed against its real unit
tests in a `--network none` container. Not self-judged.

| Mode | pass@1 | Tokens/problem | Wall clock |
|---|---:|---:|---:|
| Thinking off | 93.9% (154/164) | ~200 | 3 min |
| Thinking on | **97.0%** (159/164) | ~945 | 19 min |

Thinking fixed 7 genuine failures (77, 91, 93, 103, 116, 127, 160).

**Of the 5 remaining thinking-mode failures, only one is a wrong answer**
(HumanEval/115). The other four burned a full 16,384-token budget without
emitting code. Excluding those runaways: 99.4% (159/160).

### Two measurement traps

**Truncation reads as a quality regression.** A first thinking run at a
4,096-token cap scored 90.9% — and 14 of its 15 failures had
`finish_reason == "length"`. Always check `finish_reason` before quoting a
thinking-mode score.

**Greedy is not bitwise deterministic.** At temperature 0, repeat runs still
flip 2–3 problems, because dynamic batching plus speculative decoding changes
reduction order. Treat pass@1 as ±2 problems; do not read a sub-2% delta as a
regression.

A naive harness scored 92.7% where 5 of 12 "failures" were harness bugs:
the fence regex required a *closing* fence (truncated generations fell
through), and HumanEval/38 and /50 define a helper function above the target
that the tests need.

## Thermals

Measured at 96% GPU utilisation, sustained.

| | |
|---|---:|
| GPU temperature | 59 °C |
| SM clock | 2405 MHz |
| Throttle reasons active | `0x0` — none |
| ACPI thermal zones | 59–70 °C |
| Idle GPU temperature | 36 °C |

Full load costs ~23 °C over idle, with clocks pinned at maximum and no
throttling. SparkStation's policy suspends at 80 °C, so there is ~21 °C of
headroom.

The reported ~32 W board power under full load is implausibly low for GB10 —
that rail appears to cover only part of the SoC. Do not use it for power
budgeting.

## Memory

| Configuration | GDN state pool | Free GPU memory after load |
|---|---:|---:|
| `max-mamba-cache-size 20` | 3.8 GB | 38.1 GB |
| `max-mamba-cache-size 64` | 12.2 GB | 24.1 GB |
| `max-mamba-cache-size 80` | 15.6 GB | 20.6 GB |

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
| **This build** | **60.0 tok/s single / 480.7 aggregate** | measured here |

Different prompt shapes and output lengths make cross-source comparison
approximate; the FP8-vs-NVFP4 and vLLM-vs-SGLang ordering is the durable part.
