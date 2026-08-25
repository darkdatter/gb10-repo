# Measured results

One ASUS DGX Spark, August 2026, box otherwise idle for every figure.

**Compute** GB10, 20 cores, 128 GB unified, 916 GB NVMe. Ubuntu 24.04,
kernel 6.17.0-1031-nvidia, aarch64. Driver 580.173.02, CUDA 13.0.
Docker 29.2.1, nvidia-container-toolkit 1.20 via CDI.

**Serving.** SGLang + `RadixArk/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`,
`--kv-cache-dtype fp8_e4m3`, `--mamba-ssm-dtype bfloat16`, 262144 context,
`--mem-fraction-static 0.85`, CPU pinned to `5-9,15-19` (the Cortex-X5 cores).

**Draft tokens differ by section.** Everything below was measured at
`--speculative-num-draft-tokens 8` — TTFT, thinking-on, prose, concurrency,
prefill, long context and HumanEval — **except** the draft=16 rows, which are
labelled as such. The two settings are not interchangeable: draft=16 is +28% on
single-stream and −10% on aggregate, so mixing them in one figure is misleading.

Throughput is code generation (LRUCache prompt) at temperature 0, counting
`completion_tokens` over wall time.

## Latency and single-stream

| Metric | Result |
|---|---:|
| Time to first token | 190 ms (184–193, n=5) |
| Decode, thinking off, draft=16 | **78.6 tok/s** (n=5; 77.0–78.9) |
| Decode, thinking off, draft=8 | 61.3 tok/s |
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
hardware.

**The peak tracks the cap, it is not a hardware limit.** At cap 12 the peak sits
at 32 streams; at cap 16 it moves to 16. TTFT p50 and max are identical at the
cap because nothing queues there, and the 24/32 rows are queueing against it —
not the GPU running out. Raising `max-mamba-cache-size` moves the peak again
(pool 160 → 474.0 at 32 streams). See [REPRODUCTION.md](REPRODUCTION.md).

## Prefill / long context

| Prompt | TTFT | Prefill rate |
|---:|---:|---:|
| 2,448 tok | 1.54 s | 1,591 tok/s |
| 9,698 tok | 4.46 s | **2,173 tok/s** |
| 38,698 tok | 20.42 s | 1,895 tok/s |
| 120,865 tok | **94.89 s** | 1,274 tok/s |

Prefill peaks near 10K tokens then degrades. The 262K context is real, but a
~120K prompt costs ~95 s before the first token.

## Draft tokens

`--speculative-num-draft-tokens`, all values on the same build. `accept_len` is
mean accepted tokens per step; `accept_rate` the fraction of drafted tokens kept.

| draft | single | agg @16 | accept_len | accept_rate |
|---:|---:|---:|---:|---:|
| 6 | 49.5 | 404.2 | 5.25 | 0.85 |
| 7 | 55.7 | 386.6 | 5.89 | 0.82 |
| 8 (default) | 61.3 | 429.2 | 6.63 | 0.80 |
| 10 | 65.2 | **435.1** | 7.18 | 0.69 |
| 12 | 75.1 | 402.8 | 8.27 | 0.66 |
| 14 | 78.8 | 386.5 | 8.87 | 0.61 |
| 16 | **78.6** | 384.8 | 9.52 | 0.57 |
| 20 | 72.0 | 330.7 | 8.48 | 0.40 |
| 24 | 69.0 | 284.5 | 8.40 | 0.32 |

**+28% single-stream** from the default. The optima diverge: 16 for interactive,
**10 for concurrent serving**.

`accept_len` rises to 9.52 at draft=16 then falls at 20 and 24 — past 16 the
drafter cannot sustain longer correct runs, so the extra draft compute buys
rejected tokens. Under 16-stream saturation that wasted compute competes with
real work, which is why aggregate peaks much earlier.

draft=16 has now been measured three times: **85.7** in-sweep here, **78.6** on a
fresh boot here, **84.4** on a [second machine](REPRODUCTION.md). The table
publishes 78.6 because that was the fresh-boot re-measurement, but two of the
three cluster at 84–86, so **78.6 is probably the low end of the spread rather
than the centre** and the true figure is likely nearer +40% than +28%.

That spread is also why 12 / 14 / 16 are not distinguishable from each other —
only the broad shape is trustworthy.

Quality is unaffected: the second machine measured HumanEval at both settings and
found ±2 problems, inside the documented nondeterminism band. Expected, since
every draft token is verified against the target model — a wider window changes
throughput, not output.

Sweep values contradict the "block-7 peak" reported elsewhere for this stack;
that figure was measured on DSpark, not DFlash2.

## Long-context concurrency

Unique content per stream (`cached_tokens == 0`), so this is real KV demand.

| Streams @ 83K | Unique KV | Wall | TTFT max |
|---:|---:|---:|---:|
| 4 | 332,710 | 277.4 s | 277.1 s |
| 8 | 665,932 | 555.8 s | 555.4 s |
| 12 | 998,767 | 832.8 s | 832.2 s |

**69.4 s per stream, dead linear.** Prefill fully serialises — `#new-seq: 1` per
batch at ~1,000 tok/s — so concurrency does not help long-context work; it just
makes everyone wait proportionally longer. Decode throughput is the wrong
predictor for agent workloads over large inputs.

A 16-stream run did not fail on memory: the model was too busy to answer a 5 s
health probe, so the supervisor restarted it (see Traps in the README).

**Test design matters here.** Build every prompt from shared filler and the
radix cache deduplicates them — the first version of this test measured cache
hits, not capacity. Assert `cached_tokens == 0`.

## mem-fraction

| Config | Single-stream | 16 streams | 32 streams | Free GPU |
|---|---:|---:|---:|---:|
| 0.82 + 1M cap | 60.0 | **480.7** | 469.8 | 20.6 GB |
| 0.90 + 1M cap | 62.3 | 468.7 | 468.7 | 25.3 GB |
| 0.90, pool uncapped | 52.3 | 468.7 | 384.7 | 8.1 GB |
| **0.85 + 1M cap** | 61.8 | 472.0 | **477.0** | **26.1 GB** |

**Not a performance lever.** A 10-point spread moves throughput by less than
run-to-run noise. Uncapping the pool grows it 1,048,576 → 1,496,047 tokens, but
nothing uses the space (peak observed usage 67%) and the lost headroom cost 18%
at 32 streams. 0.85 with the cap kept is the settled default: baseline
throughput, most free memory.

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

**Four measurement traps**, all hit here:

- **Truncation reads as a quality regression.** A first thinking run at a 4,096
  cap scored 90.9%; 14 of its 15 failures had `finish_reason == "length"`.
  Always check it before quoting a thinking-mode score.
- **Boot-to-boot variance is ~8% on single-stream**, against <2% run-to-run
  within one instance. Several small deltas in this document sit inside that
  band; treat only large moves as signal.
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
| GPU temperature, decode | 59 °C (idle 36 °C) |
| GPU temperature, sustained prefill | 74 °C |
| SM clock | 2405 MHz |
| Throttle reasons active | `0x0` — none |
| ACPI thermal zones | 59–70 °C |

Prefill, not decode, is the thermally interesting workload: 74 °C leaves 6 °C of
margin to the 80 °C suspend threshold, against 21 °C on decode-heavy work.

Reported ~32 W board power under load is implausibly low for GB10 — that rail
appears to cover only part of the SoC. Don't use it for power budgeting.

## Memory

| `max-mamba-cache-size` | GDN pool | Free GPU after load | Peak aggregate |
|---:|---:|---:|---:|
| 20 | 3.8 GB | 38.1 GB | 190 tok/s |
| 64 | 12.2 GB | 24.1 GB | 369 tok/s |
| 80 | 15.6 GB | 20.6 GB | **480.7 tok/s** |

At 80 slots: `ssm_state` 5.70 GB + `intermediate_ssm_state_cache` 9.56 GB +
conv caches 0.38 GB — these hold only with `--mamba-ssm-dtype bfloat16`. Left
unset the SSM state resolves to `float32`, which doubles this to 30.9 GB and
leaves ~12 GB free. On this hybrid architecture concurrency is bought with
mamba state, not KV cache — worth remembering when sizing a second model
alongside it.

## Reference comparison

| Configuration | Reported | Source |
|---|---:|---|
| FP8 + vLLM | ~32 tok/s | widely-shared Reddit recipe |
| NVFP4 + vLLM | +29–34% over FP8 | NVIDIA developer forums |
| NVFP4 + SGLang + DFlash2 | 50–51 tok/s | MiaAI-Lab, hasso5703 |
| **This build**, draft=8 | 61.3 single / **480.7** aggregate | measured here |
| **This build**, draft=16 | **78.6** single / 384.8 aggregate | measured here |

Prompt shapes differ across sources, so absolute comparison is approximate. The
FP8-vs-NVFP4 and vLLM-vs-SGLang ordering is the durable part.
