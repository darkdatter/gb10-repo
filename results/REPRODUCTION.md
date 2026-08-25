# Independent reproduction

Second DGX Spark GB10, 2026-08-23, box idle. Same `bench/perf.py`, same pins.
Official image `lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2`
(`sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`),
`--mamba-ssm-dtype bfloat16`, `--mem-fraction-static 0.85`, pool 80 / cap 16,
pinned to `5-9,15-19`.

The table below is at **draft=8**, matching the config these RESULTS.md rows
were measured under. Draft=16 is a separate section.

| Metric | RESULTS.md | Here |
|---|---:|---:|
| Time to first token | 190 ms | 200 ms |
| Decode, thinking off (draft=8) | 61.3 tok/s | 60.3 |
| Decode, thinking on | 46.7 tok/s | 47.3 |
| Peak aggregate | 480.7 @ 16 | 474.5 @ 16 |
| Prefill 9,698 tok | 2,173 tok/s | 2,215 |
| Prefill 120,865 tok | 94.89 s | 88.68 s |
| HumanEval, thinking off | 93.9% | 94.5% |
| HumanEval, thinking on | 97.0% | 97.6% |

Draft acceptance 6.28 of a block of 8. Both HumanEval deltas are one problem.
Thinking-on failures were all truncation at the 16,384 cap, none wrong answers.

Raising the GDN pool moves the peak: 606.4 tok/s at cap 32 (pool 160, 1.07M KV
tokens), 640.9 at cap 48 (pool 240, 450k KV tokens). Zero queueing at the cap in
both.

## Draft tokens

Re-run here as a within-boot A/B, same pool 80 / cap 16, only the flag differing.

| draft | single | agg @16 | accept_len | RESULTS.md single |
|---:|---:|---:|---:|---:|
| 8 | 59.6 | 464.6 | 6.82 | 61.3 |
| 16 | **84.4** | 393.6 | 9.25 | 78.6 |

The direction reproduces and `accept_len` lands within 3% of yours at both
values, so the mechanism is the same. The magnitude does not: **+41.6% here**
against the +28% published. Our 84.4 is nearer the 85.7 you measured in-sweep
than the 78.6 from the fresh boot, which suggests 78.6 is the low end of the
boot-to-boot spread rather than the centre. The aggregate reversal reproduces
too — draft=16 costs 15% aggregate at 16 streams.

Quality is unchanged at the wider window, which had not been measured anywhere:

| HumanEval | draft=8 | draft=16 | excl. truncation |
|---|---:|---:|---:|
| thinking off | 94.5% | 95.7% (157/164) | 96.9% |
| thinking on | 97.6% | 96.3% (158/164) | 99.4% (158/159) |

+2 problems one way, -2 the other — inside the +/-2 band you document. So the
speed is free: `accept_rate` falling 0.80 -> 0.57 costs throughput, not
accuracy, since every draft is verified against the target model.

One thing to flag rather than patch, since it is your new text: the methodology
header now says 16 draft tokens, but TTFT, thinking-on, prose, concurrency,
prefill and HumanEval were all measured at 8. It surfaces in the comparison
row — `78.6 single / 480.7 aggregate` is two different configs, since draft=16
puts aggregate at 384.8 and draft=8 puts single at 61.3.

The base `lmsysorg/sglang:qwen38-27b` image cannot serve this: no
`CandidateSelector`, `DFlash2DraftModel` unregistered, and its draft sampler
multiplies against the packed NVFP4 `lm_head`. It falls back to an eager draft
head and reaches 29.7 tok/s single-stream, 218.6 aggregate.
