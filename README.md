# Qwen3.8-27B on DGX Spark (GB10)

A measured recipe for the fastest Qwen3.8-27B setup I could get on a DGX Spark:
**SGLang + NVFP4 + DFlash2 speculative decoding**, managed by
[SparkStation](https://github.com/kshetrajna12/sparkstation).

| Single-stream | Peak aggregate | TTFT | HumanEval pass@1 |
|---:|---:|---:|---:|
| **78.6 tok/s** | **480.7 tok/s** (16 streams) | **190 ms** | **97.0%** |

Those two throughput figures come from different settings: single-stream is at
`--speculative-num-draft-tokens 16`, peak aggregate at 8. The optima diverge —
see step 6.

**NVFP4 beats FP8, and SGLang beats vLLM, on this hardware.** The widely-shared
"FP8 on vLLM at ~32 tok/s" recipe is roughly half this speed.

Reproduce with [`bench/`](bench/). Full data in
[`results/RESULTS.md`](results/RESULTS.md).

## Ingredients

| Component | Pin |
|---|---|
| Target | `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9` |
| Draft | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c` |
| Image | `lmsysorg/sglang:qwen38-27b-dflash2` — **built locally (step 2)** |
| Toolkit | [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) |
| Disk | ~110 GB |

All model repos are public; no token needed. Verified on Ubuntu 24.04,
kernel 6.17-nvidia, driver 580.173.02 / CUDA 13.0, Docker 29.2.1, 128 GB unified.

---

## 1. Verify GPU passthrough

```bash
./scripts/00-verify-gpu.sh
```

DGX OS ships CDI, not a registered Docker runtime — `docker info` showing only
`runc` is normal and `--gpus all` still works.

## 2. Build the image, fetch the weights

```bash
./scripts/01-build-and-fetch.sh
```

`lmsysorg/sglang:qwen38-27b-dflash2` **is not on Docker Hub.** SparkStation's
`models.yaml` references it, but no released SGLang tag ships DFlash2 — you
build it. The build also applies an NVFP4 `lm_head` patch; without it,
draft-graph capture allocates ~2.5 GB and hard-reboots the machine.

## 3. Run standalone, get a baseline

Get a number before adding orchestration.

```bash
cd ~/spark/Qwen3.8-27B-SGLang-DGX-Spark   # required: start.sh uses WORK_DIR="$(pwd)"
cp -n .env.sample .env
DF_EXTRA="--mem-fraction-static 0.85" ./start-dflash.sh
```

Serves on **:8888**, OpenAI- and Anthropic-compatible. First boot ≈3 min.

**Use 0.85 on a first boot.** The toolkit defaults NVFP4 to `0.90` and generic
`start.sh` to `0.95`; both this project and the toolkit record machines
unrecoverably wedged during weight load at higher fractions — GB10 unified
memory starves the OS.

```bash
GB10_BASE_URL=http://127.0.0.1:8888/v1 GB10_MODEL=qwen3.8-27b-sglang \
  python3 bench/perf.py
```

## 4. Add SparkStation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --depth 1 https://github.com/kshetrajna12/sparkstation.git
cd sparkstation && uv sync
cp .env.example .env      # TOTAL_UNIFIED_MEMORY_GB=128, MEMORY_HARD_LIMIT_GB=110
```

One required `models.yaml` edit — the entry ships pinned to a second machine:

```diff
-    host: worker1
+    host: primary
```

```bash
uv run sparkstation start -d --profile generic
```

Gateway on **:8000**. The model container is bridge-networked, so its internal
:8000 maps to host **:8001** — no collision despite both reporting 8000.
Gateway overhead measured at zero.

## 5. Unlock concurrency

The shipped config caps you at 4 concurrent requests. **Three flags are
co-limiting** — raising only the obvious one measures nothing.

```diff
- "--max-running-requests"      "4"  →  "16"
- "--max-mamba-cache-size"     "20"  →  "80"    # 16 requests × 5 slots
- "--cuda-graph-max-bs-decode"  "4"  →  "16"    # else batches >4 fall back to eager
```

| Streams | cap 4 | cap 16 |
|---:|---:|---:|
| 1 | 58.3 | 52.9 |
| 8 | 190.1 | 315.9 |
| **16** | — | **480.7** |
| 32 | — | 469.8 |

16 streams is a true optimum, not a plateau: TTFT p50 and max are identical
there (0.68 s, zero queueing), while 24 and 32 are *slower* in aggregate with
worst-case TTFT past 10 s.

Leave `--max-total-tokens 1048576` in place: uncapping it grows the KV pool but
nothing uses the space, and the lost headroom cost 18% at 32 streams.

Costs ~5% single-stream and 11.8 GB of GDN state. Details in
[`patches/sparkstation-models.yaml.md`](patches/sparkstation-models.yaml.md).

## 6. Tune draft tokens — the largest single-stream lever

`--speculative-num-draft-tokens` ships at 8. The measured optimum is ~16,
worth **+28%** single-stream.

| draft | single | agg @16 | accept_len | accept_rate |
|---:|---:|---:|---:|---:|
| 6 | 49.5 | 404.2 | 5.25 | 0.85 |
| 8 (default) | 61.3 | 429.2 | 6.63 | 0.80 |
| 10 | 65.2 | **435.1** | 7.18 | 0.69 |
| 12 | 75.1 | 402.8 | 8.27 | 0.66 |
| 16 | **78.6** | 384.8 | 9.52 | 0.57 |
| 20 | 72.0 | 330.7 | 8.48 | 0.40 |
| 24 | 69.0 | 284.5 | 8.40 | 0.32 |

**The optima diverge — pick one:** 16 for interactive/single-stream, **10 for
concurrent serving** (435 vs 385 aggregate). You cannot have both.

Past 16, `accept_len` *falls* even though more tokens are drafted — the drafter
can't sustain longer correct runs, so you pay draft compute for tokens that get
rejected. `accept_rate` collapses from 0.80 to 0.32.

## 7. Benchmark quality

```bash
./scripts/run-humaneval.sh          # thinking off, ~5 min
./scripts/run-humaneval.sh think    # thinking on,  ~20 min
```

164 problems at temperature 0, each executed against its real unit tests in a
`--network none` container. Real pass@1, not self-judged.

| Mode | pass@1 | Tokens/problem | Wall |
|---|---:|---:|---:|
| Thinking off | 93.9% | ~200 | 3 min |
| Thinking on | **97.0%** | ~945 | 19 min |

Default to thinking off for well-specified functions — 93.9% at a fifth of the
tokens. Switch it on for hard cases.

---

## Traps

Each of these cost real time.

- **SparkStation restarts healthy models under load.** `HEALTH_CHECK_TIMEOUT_SECONDS=5`
  x 3 failures: a model saturated on long prefill cannot answer a 5s probe, so the
  supervisor kills it mid-job and clients get 503s. Raise it to 30.
- **mem-fraction is not a perf lever here.** 0.82 / 0.85 / 0.90 all measure the
  same. Concurrency is bound by mamba slots, long context by prefill — never by
  memory capacity.
- **The DFlash2 image doesn't exist upstream.** Build it; see step 2.
- **HF cache symlinks inside the container mount break everything.** The
  container bind-mounts `~/.cache/huggingface`; symlinks *within* it point at
  host-only paths, so it re-downloads and dies with
  `OSError: I/O error: File exists (os error 17)`. Keep real directories there.
- **`host: worker1` assumes a second Spark.** Use `primary`, and check any
  profile you use — a profile-level `host:` wins.
- **`start.sh` uses `WORK_DIR="$(pwd)"`**, not the script dir. Run it from
  inside the repo or your caches land elsewhere.
- **5 mamba slots per request, not 4.** DFlash2's verify needs an extra. SGLang
  clamps `max_running_requests = pool / 5` and logs it. Size the pool at 5×
  target.
- **uv installs `cli.py` into site-packages.** Editing the repo copy does
  nothing — the console script imports from `.venv/bin`. Copy it across too.
- **Gateway reports "not healthy" while working.** `_gateway_healthy()`
  hardcodes `Bearer dummy-key`, so a custom `LITELLM_MASTER_KEY` gets HTTP 400.
  Patch in [`patches/`](patches/).
- **Only the FLUX launcher forwards `HF_TOKEN`.** A gated model can't
  authenticate its own download; pre-pull on the host.
- **Never `docker rm -f` a managed container.** The supervisor's state goes
  stale; `sparkstation stop && start` reconciles.
- **Boot-to-boot variance is ~8% on single-stream**, against <2% run-to-run
  within one server instance. Size A/B deltas against 8%, and re-measure on a
  fresh boot before believing a small win.
- **Greedy is not bitwise deterministic.** Temperature 0 still flips 2–3
  HumanEval problems between runs. Don't read a sub-2% delta as a regression.
- **Thinking is on by default**, and a small `max_tokens` returns empty
  `content` with everything in `reasoning_content`. Always pair a token cap with
  a `finish_reason` check — truncation otherwise reads as a quality regression.
  That mistake made a 97.0% run score 90.9%.
- **Count `completion_tokens`, not stream events.** DFlash2 emits ~3.75 tokens
  per event; event-counting inflates throughput ~4×.

## Layout

```
bench/     common.py · perf.py · longctx.py · humaneval/{generate,execute,report}.py
scripts/   00-verify-gpu · 01-build-and-fetch · run-humaneval
patches/   models.yaml edits · gateway-health fix
results/   RESULTS.md — all measurements
```

Scripts read `GB10_BASE_URL`, `GB10_API_KEY`, `GB10_MODEL` from the environment
(see [`.env.example`](.env.example)); nothing is baked in.

## Caveats

Throughput figures are code generation at temperature 0 on short prompts.
Long-context behaves very differently — a ~120K-token prompt costs ~95 s before
the first token (full prefill curve in
[`results/RESULTS.md`](results/RESULTS.md)). Single machine; treat the ordering
of findings as durable and absolute numbers as indicative.

MIT — see [LICENSE](LICENSE).
