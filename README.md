# Qwen3.8-27B on DGX Spark (GB10)

A measured, reproducible recipe for the fastest configuration I could get on a
DGX Spark: **SGLang + NVFP4 + DFlash2 speculative decoding**, managed by
[SparkStation](https://github.com/kshetrajna12/sparkstation).

| | |
|---|---:|
| Single-stream decode | **60.0 tok/s** |
| Peak aggregate (16 streams) | **480.7 tok/s** |
| Time to first token | **190 ms** |
| HumanEval pass@1 (thinking on) | **97.0%** |

Every number was measured on one box and is reproducible with the scripts in
[`bench/`](bench/). Full data in [`results/RESULTS.md`](results/RESULTS.md).

> **The headline finding:** NVFP4 beats FP8, and SGLang beats vLLM, on this
> hardware. The widely-shared "Qwen3.8-27B FP8 on vLLM at ~32 tok/s" recipe is
> roughly half the speed of this one. NVIDIA's own benchmarks put NVFP4 29–34%
> ahead of FP8 on vLLM; moving to SGLang with DFlash2 roughly doubles it again.

---

## Ingredients

| Component | Exact pin |
|---|---|
| Target model | `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9` |
| Draft model | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c` |
| Serving image | `lmsysorg/sglang:qwen38-27b-dflash2` — **built locally, see step 2** |
| Launcher toolkit | [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) |
| Control plane | [kshetrajna12/sparkstation](https://github.com/kshetrajna12/sparkstation) |
| Disk | ~110 GB (39 GB image + 25 GB weights + headroom) |

All four model repos are public. No token is required to pull them.

**Verified environment.** Ubuntu 24.04, kernel 6.17-nvidia, aarch64. Driver
580.173.02 / CUDA 13.0. Docker 29.2.1, nvidia-container-toolkit 1.20 via CDI.
128 GB unified memory.

---

## 1. Prove GPU passthrough first

```bash
./scripts/00-verify-gpu.sh
```

DGX OS ships CDI rather than a registered Docker runtime, so `docker info`
listing only `runc` is **normal** and `--gpus all` still works. The script falls
back to `--device nvidia.com/gpu=all` and tells you how to register the runtime
if both fail.

## 2. Build the image and fetch the weights

```bash
./scripts/01-build-and-fetch.sh
```

> **This is the step most people trip on.**
> `lmsysorg/sglang:qwen38-27b-dflash2` **is not on Docker Hub.** SparkStation's
> `models.yaml` references it, but no released SGLang tag ships DFlash2 — you
> build it. The base `:qwen38-27b` tag does exist.

The build applies an NVFP4 `lm_head` patch that matters: the earlier
dequantize-everything approach allocated ~2.5 GB at draft-graph capture and
**hard-rebooted the machine**.

The two halves run in parallel — a ~39 GB image pull alongside ~25 GB of
weights.

## 3. Run standalone and get a baseline

Get a number before adding a control plane. If something breaks later you'll
know whether it was the model or the orchestration.

```bash
cd ~/spark/Qwen3.8-27B-SGLang-DGX-Spark   # REQUIRED: start.sh uses WORK_DIR="$(pwd)"
cp -n .env.sample .env
DF_EXTRA="--mem-fraction-static 0.85" ./start-dflash.sh
```

Serves on **:8888** as `qwen3.8-27b-sglang`, OpenAI- and Anthropic-compatible.
First boot ≈ 3 minutes.

> **Memory ceiling.** The toolkit defaults NVFP4 to `0.90`; the generic
> `start.sh` uses `0.95`. Both this project and the toolkit record machines
> *unrecoverably wedged during weight load* at high fractions — GB10 unified
> memory starves the OS. Use **0.85** on a first boot, especially if you can't
> physically reach the box to power-cycle it.

Then benchmark:

```bash
export GB10_BASE_URL=http://127.0.0.1:8888/v1
export GB10_MODEL=qwen3.8-27b-sglang
python3 bench/perf.py
```

## 4. Put SparkStation in front

Gives you one OpenAI endpoint, lifecycle management, auto-suspend and thermal
policy.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --depth 1 https://github.com/kshetrajna12/sparkstation.git
cd sparkstation && uv sync
cp .env.example .env      # TOTAL_UNIFIED_MEMORY_GB=128, MEMORY_HARD_LIMIT_GB=110
```

One **required** edit to `models.yaml` — the `qwen3.8-sglang` entry ships pinned
to a second machine:

```diff
-    host: worker1
+    host: primary
```

```bash
uv run sparkstation start -d --profile generic
uv run sparkstation status
```

Gateway on **:8000**, supervisor on :9001. The model container is
bridge-networked, so its internal :8000 maps to host **:8001** — no collision
with the gateway despite both reporting 8000. Gateway overhead measured at
**zero** (62.3 vs 62.5 tok/s direct).

See [`patches/sparkstation-models.yaml.md`](patches/sparkstation-models.yaml.md)
for both edits, and
[`patches/sparkstation-gateway-health.patch`](patches/sparkstation-gateway-health.patch)
for the status false-negative.

## 5. Unlock concurrency — the biggest single win

The shipped config caps you at 4 concurrent requests and leaves most of the box
idle. **Three flags are co-limiting**; raising only the obvious one measures
nothing.

```diff
- "--max-running-requests"      "4"  →  "16"
- "--max-mamba-cache-size"     "20"  →  "80"    # 16 requests × 5 slots
- "--cuda-graph-max-bs-decode"  "4"  →  "16"    # else batches >4 fall back to eager
```

| Streams | cap 4 | cap 12 | **cap 16** |
|---:|---:|---:|---:|
| 1 | 58.3 | 53.4 | 52.9 |
| 4 | 187.1 | 191.5 | 193.1 |
| 8 | 190.1 | 296.0 | 315.9 |
| **16** | — | 307.8 | **480.7** |
| 24 | — | 319.6 | 416.5 |
| 32 | — | 368.9 | 469.8 |

**16 streams is a true optimum, not a plateau.** TTFT p50 and max are identical
there (0.68 s — zero queueing, because streams equal the cap), while 24 and 32
are both *slower* in aggregate and push worst-case TTFT past 10 s.

Costs **~5% single-stream** (63.1 → 60.0 tok/s) from the larger decode
CUDA-graph set, and **11.8 GB** of extra GDN state. Worth it for multi-agent
work; revert if this is a single-user interactive box.

## 6. Benchmark quality

```bash
./scripts/run-humaneval.sh          # thinking off, ~5 min
./scripts/run-humaneval.sh think    # thinking on,  ~20 min
```

Generates 164 candidates at temperature 0, then executes each against its real
unit tests inside a `--network none` container. Real pass@1, not self-judged.

| Mode | pass@1 | Tokens/problem | Wall |
|---|---:|---:|---:|
| Thinking off | 93.9% | ~200 | 3 min |
| Thinking on | **97.0%** | ~945 | 19 min |

Practical default: **thinking off** for well-specified functions — 93.9% at a
fifth of the tokens and 6× the speed. Switch it on for hard cases.

---

## Traps

Each of these cost real time.

**The DFlash2 image doesn't exist upstream.** SparkStation's `models.yaml` names
a tag no released SGLang ships. Build it with `patch/build-dflash2-image.sh`.

**HF cache symlinks inside the container mount break everything.** The container
bind-mounts `~/.cache/huggingface`. Symlinks *within* it point at host paths that
don't exist inside the container, so it decides nothing is cached, re-downloads,
and dies with `OSError: I/O error: File exists (os error 17)`. Keep real
directories there; symlink the *other* direction if you must — Docker resolves
`-v` host-side.

**`host: worker1` assumes a second Spark.** Set `host: primary`. Check any
profile you use — a profile-level `host:` would win over the base spec.

**`start.sh` uses `WORK_DIR="$(pwd)"`**, not the script directory. Run it from
inside the repo or your HF and Triton caches land wherever you were standing.

**5 mamba slots per request, not 4.** DFlash2's verify step needs an extra.
SGLang clamps `max_running_requests = pool / 5` and says so in the log. Size the
pool at 5× your target, and read that line after any change.

**uv installs `cli.py` into site-packages.** Editing `cli.py` in the SparkStation
repo does *nothing* — the console script resolves `import cli` from `.venv/bin`,
not your cwd. Copy it to `.venv/lib/python3.12/site-packages/cli.py` as well.
Related: in the installed copy `__file__` is site-packages, so use
`PROJECT_ROOT` to locate `.env` / `models.yaml`.

**Gateway reports "not healthy" while working fine.** `_gateway_healthy()`
hardcodes `Bearer dummy-key`, so any custom `LITELLM_MASTER_KEY` gets HTTP 400.
CLI-only, but it costs a spurious warning and a 30 s stall on every start. Patch
included.

**Only the FLUX launcher forwards `HF_TOKEN`.** `sglang_launcher.py` and
`vllm_launcher.py` don't pass it into the container, so a *gated* model can't
authenticate its own download. Pre-pull on the host with `hf download`.

**Never `docker rm -f` a managed container.** The supervisor's state goes stale
and both stop (HTTP 500) and start (HTTP 409) then fail.
`sparkstation stop && sparkstation start` reconciles it.

**Greedy decoding is not bitwise deterministic.** At temperature 0, repeat runs
flip 2–3 HumanEval problems, because dynamic batching plus speculative decoding
changes reduction order. Don't read a sub-2% delta as a regression.

**Thinking is on by default.** A small `max_tokens` returns empty `content` with
everything in `reasoning_content`. Disable per request with
`chat_template_kwargs: {"enable_thinking": false}` — and always pair a token cap
with a `finish_reason` check, or silent truncation reads as a quality
regression. That mistake made a 97.0% run score 90.9%.

**Count `completion_tokens`, not stream events.** DFlash2 emits ~3.75 tokens per
SSE event, so event-counting inflates throughput roughly 4×.

---

## Repo layout

```
bench/
  common.py                    shared client; endpoint + key from env
  perf.py                      TTFT, single-stream, concurrency, prefill
  humaneval/
    generate.py                candidates at temperature 0
    execute.py                 runs them against real tests (sandbox)
    report.py                  separates wrong answers from truncation
scripts/
  00-verify-gpu.sh             GPU passthrough preflight
  01-build-and-fetch.sh        image build + weight download
  run-humaneval.sh             generate → execute → report
patches/
  sparkstation-models.yaml.md  host fix + concurrency tuning
  sparkstation-gateway-health.patch
results/RESULTS.md             all measurements
docs/recipe.html               single-page version of this guide
```

## Configuration

The benchmark scripts read the endpoint and key from the environment — nothing
is baked in:

```bash
export GB10_BASE_URL=http://127.0.0.1:8000/v1   # SparkStation gateway
export GB10_API_KEY=your-litellm-master-key
export GB10_MODEL=default
```

## Sanity check

```bash
curl -s "$GB10_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $GB10_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default",
       "messages":[{"role":"user","content":"Say exactly: it works"}],
       "max_tokens":200,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

## Caveats

Throughput figures are code generation at temperature 0 on short prompts.
Prefill-heavy and long-context workloads look very different — the full prefill
curve is in [`results/RESULTS.md`](results/RESULTS.md), and a ~120K-token prompt
costs ~95 seconds before the first token. Measured on a single machine; treat
the ordering of the findings as durable and the absolute numbers as indicative.

## License

MIT — see [LICENSE](LICENSE).
