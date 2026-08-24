# SparkStation `models.yaml` edits

Two changes to the `qwen3.8-sglang` entry. Back up first: `cp models.yaml models.yaml.bak`

## 1. Required on a single Spark

The entry ships pinned to a second machine.

```diff
   qwen3.8-sglang:
     backend: "sglang"
-    host: worker1
+    host: primary
```

The default `generic` profile enables this alias with `{}` (base spec), so
nothing pins it back. Check any profile you actually use — a profile-level
`host:` would win.

## 2. Optional: unlock concurrency

The shipped config caps you at 4 concurrent requests. Three flags are
**co-limiting** — raising only `--max-running-requests` measures nothing,
because the GDN state pool and the decode CUDA-graph batch set bind first.

Under `extra_args` → `sglang_flags`:

```diff
         - "--max-mamba-cache-size"
-        - "20"
+        - "80"
         - "--max-running-requests"
-        - "4"
+        - "16"
         - "--cuda-graph-max-bs-decode"
-        - "4"
+        - "16"
```

### Sizing the pool

Qwen3.8 is a hybrid (Gated DeltaNet) model, so concurrency is bought with
**mamba state, not KV cache**. Each request needs **5** GDN state slots — four,
plus one for DFlash2's verify step. SGLang clamps
`max_running_requests = max_mamba_cache_size / 5` and says so in the log:

```
max_running_requests is capped to 12 by the mamba state cache
(max_mamba_cache_size=64, 5 state slots per request).
```

**Pool = 5 × target concurrency.** 80 slots → 16 concurrent. Read that log line
after any change: the clamp is silent apart from it, and sizing on an assumed 4
slots per request lands you at 12 instead of 16.

### What it costs

| | pool 20 | pool 80 |
|---|---:|---:|
| GDN state allocation | 3.8 GB | **15.6 GB** |
| Free GPU memory after load | 38.1 GB | 20.6 GB |
| Peak aggregate throughput | 190 tok/s | **480.7 tok/s** |
| Single-stream decode | 63.1 tok/s | 60.0 tok/s |

Worth it for multi-agent work; revert if this is a single-user interactive box.

## Memory ceiling — before raising `mem_fraction`

`memory_gb: 98` resolves to `mem-fraction-static 0.82` via a launcher clamp.
Both this project's notes and the MiaAI-Lab toolkit record machines
**unrecoverably wedged during weight load** at higher fractions — GB10 unified
memory starves the OS. The toolkit defaults NVFP4 to `0.90`; generic `start.sh`
uses `0.95`.

Use `0.85` on a first boot, especially if you can't physically reach the box to
power-cycle it.

## Health check

`.env` ships `HEALTH_CHECK_TIMEOUT_SECONDS=5` with `HEALTH_CHECK_MAX_FAILURES=3`.
A model saturated on long prefill cannot answer a 5-second probe, so the
supervisor marks it failed and restarts it mid-job — in-flight requests return
503. Observed with 16 concurrent 83K-token requests; the container showed
`OOMKilled=false`, `exit=0`, so it was not a crash.

```diff
-HEALTH_CHECK_TIMEOUT_SECONDS=5
+HEALTH_CHECK_TIMEOUT_SECONDS=30
```

Same 3-failure threshold still catches a genuinely dead server.

## On mem-fraction

`mem_fraction_static: 0.85` is a headroom choice, not a performance one —
0.82 / 0.85 / 0.90 all measure the same. Keep `--max-total-tokens 1048576`:
uncapping the pool grows it but nothing uses the space, and the lost headroom
cost 18% at 32 streams.
