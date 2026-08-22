# SparkStation `models.yaml` edits

Two changes to the `qwen3.8-sglang` entry. Back up first:

```bash
cp models.yaml models.yaml.bak
```

## 1. Required on a single Spark

The entry ships pinned to a second machine. With one Spark, the supervisor will
try to reach a `worker1` host that does not exist.

```diff
   qwen3.8-sglang:
     name: "RadixArk/Qwen3.8-27B-NVFP4"
     backend: "sglang"
     model_type: "chat"
-    host: worker1
+    host: primary
```

The default `generic` profile enables this alias with `{}` (base spec, no
override), so nothing pins it back to `worker1`. Check any profile you actually
use — a profile-level `host:` would win.

## 2. Optional: unlock concurrency

The shipped config caps you at 4 concurrent requests. Three flags are
**co-limiting** — raising only `--max-running-requests` measures nothing,
because the GDN state pool and the decode CUDA-graph batch set both bind first.

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

So: **pool = 5 × target concurrency.** 80 slots → 16 concurrent.

Read that log line after any change. The clamp is silent apart from it, and a
pool sized on the assumption of 4 slots per request lands you at 12 instead
of 16.

### What it costs

| | pool 20 | pool 80 |
|---|---:|---:|
| GDN state allocation | 3.8 GB | **15.6 GB** |
| Free GPU memory after load | 38.1 GB | 20.6 GB |
| Peak aggregate throughput | 190 tok/s | **480.7 tok/s** |
| Single-stream decode | 63.1 tok/s | 60.0 tok/s |

Single-stream costs ~5% (larger decode CUDA-graph set). Worth it for
multi-agent work; revert if this is a single-user interactive box.

## Memory ceiling — read before raising `mem_fraction`

`memory_gb: 98` resolves to `mem-fraction-static 0.82` via a launcher clamp.
Both this project's notes and the MiaAI-Lab toolkit record machines
**unrecoverably wedged during weight load** at higher fractions — GB10 unified
memory starves the OS. The toolkit defaults NVFP4 to `0.90`; the generic
`start.sh` uses `0.95`.

Use `0.85` on a first boot, especially if you cannot physically reach the box
to power-cycle it.
