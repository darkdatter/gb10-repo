#!/usr/bin/env python3
"""Full performance suite: TTFT, single-stream decode, concurrency, prefill.

    python3 bench/perf.py

Run it against an otherwise idle box - any competing GPU work skews every
number here.
"""
import statistics
import threading
import time

from common import CODE_PROMPT, chat


def ttft_probe(n=5):
    print("## 1. Time to first token (streaming, short prompt)")
    vals = [chat("Say hello.", 16, stream=True)["ttft"] for _ in range(n)]
    vals = [v for v in vals if v]
    print(f"   median {statistics.median(vals) * 1000:.0f} ms   "
          f"min {min(vals) * 1000:.0f}   max {max(vals) * 1000:.0f}\n")


def single_stream(n=5):
    print("## 2. Single-stream decode (code, thinking off)")
    rates = []
    for _ in range(n):
        r = chat(CODE_PROMPT, 700)
        rates.append(r["completion_tokens"] / r["e2e"])
    print("   runs: " + ", ".join(f"{x:.1f}" for x in rates))
    print(f"   MEDIAN {statistics.median(rates):.1f} tok/s\n")


def concurrency(levels=(1, 2, 4, 8, 16, 24, 32), tokens=300):
    print("## 3. Concurrency sweep")
    print(f"   {'streams':>7} {'wall(s)':>8} {'aggregate':>12} {'per-stream':>12} "
          f"{'TTFT p50':>9} {'TTFT max':>9}")
    print("   " + "-" * 64)
    peak = (0, 0.0)
    for n in levels:
        out = [None] * n
        threads = []

        def worker(i):
            out[i] = chat(f"{CODE_PROMPT} Variant {i}.", tokens, stream=True)

        t0 = time.time()
        for i in range(n):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        wall = time.time() - t0

        total = sum(r["completion_tokens"] for r in out)
        per = statistics.median([r["completion_tokens"] / r["e2e"] for r in out])
        ttfts = [r["ttft"] for r in out if r["ttft"]]
        agg = total / wall
        if agg > peak[1]:
            peak = (n, agg)
        print(f"   {n:>7} {wall:>8.1f} {agg:>9.1f} t/s {per:>9.1f} t/s "
              f"{statistics.median(ttfts):>8.2f}s {max(ttfts):>8.2f}s")
        time.sleep(3)
    print(f"\n   peak aggregate {peak[1]:.1f} tok/s at {peak[0]} streams\n")


def prefill(targets=(2000, 8000, 32000, 100000)):
    print("## 4. Long-context prefill")
    filler = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
              "Meanwhile the engineer refactored the scheduler to reduce tail latency. ")
    for target in targets:
        body = (filler * max(1, target // 18))[:target * 6]
        prompt = ("Below is a log. Reply with only the word OK.\n\n" + body
                  + "\n\nReply with only: OK")
        r = chat(prompt, 8, stream=True)
        rate = r["prompt_tokens"] / r["ttft"] if r["ttft"] else 0
        print(f"   prompt {r['prompt_tokens']:>7} tok -> TTFT {r['ttft']:>6.2f}s   "
              f"prefill ~{rate:>7.0f} tok/s")
    print()


if __name__ == "__main__":
    print("warmup...")
    chat("hi", 8)
    print("ok\n")
    ttft_probe()
    single_stream()
    concurrency()
    prefill()
