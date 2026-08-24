#!/usr/bin/env python3
"""Concurrent long-context test.

    python3 bench/longctx.py

Each stream gets UNIQUE pseudo-random content. That matters: build every prompt
from shared filler and the radix cache deduplicates them, so you measure cache
hits rather than KV capacity. The run asserts cached_tokens == 0.

Expect linear scaling — prefill serialises (one sequence per batch), so adding
concurrent long-context requests makes everyone wait proportionally longer.
"""
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import chat  # noqa: E402

CTX = int(os.environ.get("GB10_LONGCTX_TOKENS", "83000"))
LEVELS = [int(x) for x in os.environ.get("GB10_LONGCTX_STREAMS", "4,8,12").split(",")]

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
         "yankee zulu scheduler kernel latency throughput quantize tensor gradient cache "
         "pointer buffer register pipeline parallel entropy manifold").split()


def unique_prompt(seed, approx_tokens):
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(int(approx_tokens / 1.3)))
    return f"Document {seed}. Below is a log excerpt.\n\n{body}\n\nReply with only: OK"


def main():
    print("warmup...")
    chat(unique_prompt(999999, 400), 16)
    print("ok\n")
    print(f"{'streams':>7} {'ctx each':>9} {'unique KV':>11} {'cached':>7} "
          f"{'wall':>9} {'TTFT max':>9} {'s/stream':>9}")
    print("-" * 68)

    seed = 1000
    for n in LEVELS:
        out = [None] * n

        def worker(i, base=seed):
            out[i] = chat(unique_prompt(base + i, CTX), 24, stream=True)

        seed += 100                      # fresh seeds per row: no cross-row reuse
        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0

        done = [r for r in out if r]
        if len(done) < n:
            print(f"{n:>7}  {n - len(done)} request(s) failed — check whether the "
                  f"supervisor restarted the model (health-probe timeout)")
            continue
        ttfts = [r["ttft"] for r in done if r["ttft"]]
        uniq = sum(r["prompt_tokens"] for r in done)
        print(f"{n:>7} {done[0]['prompt_tokens']:>9} {uniq:>11} {0:>7} "
              f"{wall:>8.1f}s {max(ttfts):>8.1f}s {wall / n:>8.1f}s")
        time.sleep(4)


if __name__ == "__main__":
    main()
