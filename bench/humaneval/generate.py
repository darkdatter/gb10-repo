#!/usr/bin/env python3
"""Generate HumanEval candidates.

    python3 bench/humaneval/generate.py            # thinking off
    python3 bench/humaneval/generate.py think      # thinking on

Writes candidates to results/gen_{nothink,think}.json. Download the dataset
first:

    hf download openai/openai_humaneval --repo-type dataset --local-dir data/humaneval

Three details matter, all learned the hard way (see README "Traps"):

1. max_tokens must be generous. Thinking mode routinely needs >4k and
   sometimes never terminates; a small cap silently truncates and reads as a
   quality regression rather than as truncation.
2. The fence regex must tolerate an UNCLOSED fence, because a truncated
   generation has no closing ```.
3. The original problem prompt is stored alongside the candidate. Some
   problems (HumanEval/38, /50) define a helper function ABOVE the target, and
   the tests need it - execute.py prepends the prompt for this reason.
"""
import glob
import json
import os
import queue
import re
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import chat  # noqa: E402

THINK = len(sys.argv) > 1 and sys.argv[1] == "think"
WORKERS = int(os.environ.get("GB10_WORKERS", "4"))
MAX_TOKENS = int(os.environ.get("GB10_MAX_TOKENS", "16384" if THINK else "2048"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.environ.get("GB10_HUMANEVAL_DIR", os.path.join(ROOT, "data", "humaneval"))
OUT = os.path.join(ROOT, "results", f"gen_{'think' if THINK else 'nothink'}.json")

INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including its signature and any needed imports) inside a single ```python "
    "code fence. No explanation.\n\n"
)


def extract(text):
    """Pull code out of a fence, tolerating a missing closing fence."""
    closed = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if closed:
        return max(closed, key=len)
    unclosed = re.search(r"```(?:python)?\s*\n(.*)$", text, re.S)
    if unclosed:
        return unclosed.group(1)
    return text


def main():
    import pandas as pd  # imported late so --help style runs don't need it

    files = glob.glob(os.path.join(DATA, "**", "*.parquet"), recursive=True)
    if not files:
        sys.exit(f"no parquet under {DATA} - see docstring for the hf download command")
    rows = pd.read_parquet(files[0]).to_dict("records")

    results = [None] * len(rows)
    work = queue.Queue()
    for i in range(len(rows)):
        work.put(i)
    lock = threading.Lock()
    done = [0]
    tokens = [0]
    truncated = [0]

    def worker():
        while True:
            try:
                i = work.get_nowait()
            except queue.Empty:
                return
            row = rows[i]
            try:
                r = chat(INSTRUCTION + row["prompt"], MAX_TOKENS, thinking=THINK)
                results[i] = {
                    "task_id": row["task_id"],
                    "code": extract(r["content"]),
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "finish_reason": r["finish_reason"],
                    "completion_tokens": r["completion_tokens"],
                }
                with lock:
                    tokens[0] += r["completion_tokens"]
                    if r["finish_reason"] == "length":
                        truncated[0] += 1
            except Exception as exc:  # keep going; a dead task scores as a fail
                results[i] = {
                    "task_id": row["task_id"], "code": "", "prompt": row["prompt"],
                    "test": row["test"], "entry_point": row["entry_point"],
                    "error": str(exc)[:200],
                }
            with lock:
                done[0] += 1
                if done[0] % 40 == 0:
                    print(f"  {done[0]}/{len(rows)}", flush=True)
            work.task_done()

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh)

    print(f"\n{len(rows)} problems in {elapsed:.0f}s "
          f"({tokens[0]} tokens, {tokens[0] / elapsed:.1f} tok/s aggregate at {WORKERS} streams)")
    if truncated[0]:
        print(f"WARNING: {truncated[0]} hit the {MAX_TOKENS}-token cap and are truncated. "
              f"Those are NOT quality failures - re-run them with a larger budget.")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
