#!/usr/bin/env python3
"""Execute HumanEval candidates against their real unit tests.

RUN THIS INSIDE A SANDBOX. It executes model-generated code. The wrapper in
scripts/run-humaneval.sh runs it in a `--network none` container:

    docker run --rm --network none -v "$PWD:/w:ro" -w /tmp python:3.12-slim \
      python /w/bench/humaneval/execute.py /w/results/gen_nothink.json \
      > results/exec_nothink.json

Each candidate runs in its own subprocess with a 15s timeout. The original
problem prompt is prepended so helper functions defined above the target
(HumanEval/38, /50) exist; the model's version is second, so it wins.
"""
import json
import os
import subprocess
import sys

TIMEOUT = int(os.environ.get("GB10_EXEC_TIMEOUT", "15"))
WORK = "/tmp/humaneval_run"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: execute.py <candidates.json>")
    candidates = json.load(open(sys.argv[1]))
    os.makedirs(WORK, exist_ok=True)

    passed = 0
    results = []
    for c in candidates:
        program = (
            c.get("prompt", "") + "\n\n"
            + c["code"] + "\n\n"
            + c["test"] + f"\n\ncheck({c['entry_point']})\n"
        )
        path = os.path.join(WORK, c["task_id"].replace("/", "_") + ".py")
        with open(path, "w") as fh:
            fh.write(program)
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True, timeout=TIMEOUT)
            ok = proc.returncode == 0
            err = "" if ok else proc.stderr.decode()[-200:]
        except subprocess.TimeoutExpired:
            ok, err = False, "TIMEOUT"
        except Exception as exc:
            ok, err = False, str(exc)[:200]
        passed += ok
        results.append({
            "task_id": c["task_id"],
            "pass": ok,
            "err": err,
            "finish_reason": c.get("finish_reason"),
        })

    print(json.dumps({
        "total": len(candidates),
        "passed": passed,
        "pass@1": round(100.0 * passed / len(candidates), 1),
        "results": results,
    }))


if __name__ == "__main__":
    main()
