#!/usr/bin/env python3
"""Summarise an execution result, separating real errors from truncation.

    python3 bench/humaneval/report.py results/exec_nothink.json [results/gen_nothink.json]
    python3 bench/humaneval/report.py results/exec_think.json results/gen_think.json \
        --compare results/exec_nothink.json

The distinction this script exists to draw: a candidate whose finish_reason is
"length" never emitted code. That is a BUDGET failure, not a quality failure.
Reporting the two together is how a 97% run reads as 90.9%.
"""
import json
import sys


def load(path):
    return json.load(open(path))


def classify(result, gen_entry):
    if gen_entry and gen_entry.get("finish_reason") == "length":
        return "TRUNCATED"
    err = result["err"]
    if not gen_entry or len(gen_entry.get("code", "").strip()) < 10:
        return "EMPTY"
    if "SyntaxError" in err or "IndentationError" in err:
        return "SYNTAX"
    if err == "TIMEOUT":
        return "TIMEOUT"
    if "AssertionError" in err:
        return "WRONG-ANSWER"
    return "RUNTIME-ERR"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    compare = None
    if "--compare" in sys.argv:
        compare = sys.argv[sys.argv.index("--compare") + 1]

    data = load(args[0])
    gen = {}
    if len(args) > 1:
        gen = {g["task_id"]: g for g in load(args[1])}

    print(f"pass@1 {data['pass@1']}%  ({data['passed']}/{data['total']})\n")

    fails = [r for r in data["results"] if not r["pass"]]
    counts = {}
    for r in fails:
        kind = classify(r, gen.get(r["task_id"]))
        counts[kind] = counts.get(kind, 0) + 1
        entry = gen.get(r["task_id"], {})
        print(f"  {r['task_id']:<16} {kind:<13} tokens={entry.get('completion_tokens')}")

    print(f"\nbreakdown: {counts}")

    truncated = counts.get("TRUNCATED", 0)
    if truncated:
        adjusted = data["total"] - truncated
        print(f"\n{truncated} failure(s) are truncation, not wrong answers.")
        print(f"pass@1 excluding truncated: "
              f"{100.0 * data['passed'] / adjusted:.1f}% ({data['passed']}/{adjusted})")
        print("Re-run those with a larger --max-tokens before quoting a quality number.")

    if compare:
        other = load(compare)
        mine = {r["task_id"] for r in data["results"] if not r["pass"]}
        theirs = {r["task_id"] for r in other["results"] if not r["pass"]}
        key = lambda t: int(t.split("/")[1])  # noqa: E731
        print(f"\nvs {compare}:")
        print(f"  failed in both : {sorted(mine & theirs, key=key)}")
        print(f"  fixed here     : {sorted(theirs - mine, key=key)}")
        print(f"  broken here    : {sorted(mine - theirs, key=key)}")


if __name__ == "__main__":
    main()
