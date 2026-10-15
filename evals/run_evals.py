"""Run every failure-injection scenario and print a table. Exit code 1 if any fail.

    python -m evals.run_evals
"""
from __future__ import annotations

import sys

from .scenarios import SCENARIOS


def main() -> int:
    width = max(len(n) for n, _, _ in SCENARIOS)
    failed = 0
    for name, desc, fn in SCENARIOS:
        try:
            problems = fn()
        except Exception as e:  # a crashing scenario is a failing scenario
            problems = [f"crashed: {type(e).__name__}: {e}"]
        mark = "PASS" if not problems else "FAIL"
        print(f"{mark}  {name:<{width}}  {desc}")
        for p in problems:
            print(f"        - {p}")
        failed += bool(problems)
    print(f"\n{len(SCENARIOS) - failed}/{len(SCENARIOS)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
