"""Execution layer: retries, unknown-state reconciliation, circuit breaker.

The key distinction:
  TransientError  -> definitely NOT executed  -> retry
  UnknownOutcome  -> MAYBE executed           -> never blindly retry a write.
                     reconcile against the system of record, or use an
                     idempotency key the backend honours, or stop and escalate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .core import PermanentError, Risk, Tool, TransientError, UnknownOutcome


@dataclass
class ExecResult:
    status: str                # ok | failed | unknown | circuit_open
    output: Any = None
    attempts: int = 0
    reconciled: bool = False
    error: str = ""


class CircuitBreaker:
    """Fail fast after `threshold` consecutive failures of a tool, for `cooldown` seconds."""

    def __init__(self, threshold: int = 5, cooldown: float = 30.0, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures: dict = {}
        self.opened_at: dict = {}

    def allow(self, name: str) -> bool:
        t = self.opened_at.get(name)
        if t is None:
            return True
        if self.clock() - t >= self.cooldown:   # half-open: let one attempt through
            self.opened_at.pop(name)
            self.failures[name] = self.threshold - 1
            return True
        return False

    def record(self, name: str, ok: bool) -> None:
        if ok:
            self.failures[name] = 0
            return
        self.failures[name] = self.failures.get(name, 0) + 1
        if self.failures[name] >= self.threshold:
            self.opened_at[name] = self.clock()


class Executor:
    def __init__(self, max_attempts: int = 4, base_delay: float = 0.2, sleep=time.sleep, breaker: CircuitBreaker | None = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.sleep = sleep
        self.breaker = breaker or CircuitBreaker()

    def _backoff(self, attempt: int) -> None:
        self.sleep(self.base_delay * 2 ** (attempt - 1))

    def run(self, tool: Tool, args: dict, key: str) -> ExecResult:
        if not self.breaker.allow(tool.name):
            return ExecResult("circuit_open", error=f"circuit open for {tool.name}")
        attempts, last = 0, ""
        while attempts < self.max_attempts:
            attempts += 1
            try:
                out = tool.fn(args, key)
                self.breaker.record(tool.name, True)
                return ExecResult("ok", out, attempts)
            except TransientError as e:
                last = f"transient: {e}"
                self.breaker.record(tool.name, False)
            except UnknownOutcome as e:
                last = f"unknown outcome: {e}"
                self.breaker.record(tool.name, False)
                if tool.risk == Risk.READ or tool.idempotent:
                    pass                                # safe to retry (same key for idempotent writes)
                elif tool.reconcile:
                    try:
                        prior = tool.reconcile(args, key)
                    except Exception as re_err:         # cannot even check -> state is unknown
                        return ExecResult("unknown", attempts=attempts, error=f"{last}; reconcile failed: {re_err}")
                    if prior is not None:               # it DID execute: adopt the result, do not repeat it
                        return ExecResult("ok", prior, attempts, reconciled=True)
                    # confirmed it did NOT execute -> a retry is safe
                else:
                    return ExecResult("unknown", attempts=attempts, error=f"{last}; no way to reconcile, not retrying")
            except PermanentError as e:
                self.breaker.record(tool.name, False)
                return ExecResult("failed", attempts=attempts, error=f"permanent: {e}")
            if attempts < self.max_attempts:
                self._backoff(attempts)
        return ExecResult("failed", attempts=attempts, error=f"exhausted {attempts} attempts; last: {last}")
