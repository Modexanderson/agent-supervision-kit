"""Core types: errors, risk tiers, tool definition, run context."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional


class TransientError(Exception):
    """Failed BEFORE any side effect happened. Always safe to retry."""


class UnknownOutcome(Exception):
    """Timeout / dropped connection: the side effect may or may not have happened."""


class PermanentError(Exception):
    """Retrying will not help."""


class Risk(IntEnum):
    READ = 0          # no side effects
    WRITE = 1         # bounded side effect; autonomy is tiered by value / taint / rules
    IRREVERSIBLE = 2  # always needs a human


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict, str], Any]                     # fn(args, idempotency_key)
    risk: Risk = Risk.READ
    untrusted_output: bool = False                     # output is third-party text -> quarantine + taint
    precheck: Optional[Callable[[dict, "Context"], list]] = None       # deterministic guard, returns problems
    verify: Optional[Callable[[dict, Any, str], tuple]] = None          # (args, output, key) -> (ok, detail)
    reconcile: Optional[Callable[[dict, str], Any]] = None              # (args, key) -> prior result | None
    idempotent: bool = False                           # backend dedupes on the key itself
    value: Optional[Callable[[dict], float]] = None    # money / exposure of the action
    summarize: Optional[Callable[[dict], str]] = None  # human-readable one-liner for reviewers
    evidence: Optional[Callable[[dict], dict]] = None  # facts a reviewer needs to judge the action
    confirm_field: Optional[str] = None                # reviewer must retype this field to approve
    canary: Optional[Callable[[dict], dict]] = None    # builds a deliberately-wrong twin (attention check)

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class Escalation:
    reason: str
    detail: str
    call_id: str = ""


@dataclass
class Context:
    run_id: str
    sources: dict = field(default_factory=dict)   # raw untrusted documents, for grounding checks
    tainted: bool = False                         # untrusted content has entered the context
    spend: float = 0.0
    abort: bool = False
    invalid: int = 0
    seen: dict = field(default_factory=dict)
    done: dict = field(default_factory=dict)
    precheck_failures: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)
    escalations: list = field(default_factory=list)
    denied: list = field(default_factory=list)
    rejected: list = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    status: str            # "completed" | "escalated"
    final_text: str
    steps: int
    actions: list
    escalations: list
    denied: list
    rejected: list
