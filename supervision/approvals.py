"""Human-in-the-loop gate designed so approvals are *read*, not clicked through.

Three mechanisms:
  1. Evidence-first requests: the reviewer sees the facts behind the action, not just "Approve?".
  2. Retype-to-confirm: approving requires retyping the number that matters (e.g. quantity).
  3. Canaries: every Nth request is preceded by a deliberately wrong twin. Approving a
     canary means the reviewer is not reading; the real request is then refused and flagged.
Fast approvals are also flagged so the watchdog can spot rubber-stamping over time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .core import Context, Tool
from .trace import Tracer


@dataclass
class ApprovalRequest:
    id: str
    tool: str
    args: dict
    summary: str
    reasons: list
    evidence: dict
    is_canary: bool = False
    confirm_field: Optional[str] = None
    expected_confirm: Any = None


@dataclass
class Decision:
    approved: bool
    reason: str = ""
    confirmed: Any = None
    latency_s: float = 0.0
    flag: str = ""


class Approver(Protocol):
    def decide(self, req: ApprovalRequest) -> Decision: ...


class FnApprover:
    def __init__(self, fn: Callable[[ApprovalRequest], Decision]):
        self.fn = fn

    def decide(self, req):
        return self.fn(req)


def careful_reviewer(latency_s: float = 9.0) -> FnApprover:
    """Test double: rejects canaries, approves real requests after 'reading' them."""
    def fn(req):
        if req.is_canary:
            return Decision(False, "number is far outside what the evidence supports", latency_s=latency_s)
        return Decision(True, "matches evidence", confirmed=req.expected_confirm, latency_s=latency_s)
    return FnApprover(fn)


def rubber_stamp(latency_s: float = 0.4) -> FnApprover:
    """Test double: approves everything instantly, retyping whatever number is asked for."""
    return FnApprover(lambda req: Decision(True, "ok", confirmed=req.expected_confirm, latency_s=latency_s))


def rejecting(reason: str = "not now") -> FnApprover:
    return FnApprover(lambda req: Decision(False, reason, latency_s=6.0))


def wrong_confirmation() -> FnApprover:
    """Test double: says yes but retypes the wrong number."""
    return FnApprover(lambda req: Decision(True, "ok", confirmed="999999", latency_s=6.0))


class CLIApprover:
    def decide(self, req: ApprovalRequest) -> Decision:
        print("\n=== APPROVAL REQUIRED ===")
        print(req.summary)
        for k, v in req.evidence.items():
            print(f"  {k}: {v}")
        for r in req.reasons:
            print(f"  why a human is asked: {r}")
        t0 = time.monotonic()
        ans = input("Approve? [y/N] ").strip().lower()
        confirmed, reason = None, ""
        if ans == "y" and req.confirm_field:
            raw = input(f"Retype {req.confirm_field} to confirm: ").strip()
            try:
                confirmed = type(req.expected_confirm)(raw)
            except (TypeError, ValueError):
                confirmed = raw
        elif ans != "y":
            reason = input("Reason (optional): ").strip()
        return Decision(ans == "y", reason, confirmed, time.monotonic() - t0)


class ApprovalGate:
    def __init__(self, approver: Approver, tracer: Tracer, canary_every: int = 0, min_latency_s: float = 3.0):
        self.approver = approver
        self.tracer = tracer
        self.canary_every = canary_every
        self.min_latency_s = min_latency_s
        self.n = 0

    def _request(self, tool: Tool, args: dict, reasons: list, canary: bool) -> ApprovalRequest:
        self.n_req = getattr(self, "n_req", 0) + 1
        return ApprovalRequest(
            id=f"req-{self.n_req}",
            tool=tool.name,
            args=args,
            summary=tool.summarize(args) if tool.summarize else f"{tool.name}({args})",
            reasons=list(reasons),
            evidence=tool.evidence(args) if tool.evidence else {},
            is_canary=canary,
            confirm_field=tool.confirm_field,
            expected_confirm=args.get(tool.confirm_field) if tool.confirm_field else None,
        )

    def review(self, tool: Tool, args: dict, ctx: Context, call_id: str, reasons: list) -> Decision:
        self.n += 1
        if self.canary_every and tool.canary and self.n % self.canary_every == 0:
            creq = self._request(tool, tool.canary(args), reasons, canary=True)
            cd = self.approver.decide(creq)
            self.tracer.emit("approval", call_id=call_id, tool=tool.name, canary=True,
                             approved=cd.approved, latency_s=cd.latency_s, reason=cd.reason)
            if cd.approved:
                return Decision(False, "attention check failed: reviewer approved a deliberately wrong request; "
                                       "route this to another reviewer", flag="canary_missed")

        req = self._request(tool, args, reasons, canary=False)
        d = self.approver.decide(req)
        flag = d.flag
        if d.approved and req.expected_confirm is not None and str(d.confirmed) != str(req.expected_confirm):
            d = Decision(False, f"confirmation mismatch: expected {req.confirm_field}={req.expected_confirm}, "
                                f"reviewer typed {d.confirmed}", d.confirmed, d.latency_s)
            flag = "confirmation_mismatch"
        elif d.approved and d.latency_s < self.min_latency_s:
            flag = "fast_approval"
        self.tracer.emit("approval", call_id=call_id, tool=tool.name, canary=False,
                         approved=d.approved, latency_s=d.latency_s, reason=d.reason, flag=flag)
        return Decision(d.approved, d.reason, d.confirmed, d.latency_s, flag)
