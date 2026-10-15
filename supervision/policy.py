"""Policy layer: sits between the model and the tools. The model proposes; policy decides.

Deny   = hard limits the model cannot talk its way past (permissions, caps, budget).
Review = a human must approve (irreversible, high value, tainted context, custom rules).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .core import Context, Risk, Tool


@dataclass
class PolicyDecision:
    allow: bool = True
    needs_approval: bool = False
    reasons: list = field(default_factory=list)


Rule = Callable[[Tool, dict, Context], Optional[str]]


class Policy:
    def __init__(
        self,
        allowed_tools: set,
        approve_above: float = 1000.0,
        deny_above: float = 5000.0,
        run_budget: float = 10000.0,
        hard_rules: list[Rule] | None = None,
        approval_rules: list[Rule] | None = None,
    ):
        self.allowed_tools = set(allowed_tools)
        self.approve_above = approve_above
        self.deny_above = deny_above
        self.run_budget = run_budget
        self.hard_rules = hard_rules or []
        self.approval_rules = approval_rules or []

    def check(self, tool: Tool, args: dict, ctx: Context) -> PolicyDecision:
        if tool.name not in self.allowed_tools:
            return PolicyDecision(False, False, [f"tool '{tool.name}' is not permitted for this agent"])

        value = tool.value(args) if tool.value else 0.0
        deny: list = []
        if value > self.deny_above:
            deny.append(f"value ${value:,.2f} exceeds the hard cap of ${self.deny_above:,.2f}")
        if ctx.spend + value > self.run_budget:
            deny.append(f"would exceed the run budget of ${self.run_budget:,.2f} (spent ${ctx.spend:,.2f})")
        for rule in self.hard_rules:
            msg = rule(tool, args, ctx)
            if msg:
                deny.append(msg)
        if deny:
            return PolicyDecision(False, False, deny)

        review: list = []
        if tool.risk >= Risk.IRREVERSIBLE:
            review.append("irreversible action")
        if tool.risk >= Risk.WRITE:
            if value > self.approve_above:
                review.append(f"value ${value:,.2f} is above the ${self.approve_above:,.2f} auto-approve limit")
            if ctx.tainted:
                review.append("untrusted content is in this run's context (possible prompt injection)")
            for rule in self.approval_rules:
                msg = rule(tool, args, ctx)
                if msg:
                    review.append(msg)
        return PolicyDecision(True, bool(review), review)
