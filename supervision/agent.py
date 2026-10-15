"""The supervised agent loop.

Every tool call the model proposes goes through the same pipeline:

  validate args -> duplicate/loop guard -> deterministic precheck -> policy
  -> (human approval) -> execute (retry / reconcile / breaker) -> verify against
  the system of record -> result back to the model.

The model never touches a tool directly, and its own claim of success is never
treated as proof. Anything we cannot resolve safely becomes an Escalation.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Optional

from .approvals import ApprovalGate
from .core import Context, Escalation, Risk, RunResult, Tool, TransientError
from .executor import Executor
from .llm import LLM, LLMResponse
from .policy import Policy
from .sanitize import wrap_untrusted
from .schema import validate
from .trace import Tracer


def _canon(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def idem_key(run_id: str, tool: str, args: dict) -> str:
    """Deterministic: the same logical action in the same run always gets the same key."""
    return hashlib.sha256(f"{run_id}|{tool}|{_canon(args)}".encode()).hexdigest()[:16]


def _text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, default=str)


class Agent:
    def __init__(self, llm: LLM, tools: list, policy: Policy, gate: ApprovalGate, executor: Executor,
                 tracer: Tracer, system: str, max_steps: int = 12, max_invalid: int = 3,
                 max_repeat: int = 3, max_llm_retries: int = 2, sleep=time.sleep):
        self.llm, self.policy, self.gate, self.executor, self.tracer = llm, policy, gate, executor, tracer
        self.tools = {t.name: t for t in tools}
        self.system = system
        self.max_steps, self.max_invalid, self.max_repeat = max_steps, max_invalid, max_repeat
        self.max_llm_retries, self.sleep = max_llm_retries, sleep

    # ------------------------------------------------------------------ run
    def run(self, task: str, untrusted_sources: Optional[dict] = None) -> RunResult:
        ctx = Context(run_id=uuid.uuid4().hex[:8], sources=dict(untrusted_sources or {}))
        self.tracer.run_id = ctx.run_id
        self.tracer.emit("run_start", task=task[:300], max_steps=self.max_steps)

        content = task
        for name, doc in ctx.sources.items():
            wrapped, hits = wrap_untrusted(name, doc)
            content += "\n\n" + wrapped
            if hits:
                ctx.tainted = True
                self.tracer.emit("quarantine", call_id="task", source=name, patterns=hits)

        messages: list = [{"role": "user", "content": content}]
        final, steps, finished = "", 0, False
        while steps < self.max_steps and not ctx.abort:
            steps += 1
            try:
                resp = self._llm(messages)
            except Exception as e:  # noqa: BLE001 - any model failure must fail closed
                self._escalate(ctx, "llm_failure", f"{type(e).__name__}: {e}")
                break
            self.tracer.emit("llm_call", step=steps, text=resp.text[:300],
                             tool_calls=[{"name": c.name, "args": c.args} for c in resp.tool_calls])
            if not resp.tool_calls:
                final, finished = resp.text, True
                break
            messages.append({"role": "assistant", "content": resp.blocks()})
            results = []
            for c in resp.tool_calls:
                if ctx.abort:
                    results.append({"type": "tool_result", "tool_use_id": c.id,
                                    "content": "skipped: run aborted", "is_error": True})
                else:
                    results.append(self._handle(c, ctx))
            messages.append({"role": "user", "content": results})

        if not finished and not ctx.abort:
            self._escalate(ctx, "step_budget", f"no final answer within {self.max_steps} steps")
        status = "escalated" if ctx.escalations else "completed"
        self.tracer.emit("run_end", status=status, steps=steps, actions=len(ctx.actions),
                         escalations=[e.reason for e in ctx.escalations])
        return RunResult(ctx.run_id, status, final, steps, ctx.actions, ctx.escalations, ctx.denied, ctx.rejected)

    # -------------------------------------------------------------- helpers
    def _llm(self, messages: list) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(1, self.max_llm_retries + 2):
            try:
                return self.llm.complete(self.system, messages, [t.spec() for t in self.tools.values()])
            except TransientError as e:
                last = e
                self.tracer.emit("llm_retry", attempt=attempt, error=str(e))
                self.sleep(0.5 * 2 ** (attempt - 1))
        assert last is not None
        raise last

    def _escalate(self, ctx: Context, reason: str, detail: str, call_id: str = "", abort: bool = True) -> None:
        ctx.escalations.append(Escalation(reason, detail, call_id))
        self.tracer.emit("escalation", call_id=call_id, reason=reason, detail=detail)
        if abort:
            ctx.abort = True

    def _invalid(self, ctx: Context, call, reply, msg: str) -> dict:
        ctx.invalid += 1
        self.tracer.emit("invalid_call", call_id=call.id, tool=call.name, problem=msg)
        if ctx.invalid >= self.max_invalid:
            self._escalate(ctx, "repeated_invalid_calls", msg, call.id)
        return reply(f"INVALID CALL (nothing was executed): {msg}", True)

    # --------------------------------------------------------- one tool call
    def _handle(self, call, ctx: Context) -> dict:
        def reply(text: str, err: bool = False) -> dict:
            return {"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": err}

        tool: Optional[Tool] = self.tools.get(call.name)
        if tool is None:
            return self._invalid(ctx, call, reply, f"unknown tool '{call.name}'. Available: {sorted(self.tools)}")
        problems = validate(tool.input_schema, call.args)
        if problems:
            return self._invalid(ctx, call, reply, "; ".join(problems))

        canon = _canon(call.args)
        key = idem_key(ctx.run_id, tool.name, call.args)

        if (tool.name, canon) in ctx.done:
            self.tracer.emit("duplicate_write_blocked", call_id=call.id, tool=tool.name)
            return reply("ALREADY EXECUTED earlier in this run with identical arguments; not executing again. "
                         "Previous result: " + ctx.done[(tool.name, canon)])

        n = ctx.seen[(tool.name, canon)] = ctx.seen.get((tool.name, canon), 0) + 1
        if n > self.max_repeat:
            self._escalate(ctx, "loop_detected", f"{tool.name} called {n}x with identical arguments", call.id)
            return reply("Run aborted: repeated identical call.", True)

        if tool.precheck:
            found = tool.precheck(call.args, ctx)
            if found:
                cnt = ctx.precheck_failures[tool.name] = ctx.precheck_failures.get(tool.name, 0) + 1
                self.tracer.emit("precheck_failed", call_id=call.id, tool=tool.name, problems=found)
                if cnt >= 3:
                    self._escalate(ctx, "repeated_precheck_failure", "; ".join(found), call.id)
                return reply("PRECHECK FAILED (nothing was executed): " + "; ".join(found) +
                             ". Correct the arguments using only facts from the source data, or stop and explain.", True)

        d = self.policy.check(tool, call.args, ctx)
        self.tracer.emit("policy", call_id=call.id, tool=tool.name, risk=int(tool.risk),
                         allow=d.allow, needs_approval=d.needs_approval, reasons=d.reasons)
        if not d.allow:
            ctx.denied.append({"tool": tool.name, "args": call.args, "reasons": d.reasons})
            return reply("DENIED by policy: " + "; ".join(d.reasons) +
                         ". Do not retry the same request; propose a compliant alternative or stop.", True)

        if d.needs_approval:
            dec = self.gate.review(tool, call.args, ctx, call.id, d.reasons)
            if not dec.approved:
                ctx.rejected.append({"tool": tool.name, "args": call.args, "reason": dec.reason})
                if dec.flag == "canary_missed":
                    self._escalate(ctx, "reviewer_attention_check_failed", dec.reason, call.id, abort=False)
                return reply(f"REJECTED by human reviewer: {dec.reason or 'no reason given'}. Do not retry.", True)

        res = self.executor.run(tool, call.args, key)
        self.tracer.emit("exec", call_id=call.id, tool=tool.name, risk=int(tool.risk), status=res.status,
                         attempts=res.attempts, reconciled=res.reconciled, error=res.error)
        if res.status == "unknown":
            self._escalate(ctx, "unknown_outcome", f"{tool.name}: {res.error}", call.id)
            return reply("EXECUTION STATE UNKNOWN. Do not retry. A human has been alerted to reconcile.", True)
        if res.status == "circuit_open":
            self._escalate(ctx, "circuit_open", f"{tool.name}: {res.error}", call.id)
            return reply("Dependency is failing (circuit open). Stopping.", True)
        if res.status == "failed":
            if tool.risk >= Risk.WRITE:
                self._escalate(ctx, "write_failed", f"{tool.name}: {res.error}", call.id)
            return reply(f"FAILED after {res.attempts} attempt(s): {res.error}", True)

        if tool.risk >= Risk.WRITE and tool.verify:
            try:
                ok, detail = tool.verify(call.args, res.output, key)
            except Exception as e:  # noqa: BLE001
                ok, detail = False, f"verifier crashed: {e}"
            self.tracer.emit("verify", call_id=call.id, tool=tool.name, ok=ok, detail=detail)
            if not ok:
                self._escalate(ctx, "verification_failed", f"{tool.name}: {detail}", call.id)
                return reply(f"VERIFICATION FAILED: the tool reported success but the system of record disagrees "
                             f"({detail}). Do not retry. A human has been alerted.", True)

        text = _text(res.output)
        if tool.untrusted_output:
            text, hits = wrap_untrusted(tool.name, text)
            if hits:
                ctx.tainted = True
                self.tracer.emit("quarantine", call_id=call.id, source=tool.name, patterns=hits)

        if tool.risk >= Risk.WRITE:
            ctx.spend += tool.value(call.args) if tool.value else 0.0
            ctx.actions.append({"tool": tool.name, "args": call.args, "output": res.output,
                                "reconciled": res.reconciled, "human_approved": d.needs_approval})
            ctx.done[(tool.name, canon)] = text
        return reply(("OK (reconciled after timeout) " if res.reconciled else "OK ") + text)
