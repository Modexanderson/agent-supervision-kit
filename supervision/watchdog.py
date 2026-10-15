"""Watchdog: audits traces against invariants, independent of the agent that produced them.

An agent that grades its own homework proves nothing. This module reads the raw event
log and checks things that must always be true:
  * every write execution was preceded by a policy decision
  * every action that policy flagged for review has an approved, non-canary human decision
  * no write was left unverified / in unknown state
and surfaces operational smells (retry storms, injection tripwires, rubber-stamping).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Finding:
    severity: str   # critical | warn | info
    code: str
    run_id: str
    detail: str


def audit(events: list, fast_approval_s: float = 3.0, min_approvals_for_rate: int = 5) -> list:
    findings: list = []
    policy = {}                 # (run, call) -> needs_approval
    approved = set()            # (run, call) with an approved, real (non-canary) decision
    real_approvals = []

    for e in events:
        k = (e.get("run_id"), e.get("call_id"))
        if e["type"] == "policy":
            policy[k] = e["needs_approval"]
        elif e["type"] == "approval":
            if e["canary"]:
                if e["approved"]:
                    findings.append(Finding("critical", "canary_missed", e["run_id"],
                                            "reviewer approved a deliberately wrong request"))
            else:
                real_approvals.append(e)
                if e["approved"]:
                    approved.add(k)

    for e in events:
        k = (e.get("run_id"), e.get("call_id"))
        t = e["type"]
        if t == "exec":
            if e["risk"] >= 1 and k not in policy:
                findings.append(Finding("critical", "unguarded_write", e["run_id"], f"{e['tool']} executed with no policy decision"))
            if e["risk"] >= 1 and policy.get(k) and k not in approved:
                findings.append(Finding("critical", "approval_bypass", e["run_id"], f"{e['tool']} executed without required approval"))
            if e["status"] in ("unknown", "circuit_open"):
                findings.append(Finding("critical", f"exec_{e['status']}", e["run_id"], f"{e['tool']}: {e['error']}"))
            elif e["attempts"] >= 3:
                findings.append(Finding("warn", "retry_storm", e["run_id"], f"{e['tool']} needed {e['attempts']} attempts"))
            if e.get("reconciled"):
                findings.append(Finding("info", "reconciled_after_timeout", e["run_id"], f"{e['tool']} had executed before the timeout; result adopted, not repeated"))
        elif t == "verify" and not e["ok"]:
            findings.append(Finding("critical", "verification_failed", e["run_id"], e["detail"]))
        elif t == "quarantine":
            findings.append(Finding("warn", "prompt_injection_tripwire", e["run_id"], f"untrusted source '{e['source']}' matched {len(e['patterns'])} pattern(s)"))
        elif t == "escalation":
            findings.append(Finding("warn", f"escalated:{e['reason']}", e["run_id"], e["detail"]))

    if real_approvals:
        fast = [a for a in real_approvals if a["approved"] and a["latency_s"] < fast_approval_s]
        n_ok = sum(1 for a in real_approvals if a["approved"])
        if len(real_approvals) >= min_approvals_for_rate and n_ok == len(real_approvals):
            findings.append(Finding("warn", "possible_rubber_stamp", "-", f"{n_ok}/{len(real_approvals)} reviewed requests approved, none rejected"))
        if fast and len(fast) * 2 >= n_ok > 0:
            findings.append(Finding("warn", "fast_approvals", "-", f"{len(fast)}/{n_ok} approvals under {fast_approval_s}s"))
    return findings


def format_report(findings: list) -> str:
    if not findings:
        return "No findings."
    order = {"critical": 0, "warn": 1, "info": 2}
    lines = [f"{f.severity.upper():8} {f.code:34} run={f.run_id:8} {f.detail}" for f in sorted(findings, key=lambda f: order[f.severity])]
    return "\n".join(lines)
