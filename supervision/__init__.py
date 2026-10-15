"""agent-supervision-kit: run tool-using agents with verification, observability,
retries, permissions, failure recovery and human-in-the-loop built in."""
from .agent import Agent
from .approvals import ApprovalGate
from .core import Risk, Tool
from .executor import CircuitBreaker, Executor
from .policy import Policy
from .trace import Tracer
from .watchdog import audit, format_report

__all__ = ["Agent", "ApprovalGate", "Risk", "Tool", "CircuitBreaker", "Executor", "Policy", "Tracer", "audit", "format_report"]
