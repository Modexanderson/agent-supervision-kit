"""Model interface.

* ScriptedLLM  - deterministic stand-in used by the evals and tests.
* AnthropicLLM - stdlib-only adapter for the Messages API with tool use.

Messages use Anthropic's content-block format throughout, so the adapter is a
thin pass-through.
"""
from __future__ import annotations

import itertools
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol, Union

from .core import PermanentError, TransientError


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)

    def blocks(self) -> list:
        out = []
        if self.text:
            out.append({"type": "text", "text": self.text})
        for c in self.tool_calls:
            out.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
        return out


class LLM(Protocol):
    def complete(self, system: str, messages: list, tools: list) -> LLMResponse: ...


_ids = itertools.count(1)


def call(name: str, text: str = "", **args) -> LLMResponse:
    """Script helper: a model turn that calls one tool."""
    return LLMResponse(text=text, tool_calls=[ToolCall(f"toolu_{next(_ids):04d}", name, args)])


def say(text: str) -> LLMResponse:
    """Script helper: a model turn with a final answer."""
    return LLMResponse(text=text)


class ScriptedLLM:
    """Replays fixed turns. A turn is an LLMResponse or a callable(messages) -> LLMResponse."""

    def __init__(self, turns: list[Union[LLMResponse, Callable]]):
        self.turns = list(turns)
        self.calls = 0

    def complete(self, system, messages, tools):
        if self.calls >= len(self.turns):
            return LLMResponse(text="(script exhausted)")
        turn = self.turns[self.calls]
        self.calls += 1
        return turn(messages) if callable(turn) else turn


class AnthropicLLM:
    """Messages API adapter. Needs ANTHROPIC_API_KEY. Override the model with ANTHROPIC_MODEL."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 1024, timeout: float = 60.0):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system, messages, tools) -> LLMResponse:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        req = urllib.request.Request(
            self.URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code}"
            if e.code in (408, 409, 429) or e.code >= 500:
                raise TransientError(msg) from e
            raise PermanentError(f"{msg}: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise TransientError(str(e)) from e

        text, calls = [], []
        for block in data.get("content", []):
            if block["type"] == "text":
                text.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block["input"]))
        return LLMResponse(text="".join(text), tool_calls=calls)
