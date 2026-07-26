"""Open-source LLM client with tool calling.

Three providers behind one interface:

* ``ollama`` — native ``/api/chat``. Tool arguments arrive already parsed as a
  dict.
* ``openai_compat`` — ``/v1/chat/completions``, which covers vLLM, llama.cpp
  ``--server``, LM Studio, TGI and Ollama's compatibility endpoint. Tool
  arguments arrive as a JSON *string*.
* ``mock`` — a deterministic stand-in used by the tests so the suite needs
  neither a GPU nor a network.

Everything about talking to a small local model that turned out to matter is
handled here rather than in the control logic:

* **arguments may be a dict or a string** — normalised either way;
* **models emit fenced JSON instead of a tool call** — recovered by
  :func:`extract_tool_calls_from_text`, which is the difference between a 3B
  model being usable and not;
* **a hard wall-clock deadline** — the caller passes the remaining budget, and
  a request that would exceed it is not even attempted.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from ..config import LLMConfig


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "id": self.call_id}


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    ok: bool = True
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    recovered_from_text: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content[:600],
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "recovered_from_text": self.recovered_from_text,
        }


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_BARE_OBJECT = re.compile(r"\{[^{}]*\"(?:name|tool|function|cooling_setpoint_c)\"[^{}]*\}", re.S)


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a dict (Ollama) or a JSON string (OpenAI).
    Small models also produce double-encoded strings and stray prose."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        for _ in range(2):   # unwrap double encoding
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                break
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                text = parsed
                continue
            break
    return {}


def extract_tool_calls_from_text(text: str, valid_names: set[str]) -> list[ToolCall]:
    """Recover a tool call from prose. Small models frequently answer with a
    fenced JSON object instead of using the tool protocol; discarding those
    replies would throw away a usable decision, so we parse them.

    Accepts either ``{"name": ..., "arguments": {...}}`` or a bare argument
    object, in which case the tool is inferred from the keys present.
    """
    if not text:
        return []
    candidates: list[str] = [m.group(1) for m in _JSON_BLOCK.finditer(text)]
    candidates.extend(m.group(0) for m in _BARE_OBJECT.finditer(text))
    # Also try the whole reply, in case it is bare JSON.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    out: list[ToolCall] = []
    for blob in candidates:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name") or data.get("tool") or data.get("function")
        if isinstance(name, dict):
            name = name.get("name")
        args = _coerce_arguments(data.get("arguments", data.get("parameters", {})))
        if not args and not isinstance(name, str):
            # A bare argument object: infer the tool from its shape.
            if any(k in data for k in ("cooling_setpoint_c", "heating_setpoint_c", "oa_fraction")):
                name, args = "set_zone_setpoints", data
        if isinstance(name, str) and name in valid_names:
            if not args:
                args = {k: v for k, v in data.items() if k not in ("name", "tool", "function")}
            out.append(ToolCall(name=name, arguments=args, call_id="recovered"))
            break   # one recovered decision is enough
    return out


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self._mock_step = 0

    # ------------------------------------------------------------------ api
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        deadline: float | None = None,
        valid_names: set[str] | None = None,
    ) -> LLMReply:
        """One model round trip. ``deadline`` is an absolute ``time.monotonic``
        value; the request is abandoned rather than started if it has passed."""
        budget = self.cfg.timeout_s
        if deadline is not None:
            budget = min(budget, max(0.0, deadline - time.monotonic()))
            if budget < 0.35:
                return LLMReply(ok=False, error="decision deadline exhausted", model=self.cfg.model)

        started = time.monotonic()
        try:
            if self.cfg.provider == "mock":
                reply = self._chat_mock(messages, tools)
            elif self.cfg.provider == "openai_compat":
                reply = self._chat_openai(messages, tools, budget)
            else:
                reply = self._chat_ollama(messages, tools, budget)
        except requests.Timeout:
            reply = LLMReply(ok=False, error=f"timeout after {budget:.1f}s", model=self.cfg.model)
        except requests.RequestException as exc:
            reply = LLMReply(ok=False, error=f"{type(exc).__name__}: {exc}", model=self.cfg.model)
        except (ValueError, KeyError, TypeError) as exc:
            reply = LLMReply(ok=False, error=f"malformed response: {exc}", model=self.cfg.model)

        reply.latency_ms = (time.monotonic() - started) * 1000.0
        if reply.ok and not reply.tool_calls and reply.content:
            recovered = extract_tool_calls_from_text(reply.content, valid_names or set())
            if recovered:
                reply.tool_calls = recovered
                reply.recovered_from_text = True
        return reply

    def assistant_turn(self, reply: LLMReply) -> dict[str, Any]:
        """Build the assistant message that replays ``reply``'s tool calls.

        The two providers disagree about one field: Ollama's ``/api/chat``
        requires ``function.arguments`` to be an **object** and answers 400 to a
        string, while the OpenAI schema specifies a **JSON string**. Getting this
        wrong only fails on the second round trip of a decision — i.e. only when
        the model used a read-only tool before acting — which is exactly the
        intermittent, hard-to-spot failure worth encoding in one place.
        """
        as_object = self.cfg.provider != "openai_compat"
        return {
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments if as_object else json.dumps(call.arguments),
                    },
                }
                for call in reply.tool_calls
            ],
        }

    def health(self) -> dict[str, Any]:
        """Is the model actually reachable and loaded? Called once per run so a
        misconfigured endpoint is reported up front instead of as 144 timeouts."""
        if self.cfg.provider == "mock":
            return {"ok": True, "provider": "mock", "model": self.cfg.model}
        try:
            if self.cfg.provider == "openai_compat":
                resp = self.session.get(f"{self.cfg.base_url.rstrip('/')}/v1/models", timeout=6)
                resp.raise_for_status()
                names = [m.get("id", "") for m in resp.json().get("data", [])]
            else:
                resp = self.session.get(f"{self.cfg.base_url.rstrip('/')}/api/tags", timeout=6)
                resp.raise_for_status()
                names = [m.get("name", "") for m in resp.json().get("models", [])]
            loaded = any(self.cfg.model.split(":")[0] in n for n in names)
            return {
                "ok": True,
                "provider": self.cfg.provider,
                "model": self.cfg.model,
                "model_available": loaded,
                "available": names[:12],
                "base_url": self.cfg.base_url,
            }
        except requests.RequestException as exc:
            return {
                "ok": False,
                "provider": self.cfg.provider,
                "model": self.cfg.model,
                "base_url": self.cfg.base_url,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # --------------------------------------------------------------- ollama
    def _chat_ollama(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, budget: float
    ) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",   # keep weights resident between decisions
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        resp = self.session.post(
            f"{self.cfg.base_url.rstrip('/')}/api/chat", json=payload, timeout=budget
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") or {}
        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            calls.append(
                ToolCall(
                    name=str(name),
                    arguments=_coerce_arguments(fn.get("arguments", {})),
                    call_id=str(raw.get("id", "")),
                )
            )
        return LLMReply(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            model=str(data.get("model") or self.cfg.model),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )

    # --------------------------------------------------------------- openai
    def _chat_openai(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, budget: float
    ) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        resp = self.session.post(
            f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=budget,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            calls.append(
                ToolCall(
                    name=str(name),
                    arguments=_coerce_arguments(fn.get("arguments", "{}")),
                    call_id=str(raw.get("id", "")),
                )
            )
        usage = data.get("usage") or {}
        return LLMReply(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            model=str(data.get("model") or self.cfg.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    # ----------------------------------------------------------------- mock
    def _chat_mock(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> LLMReply:
        """A stand-in that behaves like a competent small model: it reads the
        state block out of the prompt and returns a plausible tool call."""
        self._mock_step += 1
        text = messages[-1].get("content", "") if messages else ""
        occupied = '"occupied": true' in text.lower() or "occupied: true" in text.lower()
        peak = '"peak_window": true' in text.lower()
        cooling = 26.5 if peak else (25.0 if occupied else 29.0)
        args = {
            "zone": "ALL",
            "cooling_setpoint_c": cooling,
            "heating_setpoint_c": 20.0 if occupied else 16.0,
            "oa_fraction": 0.6 if occupied else 0.35,
            "rationale": "mock provider: mid-band when occupied, setback otherwise",
        }
        return LLMReply(
            content="",
            tool_calls=[ToolCall(name="set_zone_setpoints", arguments=args, call_id=f"mock-{self._mock_step}")],
            model="mock",
        )
