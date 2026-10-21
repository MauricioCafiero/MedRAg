"""Switchable LLM backends for the agent: Ollama cloud or the Anthropic API.

Both backends take the same neutral message list and tool functions, so the
agent loop never knows which one it is talking to:

    {"role": "user",      "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool",      "id": str, "name": str, "content": str}

Tools are plain Python functions; their signature and docstring become the
tool schema (the same convention MoDrAg_CLI relies on with ollama).
"""
from __future__ import annotations

import inspect
import json
import os
import types
import typing
from dataclasses import dataclass, field

DEFAULT_MODELS = {
    "ollama": "glm-5.3",             # cloud model (MoDrAg_CLI uses glm-5.2; 5.3 is served on ollama.com)
    "anthropic": "claude-sonnet-5-5",
}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# --- function -> JSON schema -------------------------------------------------

_SIMPLE = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(tp) -> dict:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _json_type(args[0]) if args else {"type": "string"}
    if origin in (list, tuple, set):
        inner = typing.get_args(tp)
        return {"type": "array", "items": _json_type(inner[0]) if inner else {"type": "string"}}
    if origin is dict or tp is dict:
        return {"type": "object"}
    if origin is typing.Literal:
        return {"type": "string", "enum": list(typing.get_args(tp))}
    return {"type": _SIMPLE.get(tp, "string")}


def _param_docs(doc: str) -> dict[str, str]:
    """Parse the 'Args:' block of a Google-style docstring into {name: text}."""
    out: dict[str, str] = {}
    in_args, cur = False, None
    for line in doc.splitlines():
        s = line.strip()
        if s.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if s.lower().startswith(("returns:", "return:", "raises:")):
                break
            if ":" in s and s.split(":", 1)[0].replace("_", "").isalnum() and " " not in s.split(":", 1)[0]:
                cur = s.split(":", 1)[0]
                out[cur] = s.split(":", 1)[1].strip()
            elif cur and s:
                out[cur] += " " + s
    return out


def tool_schema(fn) -> dict:
    """{'name', 'description', 'parameters'} (JSON-schema) for a plain function."""
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    doc = inspect.getdoc(fn) or ""
    pdocs = _param_docs(doc)
    props, required = {}, []
    for name, p in sig.parameters.items():
        prop = _json_type(hints.get(name, str))
        if name in pdocs:
            prop["description"] = pdocs[name]
        props[name] = prop
        if p.default is inspect.Parameter.empty:
            required.append(name)
    desc = doc.split("Args:")[0].strip() or fn.__name__
    return {"name": fn.__name__, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}


# --- backends ------------------------------------------------------------------

class Backend:
    name = "base"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODELS[self.name]

    def step(self, system: str, messages: list[dict], tools: list) -> Reply:
        raise NotImplementedError


class OllamaBackend(Backend):
    """Ollama cloud (https://ollama.com, OLLAMA_API_KEY). Nothing runs locally;
    set OLLAMA_HOST only to point at a different cloud-compatible endpoint."""
    name = "ollama"

    def __init__(self, model: str | None = None):
        super().__init__(model)
        from ollama import Client
        key = os.getenv("OLLAMA_API_KEY")
        if not key:
            raise RuntimeError("OLLAMA_API_KEY is not set (needed for the ollama cloud backend).")
        self.client = Client(host=os.getenv("OLLAMA_HOST", "https://ollama.com"),
                             headers={"Authorization": f"Bearer {key}"})

    @staticmethod
    def _convert(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content", ""),
                            "tool_calls": [{"function": {"name": c["name"], "arguments": c["args"]}}
                                           for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def step(self, system, messages, tools):
        resp = self.client.chat(model=self.model, messages=self._convert(system, messages),
                                tools=tools, think=True)
        msg = resp.message
        calls = [ToolCall(id=f"call_{i}", name=tc.function.name, args=dict(tc.function.arguments))
                 for i, tc in enumerate(msg.tool_calls or [])]
        return Reply(text=msg.content or "", thinking=getattr(msg, "thinking", "") or "", tool_calls=calls)


class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        super().__init__(model)
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for the anthropic backend).")
        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no text)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}
                # consecutive tool results must share ONE user message
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def step(self, system, messages, tools):
        specs = [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
                 for s in map(tool_schema, tools)]
        kwargs = {"tools": specs} if specs else {}          # the API rejects an empty tools list
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                           messages=self._convert(messages), **kwargs)
        text, calls = "", []
        for b in resp.content:
            if b.type == "text":
                text += b.text
            elif b.type == "tool_use":
                calls.append(ToolCall(id=b.id, name=b.name, args=dict(b.input)))
        return Reply(text=text, tool_calls=calls)


def make_backend(name: str, model: str | None = None) -> Backend:
    backends = {"ollama": OllamaBackend, "anthropic": AnthropicBackend}
    if name not in backends:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(backends)}")
    return backends[name](model)


def run_turn(backend: Backend, system: str, messages: list[dict], tools: list,
             on_event=None, max_steps: int = 30) -> str:
    """Drive one user turn to completion: call the model, execute tool calls,
    feed results back, stop when the model answers without a tool call."""
    registry = {fn.__name__: fn for fn in tools}
    for _ in range(max_steps):
        reply = backend.step(system, messages, tools)
        messages.append({"role": "assistant", "content": reply.text,
                         "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in reply.tool_calls]})
        if on_event and reply.thinking:
            on_event("thinking", reply.thinking)
        if not reply.tool_calls:
            return reply.text
        for c in reply.tool_calls:
            if on_event:
                on_event("call", f"{c.name}({json.dumps(c.args)[:300]})")
            fn = registry.get(c.name)
            try:
                result = f"Error: unknown tool {c.name!r}." if fn is None else str(fn(**c.args))
            except Exception as exc:  # surface to the model so it can adjust
                result = f"Error in {c.name}: {exc!r}"
            if on_event:
                on_event("result", result[:500])
            messages.append({"role": "tool", "id": c.id, "name": c.name, "content": result})
    return "Stopped: step limit reached without a final answer."
