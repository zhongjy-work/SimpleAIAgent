"""最小 Agent 循环，支持两种工具暴露策略：

  eager —— 传统做法：每一轮都把全部工具定义塞进 `tools` 参数。
  lazy  —— 工具搜索：初始只暴露一个轻量的 `search_tools`；
           模型调用它后，harness 检索注册表，把命中工具的完整定义
           "加载"进后续轮次的 `tools` 列表（tool_result 里只回传简短的名称+摘要）。

模型后端：
  MockLLM       —— 规则驱动的"假模型"，离线即可运行，用来演示协议流程与 token 统计。
  AnthropicLLM  —— 真实模型（需要 ANTHROPIC_API_KEY 和 `pip install anthropic`）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from tool_search import ToolIndex
from tools_registry import TOOLS, execute_tool

SYSTEM_EAGER = "You are a helpful assistant. Use the provided tools when needed."

SYSTEM_LAZY = (
    "You are a helpful assistant. You only see a small set of tools by default. "
    "If the task needs a capability you don't currently have, call `search_tools` "
    "with a short English description of the capability (e.g. 'send email', "
    "'github issues'). Matching tools will become available on your next turn. "
    "Do not search for tools if you can answer directly."
)

# 唯一常驻的"入口工具"：定义很短，成本固定，与注册表规模无关
SEARCH_TOOL = {
    "name": "search_tools",
    "description": "Search the tool catalog for tools that can perform a capability. "
                   "Returns matching tool names; their full definitions are loaded "
                   "automatically for your next turn.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Capability description, e.g. 'send email'"}},
        "required": ["query"],
    },
}


def estimate_tokens(obj) -> int:
    """粗略估算 token：ASCII 约 4 字符/token，非 ASCII（中文）约 1.5 token/字。
    仅用于演示对比；真实模式下以 API 返回的 usage 为准。"""
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    ascii_chars = sum(1 for c in s if ord(c) < 128)
    return round(ascii_chars / 4 + (len(s) - ascii_chars) * 1.5)


def public_def(tool: dict) -> dict:
    """去掉仅用于检索的 hint 字段，得到真正发给模型的定义。"""
    return {k: tool[k] for k in ("name", "description", "input_schema")}


@dataclass
class Turn:
    exposed_tools: list[str]
    tool_def_tokens: int
    message_tokens: int
    calls: list[str]
    real_input_tokens: int | None = None


@dataclass
class Trace:
    task: str
    mode: str
    turns: list[Turn] = field(default_factory=list)
    final_text: str = ""
    called: list[str] = field(default_factory=list)   # 实际执行的业务工具

    @property
    def total_def_tokens(self):
        return sum(t.tool_def_tokens for t in self.turns)

    @property
    def total_input_tokens(self):
        return sum(t.tool_def_tokens + t.message_tokens for t in self.turns)


# --------------------------------------------------------------------------
# 模型后端
# --------------------------------------------------------------------------
class MockLLM:
    """规则驱动的假模型。script 是该任务需要依次调用的 [(搜索词, 工具名, 参数)]。
    行为模拟一个"守规矩"的模型：有工具就直接调用；没有就先 search_tools；
    搜不到就放弃。它不是真正的 LLM，只用来让演示可离线复现。"""

    def __init__(self, script: list[tuple[str, str, dict]], answer: str):
        self.script, self.answer = script, answer
        self._n = 0

    def step(self, system, messages, tools):
        names = {t["name"] for t in tools}
        done = sum(1 for m in messages if m["role"] == "assistant"
                   for b in m["content"] if b["type"] == "tool_use" and b["name"] != "search_tools")
        if done >= len(self.script):
            return [{"type": "text", "text": self.answer}], None
        query, tool, args = self.script[done]
        self._n += 1
        if tool in names:
            return [{"type": "tool_use", "id": f"mock_{self._n}", "name": tool, "input": args}], None
        searched = sum(1 for m in messages if m["role"] == "assistant"
                       for b in m["content"] if b["type"] == "tool_use"
                       and b["name"] == "search_tools" and b["input"].get("query") == query)
        if "search_tools" in names and searched == 0:
            return [{"type": "tool_use", "id": f"mock_{self._n}", "name": "search_tools",
                     "input": {"query": query}}], None
        return [{"type": "text", "text": f"（找不到可用工具：{query}）"}], None


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5-5"):
        import anthropic  # 延迟导入，离线模式不需要
        self.client = anthropic.Anthropic()
        self.model = model

    def step(self, system, messages, tools):
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=system, messages=messages,
            tools=[public_def(t) if "hint" in t else t for t in tools],
        )
        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return blocks, {"input_tokens": resp.usage.input_tokens}


# --------------------------------------------------------------------------
# Agent 循环
# --------------------------------------------------------------------------
def run_agent(task: str, llm, mode: str, top_k: int = 3, max_turns: int = 8,
              verbose: bool = False) -> Trace:
    assert mode in ("eager", "lazy")
    index = ToolIndex(TOOLS)
    registry = {t["name"]: t for t in TOOLS}
    active: dict[str, dict] = {}          # lazy 模式下已加载的工具
    system = SYSTEM_EAGER if mode == "eager" else SYSTEM_LAZY
    messages = [{"role": "user", "content": task}]
    trace = Trace(task, mode)

    for _ in range(max_turns):
        # ① 决定本轮暴露给模型的工具
        tools = list(TOOLS) if mode == "eager" else [SEARCH_TOOL, *active.values()]
        defs = [public_def(t) if "hint" in t else t for t in tools]

        blocks, usage = llm.step(system, messages, tools)
        messages.append({"role": "assistant", "content": blocks})

        turn = Turn(
            exposed_tools=[t["name"] for t in tools],
            tool_def_tokens=estimate_tokens(defs),
            message_tokens=estimate_tokens(system) + estimate_tokens(messages[:-1]),
            calls=[b["name"] for b in blocks if b["type"] == "tool_use"],
            real_input_tokens=usage["input_tokens"] if usage else None,
        )
        trace.turns.append(turn)

        uses = [b for b in blocks if b["type"] == "tool_use"]
        if not uses:
            trace.final_text = "".join(b["text"] for b in blocks if b["type"] == "text")
            break

        # ② 执行工具调用；search_tools 由 harness 本地处理
        results = []
        for u in uses:
            if u["name"] == "search_tools":
                hits = index.search(u["input"].get("query", ""), top_k)
                for t, _score in hits:
                    active[t["name"]] = t                    # ← 真正的"加载"动作
                content = json.dumps(
                    [{"name": t["name"], "summary": t["description"].split(".")[0]} for t, _ in hits]
                    or "No matching tools found.", ensure_ascii=False)
            elif mode == "lazy" and u["name"] not in active:
                content = f"Error: tool '{u['name']}' is not loaded. Use search_tools first."
            else:
                content = execute_tool(u["name"], u["input"])
                trace.called.append(u["name"])
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": content})
        messages.append({"role": "user", "content": results})

        if verbose:
            print(f"    turn {len(trace.turns)}: 暴露 {len(tools)} 个工具 -> 调用 {turn.calls}")
    return trace
