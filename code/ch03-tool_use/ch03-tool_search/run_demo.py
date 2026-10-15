"""演示入口。

    python run_demo.py                 # 离线 mock 后端（默认）
    python run_demo.py --backend anthropic   # 真实模型，需要 ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse

from agent import SEARCH_TOOL, AnthropicLLM, MockLLM, estimate_tokens, public_def, run_agent
from tool_search import ToolIndex
from tools_registry import TOOLS

# (任务描述, mock 脚本 [(搜索词, 工具, 参数)], mock 最终回答)
TASKS = [
    ("查一下东京 2026-10-16 的天气，如果下雨就给 alice@example.com 发邮件提醒她带伞。",
     [("weather forecast for a city", "get_weather_forecast", {"city": "Tokyo", "date": "2026-10-16"}),
      ("send email", "send_email", {"to": "alice@example.com", "subject": "带伞提醒", "body": "明天东京有雨"})],
     "明天东京有雨，已发送邮件提醒 Alice。"),
    ("列出 foo/bar 仓库的 open issues，并把结果发到 Slack 的 #dev 频道。",
     [("github issues of a repository", "github_list_issues", {"repo": "foo/bar", "state": "open"}),
      ("post message to slack channel", "slack_post_message", {"channel": "#dev", "text": "2 open issues"})],
     "已把 2 个 open issue 发到 #dev。"),
    ("100 美元大概能换多少日元？",
     [("currency exchange rate conversion", "currency_convert", {"amount": 100, "from_ccy": "USD", "to_ccy": "JPY"})],
     "约 15,120 日元。"),
    ("1+1 等于几？", [], "等于 2。"),   # 不需要任何工具：lazy 模式不应触发搜索
]


def fmt_row(cells, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    args = ap.parse_args()

    full = [public_def(t) for t in TOOLS]
    print(f"注册表共 {len(TOOLS)} 个工具")
    print(f"  全量工具定义 ≈ {estimate_tokens(full)} tokens（eager：每一轮都要付）")
    print(f"  搜索入口定义 ≈ {estimate_tokens(SEARCH_TOOL)} tokens（lazy：常驻成本，与工具数量无关）\n")

    rows = []
    shown = False
    for task, script, answer in TASKS:
        need = [s[1] for s in script]
        res = {}
        for mode in ("eager", "lazy"):
            llm = MockLLM(script, answer) if args.backend == "mock" else AnthropicLLM(args.model)
            if mode == "lazy" and not shown:
                print(f"▶ 任务：{task}\n  lazy 模式逐轮过程：")
                tr = run_agent(task, llm, mode, verbose=True)
                print(f"  最终回答：{tr.final_text}\n")
                shown = True
            else:
                tr = run_agent(task, llm, mode)
            res[mode] = tr
        e, l = res["eager"], res["lazy"]
        ok = lambda tr: "✓" if tr.called == need else "✗"
        saved = 100 * (1 - l.total_input_tokens / e.total_input_tokens)
        rows.append([task[:16] + "…", len(e.turns), e.total_input_tokens, ok(e),
                     len(l.turns), l.total_input_tokens, ok(l), f"{saved:.0f}%"])

    widths = [20, 8, 12, 6, 8, 12, 6, 8]
    print(fmt_row(["任务", "eager轮数", "eager输入tok", "正确", "lazy轮数", "lazy输入tok", "正确", "节省"], widths))
    print(fmt_row(["-" * w for w in widths], widths))
    for r in rows:
        print(fmt_row(r, widths))
    print("\n注：tok 为估算值（含系统提示、历史消息与工具定义，累计所有轮次）；"
          "mock 后端不是真实模型，只用于演示机制。")

    # 检索质量自检
    print("\n检索质量自检（top-3 是否包含目标工具）：")
    index = ToolIndex(TOOLS)
    probes = [("send email", "send_email"), ("发邮件通知", "send_email"),
              ("天气预报", "get_weather_forecast"), ("github issues", "github_list_issues"),
              ("汇率换算", "currency_convert"), ("book a flight", "search_flights")]
    for q, target in probes:
        names = [t["name"] for t, _ in index.search(q, 3)]
        print(f"  {'✓' if target in names else '✗'} {q!r:14} -> {names}")


if __name__ == "__main__":
    main()
