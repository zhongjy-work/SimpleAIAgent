# 工具动态发现（Tool Search）演示实验

演示：不把全部工具定义一次性塞进上下文，而是先只给模型一个轻量的 `search_tools` 入口，
需要某项能力时再检索并加载对应工具的完整定义。

## 运行

```bash
cd tool_search_demo
python3 run_demo.py                      # 离线 mock，无需 API key、无第三方依赖
pip install anthropic && export ANTHROPIC_API_KEY=...
python3 run_demo.py --backend anthropic  # 真实模型（默认 claude-sonnet-5-5，可用 --model 指定）
```

## 文件

| 文件 | 作用 |
|---|---|
| `tools_registry.py` | 43 个模拟工具（真实场景中来自多个 MCP Server 的 `tools/list`） |
| `tool_search.py` | 纯 Python BM25 检索，支持中英文查询 |
| `agent.py` | Agent 循环：`eager`（全量）与 `lazy`（按需加载）两种策略 + token 统计 |
| `run_demo.py` | 对比各类任务的 token 消耗，并自检检索质量 |

## 这个机制"放在哪里"

不是 prompt，也不是 skill，而是 **Agent 运行时（harness / client）层**：
每一轮请求前由 harness 决定 `tools` 参数里放哪些定义。prompt 只负责一小句引导
（`SYSTEM_LAZY`：告诉模型"你看到的不是全部工具，缺能力时先搜索"）。

```
用户任务 ─► [tools = search_tools]            ← 常驻成本约 90 tokens，与工具总数无关
            模型：需要发邮件 → search_tools("send email")
harness：BM25 检索注册表 → 命中 top-k
         ├ tool_result 只回简短名称 + 摘要
         └ 把命中工具的完整定义加入 active 集合
下一轮 ─► [tools = search_tools + send_email + ...]
            模型：调用 send_email(...)
```

Skill 的"渐进式披露"（先只放 name+description，用到再读 SKILL.md）与此是同一思想，
只是对象是技能说明而不是工具 schema。

## 业界做法对照

- **Anthropic API 原生 Tool Search Tool**：把工具标记 `defer_loading: true`，并提供 regex 或 BM25 版本的搜索工具；
  检索由服务端完成，命中的工具以引用形式展开，且不破坏 prompt cache。
  接入时请以官方文档为准核对 beta 头与 tool type 的当前名称。
- **Claude Code**：MCP 工具数量多时默认延迟加载，上下文里只保留工具名，用 ToolSearch 按需取回 schema。
- **自己实现（本实验）**：适合讲清原理；任何不支持原生能力的模型 / 框架都可以照此落地。

## 实验结论与工程注意点（可写进书里）

1. **成本模型变化**：eager 的成本 ≈ 工具总数 × 每轮；lazy 的成本 ≈ 固定入口 + 实际用到的工具。工具越多、单任务用到的越少，收益越大。
2. **有代价**：多了搜索往返（本实验每个工具多 1 轮），延迟变高；工具很少（如 <10 个）时反而不划算。
3. **检索质量决定上限**：工具名和描述要写清楚；`hint` 字段补同义词；也可换成向量检索，或让模型多次改写查询。
4. **Prompt cache**：手动改 `tools` 列表会让缓存失效。生产里应追加而非重排，或使用服务端原生机制。
5. **兜底**：模型直接调用未加载的工具时，返回"请先 search_tools"的错误，而不是静默失败（见 `agent.py`）。
6. **安全**：被搜索到不等于被授权——权限校验仍应在工具执行层做。

## 局限

- mock 后端是规则驱动的假模型，只演示机制与 token 统计；准确率、模型是否会正确发起搜索等结论需用 `--backend anthropic` 实测。
- token 数为字符估算，真实模式下 `Turn.real_input_tokens` 记录 API 返回的 usage。
