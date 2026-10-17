# -*- coding: utf-8 -*-
"""
实验：搭建可重置的评估环境

实验目的：
  1. 验证同一用例在多次运行中具有可比性（初始状态一致、验收结果一致）。
  2. 确认评估环境不会把隐藏答案泄漏给 Agent。
  3. 故意跳过状态重置，观察上一轮运行如何污染下一轮的结果。

运行方式：
  python eval_env_experiment.py                  # 使用固定策略 Agent（无需 LLM，结果可复现）
  python eval_env_experiment.py --agent llm      # 使用本地 llama.cpp 服务中的模型作为 Agent

依赖：
  固定策略模式只需要标准库；llm 模式需要 pip install openai
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile

# ----------------------------------------------------------------------
# 一、实验数据：初始订单、任务描述、隐藏答案
# ----------------------------------------------------------------------

# 初始订单数据（每次重置都会还原成这份数据）
SEED_ORDERS = {
    "O-1001": {
        "customer": "Alice",
        "status": "pending",
        "items": [{"sku": "apple", "qty": 2}, {"sku": "milk", "qty": 1}],
    },
    "O-1002": {
        "customer": "Bob",
        "status": "pending",
        "items": [{"sku": "pen", "qty": 3}, {"sku": "notebook", "qty": 1}],
    },
    "O-1003": {
        "customer": "Carol",
        "status": "shipped",
        "items": [{"sku": "lamp", "qty": 1}],
    },
}

# 金丝雀字符串：只写在隐藏答案文件里。
# 如果它出现在 Agent 看到的任何内容中，就说明隐藏答案泄漏了。
CANARY = "CANARY-7f3a91c2"

# 交给 Agent 的任务描述（注意：这里是"增量"任务，不是"设为某个值"，
# 所以任务不是幂等的，重复执行会累加，这正是观察状态污染的关键）
TASK = (
    "订单 O-1002 的客户追加购买：请把该订单中 sku 为 pen 的商品数量"
    "在现有基础上增加 2 件，并把订单状态设置为 confirmed，"
    "完成后调用 finish 工具结束任务。"
)

# 工具说明（会同时展示给 Agent，也会纳入泄漏扫描）
TOOL_DOC = """可用工具（每次只能调用一个）：
1. list_orders()                              列出所有订单的编号、客户、状态
2. get_order(order_id)                        查看某个订单的详情
3. add_quantity(order_id, sku, delta)         把订单中某商品数量增加 delta（整数，可为负）
4. set_status(order_id, status)               设置订单状态，可选 pending/confirmed/shipped/cancelled
5. read_file(path)                            读取工作区内的文件（如 orders.json、audit.log）
6. finish(summary)                            提交结果并结束任务（终止信号）"""

VALID_STATUS = ["pending", "confirmed", "shipped", "cancelled"]


def build_expected_orders():
    """根据初始数据推导出"正确答案"：pen 数量 +2，订单状态变为 confirmed。"""
    expected = copy.deepcopy(SEED_ORDERS)
    items = expected["O-1002"]["items"]
    for item in items:
        if item["sku"] == "pen":
            item["qty"] = item["qty"] + 2
    expected["O-1002"]["status"] = "confirmed"
    return expected


def short_hash(text):
    """计算文本的 sha256，并只保留前 12 位，便于打印比较。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_json(path):
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """写入 JSON 文件（键排序，保证内容稳定）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


# ----------------------------------------------------------------------
# 二、评估环境：初始化脚本 + 工具接口 + 终止协议 + 验收函数
# ----------------------------------------------------------------------

class OrderEnv:
    """订单编辑任务的评估环境。

    目录结构：
      root/workspace/   Agent 可见的工作区（orders.json、audit.log）
      root/hidden/      评估方私有目录，存放隐藏答案，Agent 的工具永远不能访问
    """

    def __init__(self, root):
        """记录各个目录和文件路径，并初始化回合内的内存状态。"""
        self.root = root
        self.workspace = os.path.join(root, "workspace")
        self.hidden = os.path.join(root, "hidden")
        self.orders_path = os.path.join(self.workspace, "orders.json")
        self.audit_path = os.path.join(self.workspace, "audit.log")
        self.expected_path = os.path.join(self.hidden, "expected.json")
        self.finished = False
        self.final_summary = ""

    # ---------------- 初始化脚本 ----------------

    def is_initialized(self):
        """判断工作区是否已经存在（用于"跳过重置"模式下的首次初始化）。"""
        return os.path.exists(self.orders_path)

    def reset(self):
        """初始化脚本：删除旧的工作区和隐藏目录，重新写入初始订单、空审计日志和隐藏答案。"""
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        if os.path.exists(self.hidden):
            shutil.rmtree(self.hidden)
        os.makedirs(self.workspace)
        os.makedirs(self.hidden)

        save_json(self.orders_path, SEED_ORDERS)
        with open(self.audit_path, "w", encoding="utf-8") as f:
            f.write("")
        save_json(self.expected_path, {"canary": CANARY, "orders": build_expected_orders()})

    def begin_episode(self):
        """每个回合开始时清空"回合级"内存状态（是否已 finish）。
        注意：它只清内存标记，不碰工作区文件；文件状态由 reset() 负责。"""
        self.finished = False
        self.final_summary = ""

    # ---------------- 状态观测（供实验比较使用） ----------------

    def state_hash(self):
        """对 Agent 可见的全部文件状态（orders.json + audit.log）求指纹，用来比较初始状态是否一致。"""
        orders = load_json(self.orders_path)
        orders_text = json.dumps(orders, sort_keys=True, ensure_ascii=False)
        with open(self.audit_path, "r", encoding="utf-8") as f:
            audit_text = f.read()
        return short_hash(orders_text + "||" + audit_text)

    def pen_qty(self):
        """读取 O-1002 订单中 pen 当前的数量（实验中用来直观展示状态漂移）。"""
        orders = load_json(self.orders_path)
        for item in orders["O-1002"]["items"]:
            if item["sku"] == "pen":
                return item["qty"]
        return None

    def order_status(self):
        """读取 O-1002 订单当前的状态。"""
        orders = load_json(self.orders_path)
        return orders["O-1002"]["status"]

    def audit_line_count(self):
        """统计 audit.log 的行数，即"上一轮留下了多少痕迹"。"""
        with open(self.audit_path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = text.strip().split("\n")
        if lines == [""]:
            return 0
        return len(lines)

    # ---------------- 工具接口 ----------------

    def call_tool(self, name, args):
        """统一的工具入口：根据名字分发，返回 JSON 字符串。
        任何错误都以文本形式返回给 Agent，而不是抛异常中断评估。"""
        if not isinstance(args, dict):
            return json.dumps({"error": "参数必须是 JSON 对象"}, ensure_ascii=False)

        if name == "list_orders":
            result = self.tool_list_orders()
        elif name == "get_order":
            result = self.tool_get_order(args)
        elif name == "add_quantity":
            result = self.tool_add_quantity(args)
        elif name == "set_status":
            result = self.tool_set_status(args)
        elif name == "read_file":
            result = self.tool_read_file(args)
        elif name == "finish":
            result = self.tool_finish(args)
        else:
            result = {"error": "未知工具: " + str(name)}
        return json.dumps(result, ensure_ascii=False)

    def write_audit(self, line):
        """向 audit.log 追加一行操作记录（这是一个会跨回合累积的文件）。"""
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tool_list_orders(self):
        """列出所有订单的摘要信息。"""
        orders = load_json(self.orders_path)
        result = []
        for order_id in sorted(orders.keys()):
            order = orders[order_id]
            result.append({
                "order_id": order_id,
                "customer": order["customer"],
                "status": order["status"],
            })
        return {"orders": result}

    def tool_get_order(self, args):
        """查看单个订单详情。"""
        order_id = args.get("order_id")
        orders = load_json(self.orders_path)
        if order_id not in orders:
            return {"error": "订单不存在: " + str(order_id)}
        return {"order_id": order_id, "order": orders[order_id]}

    def tool_add_quantity(self, args):
        """给订单中的某个商品增加数量，并写审计日志。"""
        order_id = args.get("order_id")
        sku = args.get("sku")
        delta = args.get("delta")
        if isinstance(delta, bool) or not isinstance(delta, int):
            return {"error": "delta 必须是整数"}

        orders = load_json(self.orders_path)
        if order_id not in orders:
            return {"error": "订单不存在: " + str(order_id)}

        for item in orders[order_id]["items"]:
            if item["sku"] == sku:
                old_qty = item["qty"]
                new_qty = old_qty + delta
                if new_qty < 0:
                    return {"error": "数量不能为负数"}
                item["qty"] = new_qty
                save_json(self.orders_path, orders)
                self.write_audit("add_quantity %s %s %d->%d" % (order_id, sku, old_qty, new_qty))
                return {"ok": True, "order_id": order_id, "sku": sku, "old_qty": old_qty, "new_qty": new_qty}
        return {"error": "订单中没有该商品: " + str(sku)}

    def tool_set_status(self, args):
        """设置订单状态，并写审计日志。"""
        order_id = args.get("order_id")
        status = args.get("status")
        if status not in VALID_STATUS:
            return {"error": "非法状态: " + str(status)}

        orders = load_json(self.orders_path)
        if order_id not in orders:
            return {"error": "订单不存在: " + str(order_id)}

        old_status = orders[order_id]["status"]
        orders[order_id]["status"] = status
        save_json(self.orders_path, orders)
        self.write_audit("set_status %s %s->%s" % (order_id, old_status, status))
        return {"ok": True, "order_id": order_id, "old_status": old_status, "new_status": status}

    def safe_path(self, rel_path):
        """路径沙箱：把路径解析为真实路径，只允许落在 workspace 之内。
        路径穿越（../）、绝对路径都会被解析到 workspace 之外，从而被拒绝。"""
        if not isinstance(rel_path, str):
            return None
        full = os.path.realpath(os.path.join(self.workspace, rel_path))
        ws = os.path.realpath(self.workspace)
        if full == ws or full.startswith(ws + os.sep):
            return full
        return None

    def tool_read_file(self, args):
        """读取工作区内的文件；越界路径一律拒绝。"""
        full = self.safe_path(args.get("path"))
        if full is None:
            return {"error": "拒绝访问：路径超出工作区"}
        if not os.path.isfile(full):
            return {"error": "文件不存在"}
        with open(full, "r", encoding="utf-8") as f:
            return {"path": args.get("path"), "content": f.read()}

    def tool_finish(self, args):
        """终止协议：Agent 调用 finish 表示"我做完了"，评估方据此结束本回合。"""
        self.finished = True
        self.final_summary = str(args.get("summary", ""))
        return {"ok": True, "message": "任务已提交，回合结束"}

    # ---------------- 验收函数 ----------------

    def accept(self):
        """验收函数：读取隐藏答案，与最终状态逐项比较，返回详细结果。
        通过条件：Agent 主动 finish，且最终订单数据与隐藏答案完全一致。"""
        expected = load_json(self.expected_path)["orders"]
        actual = load_json(self.orders_path)

        # 检查 1：目标订单的 pen 数量
        expected_qty = 0
        for item in expected["O-1002"]["items"]:
            if item["sku"] == "pen":
                expected_qty = item["qty"]
        qty_ok = (self.pen_qty() == expected_qty)

        # 检查 2：目标订单状态
        status_ok = (actual["O-1002"]["status"] == expected["O-1002"]["status"])

        # 检查 3：其他订单没有被误改
        others_ok = True
        for order_id in expected.keys():
            if order_id != "O-1002" and actual[order_id] != expected[order_id]:
                others_ok = False

        # 检查 4：整体状态与隐藏答案完全一致
        all_ok = (actual == expected)

        passed = self.finished and all_ok
        return {
            "passed": passed,
            "finished": self.finished,
            "qty_ok": qty_ok,
            "status_ok": status_ok,
            "others_ok": others_ok,
            "actual_pen_qty": self.pen_qty(),
            "expected_pen_qty": expected_qty,
        }


# ----------------------------------------------------------------------
# 三、Agent：固定策略 Agent（可复现）与 LLM Agent（连接本地服务）
# ----------------------------------------------------------------------

class ScriptedAgent:
    """固定策略 Agent：每次都执行同样的 5 步操作，完全不看观测结果。
    用它做实验，可以排除模型随机性，保证"结果差异只来自环境"。"""

    def __init__(self):
        """初始化步骤计数。"""
        self.step = 0

    def start(self, task, tool_doc):
        """每个回合开始前调用，重置步骤计数。"""
        self.step = 0

    def next_action(self, observation):
        """按预设顺序给出下一步动作。"""
        actions = [
            {"tool": "get_order", "args": {"order_id": "O-1002"}},
            {"tool": "add_quantity", "args": {"order_id": "O-1002", "sku": "pen", "delta": 2}},
            {"tool": "set_status", "args": {"order_id": "O-1002", "status": "confirmed"}},
            {"tool": "get_order", "args": {"order_id": "O-1002"}},
            {"tool": "finish", "args": {"summary": "已将 pen 数量增加 2 并确认订单"}},
        ]
        if self.step >= len(actions):
            return {"tool": "finish", "args": {"summary": "无更多动作"}}
        action = actions[self.step]
        self.step = self.step + 1
        return action


class LLMAgent:
    """LLM Agent：通过 openai 兼容 SDK 连接本地 llama.cpp 服务。
    动作协议为"每轮输出一个 JSON"，不依赖服务端的 function calling 能力。"""

    def __init__(self, base_url, model, api_key):
        """创建 openai 客户端。"""
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.messages = []

    def start(self, task, tool_doc):
        """每个回合开始前调用：清空对话历史，写入系统提示和任务。"""
        system_prompt = (
            "你是订单处理助手，通过调用工具完成任务。\n"
            + tool_doc + "\n\n"
            "每一轮你必须只输出一个 JSON 对象，不要输出其他文字，格式如下：\n"
            '{"tool": "工具名", "args": {"参数名": "参数值"}}\n'
            "例如：{\"tool\": \"get_order\", \"args\": {\"order_id\": \"O-1002\"}}\n"
            "任务完成后必须调用 finish 工具。"
        )
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "任务：" + task},
        ]

    def next_action(self, observation):
        """把上一步的工具返回交给模型，解析模型输出的下一个动作。"""
        if observation is not None:
            self.messages.append({"role": "user", "content": "工具返回：" + observation})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0,
            max_tokens=512,
        )
        text = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": text})
        action = parse_action(text)
        if action is None:
            return {"tool": "__invalid__", "args": {"raw": text[:100]}}
        return action


def parse_action(text):
    """从模型输出文本中提取 JSON 动作；解析失败返回 None。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or "tool" not in data:
        return None
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return {"tool": data["tool"], "args": args}


# ----------------------------------------------------------------------
# 四、回合运行器
# ----------------------------------------------------------------------

def clip(text, limit=90):
    """截断过长文本，避免打印刷屏。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def run_episode(env, agent, do_reset, max_steps=8):
    """运行一个完整回合并返回记录。

    do_reset=True  ：回合开始前执行初始化脚本（正确做法）
    do_reset=False ：跳过重置，沿用上一轮留下的状态（故意制造污染）；
                     只有环境从未初始化时才会做首次初始化。
    """
    if do_reset:
        env.reset()
        print("  [环境] 已执行 reset()：工作区与隐藏答案已还原")
    elif not env.is_initialized():
        env.reset()
        print("  [环境] 首次运行，执行初始化")
    else:
        print("  [环境] 跳过 reset()：沿用上一轮留下的状态")

    env.begin_episode()

    # 记录回合开始时 Agent 可见的初始状态
    init_hash = env.state_hash()
    init_qty = env.pen_qty()
    init_status = env.order_status()
    init_audit = env.audit_line_count()
    print("  [初始状态] 指纹=%s  pen数量=%s  订单状态=%s  audit.log行数=%d"
          % (init_hash, init_qty, init_status, init_audit))

    agent.start(TASK, TOOL_DOC)

    observation = None
    trace = []              # 动作轨迹（工具名 + 参数）
    seen_texts = [TASK, TOOL_DOC]   # Agent 能看到的全部内容，用于泄漏扫描
    invalid_count = 0
    termination = "max_steps"

    for step_no in range(1, max_steps + 1):
        action = agent.next_action(observation)
        tool = action["tool"]
        args = action["args"]

        if tool == "__invalid__":
            invalid_count = invalid_count + 1
            observation = json.dumps({"error": "输出格式不合法，请只输出一个 JSON 动作"}, ensure_ascii=False)
            print("  step %d: 无效动作 -> %s" % (step_no, clip(args.get("raw", ""))))
            if invalid_count >= 3:
                termination = "too_many_invalid"
                break
            continue

        observation = env.call_tool(tool, args)
        trace.append(tool + json.dumps(args, sort_keys=True, ensure_ascii=False))
        seen_texts.append(observation)
        print("  step %d: %s %s -> %s"
              % (step_no, tool, json.dumps(args, ensure_ascii=False), clip(observation)))

        if env.finished:
            termination = "finished"
            break

    result = env.accept()
    leaked = False
    for text in seen_texts:
        if CANARY in text:
            leaked = True

    print("  [终止] 原因=%s" % termination)
    print("  [验收] 通过=%s  pen数量=%s(期望%s)  状态正确=%s  其他订单未被改动=%s"
          % (result["passed"], result["actual_pen_qty"], result["expected_pen_qty"],
             result["status_ok"], result["others_ok"]))
    print("  [泄漏扫描] Agent 可见内容中出现金丝雀=%s" % leaked)

    return {
        "init_hash": init_hash,
        "init_qty": init_qty,
        "init_audit": init_audit,
        "trace_hash": short_hash("|".join(trace)),
        "termination": termination,
        "passed": result["passed"],
        "final_qty": result["actual_pen_qty"],
        "leaked": leaked,
    }


# ----------------------------------------------------------------------
# 五、实验的三个部分
# ----------------------------------------------------------------------

def part1_reset_every_time(env, agent, runs):
    """实验一：每轮都重置，验证初始状态与验收结果可比。"""
    print("\n" + "=" * 70)
    print("实验一：每轮运行前都执行 reset()，共 %d 轮" % runs)
    print("=" * 70)
    records = []
    for i in range(1, runs + 1):
        print("\n--- 第 %d 轮（重置）---" % i)
        records.append(run_episode(env, agent, do_reset=True))

    hashes = set()
    for r in records:
        hashes.add(r["init_hash"])
    passes = 0
    for r in records:
        if r["passed"]:
            passes = passes + 1

    print("\n[实验一小结]")
    print("  各轮初始状态指纹种类数: %d（1 表示每轮起点完全一致）" % len(hashes))
    print("  各轮通过情况: %d/%d" % (passes, runs))
    same_start = (len(hashes) == 1)
    print("  结论: 初始状态一致=%s，结果可比=%s" % (same_start, same_start))
    return records


def part2_leak_probe(env):
    """实验二：模拟 Agent 刺探隐藏答案，确认每一种越界读取都被拒绝。"""
    print("\n" + "=" * 70)
    print("实验二：隐藏答案泄漏探测")
    print("=" * 70)
    env.reset()
    env.begin_episode()
    print("隐藏答案的真实位置（评估方私有）: %s" % env.expected_path)

    # 对照组：正常文件应当可以读取，证明 read_file 本身是可用的
    out = env.call_tool("read_file", {"path": "orders.json"})
    print("\n[对照] read_file('orders.json') -> %s" % clip(out, 70))

    probes = [
        "../hidden/expected.json",
        "hidden/expected.json",
        "orders.json/../../hidden/expected.json",
        env.expected_path,
    ]
    leaked_count = 0
    for path in probes:
        out = env.call_tool("read_file", {"path": path})
        leaked = CANARY in out
        if leaked:
            leaked_count = leaked_count + 1
        print("[探测] read_file(%r)\n        -> %s  | 泄漏=%s" % (path, clip(out, 60), leaked))

    # 还要扫描其他工具的输出
    other_outputs = [
        env.call_tool("list_orders", {}),
        env.call_tool("get_order", {"order_id": "O-1002"}),
    ]
    for out in other_outputs:
        if CANARY in out:
            leaked_count = leaked_count + 1

    print("\n[实验二小结]")
    print("  探测 %d 条越界路径 + 其他工具输出，泄漏次数: %d" % (len(probes), leaked_count))
    print("  结论: 隐藏答案未泄漏=%s" % (leaked_count == 0))
    return leaked_count


def part3_skip_reset(env, agent, runs):
    """实验三：故意跳过重置，观察上一轮对下一轮的污染。"""
    print("\n" + "=" * 70)
    print("实验三：故意跳过重置，连续运行 %d 轮" % runs)
    print("=" * 70)
    env.reset()  # 先清理，保证第一轮从干净状态开始
    records = []
    for i in range(1, runs + 1):
        print("\n--- 第 %d 轮（不重置）---" % i)
        records.append(run_episode(env, agent, do_reset=False))

    hashes = set()
    traces = set()
    for r in records:
        hashes.add(r["init_hash"])
        traces.add(r["trace_hash"])

    print("\n[实验三小结]")
    print("  轮次 | 起始pen数量 | 起始audit行数 | 结束pen数量 | 验收")
    for i in range(len(records)):
        r = records[i]
        mark = "通过" if r["passed"] else "失败"
        print("  %4d | %11s | %13d | %11s | %s"
              % (i + 1, r["init_qty"], r["init_audit"], r["final_qty"], mark))
    print("  各轮初始状态指纹种类数: %d（>1 表示起点已不一致）" % len(hashes))
    print("  各轮动作轨迹种类数: %d（1 表示 Agent 做的事完全相同）" % len(traces))
    print("  结论:")
    print("    - 任务是'在现有基础上 +2'，上一轮的结果成了下一轮的起点，数量逐轮漂移；")
    print("    - 第 2 轮起验收失败，但 Agent 的行为并没有变，失败来自环境污染，而非能力下降；")
    print("    - audit.log 还把上一轮的操作历史暴露给了 Agent，等于泄漏了'答案线索'。")
    return records


# ----------------------------------------------------------------------
# 六、主程序
# ----------------------------------------------------------------------

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="可重置评估环境实验")
    parser.add_argument("--agent", choices=["scripted", "llm"], default="scripted",
                        help="scripted=固定策略（默认）；llm=本地 llama.cpp 模型")
    parser.add_argument("--runs", type=int, default=5, help="实验一的重复轮数")
    parser.add_argument("--stale-runs", type=int, default=4, help="实验三（不重置）的轮数")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                        help="llama.cpp 服务的 openai 兼容地址")
    parser.add_argument("--model", default="gemma-4", help="模型名（llama.cpp 通常忽略该值）")
    parser.add_argument("--api-key", default="sk-no-key-required", help="占位 api key")
    return parser.parse_args()


def main():
    """主流程：创建临时目录 -> 创建 Agent -> 依次运行三个实验 -> 清理。"""
    args = parse_args()
    root = tempfile.mkdtemp(prefix="agent_eval_")
    env = OrderEnv(root)

    print("实验根目录: %s" % root)
    print("任务描述: %s" % TASK)
    print("Agent 类型: %s" % args.agent)

    if args.agent == "llm":
        print("LLM 服务: %s  模型: %s" % (args.base_url, args.model))
        agent = LLMAgent(args.base_url, args.model, args.api_key)
    else:
        agent = ScriptedAgent()

    try:
        part1_reset_every_time(env, agent, args.runs)
        part2_leak_probe(env)
        part3_skip_reset(env, agent, args.stale_runs)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        print("\n实验结束，临时目录已清理。")


if __name__ == "__main__":
    main()
