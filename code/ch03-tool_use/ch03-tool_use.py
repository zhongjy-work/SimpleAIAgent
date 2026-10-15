"""
AI Agent 多轮工具调用实验：电商售后退款

任务：用户 u_alice 想给买的耳机退款，并要求发邮件告知结果。
Agent 需要多轮调用：get_user_profile -> search_orders -> refund_order -> send_email

本实验覆盖：
  1. 参数分层校验   结构校验 -> 权限/审批 -> 业务校验（见 run_tool 与各工具函数）
  2. 超时/重试/退避 线程池 + 超时；只对可重试错误做指数退避重试
  3. 幂等性         运行时用 (工具名+参数) 生成幂等键，后端据此去重
  4. 错误指导性     错误统一返回 {error_code, message, hint}，hint 告诉模型下一步怎么做
  5. 可观测性       每次 LLM 调用 / 工具尝试 / 工具结果都记录结构化日志，写入 trace.jsonl
  6. 有效性评估     以最终系统状态判断任务是否做对 + 过程指标（成功率、重试、错误恢复等）

运行：
  llama-server -m <gemma 模型>.gguf --jinja --port 8080     # 必须带 --jinja 才支持工具调用
  pip install openai
  LLM_BASE_URL=http://localhost:8080/v1 LLM_MODEL=gemma python agent_lab.py
"""
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from openai import OpenAI

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
MODEL = os.getenv("LLM_MODEL", "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf")
MAX_STEPS = 8          # Agent 最多循环轮数，防止死循环
TOOL_TIMEOUT = 1.0     # 单次工具调用超时（秒）
MAX_ATTEMPTS = 3       # 每个工具最多尝试次数（含首次）
BACKOFF_BASE = 0.3     # 指数退避基数（秒）：0.3, 0.6, ...
APPROVE_LIMIT = 200    # 退款金额 >= 该值需要审批
ROLE_ALLOWED = {"get_user_profile", "search_orders", "refund_order", "send_email"}  # 当前角色可用工具

TASK = "用户 u_alice 说她买的耳机想全额退款，请帮她处理，并发一封邮件告知她处理结果。"
SYSTEM = (
    "你是电商售后客服 Agent。只能通过工具获取信息和执行操作，不要编造订单号、邮箱等信息。\n"
    "工具返回 ok=false 时，认真阅读 message 和 hint，按 hint 修正后再继续。\n"
    "retryable 类错误运行时已自动重试过，不要原样重复同一调用。\n"
    "全部完成后，用中文简要汇报结果。"
)

# ============================================================
# 1. 模拟业务后端（内存数据 + 故障注入）
# ============================================================
USERS = {"u_alice": {"email": "alice@example.com"}, "u_bob": {"email": "bob@example.com"}}
ORDERS = {
    "O1001": {"user_id": "u_alice", "item": "无线耳机(旧款)", "amount": 199.0, "status": "refunded"},
    "O1002": {"user_id": "u_alice", "item": "降噪耳机 Pro", "amount": 299.0, "status": "delivered"},
    "O2001": {"user_id": "u_bob", "item": "机械键盘", "amount": 459.0, "status": "delivered"},
}
LEDGER, OUTBOX, IDEM = [], [], {}   # 退款流水、已发邮件、幂等键 -> 首次结果
LOCK = threading.Lock()
FAULTS = {"search_orders": "transient", "refund_order": "slow"}  # 每个故障只触发一次


class ToolError(Exception):
    """工具失败：code 供程序判断，message 说明发生了什么，hint 指导模型下一步。"""
    def __init__(self, code, message, hint="", retryable=False):
        super().__init__(message)
        self.code, self.message, self.hint, self.retryable = code, message, hint, retryable


def inject_fault(name):
    fault = FAULTS.pop(name, None)
    if fault == "transient":      # 上游暂时不可用（可重试）
        raise ToolError("UPSTREAM_503", "订单服务暂时不可用", retryable=True)
    if fault == "slow":           # 操作已提交，但响应迟迟不返回 -> 调用方超时
        time.sleep(TOOL_TIMEOUT * 2)


# ============================================================
# 2. 工具实现（第 3 层：业务校验在这里，因为需要读真实数据）
#    所有工具都接收 idem_key；写操作用它去重，保证幂等
# ============================================================
def get_user_profile(user_id, idem_key=None):
    user = USERS.get(user_id)
    if not user:
        raise ToolError("USER_NOT_FOUND", f"用户 {user_id} 不存在", "user_id 形如 u_xxx，请核对用户消息中的标识")
    return {"user_id": user_id, "email": user["email"]}


def search_orders(user_id, keyword="", idem_key=None):
    inject_fault("search_orders")
    if user_id not in USERS:
        raise ToolError("USER_NOT_FOUND", f"用户 {user_id} 不存在", "user_id 形如 u_xxx，请核对用户消息中的标识")
    return {"orders": [{"order_id": k, "item": o["item"], "amount": o["amount"], "status": o["status"]}
                       for k, o in ORDERS.items() if o["user_id"] == user_id and keyword in o["item"]]}


def refund_order(order_id, amount, reason, idem_key=None):
    with LOCK:
        if idem_key in IDEM:                       # 相同请求重复到达：直接返回首次结果，不重复退款
            return {**IDEM[idem_key], "duplicate": True}
        order = ORDERS.get(order_id)
        if not order:
            raise ToolError("ORDER_NOT_FOUND", f"订单 {order_id} 不存在", "请先用 search_orders 查到正确的 order_id")
        if order["status"] != "delivered":
            raise ToolError("ORDER_NOT_REFUNDABLE", f"订单 {order_id} 状态为 {order['status']}，不可退款",
                            "这不是目标订单；请用 search_orders 查看该用户其他状态为 delivered 的订单")
        if amount > order["amount"]:
            raise ToolError("AMOUNT_TOO_LARGE", f"退款金额 {amount} 超过订单金额 {order['amount']}",
                            f"请把 amount 改为不超过 {order['amount']} 后重试")
        order["status"] = "refunded"
        LEDGER.append({"order_id": order_id, "amount": amount, "reason": reason})
        IDEM[idem_key] = {"refund_id": f"R{len(LEDGER):04d}", "order_id": order_id, "amount": amount}
    inject_fault("refund_order")                   # 模拟：退款已落库，但响应超时
    return IDEM[idem_key]


def send_email(to, subject, body, idem_key=None):
    with LOCK:
        if idem_key in IDEM:
            return {**IDEM[idem_key], "duplicate": True}
        if to not in [u["email"] for u in USERS.values()]:
            raise ToolError("RECIPIENT_UNKNOWN", f"{to} 不在用户档案中",
                            "不要猜测邮箱；请先调用 get_user_profile 获取用户真实邮箱")
        OUTBOX.append({"to": to, "subject": subject, "body": body})
        IDEM[idem_key] = {"message_id": f"M{len(OUTBOX):04d}", "to": to}
        return IDEM[idem_key]


# ============================================================
# 3. 工具登记：一份定义同时用于「告诉模型」和「校验参数」
# ============================================================
def P(type_, desc, **extra):
    return {"type": type_, "description": desc, **extra}


TOOLS = {
    "get_user_profile": dict(
        fn=get_user_profile, desc="查询用户资料，返回用户邮箱",
        props={"user_id": P("string", "用户ID，如 u_alice")}, required=["user_id"], idempotent=True),
    "search_orders": dict(
        fn=search_orders, desc="按商品关键词查询某用户的订单，返回订单号、商品、金额、状态",
        props={"user_id": P("string", "用户ID，如 u_alice"), "keyword": P("string", "商品名关键词，可省略")},
        required=["user_id"], idempotent=True),
    "refund_order": dict(
        fn=refund_order, desc=f"对订单发起退款（高风险写操作，金额 >= {APPROVE_LIMIT} 需审批）",
        props={"order_id": P("string", "订单号，如 O1002"),
               "amount": P("number", "退款金额（元），必须大于 0", minimum=0.01),
               "reason": P("string", "退款原因")},
        required=["order_id", "amount", "reason"], idempotent=True,
        needs_approval=lambda a: a["amount"] >= APPROVE_LIMIT),
    "send_email": dict(
        fn=send_email, desc="给用户发送邮件",
        props={"to": P("string", "收件人邮箱"), "subject": P("string", "邮件主题"), "body": P("string", "邮件正文")},
        required=["to", "subject", "body"], idempotent=True),
}


def openai_tools():
    """转成 OpenAI 兼容的 tools 参数。"""
    return [{"type": "function", "function": {
        "name": n, "description": s["desc"],
        "parameters": {"type": "object", "properties": s["props"], "required": s["required"]}}}
        for n, s in TOOLS.items()]


# ============================================================
# 4. 可观测性：结构化日志（同时打印并保存在内存，最后写 trace.jsonl）
# ============================================================
TRACE = []


def log(kind, **fields):
    TRACE.append({"ts": round(time.time(), 3), "kind": kind, **fields})
    print(f"[{kind}] " + " ".join(f"{k}={str(v)[:100]}" for k, v in fields.items()))


# ============================================================
# 5. 运行时：分层校验 -> 超时 -> 重试退避 -> 统一错误格式
# ============================================================
POOL = ThreadPoolExecutor(max_workers=4)
PY_TYPES = {"string": str, "number": (int, float)}


def validate_schema(spec, args):
    """第 1 层：结构校验（必填、未知字段、类型、范围）。返回错误描述或 None。"""
    if not isinstance(args, dict):
        return "arguments 必须是 JSON 对象"
    for name in spec["required"]:
        if name not in args:
            return f"缺少必填参数 {name}"
    for name, value in args.items():
        prop = spec["props"].get(name)
        if prop is None:
            return f"未知参数 {name}，允许的参数: {list(spec['props'])}"
        if isinstance(value, bool) or not isinstance(value, PY_TYPES[prop["type"]]):
            return f"参数 {name} 应为 {prop['type']}，实际是 {type(value).__name__}"
        if "minimum" in prop and value < prop["minimum"]:
            return f"参数 {name} 不能小于 {prop['minimum']}"
    return None


def approve(name, args):
    """第 2 层的审批钩子。真实系统里这里是人工确认；实验里用环境变量 APPROVE=no 模拟拒绝。"""
    return os.getenv("APPROVE", "yes") != "no"


def run_tool(step, name, raw_args):
    """执行一次模型提出的工具调用，永远返回 JSON 字符串（成功或失败都交还给模型）。"""
    t0, attempts, args = time.time(), 0, {}

    def finish(ok, code=None, message="", hint="", data=None):
        log("tool_done", step=step, tool=name, args=args, ok=ok, code=code, attempts=attempts,
            ms=int((time.time() - t0) * 1000))
        out = {"ok": True, "data": data} if ok else {"ok": False, "error_code": code, "message": message, "hint": hint}
        return json.dumps(out, ensure_ascii=False)

    spec = TOOLS.get(name)
    if not spec:
        return finish(False, "UNKNOWN_TOOL", f"不存在工具 {name}", f"只能使用: {list(TOOLS)}")
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as e:
        return finish(False, "BAD_JSON", f"arguments 不是合法 JSON: {e}", "重新生成合法的 JSON 参数")

    # —— 第 1 层：结构校验
    if (err := validate_schema(spec, args)):
        return finish(False, "INVALID_ARGS", err, "按工具定义修正参数后重新调用")
    # —— 第 2 层：权限与审批
    if name not in ROLE_ALLOWED:
        return finish(False, "PERMISSION_DENIED", f"当前角色无权使用 {name}", "不要再尝试该工具，告知用户需转人工")
    if "needs_approval" in spec and spec["needs_approval"](args) and not approve(name, args):
        return finish(False, "APPROVAL_DENIED", "该操作未获审批", "不要重试；告知用户此操作需人工审批后才能执行")

    # —— 执行：超时 + 重试退避。幂等键由「工具名+参数」决定，重试时保持不变
    idem_key = hashlib.sha256(f"{name}{json.dumps(args, sort_keys=True)}".encode()).hexdigest()[:16]
    err = None
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        ta = time.time()
        try:
            data = POOL.submit(spec["fn"], idem_key=idem_key, **args).result(timeout=TOOL_TIMEOUT)
            log("attempt", step=step, tool=name, n=attempts, result="ok", ms=int((time.time() - ta) * 1000))
            return finish(True, data=data)
        except FutureTimeout:   # 线程无法被强杀，所以「超时 = 结果未知」，只有幂等工具才敢重试
            err = ToolError("TIMEOUT", f"工具在 {TOOL_TIMEOUT}s 内未响应，结果未知", retryable=True)
        except ToolError as e:  # 第 3 层：业务校验失败也走这里
            err = e
        except Exception as e:
            err = ToolError("INTERNAL_ERROR", repr(e))
        log("attempt", step=step, tool=name, n=attempts, result=err.code, ms=int((time.time() - ta) * 1000))
        if not (err.retryable and spec["idempotent"]) or attempts == MAX_ATTEMPTS:
            break
        time.sleep(BACKOFF_BASE * 2 ** (attempts - 1) + random.uniform(0, 0.1))   # 指数退避 + 抖动

    hint = err.hint
    if err.retryable:
        hint = f"运行时已自动重试 {attempts} 次仍失败。不要原样重复调用，请向用户说明并建议稍后再试。{hint}"
    return finish(False, err.code, err.message, hint)


# ============================================================
# 6. Agent 循环：模型 -> tool_calls -> 运行时 -> tool 消息 -> 模型 ...
# ============================================================
def run_agent(client, task):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for step in range(1, MAX_STEPS + 1):
        t0 = time.time()
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=openai_tools(),
                                              tool_choice="auto", temperature=0)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        log("llm", step=step, ms=int((time.time() - t0) * 1000),
            tokens=resp.usage.total_tokens if resp.usage else None, tool_calls=[c.function.name for c in calls])
        if not calls:
            return msg.content                      # 没有工具调用 = 模型认为任务完成
        ids = [c.id or f"call_{step}_{i}" for i, c in enumerate(calls)]
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": i, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for i, c in zip(ids, calls)]})
        for call_id, c in zip(ids, calls):          # 并行提出的多个调用，这里按顺序执行
            result = run_tool(step, c.function.name, c.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
    return None                                     # 超过 MAX_STEPS 仍未结束


# ============================================================
# 7. 评估：结果正确性（看系统最终状态，不看模型怎么说）+ 过程指标
# ============================================================
def evaluate(final_answer):
    checks = {
        "O1002 恰好退款一次且金额为 299": [x["amount"] for x in LEDGER if x["order_id"] == "O1002"] == [299.0],
        "没有误退其他订单": all(x["order_id"] == "O1002" for x in LEDGER),
        "只给 alice 发了一封邮件": [m["to"] for m in OUTBOX] == ["alice@example.com"],
        "邮件提到订单号和金额": bool(OUTBOX) and "O1002" in OUTBOX[0]["body"] and "299" in OUTBOX[0]["body"],
        "在步数限制内给出最终答复": final_answer is not None,
    }
    done = [e for e in TRACE if e["kind"] == "tool_done"]
    llm = [e for e in TRACE if e["kind"] == "llm"]
    failed = [e for e in done if not e["ok"]]
    seen, duplicates = set(), 0
    for e in done:                                  # 模型原样重复同一调用 = 浪费
        key = (e["tool"], json.dumps(e["args"], sort_keys=True))
        duplicates += key in seen
        seen.add(key)
    recovered = sum(any(c["ok"] and c["tool"] == e["tool"] for c in done[i + 1:])
                    for i, e in enumerate(done) if not e["ok"])   # 失败后同一工具后来成功

    print("\n========== 评估报告 ==========")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"  LLM 轮数: {len(llm)}   工具调用: {len(done)}   成功: {len(done) - len(failed)}")
    print(f"  调用一次成功率: {sum(e['ok'] and e['attempts'] == 1 for e in done)}/{len(done)}")
    print(f"  被拒绝/失败的调用: {[e['code'] for e in failed]}")
    print(f"  运行时自动重试次数: {sum(e['attempts'] - 1 for e in done)}")
    print(f"  模型重复调用数: {duplicates}")
    print(f"  错误恢复率: {recovered}/{len(failed)}")
    print(f"  耗时 LLM {sum(e['ms'] for e in llm)}ms / 工具 {sum(e['ms'] for e in done)}ms")
    with open("trace.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in TRACE)
    return all(checks.values())


if __name__ == "__main__":
    client = OpenAI(base_url=BASE_URL, api_key="sk-no-key")   # llama.cpp 不校验 key，随便填
    answer = run_agent(client, TASK)
    print("\n最终答复:", answer)
    raise SystemExit(0 if evaluate(answer) else 1)
