# -*- coding: utf-8 -*-
"""
上下文压缩策略对比实验：直接截断 vs 结构化压缩
==================================================

【实验目的】
在 AI Agent 与 LLM 的长对话场景中，对话越长，每次请求携带的上下文（prompt）就越长。
本脚本对比三种上下文处理策略，观察它们对 "上下文长度" 和 "回答质量" 的影响：

  1. 完整上下文（基线）：不做任何压缩，每次都把全部历史发给模型。
  2. 直接截断：超过预算后，直接丢掉最早的对话，只保留最近的部分。
  3. 结构化压缩：超过阈值后，让模型把较早的对话压缩成一份"结构化记忆"
     （目标 / 关键事实 / 决策与约束 / 人员分工 / 待办），之后只发送
     "结构化记忆 + 最近几条原始消息"。

【实验方法】
  - 构造一段固定的 22 轮对话脚本：开头 4 轮交代项目的关键事实（代号、预算、技术栈、负责人等），
    中间穿插大量无关闲聊，并在不同位置插入"探测题"，询问早期交代过的事实。
  - 三种策略各自完整跑一遍同样的脚本（模型的回答由各策略自己生成）。
  - 上下文长度：取服务端返回的 usage.prompt_tokens（真实 token 数）。
  - 回答质量：探测题的回答中是否包含标准答案关键词（关键词召回率）。

【运行方式】
  先启动 llama.cpp 服务（上下文建议 -c 8192 以上，否则"完整上下文"基线可能超长），例如：
      llama-server -m gemma-4.gguf -c 8192 --port 8080
  再运行：
      pip install openai
      python context_compression_experiment.py
  可通过环境变量修改连接信息：LLM_BASE_URL、LLM_API_KEY、LLM_MODEL；
  也可修改截断预算与压缩触发阈值：TRUNCATE_BUDGET、COMPRESS_TRIGGER（单位：估算 token）
"""

import os
from openai import OpenAI

# ====================== 一、配置区 ======================

# llama.cpp 的 OpenAI 兼容接口地址（llama-server 默认端口 8080，路径前缀 /v1）
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
# llama.cpp 默认不校验 key，随便填一个非空字符串即可
API_KEY = os.environ.get("LLM_API_KEY", "123")
# 模型名：llama.cpp 单模型部署时会忽略该字段，按实际情况填写即可
MODEL = os.environ.get("LLM_MODEL", "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf")

# 每次回答最多生成多少 token
MAX_REPLY_TOKENS = 400
# 直接截断策略：历史消息的 token 预算（估算值），超过就丢弃最早的消息
TRUNCATE_BUDGET = int(os.environ.get("TRUNCATE_BUDGET", "1200"))
# 结构化压缩策略：未压缩的历史超过该估算 token 数时，触发一次压缩
COMPRESS_TRIGGER = int(os.environ.get("COMPRESS_TRIGGER", "1200"))
# 结构化压缩策略：压缩时保留最近多少条原始消息不压缩（取奇数，保证剩余消息以用户消息开头）
KEEP_RECENT = 3

# 系统提示词：三种策略共用，保证对比公平
SYSTEM_PROMPT = "你是一个项目管理助手。请用中文回答，回答要简洁，控制在 80 字以内。"

# 创建 OpenAI 客户端（使用 openai 官方 SDK，指向本地 llama.cpp 服务）
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ====================== 二、对话脚本 ======================
# 每一项是一轮用户发言；expect 非空表示这是"探测题"，
# expect 里每个元素是一个必须出现在回答里的关键词，用 "|" 分隔表示"任意一个出现即可"。

SCRIPT = [
    # ---- 第 1~4 轮：交代关键事实（后面会被反复追问）----
    {"user": "我们要启动一个内部项目，代号叫『青鸾』，目标是做一个客服工单自动分类系统。请简短确认。", "expect": []},
    {"user": "项目预算上限是 86 万元，截止日期是 2026 年 12 月 18 日。请简短确认。", "expect": []},
    {"user": "技术栈定为 FastAPI + PostgreSQL，明确禁止使用 MongoDB，因为运维团队不支持。请简短确认。", "expect": []},
    {"user": "项目负责人是李明，测试负责人是王芳，每周三上午 10 点开周会。请简短确认。", "expect": []},
    # ---- 第 5~10 轮：无关闲聊，让上下文变长 ----
    {"user": "管理后台的界面想做得简洁一些，配色上你有什么建议？请简要回答。", "expect": []},
    {"user": "工单分类的类别我们初步定了十几个，类别命名怎么设计才不容易混淆？", "expect": []},
    {"user": "训练用的历史工单数据里有不少重复和脏数据，清洗时要注意什么？", "expect": []},
    {"user": "接口的错误码想统一规范，你建议怎么设计？", "expect": []},
    {"user": "代码评审流程想轻量一点，每个 PR 需要几个人评审比较合适？", "expect": []},
    {"user": "分类模型上线后，如何监控它的准确率是否下降？", "expect": []},
    # ---- 第 11 轮：探测题 1（考察第 1、2 轮的事实）----
    {"user": "提醒我一下，项目代号和预算上限分别是什么？", "expect": ["青鸾", "86"]},
    # ---- 第 12~14 轮：继续闲聊 ----
    {"user": "我们想给客服加一个反馈按钮，让他们纠正错误的分类，交互上怎么做比较好？", "expect": []},
    {"user": "部署打算用容器，滚动发布时有哪些常见的坑需要提前规避？", "expect": []},
    {"user": "文档要写哪些内容才算够用？不想写太多，但又怕交接困难。", "expect": []},
    # ---- 第 15 轮：中途新增一个事实（较新的信息，用来考察"近期信息"是否保留）----
    {"user": "补充一个决定：系统日志统一保留 90 天，超过自动清理。请简短确认。", "expect": []},
    # ---- 第 16~18 轮：继续闲聊 ----
    {"user": "团队有两位新人，怎么安排上手任务比较合适？", "expect": []},
    {"user": "接口限流策略，你觉得按用户还是按 IP 限流更合理？", "expect": []},
    {"user": "如果工单里涉及客户隐私，在日志里应该如何脱敏？", "expect": []},
    # ---- 第 19~22 轮：探测题 2~5 ----
    {"user": "我们的数据库选型是什么？有什么明确禁止使用的技术？", "expect": ["PostgreSQL", "MongoDB"]},
    {"user": "项目负责人是谁？每周的周会安排在什么时候？", "expect": ["李明", "周三"]},
    {"user": "我们的系统日志要保留多少天？", "expect": ["90"]},
    {"user": "项目的截止日期是哪一天？", "expect": ["12月18|12-18|12/18|12 月 18"]},
]


# ====================== 三、工具函数 ======================

def estimate_tokens(text):
    """
    粗略估算一段文本的 token 数（仅用于"决定何时压缩/截断"，不用于最终统计）。
    规则：非 ASCII 字符（主要是中文）按 1 个 token 算，ASCII 字符按 0.25 个 token 算。
    最终报告里的上下文长度使用服务端返回的真实 usage.prompt_tokens。
    """
    total = 0.0
    for ch in text:
        if ord(ch) > 127:
            total += 1
        else:
            total += 0.25
    return int(total)


def estimate_messages_tokens(messages):
    """估算一组消息的总 token 数：把每条消息的内容估算值累加，每条消息再加 4 个 token 的格式开销。"""
    total = 0
    for m in messages:
        total += estimate_tokens(m["content"]) + 4
    return total


def chat(messages, max_tokens=MAX_REPLY_TOKENS):
    """
    调用本地 llama.cpp 服务（OpenAI 兼容的 chat.completions 接口）。
    参数：messages 为标准的 [{"role":..., "content":...}] 列表。
    返回：(回答文本, 服务端统计的 prompt_tokens, 服务端统计的 completion_tokens)
    说明：temperature=0 让输出尽量稳定，便于对比；
         enable_thinking=False 是 llama.cpp 的模板参数，用来关闭"思考模式"，避免思考内容占满 max_tokens，
         不支持该参数的模型会直接忽略它。
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = resp.choices[0].message.content or ""
    if resp.usage is not None:
        prompt_tokens = resp.usage.prompt_tokens
        completion_tokens = resp.usage.completion_tokens
    else:
        # 服务端没返回 usage 时，退回到估算值
        prompt_tokens = estimate_messages_tokens(messages)
        completion_tokens = estimate_tokens(text)
    return text.strip(), prompt_tokens, completion_tokens


def score_answer(answer, expect):
    """
    给探测题的回答打分：统计标准答案关键词命中了几个。
    expect 中每个元素形如 "12月18|12-18"，"|" 分隔的多个写法只要命中任意一个就算该关键词命中。
    返回：(命中个数, 关键词总数)
    """
    hit = 0
    for item in expect:
        candidates = item.split("|")
        for c in candidates:
            if c in answer:
                hit += 1
                break
    return hit, len(expect)


# ====================== 四、三种上下文策略 ======================
# 每个策略都实现 build_context(history)：
#   输入：完整的原始对话历史 history（最后一条是刚提出的用户问题）
#   输出：真正发送给模型的 messages 列表

class FullStrategy:
    """策略 1：完整上下文（基线）。不做任何压缩，系统提示词 + 全部历史原样发送。"""

    def __init__(self):
        # 压缩额外开销（基线没有压缩，恒为 0）
        self.extra_prompt_tokens = 0
        self.extra_completion_tokens = 0
        self.compress_count = 0

    def build_context(self, history):
        """返回：系统提示词 + 全部历史。"""
        return [{"role": "system", "content": SYSTEM_PROMPT}] + history


class TruncateStrategy:
    """策略 2：直接截断。历史超过 token 预算时，从最早的消息开始整轮丢弃，不留任何痕迹。"""

    def __init__(self, budget):
        self.budget = budget  # 历史消息允许的最大估算 token 数
        self.extra_prompt_tokens = 0
        self.extra_completion_tokens = 0
        self.compress_count = 0

    def build_context(self, history):
        """
        从头部每次丢弃 2 条消息（一问一答），直到估算 token 数不超过预算，
        或只剩最后 1 条（当前用户问题）为止。成对丢弃能保证剩余消息始终以用户消息开头。
        """
        kept = list(history)
        while len(kept) > 1 and estimate_messages_tokens(kept) > self.budget:
            kept = kept[2:]
        return [{"role": "system", "content": SYSTEM_PROMPT}] + kept


class StructuredStrategy:
    """
    策略 3：结构化压缩。
    维护一份"结构化记忆"（self.memory）。未压缩的历史过长时，
    把较早的那部分消息连同旧记忆一起交给模型，让它产出更新后的结构化记忆；
    之后发送给模型的是：系统提示词 + 结构化记忆 + 最近几条原始消息。
    """

    def __init__(self, trigger, keep_recent):
        self.trigger = trigger          # 触发压缩的估算 token 阈值
        self.keep_recent = keep_recent  # 压缩时保留的最近原始消息条数
        self.memory = ""                # 当前的结构化记忆文本
        self.compressed_upto = 0        # history 中前多少条消息已被压缩进记忆
        # 压缩过程本身也要调用模型，这部分额外开销单独统计，报告中会计入总成本
        self.extra_prompt_tokens = 0
        self.extra_completion_tokens = 0
        self.compress_count = 0

    def update_memory(self, old_messages):
        """
        调用模型，把"旧记忆 + 需要被压缩的旧消息"合并成一份新的结构化记忆。
        提示词要求固定的五个栏目，并强调不能丢失数字、名称、日期、禁止事项。
        返回新的记忆文本，同时累加压缩调用的 token 开销。
        """
        # 把待压缩的消息拼成纯文本
        lines = []
        for m in old_messages:
            role = "用户" if m["role"] == "user" else "助手"
            lines.append(role + "：" + m["content"])
        dialog_text = "\n".join(lines)

        old_memory = self.memory if self.memory else "（暂无）"
        prompt = (
            "你是对话记忆整理器。请把【已有记忆】和【新增对话】合并成一份更新后的结构化记忆。\n"
            "必须严格按下面五个栏目输出，没有内容的栏目写“无”：\n"
            "【项目目标】\n【关键事实】（名称、数字、日期、金额等，原样保留）\n"
            "【决策与约束】（含明确禁止的事项）\n【人员与分工】\n【待办与未决问题】\n"
            "要求：不得丢失任何具体的名称、数字、日期、禁止事项；省略寒暄和无关的闲聊细节；总长度不超过 400 字。\n\n"
            "【已有记忆】\n" + old_memory + "\n\n【新增对话】\n" + dialog_text
        )
        # 压缩请求只用一条 user 消息，避免部分聊天模板对 system 角色的限制
        new_memory, p, c = chat([{"role": "user", "content": prompt}], max_tokens=700)
        self.extra_prompt_tokens += p
        self.extra_completion_tokens += c
        self.compress_count += 1
        return new_memory

    def build_context(self, history):
        """
        1) 取出尚未压缩的历史 pending；
        2) 如果 pending 的估算 token 数超过阈值，就把除了最近 keep_recent 条之外的部分压缩进记忆；
        3) 组装请求：系统提示词（附带结构化记忆）+ 剩余的原始消息。
        """
        pending = history[self.compressed_upto:]
        if estimate_messages_tokens(pending) > self.trigger and len(pending) > self.keep_recent:
            cut = len(pending) - self.keep_recent
            # 保证压缩的消息条数为偶数（整轮问答），这样剩余消息一定以用户消息开头
            if cut % 2 == 1:
                cut -= 1
            if cut > 0:
                self.memory = self.update_memory(pending[:cut])
                self.compressed_upto += cut
                pending = history[self.compressed_upto:]

        # 记忆并入系统提示词（只用一条 system 消息，兼容 Gemma 的聊天模板）
        system_text = SYSTEM_PROMPT
        if self.memory:
            system_text += "\n\n以下是此前对话的结构化记忆，回答时请以此为准：\n" + self.memory
        return [{"role": "system", "content": system_text}] + pending


# ====================== 五、实验流程 ======================

def run_experiment(name, strategy):
    """
    用指定策略把 SCRIPT 完整跑一遍。
    每一轮：加入用户消息 -> 由策略构造上下文 -> 调用模型 -> 把回答追加回历史。
    返回一个字典，记录每轮的 prompt_tokens、探测题得分与回答，以及压缩开销。
    """
    print("\n>>> 开始运行策略：" + name)
    history = []          # 完整的原始对话历史（两种压缩策略只是"发送"时加工，不会改动它）
    turn_tokens = []      # 每一轮请求的 prompt_tokens
    reply_tokens = 0      # 所有回答的 completion_tokens 累计
    probes = []           # 探测题记录：(轮次, 问题, 回答, 命中数, 总数)

    for i in range(len(SCRIPT)):
        turn = SCRIPT[i]
        history.append({"role": "user", "content": turn["user"]})

        messages = strategy.build_context(history)
        reply, p, c = chat(messages)

        history.append({"role": "assistant", "content": reply})
        turn_tokens.append(p)
        reply_tokens += c

        if turn["expect"]:
            hit, total = score_answer(reply, turn["expect"])
            probes.append((i + 1, turn["user"], reply, hit, total))
        print("  第 %2d/%d 轮  prompt_tokens=%d" % (i + 1, len(SCRIPT), p))

    return {
        "name": name,
        "turn_tokens": turn_tokens,
        "reply_tokens": reply_tokens,
        "probes": probes,
        "extra_prompt": strategy.extra_prompt_tokens,
        "extra_completion": strategy.extra_completion_tokens,
        "compress_count": strategy.compress_count,
        "memory": getattr(strategy, "memory", ""),
    }


def print_report(results):
    """打印对比报告：每轮上下文长度表、上下文汇总、回答质量汇总、探测题详细回答。"""
    line = "=" * 78

    # ---- 1. 每轮 prompt_tokens 对照表 ----
    print("\n" + line)
    print("一、每轮请求的上下文长度（prompt_tokens，服务端统计）")
    print(line)
    header = "轮次 | " + " | ".join([r["name"] for r in results])
    print(header)
    for i in range(len(SCRIPT)):
        mark = "  <- 探测题" if SCRIPT[i]["expect"] else ""
        cells = [str(r["turn_tokens"][i]) for r in results]
        print("%4d | %s%s" % (i + 1, " | ".join(cells), mark))

    # ---- 2. 上下文长度汇总 ----
    print("\n" + line)
    print("二、上下文长度汇总")
    print(line)
    base_total = sum(results[0]["turn_tokens"])
    for r in results:
        tokens = r["turn_tokens"]
        avg = sum(tokens) / len(tokens)
        peak = max(tokens)
        main_total = sum(tokens)
        # 总成本 = 对话请求的 prompt 之和 + 压缩调用额外消耗的 prompt 与 completion
        all_total = main_total + r["extra_prompt"] + r["extra_completion"]
        saving = (1 - all_total / base_total) * 100
        print("[%s]" % r["name"])
        print("  平均每轮 prompt_tokens : %.0f" % avg)
        print("  峰值 prompt_tokens     : %d" % peak)
        print("  对话请求 prompt 总和   : %d" % main_total)
        print("  压缩调用次数 / 额外开销: %d 次 / %d tokens" % (r["compress_count"], r["extra_prompt"] + r["extra_completion"]))
        print("  含压缩开销的总 token   : %d（相对基线节省 %.1f%%）" % (all_total, saving))

    # ---- 3. 回答质量汇总 ----
    print("\n" + line)
    print("三、回答质量（探测题关键词召回率）")
    print(line)
    for r in results:
        hit_sum = 0
        total_sum = 0
        detail = []
        for p in r["probes"]:
            hit_sum += p[3]
            total_sum += p[4]
            detail.append("第%d轮:%d/%d" % (p[0], p[3], p[4]))
        rate = hit_sum / total_sum * 100 if total_sum > 0 else 0
        print("[%s] 总召回率 %.1f%% (%d/%d)   %s" % (r["name"], rate, hit_sum, total_sum, "  ".join(detail)))

    # ---- 4. 探测题的详细回答 ----
    print("\n" + line)
    print("四、探测题的具体回答")
    print(line)
    for k in range(len(results[0]["probes"])):
        p0 = results[0]["probes"][k]
        print("\n[第 %d 轮] 问：%s   标准关键词：%s" % (p0[0], p0[1], SCRIPT[p0[0] - 1]["expect"]))
        for r in results:
            p = r["probes"][k]
            answer = p[2].replace("\n", " ")
            print("  - %s（%d/%d）：%s" % (r["name"], p[3], p[4], answer))

    # ---- 5. 结构化压缩最终生成的记忆 ----
    for r in results:
        if r["memory"]:
            print("\n" + line)
            print("五、[%s] 最终的结构化记忆" % r["name"])
            print(line)
            print(r["memory"])


def main():
    """入口：依次用三种策略跑同一段对话脚本，最后输出对比报告。"""
    strategies = [
        ("完整上下文", FullStrategy()),
        ("直接截断", TruncateStrategy(TRUNCATE_BUDGET)),
        ("结构化压缩", StructuredStrategy(COMPRESS_TRIGGER, KEEP_RECENT)),
    ]
    results = []
    for name, strategy in strategies:
        results.append(run_experiment(name, strategy))
    print_report(results)


if __name__ == "__main__":
    main()
