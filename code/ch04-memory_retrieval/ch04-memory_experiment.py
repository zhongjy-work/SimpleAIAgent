# -*- coding: utf-8 -*-
"""
实验:记忆提取、召回与冲突更新
=====================================================================
实验目的:
    验证系统能否从多轮对话中提炼稳定偏好和项目事实,在后续会话中准确召回,
    并在用户纠正旧信息后正确更新。

对比三种记忆策略:
    策略 A:直接保存全部历史(每一轮对话原样保存,查询时全部注入)
    策略 B:仅保存摘要(每个会话结束后,用 LLM 滚动更新一份摘要)
    策略 C:结构化提取并带元数据管理(提取 -> 去重 -> 冲突更新 -> 过期过滤 -> 来源追溯)

观测指标:
    1. 记忆提取准确率      :稳定信息是否被保存、临时指令是否被拒绝
    2. 关键记忆召回率      :不同措辞的提问,能否找回关键事实
    3. 冲突处理正确率      :被纠正的信息,查询时只给新值、不给旧值
    4. 过期记忆误召回率    :旧值、已过期事件、临时指令是否被错误注入
    5. 每次查询注入 Token 数:注入到提示词里的记忆文本的估算 Token 数
    6. 来源可追溯性        :每条记忆能否定位回原始对话(会话、轮次、原话)

运行方式:
    pip install openai
    python memory_experiment.py

可用环境变量(可选):
    LLM_BASE_URL  默认 http://127.0.0.1:8080/v1   (llama.cpp 服务地址)
    LLM_API_KEY   默认 sk-no-key-required
    LLM_MODEL     默认 gemma-4                     (llama.cpp 通常忽略该名字)
"""

import json
import os
import time

from openai import OpenAI

# ---------------------------------------------------------------------
# 第 0 部分:基本配置
# ---------------------------------------------------------------------

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.environ.get("LLM_API_KEY", "sk-no-key-required")
MODEL_NAME = os.environ.get("LLM_MODEL", "gemma-4")

# 实验中的"今天"。固定下来,保证过期判断每次运行结果一致。
TODAY = "2026-10-16"

# 实验结果会保存到这个文件
RESULT_FILE = "memory_experiment_result.json"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600)


# ---------------------------------------------------------------------
# 第 1 部分:实验数据(多轮、跨会话的对话 + 标准答案)
# ---------------------------------------------------------------------

# 三个会话,时间跨度约一个半月。
# 数据中刻意混入了:明确偏好、临时指令、重复陈述、会过期的事件、显式纠正。
SESSIONS = [
    {
        "session_id": "S1",
        "date": "2026-09-01",
        "turns": [
            {"role": "user", "text": "你好,我在做一个后端项目,代号叫 Atlas。"},
            {"role": "assistant", "text": "好的,Atlas 项目。请问用什么技术栈?"},
            {"role": "user", "text": "后端用 Python 3.10 加 FastAPI,数据库用 MySQL。"},
            {"role": "assistant", "text": "明白了。需要我帮你做什么?"},
            {"role": "user", "text": "我喜欢简洁的回答,请用中文,不要使用表情符号。"},
            {"role": "assistant", "text": "收到,我会保持简洁。"},
            {"role": "user", "text": "今天先帮我把这段代码的注释改成英文注释,就这一次。"},
            {"role": "assistant", "text": "好的,我来改成英文注释。"},
            {"role": "user", "text": "对了,9 月 9 日我们要做项目评审,要准备材料。"},
            {"role": "assistant", "text": "记下了,9 月 9 日项目评审。"},
            {"role": "user", "text": "再强调一遍,我真的喜欢简洁的回答,别写太长。"},
            {"role": "assistant", "text": "好的,我会尽量简短。"},
        ],
    },
    {
        "session_id": "S2",
        "date": "2026-09-15",
        "turns": [
            {"role": "user", "text": "继续聊 Atlas。我们团队一共 4 个人。"},
            {"role": "assistant", "text": "好的,4 人团队。"},
            {"role": "user", "text": "服务现在部署在 AWS 东京区域。"},
            {"role": "assistant", "text": "了解,东京区域。"},
            {"role": "user", "text": "这次对话里请临时用英文回复我,我想练习英语,下次就不用了。"},
            {"role": "assistant", "text": "Sure, I will reply in English in this conversation."},
            {"role": "user", "text": "Please help me write a short deployment checklist."},
            {"role": "assistant", "text": "1. Check config. 2. Run tests. 3. Deploy. 4. Verify."},
        ],
    },
    {
        "session_id": "S3",
        "date": "2026-10-05",
        "turns": [
            {"role": "user", "text": "有个更正:之前我说数据库用 MySQL,那是错的,我们已经迁移到 PostgreSQL 15 了。"},
            {"role": "assistant", "text": "好的,已更正为 PostgreSQL 15。"},
            {"role": "user", "text": "Python 也升级了,现在用 3.12。"},
            {"role": "assistant", "text": "明白,Python 3.12。"},
            {"role": "user", "text": "部署区域也换了,不在东京了,现在改到新加坡区域。"},
            {"role": "assistant", "text": "收到,新加坡区域。"},
            {"role": "user", "text": "团队人数没变,还是 4 个人。"},
            {"role": "assistant", "text": "好的,团队仍是 4 人。"},
        ],
    },
]

# 应当被长期记住的信息(以"当前最新值"为准)。
# keywords:只要保存的文本里出现其中任意一个词,就认为这条信息被保存了。
STABLE_FACTS = [
    {"id": "pref_style", "desc": "偏好:回答要简洁(重复陈述过两次)", "keywords": ["简洁"]},
    {"id": "pref_emoji", "desc": "偏好:不使用表情符号", "keywords": ["表情"]},
    {"id": "project_name", "desc": "项目事实:项目代号 Atlas", "keywords": ["Atlas"]},
    {"id": "framework", "desc": "项目事实:后端框架 FastAPI", "keywords": ["FastAPI"]},
    {"id": "team_size", "desc": "项目事实:团队 4 人(重复陈述过)",
     "keywords": ["4人", "4 人", "4个", "4 个", "四人", "四个"]},
    {"id": "database", "desc": "项目事实:数据库 PostgreSQL(纠正了旧值 MySQL)", "keywords": ["PostgreSQL"]},
    {"id": "python", "desc": "项目事实:Python 3.12(升级了旧值 3.10)", "keywords": ["3.12"]},
    {"id": "region", "desc": "项目事实:部署在新加坡(替代旧值 东京)", "keywords": ["新加坡"]},
    {"id": "event_review", "desc": "带日期的事件:9 月 9 日项目评审(应带过期时间保存)", "keywords": ["评审"]},
]

# 不应该被长期记住的临时指令。
TEMP_ITEMS = [
    {"id": "temp_comment", "desc": "临时指令:这一次把注释改成英文", "keywords": ["注释"]},
    {"id": "temp_english", "desc": "临时指令:本次会话用英文回复", "keywords": ["用英文", "英文回复", "英语"]},
]

# 查询题目。每个问题的措辞都和原话不同,用来检验"换个说法也能找到"。
#   type = recall   :考察能否召回关键记忆
#   type = conflict :考察被纠正的信息,要给新值、不能给旧值
#   type = expired  :考察已过期的事件不应被召回
#   type = temp     :考察临时指令不应被召回
# expect :期望在注入内容中出现的关键词(任意一个即可)
# forbid :不应出现在注入内容中的关键词(出现任意一个即算误召回)
QUERIES = [
    {"id": "Q1", "type": "conflict", "question": "我们项目现在用的是哪种数据库?",
     "expect": ["PostgreSQL"], "forbid": ["MySQL"]},
    {"id": "Q2", "type": "conflict", "question": "后端的数据存储用的什么方案?",
     "expect": ["PostgreSQL"], "forbid": ["MySQL"]},
    {"id": "Q3", "type": "conflict", "question": "Python 目前是哪个版本?",
     "expect": ["3.12"], "forbid": ["3.10"]},
    {"id": "Q4", "type": "conflict", "question": "服务部署在哪个云区域?",
     "expect": ["新加坡"], "forbid": ["东京"]},
    {"id": "Q5", "type": "conflict", "question": "我们的线上环境落在哪个城市的机房?",
     "expect": ["新加坡"], "forbid": ["东京"]},
    {"id": "Q6", "type": "recall", "question": "写回复的时候,我希望是什么风格?",
     "expect": ["简洁"], "forbid": []},
    {"id": "Q7", "type": "recall", "question": "回答里可以带 emoji 吗?",
     "expect": ["表情"], "forbid": []},
    {"id": "Q8", "type": "recall", "question": "我们这个项目叫什么名字?",
     "expect": ["Atlas"], "forbid": []},
    {"id": "Q9", "type": "recall", "question": "后端用哪个 Web 框架?",
     "expect": ["FastAPI"], "forbid": []},
    {"id": "Q10", "type": "recall", "question": "咱们组有几个人?",
     "expect": ["4人", "4 人", "4个", "4 个", "四人", "四个"], "forbid": []},
    {"id": "Q11", "type": "expired", "question": "最近有什么需要准备的评审安排吗?",
     "expect": [], "forbid": ["评审"]},
    {"id": "Q12", "type": "temp", "question": "现在回复我的时候,需要用英文吗?",
     "expect": [], "forbid": ["用英文", "英文回复", "英语"]},
]


# ---------------------------------------------------------------------
# 第 2 部分:通用工具函数
# ---------------------------------------------------------------------

def call_llm(system_prompt, user_prompt, max_tokens=3000):
    """调用本地 llama.cpp 模型,返回文本。失败会重试 2 次,最终失败返回空字符串。"""
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,  # 温度设为 0,让结果尽量稳定
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                content = ""
            return content.strip()
        except Exception as error:
            print("    [LLM 调用失败,第 %d 次] %s" % (attempt + 1, error))
            time.sleep(1)
    return ""


def cut_json_text(raw_text, open_char, close_char):
    """从模型输出中截取第一个 open_char 到最后一个 close_char 之间的内容。
    模型有时会在 JSON 外面加上 ```json 之类的标记,这里把它们去掉。"""
    start = raw_text.find(open_char)
    end = raw_text.rfind(close_char)
    if start == -1 or end == -1 or end <= start:
        return ""
    return raw_text[start:end + 1]


def parse_json_list(raw_text):
    """把模型输出解析成 Python 列表。解析失败返回 None(区别于空列表 [])。"""
    json_text = cut_json_text(raw_text, "[", "]")
    if json_text == "":
        return None
    try:
        result = json.loads(json_text)
    except Exception:
        return None
    if isinstance(result, list):
        return result
    return None


def estimate_tokens(text):
    """粗略估算 Token 数(不依赖具体分词器,只用于策略之间的横向对比):
    非 ASCII 字符(如汉字)按 1 字符 ≈ 1 Token,ASCII 字符按 4 字符 ≈ 1 Token。"""
    wide_count = 0
    ascii_count = 0
    for ch in text:
        if ord(ch) > 127:
            wide_count = wide_count + 1
        else:
            ascii_count = ascii_count + 1
    return wide_count + (ascii_count + 3) // 4


def remove_spaces(text):
    """去掉所有空白字符,用于宽松地比较两段文字。"""
    result = ""
    for ch in text:
        if ch != " " and ch != "\n" and ch != "\t":
            result = result + ch
    return result


def has_any_keyword(text, keywords):
    """文本中是否出现了关键词列表里的任意一个(不区分大小写)。"""
    lower_text = text.lower()
    for keyword in keywords:
        if keyword.lower() in lower_text:
            return True
    return False


def format_dialogue(session):
    """把一个会话格式化成带编号的文本,交给 LLM 阅读。"""
    lines = []
    index = 0
    for turn in session["turns"]:
        if turn["role"] == "user":
            role_name = "用户"
        else:
            role_name = "助手"
        lines.append("[%d] %s:%s" % (index, role_name, turn["text"]))
        index = index + 1
    return "\n".join(lines)


def find_turn_by_quote(session, quote):
    """在会话的"用户发言"里查找原话片段,返回轮次编号;找不到返回 -1。
    这一步由代码完成而不是由 LLM 完成,所以来源信息是可验证的。"""
    clean_quote = remove_spaces(quote)
    if clean_quote == "":
        return -1
    index = 0
    for turn in session["turns"]:
        if turn["role"] == "user":
            if clean_quote in remove_spaces(turn["text"]):
                return index
        index = index + 1
    return -1


def build_source(session, quote):
    """构造一条"来源记录":来自哪个会话、哪一轮、原话是什么、是否验证通过。"""
    turn_index = find_turn_by_quote(session, quote)
    source = {
        "session_id": session["session_id"],
        "session_date": session["date"],
        "turn_index": turn_index,
        "quote": quote,
        "verified": turn_index >= 0,
    }
    return source


def short_text(text, limit):
    """把多行文本压成一行并截断,方便在控制台预览。"""
    one_line = text.replace("\n", " / ")
    if len(one_line) > limit:
        return one_line[:limit] + "..."
    return one_line


def display_width(text):
    """估算文本在终端中的显示宽度(汉字占 2 格)。"""
    width = 0
    for ch in text:
        if ord(ch) > 127:
            width = width + 2
        else:
            width = width + 1
    return width


def pad(text, width):
    """用空格把文本补齐到指定显示宽度,用于对齐表格。"""
    gap = width - display_width(text)
    if gap < 0:
        gap = 0
    return text + " " * gap


def print_title(title):
    """打印一个醒目的章节标题。"""
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------
# 第 3 部分:三种记忆策略
# 每个策略都实现同样的四个方法:
#   ingest_session(session)        写入一个会话
#   get_stored_texts()             返回所有已保存的记忆文本(用于评估提取准确率)
#   retrieve(question)             针对问题返回要注入提示词的文本
#   get_traceability()             返回(可追溯的记忆条数, 记忆总条数)
# ---------------------------------------------------------------------

class FullHistoryMemory:
    """策略 A:直接保存全部历史。
    写入:每一轮对话原样保存。
    召回:不做筛选,把全部历史注入提示词(最朴素的做法)。"""

    def __init__(self):
        self.name = "A.全部历史"
        self.records = []

    def ingest_session(self, session):
        """逐轮保存对话原文,并记录它来自哪个会话、哪一轮。"""
        index = 0
        for turn in session["turns"]:
            if turn["role"] == "user":
                role_name = "用户"
            else:
                role_name = "助手"
            record = {
                "text": "(%s)%s:%s" % (session["date"], role_name, turn["text"]),
                "source": {
                    "session_id": session["session_id"],
                    "session_date": session["date"],
                    "turn_index": index,
                    "quote": turn["text"],
                    "verified": True,
                },
            }
            self.records.append(record)
            index = index + 1

    def get_stored_texts(self):
        """返回所有保存的文本。"""
        texts = []
        for record in self.records:
            texts.append(record["text"])
        return texts

    def retrieve(self, question):
        """不做任何筛选,把全部历史拼起来注入。"""
        lines = []
        for record in self.records:
            lines.append(record["text"])
        context = "\n".join(lines)
        info = {"filtered_superseded": 0, "filtered_expired": 0}
        return context, info

    def get_traceability(self):
        """每一条都保留了会话编号和轮次,因此全部可追溯。"""
        return len(self.records), len(self.records)


class SummaryMemory:
    """策略 B:仅保存摘要。
    写入:每个会话结束后,让 LLM 把"旧摘要 + 新会话"合并成一份新摘要。
    召回:每次都把整份摘要注入提示词。"""

    def __init__(self):
        self.name = "B.仅摘要"
        self.summary = ""

    def ingest_session(self, session):
        """调用 LLM,滚动更新摘要。"""
        system_prompt = "你是对话摘要助手。请把对话压缩成简短摘要,供以后的会话作为背景使用。"
        old_summary = self.summary
        if old_summary == "":
            old_summary = "(暂无)"
        user_prompt = (
            "已有摘要:\n" + old_summary + "\n\n"
            "新会话(日期:" + session["date"] + "):\n" + format_dialogue(session) + "\n\n"
            "请输出更新后的完整摘要,不超过 200 字。"
            "如果新对话与已有摘要冲突,以新对话为准。只输出摘要正文。"
        )
        new_summary = call_llm(system_prompt, user_prompt, max_tokens=1500)
        if new_summary != "":
            self.summary = new_summary

    def get_stored_texts(self):
        """摘要只有一段文本。"""
        if self.summary == "":
            return []
        return [self.summary]

    def retrieve(self, question):
        """整份摘要都注入。"""
        info = {"filtered_superseded": 0, "filtered_expired": 0}
        return self.summary, info

    def get_traceability(self):
        """摘要是融合后的一段文字,无法定位到具体的原始对话,所以可追溯条数为 0。"""
        if self.summary == "":
            return 0, 0
        return 0, 1


EXTRACT_SYSTEM_PROMPT = """你是一个"长期记忆提取器"。请从对话中提取值得长期保存的信息。
规则:
1. 只提取【用户本人】说的内容,分三类 type:
   preference(稳定偏好)、project_fact(项目事实)、event(带明确日期的事件)。
2. 只对当前会话有效的临时指令(例如"这次""今天""本次对话""就这一次"),scope 填 session;
   其他信息 scope 填 long_term。
3. 同一信息在对话中重复出现,只输出一条。
4. 用户明确纠正或更新旧信息时,输出新值,并设置 is_correction 为 true。
5. key 使用简短的英文小写下划线命名。每个独立的信息单独一条,不要把不同信息合并到一个 key 里。
   如果"已有记忆"中已有含义相同的 key,必须原样复用该 key。
6. event 类型必须给出 expires_at(事件当天,格式 YYYY-MM-DD,年份根据会话日期推算);
   其他类型 expires_at 填空字符串。
7. evidence_quote 必须是用户原话中的连续片段,逐字复制,不要改写。
8. 只输出 JSON 数组,不要输出任何其他文字。没有可提取的信息时输出 []。
输出格式:
[{"key": "", "value": "", "type": "preference", "scope": "long_term",
  "expires_at": "", "is_correction": false, "evidence_quote": ""}]"""

SELECT_SYSTEM_PROMPT = """你是记忆检索器。下面有一份编号的记忆清单和一个用户问题。
请找出回答该问题需要用到的记忆编号。
只输出 JSON 数组,例如 [1, 3]。没有相关记忆时输出 []。不要输出其他文字。"""


class StructuredMemory:
    """策略 C:结构化提取 + 元数据管理。
    写入流程:
        1. LLM 把对话提取成结构化条目(key / value / 类型 / 范围 / 过期时间 / 原话)。
        2. 代码丢弃"仅当前会话有效"的临时指令。
        3. 代码按 key 做去重:同 key 同值 -> 重复次数加 1,并追加来源。
        4. 代码按 key 做冲突更新:同 key 不同值 -> 旧记录标记为 superseded(已被替代),新记录生效。
    召回流程:
        1. 代码先过滤掉:已被替代的、已过期的记录。
        2. 把剩下的记录交给 LLM,让它按问题挑选相关编号(这样换措辞提问也能找到)。
        3. 只把被选中记录的 "key: value" 注入提示词(来源等元数据不注入,只保存在库里)。"""

    def __init__(self):
        self.name = "C.结构化"
        self.records = []        # 全部记忆记录(包含已被替代的旧记录)
        self.rejected = []       # 被拒绝保存的临时指令(用于展示)
        self.conflict_log = []   # 冲突更新日志(用于展示)
        self.next_number = 1     # 用于生成记忆编号 M1, M2, ...

    # ----- 写入部分 -----

    def ingest_session(self, session):
        """写入一个会话:调用 LLM 提取,再由代码管理。"""
        existing_text = self.format_active_memories()
        user_prompt = (
            "会话日期:" + session["date"] + "\n\n"
            "已有记忆(提取时请复用其中的 key):\n" + existing_text + "\n\n"
            "对话内容:\n" + format_dialogue(session) + "\n\n"
            "请输出 JSON 数组。"
        )

        items = None
        for attempt in range(2):
            raw_text = call_llm(EXTRACT_SYSTEM_PROMPT, user_prompt, max_tokens=3000)
            items = parse_json_list(raw_text)
            if items is not None:
                break
            print("    [提取结果无法解析为 JSON,重试]")
        if items is None:
            print("    [提取失败,本会话没有写入任何记忆]")
            return

        for item in items:
            if isinstance(item, dict):
                self.apply_item(item, session)

    def format_active_memories(self):
        """把当前有效的记忆格式化成文本,提取时提供给 LLM 参考。"""
        lines = []
        for record in self.records:
            if record["status"] == "active":
                lines.append("- %s = %s" % (record["key"], record["value"]))
        if len(lines) == 0:
            return "(暂无)"
        return "\n".join(lines)

    def apply_item(self, item, session):
        """把一条提取结果写入记忆库:丢弃临时、去重、冲突更新。"""
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        scope = str(item.get("scope", "long_term")).strip()
        kind = str(item.get("type", "project_fact")).strip()
        expires_at = str(item.get("expires_at", "")).strip()
        quote = str(item.get("evidence_quote", "")).strip()
        is_correction = item.get("is_correction", False)

        if key == "" or value == "":
            return

        # 规则 1:仅当前会话有效的临时指令,不进入长期记忆
        if scope == "session":
            self.rejected.append({
                "key": key, "value": value,
                "session_id": session["session_id"], "reason": "临时指令,仅当前会话有效",
            })
            return

        source = build_source(session, quote)
        old_record = self.find_active_by_key(key)

        # 规则 2:没有同 key 的有效记录 -> 直接新增
        if old_record is None:
            self.add_record(key, value, kind, expires_at, source, session)
            return

        # 规则 3:同 key 且值相同 -> 重复陈述,只增加计数并追加来源
        if self.normalize_value(old_record["value"]) == self.normalize_value(value):
            old_record["repeat_count"] = old_record["repeat_count"] + 1
            old_record["sources"].append(source)
            return

        # 规则 4:同 key 但值不同 -> 冲突更新。旧记录保留但标记为已被替代。
        new_record = self.add_record(key, value, kind, expires_at, source, session)
        old_record["status"] = "superseded"
        old_record["superseded_by"] = new_record["id"]
        new_record["supersedes"] = old_record["id"]
        self.conflict_log.append({
            "key": key,
            "old_id": old_record["id"], "old_value": old_record["value"],
            "new_id": new_record["id"], "new_value": new_record["value"],
            "session_id": session["session_id"],
            "is_correction": bool(is_correction),
        })

    def add_record(self, key, value, kind, expires_at, source, session):
        """创建一条新的记忆记录,并带上元数据。"""
        record = {
            "id": "M%d" % self.next_number,
            "key": key,
            "value": value,
            "type": kind,
            "status": "active",          # active 生效 / superseded 已被替代
            "expires_at": expires_at,    # 空字符串表示不过期
            "created_session": session["session_id"],
            "created_date": session["date"],
            "repeat_count": 1,
            "sources": [source],
            "supersedes": "",
            "superseded_by": "",
        }
        self.next_number = self.next_number + 1
        self.records.append(record)
        return record

    def find_active_by_key(self, key):
        """按 key 查找当前有效的记录,找不到返回 None。"""
        for record in self.records:
            if record["key"] == key and record["status"] == "active":
                return record
        return None

    def normalize_value(self, value):
        """把值规范化(去空格、转小写),用于判断两个值是否相同。"""
        return remove_spaces(value).lower()

    # ----- 召回部分 -----

    def is_expired(self, record):
        """判断记录是否已过期。日期格式 YYYY-MM-DD 可以直接按字符串比较大小。"""
        expires_at = record["expires_at"]
        if len(expires_at) != 10:
            return False
        return expires_at < TODAY

    def retrieve(self, question):
        """先过滤(替代/过期),再让 LLM 挑选相关记忆,最后拼成注入文本。"""
        candidates = []
        superseded_count = 0
        expired_count = 0
        for record in self.records:
            if record["status"] != "active":
                superseded_count = superseded_count + 1
            elif self.is_expired(record):
                expired_count = expired_count + 1
            else:
                candidates.append(record)

        info = {"filtered_superseded": superseded_count, "filtered_expired": expired_count}
        if len(candidates) == 0:
            return "", info

        # 构造"编号 + key: value"的候选清单,让 LLM 挑选
        menu_lines = []
        number = 1
        for record in candidates:
            menu_lines.append("%d. %s: %s" % (number, record["key"], record["value"]))
            number = number + 1
        user_prompt = "记忆清单:\n" + "\n".join(menu_lines) + "\n\n用户问题:" + question

        chosen_numbers = None
        for attempt in range(2):
            raw_text = call_llm(SELECT_SYSTEM_PROMPT, user_prompt, max_tokens=1500)
            chosen_numbers = parse_json_list(raw_text)
            if chosen_numbers is not None:
                break
        if chosen_numbers is None:
            chosen_numbers = []

        # 把编号换回记录,并拼成注入文本
        context_lines = []
        for number in chosen_numbers:
            if isinstance(number, int) and 1 <= number <= len(candidates):
                record = candidates[number - 1]
                context_lines.append("- %s: %s" % (record["key"], record["value"]))
        return "\n".join(context_lines), info

    # ----- 评估辅助 -----

    def get_stored_texts(self):
        """返回库中所有记忆(包含已被替代的旧记录)的 "key: value" 文本。"""
        texts = []
        for record in self.records:
            texts.append("%s: %s" % (record["key"], record["value"]))
        return texts

    def get_traceability(self):
        """统计有多少条记忆至少有一个"验证通过"的来源(原话确实出现在对应会话中)。"""
        traceable = 0
        for record in self.records:
            has_verified = False
            for source in record["sources"]:
                if source["verified"]:
                    has_verified = True
            if has_verified:
                traceable = traceable + 1
        return traceable, len(self.records)


# ---------------------------------------------------------------------
# 第 4 部分:评估
# ---------------------------------------------------------------------

def evaluate_extraction(strategy):
    """评估"记忆提取准确率"。
    共 11 个判断点:
        - 9 条应保存的稳定信息:被保存算对。
        - 2 条不应保存的临时指令:没有被保存算对。
    准确率 = 判断正确的个数 / 11。"""
    all_text = "\n".join(strategy.get_stored_texts())
    details = []
    correct_count = 0

    for fact in STABLE_FACTS:
        found = has_any_keyword(all_text, fact["keywords"])
        if found:
            correct_count = correct_count + 1
        details.append({"desc": fact["desc"], "expect": "应保存", "correct": found})

    for temp in TEMP_ITEMS:
        leaked = has_any_keyword(all_text, temp["keywords"])
        if not leaked:
            correct_count = correct_count + 1
        details.append({"desc": temp["desc"], "expect": "不应保存", "correct": not leaked})

    total = len(STABLE_FACTS) + len(TEMP_ITEMS)
    accuracy = correct_count / total
    return {"accuracy": accuracy, "correct": correct_count, "total": total, "details": details}


def evaluate_queries(strategy):
    """对每个问题执行召回,并判断是否通过。返回每题的详细结果。"""
    results = []
    for query in QUERIES:
        context, info = strategy.retrieve(query["question"])
        tokens = estimate_tokens(context)

        expect_hit = False
        if len(query["expect"]) > 0:
            expect_hit = has_any_keyword(context, query["expect"])
        forbid_hit = False
        if len(query["forbid"]) > 0:
            forbid_hit = has_any_keyword(context, query["forbid"])

        # 不同类型的题目,"通过"的标准不同
        if query["type"] == "recall":
            passed = expect_hit
        elif query["type"] == "conflict":
            passed = expect_hit and (not forbid_hit)
        else:
            passed = not forbid_hit

        results.append({
            "id": query["id"], "type": query["type"], "question": query["question"],
            "context": context, "tokens": tokens,
            "expect_hit": expect_hit, "forbid_hit": forbid_hit, "passed": passed,
            "filtered_superseded": info["filtered_superseded"],
            "filtered_expired": info["filtered_expired"],
        })
    return results


def compute_metrics(extraction, query_results, strategy):
    """根据提取评估和查询评估,计算六项指标。"""
    recall_total = 0
    recall_hit = 0
    conflict_total = 0
    conflict_pass = 0
    stale_total = 0
    stale_hit = 0
    token_sum = 0

    for result in query_results:
        token_sum = token_sum + result["tokens"]
        # 召回率:recall 和 conflict 两类题都需要找回关键记忆
        if result["type"] == "recall" or result["type"] == "conflict":
            recall_total = recall_total + 1
            if result["expect_hit"]:
                recall_hit = recall_hit + 1
        # 冲突处理正确率:只看 conflict 类题
        if result["type"] == "conflict":
            conflict_total = conflict_total + 1
            if result["passed"]:
                conflict_pass = conflict_pass + 1
        # 误召回率:conflict / expired / temp 三类题都有"不该出现"的内容
        if result["type"] == "conflict" or result["type"] == "expired" or result["type"] == "temp":
            stale_total = stale_total + 1
            if result["forbid_hit"]:
                stale_hit = stale_hit + 1

    traceable, memory_total = strategy.get_traceability()
    trace_rate = 0.0
    if memory_total > 0:
        trace_rate = traceable / memory_total

    store_tokens = 0
    for text in strategy.get_stored_texts():
        store_tokens = store_tokens + estimate_tokens(text)

    metrics = {
        "extraction_accuracy": extraction["accuracy"],
        "recall_rate": recall_hit / recall_total,
        "conflict_rate": conflict_pass / conflict_total,
        "stale_rate": stale_hit / stale_total,
        "avg_tokens": token_sum / len(query_results),
        "trace_rate": trace_rate,
        "memory_count": memory_total,
        "store_tokens": store_tokens,
    }
    return metrics


# ---------------------------------------------------------------------
# 第 5 部分:打印(让结果具有说明性)
# ---------------------------------------------------------------------

def print_dataset_overview():
    """打印实验数据的设计,让读者知道每段对话想检验什么。"""
    print_title("【实验数据说明】")
    print("当前日期(用于过期判断):%s" % TODAY)
    print("")
    for session in SESSIONS:
        print("会话 %s(%s),共 %d 轮发言" % (session["session_id"], session["date"], len(session["turns"])))
    print("")
    print("数据中埋入的各类信息:")
    print("  - 稳定偏好      :回答简洁、不用表情符号(其中'简洁'重复陈述了两次)")
    print("  - 项目事实      :项目 Atlas、FastAPI、团队 4 人(重复陈述两次)")
    print("  - 临时指令      :'这次把注释改英文'、'本次对话用英文回复'(不应长期记住)")
    print("  - 会过期的事件  :9 月 9 日项目评审(今天是 10 月 16 日,已过期)")
    print("  - 过时事实+纠正 :MySQL -> PostgreSQL(显式纠正);Python 3.10 -> 3.12;东京 -> 新加坡")
    print("")
    print("共 %d 个查询,措辞都和原话不同:" % len(QUERIES))
    for query in QUERIES:
        print("  %s [%s] %s" % (query["id"], query["type"], query["question"]))


def print_structured_store(strategy):
    """打印结构化记忆库里的全部记录,以及被拒绝的临时指令、冲突日志。"""
    print("")
    print("  结构化记忆库内容:")
    header = "  " + pad("编号", 5) + pad("key", 20) + pad("value", 22) + pad("状态", 12) + pad("过期日", 12) + pad("重复", 5) + "来源"
    print(header)
    for record in strategy.records:
        source_text = ""
        for source in record["sources"]:
            if source["verified"]:
                source_text = source_text + "%s#%d " % (source["session_id"], source["turn_index"])
            else:
                source_text = source_text + "%s#未验证 " % source["session_id"]
        expires_text = record["expires_at"]
        if expires_text == "":
            expires_text = "-"
        line = "  " + pad(record["id"], 5) + pad(record["key"], 20) + pad(short_text(record["value"], 20), 22)
        line = line + pad(record["status"], 12) + pad(expires_text, 12) + pad(str(record["repeat_count"]), 5) + source_text
        print(line)

    print("")
    print("  被拒绝保存的临时指令(%d 条):" % len(strategy.rejected))
    for item in strategy.rejected:
        print("    - [%s] %s = %s(%s)" % (item["session_id"], item["key"], item["value"], item["reason"]))

    print("")
    print("  冲突更新日志(%d 条):" % len(strategy.conflict_log))
    for item in strategy.conflict_log:
        print("    - [%s] %s:%s(%s) -> %s(%s) 显式纠正=%s" % (
            item["session_id"], item["key"], item["old_value"], item["old_id"],
            item["new_value"], item["new_id"], item["is_correction"]))


def print_store_overview(strategy):
    """按策略类型打印写入后的记忆内容。"""
    if isinstance(strategy, FullHistoryMemory):
        print("  共保存 %d 条原始发言,没有任何筛选。" % len(strategy.records))
    elif isinstance(strategy, SummaryMemory):
        print("  最终摘要如下:")
        print("  " + short_text(strategy.summary, 600))
    elif isinstance(strategy, StructuredMemory):
        print_structured_store(strategy)


def print_extraction_result(extraction):
    """打印提取准确率的逐项判断。"""
    print("")
    print("  记忆提取准确率:%d/%d = %.1f%%" % (extraction["correct"], extraction["total"], extraction["accuracy"] * 100))
    for detail in extraction["details"]:
        if detail["correct"]:
            mark = "对"
        else:
            mark = "错"
        print("    [%s] %s(%s)" % (mark, detail["desc"], detail["expect"]))


def print_query_results(query_results):
    """打印每个查询的召回结果。"""
    print("")
    print("  查询结果:")
    for result in query_results:
        if result["passed"]:
            mark = "通过"
        else:
            mark = "失败"
        reasons = ""
        if result["type"] == "recall" or result["type"] == "conflict":
            if not result["expect_hit"]:
                reasons = reasons + " 漏召回;"
        if result["forbid_hit"]:
            reasons = reasons + " 误召回旧值/临时/过期内容;"
        print("    %s [%s] %s -> %s  注入 %d Token%s" % (
            result["id"], result["type"], result["question"], mark, result["tokens"], reasons))
        if result["context"] == "":
            preview = "(没有注入任何记忆)"
        else:
            preview = short_text(result["context"], 90)
        print("        注入内容:" + preview)
        if result["filtered_superseded"] > 0 or result["filtered_expired"] > 0:
            print("        过滤掉:已被替代 %d 条,已过期 %d 条" % (result["filtered_superseded"], result["filtered_expired"]))


def print_version_chains(strategy):
    """展示结构化记忆的"版本链":新值替代了哪个旧值,各自来自哪句原话。"""
    print_title("【来源可追溯性示例】结构化记忆的版本链")
    printed = False
    for record in strategy.records:
        if record["supersedes"] != "":
            old_record = None
            for other in strategy.records:
                if other["id"] == record["supersedes"]:
                    old_record = other
            if old_record is None:
                continue
            printed = True
            print("key = %s" % record["key"])
            old_source = old_record["sources"][0]
            new_source = record["sources"][0]
            print("  旧值 %s(%s,已被替代)  来源 %s 第 %d 轮:%s" % (
                old_record["value"], old_record["id"], old_source["session_id"],
                old_source["turn_index"], short_text(old_source["quote"], 40)))
            print("  新值 %s(%s,当前生效)  来源 %s 第 %d 轮:%s" % (
                record["value"], record["id"], new_source["session_id"],
                new_source["turn_index"], short_text(new_source["quote"], 40)))
    if not printed:
        print("(没有发生冲突更新)")


def format_percent(value):
    """把小数格式化成百分比文本。"""
    return "%.1f%%" % (value * 100)


def print_comparison_table(all_results):
    """打印三种策略的指标对比表。"""
    print_title("【三种策略指标对比】")
    names = []
    for item in all_results:
        names.append(item["name"])

    rows = [
        ("记忆提取准确率(越高越好)", "extraction_accuracy", "percent"),
        ("关键记忆召回率(越高越好)", "recall_rate", "percent"),
        ("冲突处理正确率(越高越好)", "conflict_rate", "percent"),
        ("过期/旧值误召回率(越低越好)", "stale_rate", "percent"),
        ("每次查询平均注入 Token(越低越好)", "avg_tokens", "number"),
        ("来源可追溯率(越高越好)", "trace_rate", "percent"),
        ("保存的记忆条数", "memory_count", "integer"),
        ("记忆库总 Token", "store_tokens", "integer"),
    ]

    header = pad("指标", 38)
    for name in names:
        header = header + pad(name, 14)
    print(header)
    print("-" * 78)

    for row in rows:
        line = pad(row[0], 38)
        for item in all_results:
            value = item["metrics"][row[1]]
            if row[2] == "percent":
                text = format_percent(value)
            elif row[2] == "number":
                text = "%.1f" % value
            else:
                text = str(int(value))
            line = line + pad(text, 14)
        print(line)


def find_best(all_results, metric_key, higher_is_better):
    """找出某项指标上表现最好的策略名字(并列时返回第一个)。"""
    best_item = all_results[0]
    for item in all_results:
        value = item["metrics"][metric_key]
        best_value = best_item["metrics"][metric_key]
        if higher_is_better and value > best_value:
            best_item = item
        if (not higher_is_better) and value < best_value:
            best_item = item
    return best_item["name"]


def print_interpretation(all_results):
    """打印结论与解读。"""
    print_title("【结果解读】")
    print("说明:模型输出存在波动,请以上方实际数字为准;下面是对数字背后原因的解释。")
    print("")
    print("按指标看谁最好:")
    print("  提取准确率最高    :%s" % find_best(all_results, "extraction_accuracy", True))
    print("  关键记忆召回率最高:%s" % find_best(all_results, "recall_rate", True))
    print("  冲突处理最正确    :%s" % find_best(all_results, "conflict_rate", True))
    print("  误召回率最低      :%s" % find_best(all_results, "stale_rate", False))
    print("  注入 Token 最少   :%s" % find_best(all_results, "avg_tokens", False))
    print("  来源可追溯率最高  :%s" % find_best(all_results, "trace_rate", True))
    print("")
    print("原因分析:")
    print("  A.全部历史:什么都保留,所以召回率很高;但旧值(MySQL、东京)和新值同时被注入,")
    print("    模型看到的是互相矛盾的材料,冲突处理无从谈起;临时指令和过期事件也一并带入。")
    print("    注入 Token 随对话轮数线性增长,会话越多越贵。来源可追溯是它唯一的优势。")
    print("  B.仅摘要:Token 很省,新对话能覆盖旧摘要,所以可能处理好一部分冲突。")
    print("    但摘要是有损压缩:细节可能丢失,临时指令或过期事件可能被写进摘要,")
    print("    而且融合后的文字无法追溯到哪一句原话,出错后很难排查。")
    print("  C.结构化:临时指令在写入阶段就被拒绝;同 key 的新值替代旧值,旧值标记为 superseded;")
    print("    带日期的事件在召回时按过期日过滤;每条记忆保存原话、会话、轮次,可以追溯。")
    print("    召回时只注入与问题相关的少量条目,所以 Token 最少、噪声最小。")
    print("    代价:写入阶段需要额外的 LLM 调用,提取质量依赖提示词和模型;")
    print("    如果 key 命名不一致(如 db 与 database),冲突检测会失效,这是工程上要重点关注的点。")


# ---------------------------------------------------------------------
# 第 6 部分:主流程
# ---------------------------------------------------------------------

def check_llm_ready():
    """实验开始前先确认本地模型服务可用,避免跑到一半才发现连不上。"""
    answer = call_llm("你是助手。", "请只回复两个字母:OK", max_tokens=200)
    return answer != ""


def run_one_strategy(strategy):
    """对一个策略完整执行:写入全部会话 -> 评估提取 -> 评估查询 -> 计算指标。"""
    print_title("【策略】" + strategy.name)

    print("步骤 1:按时间顺序写入各会话")
    for session in SESSIONS:
        print("  写入会话 %s(%s)..." % (session["session_id"], session["date"]))
        strategy.ingest_session(session)

    print("")
    print("步骤 2:查看写入后的记忆")
    print_store_overview(strategy)

    print("")
    print("步骤 3:评估记忆提取")
    extraction = evaluate_extraction(strategy)
    print_extraction_result(extraction)

    print("")
    print("步骤 4:用不同措辞提问,评估召回与冲突处理")
    query_results = evaluate_queries(strategy)
    print_query_results(query_results)

    metrics = compute_metrics(extraction, query_results, strategy)
    result = {
        "name": strategy.name,
        "metrics": metrics,
        "extraction": extraction,
        "queries": query_results,
    }
    return result


def save_results(all_results, structured_strategy):
    """把全部结果和结构化记忆库保存成 JSON 文件,方便写书时引用数据。"""
    output = {
        "today": TODAY,
        "model": MODEL_NAME,
        "results": all_results,
        "structured_memory_records": structured_strategy.records,
        "structured_memory_rejected": structured_strategy.rejected,
        "structured_memory_conflicts": structured_strategy.conflict_log,
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print("")
    print("完整结果已保存到:%s" % RESULT_FILE)


def main():
    print_title("实验:记忆提取、召回与冲突更新")
    print("模型服务:%s  模型名:%s" % (BASE_URL, MODEL_NAME))

    if not check_llm_ready():
        print("")
        print("无法连接本地模型服务。请确认 llama.cpp 已启动,并检查环境变量 LLM_BASE_URL。")
        return

    print_dataset_overview()

    full_history = FullHistoryMemory()
    summary_memory = SummaryMemory()
    structured_memory = StructuredMemory()
    strategies = [full_history, summary_memory, structured_memory]

    all_results = []
    for strategy in strategies:
        result = run_one_strategy(strategy)
        all_results.append(result)

    print_version_chains(structured_memory)
    print_comparison_table(all_results)
    print_interpretation(all_results)
    save_results(all_results, structured_memory)


if __name__ == "__main__":
    main()
