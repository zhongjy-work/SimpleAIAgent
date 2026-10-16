# -*- coding: utf-8 -*-
"""
实验:关键词检索、向量检索与混合检索(以及重排序)的对比
=====================================================================
实验目的:
    在相同语料、相同问题集、相同生成模型下,对比四种检索策略的召回效果,
    并观察"重排序"能否提高最终证据质量。

四种检索策略:
    keyword :关键词检索(BM25,纯 Python 实现)
    vector  :向量检索(嵌入 + 余弦相似度)
    hybrid  :混合检索(把关键词和向量两路结果用 RRF 倒数排名融合)
    rerank  :混合检索 + 重排序(让 LLM 对混合检索的候选重新排序)

公平性设计:
    1. 所有策略最终都输出 Top-10 候选(关键词检索命中词太少时可能不足 10 条,会如实记录候选数)。
    2. 混合检索内部,关键词和向量各取前 20 名参与融合,最后截取 Top-10。
    3. 重排序只对混合检索的 Top-10 重新排序,不引入新候选,所以它的 Recall@10 与混合检索相同,
       差异只体现在排序质量和送入生成器的证据质量上。
    4. 生成阶段:每个策略都取自己的前 3 条材料,交给同一个模型、同一个提示词回答。

测试集(共 24 题,4 类各 6 题):
    精确编号 :错误码、配置项、变更单编号等,考察词面精确匹配
    语义表达 :换了说法的提问,和原文几乎没有共同关键词
    跨段落   :答案需要同时用到两段材料
    版本约束 :同一事项在 1.8 版和 2.0 版数值不同,提问中指定了版本

观测指标:
    Recall@K、MRR、nDCG@10、检索延迟、答案正确性、答案忠实度、引用正确率、单次查询成本。
    既给总体平均值,也按问题类型拆分。

运行方式:
    pip install openai
    python retrieval_experiment.py

可用环境变量(可选):
    LLM_BASE_URL  默认 http://127.0.0.1:8080/v1
    LLM_API_KEY   默认 sk-no-key-required
    LLM_MODEL     默认 gemma-4
    EMBED_MODE    auto(默认,服务嵌入可用就用,否则退回本地哈希嵌入)/ local / api
    USE_LLM       默认 1;设为 0 则只评估检索指标,跳过重排序和答案生成
"""

import json
import math
import os
import time

from openai import OpenAI

# ---------------------------------------------------------------------
# 第 0 部分:配置
# ---------------------------------------------------------------------

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.environ.get("LLM_API_KEY", "sk-no-key-required")
MODEL_NAME = os.environ.get("LLM_MODEL", "gemma-4")
EMBED_MODE = os.environ.get("EMBED_MODE", "auto")
USE_LLM = os.environ.get("USE_LLM", "1") == "1"

RESULT_FILE = "retrieval_experiment_result.json"

TOP_K_FINAL = 10     # 每个策略最终输出的候选数量(固定)
FUSE_N = 20          # 混合检索内部:每一路取前多少名参与融合
K_CONTEXT = 3        # 生成答案时,取排名前几的材料
RRF_K = 60           # RRF 融合公式里的常数
BM25_K1 = 1.5        # BM25 参数
BM25_B = 0.75        # BM25 参数
LOCAL_DIM = 1024     # 本地哈希嵌入的维度
API_BATCH = 8        # 调用嵌入接口时,每批文本数量

# 成本折算用的"示例单价"(元/千 Token)。本地模型没有真实计费,请按需要修改。
PRICE_INPUT_PER_1K = 0.001
PRICE_OUTPUT_PER_1K = 0.002

STRATEGIES = [
    ("keyword", "关键词检索(BM25)"),
    ("vector", "向量检索"),
    ("hybrid", "混合检索(RRF)"),
    ("rerank", "混合+重排序"),
]
QUESTION_TYPES = ["精确编号", "语义表达", "跨段落", "版本约束"]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600)


# ---------------------------------------------------------------------
# 第 1 部分:通用小工具
# ---------------------------------------------------------------------

def print_title(title):
    """打印醒目的章节标题。"""
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


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


def print_table(headers, rows, widths):
    """打印一个简单的文本表格。"""
    line = ""
    for i in range(len(headers)):
        line = line + pad(headers[i], widths[i])
    print(line)
    print("-" * 78)
    for row in rows:
        line = ""
        for i in range(len(row)):
            line = line + pad(str(row[i]), widths[i])
        print(line)


def percent(value):
    """把小数格式化成百分比文本;None 显示为 -。"""
    if value is None:
        return "-"
    return "%.1f%%" % (value * 100)


def three(value):
    """保留三位小数;None 显示为 -。"""
    if value is None:
        return "-"
    return "%.3f" % value


def short_text(text, limit):
    """把多行文本压成一行并截断。"""
    one_line = text.replace("\n", " / ")
    if len(one_line) > limit:
        return one_line[:limit] + "..."
    return one_line


def remove_spaces(text):
    """去掉所有空白字符,用于宽松地比较文字。"""
    result = ""
    for ch in text:
        if ch != " " and ch != "\n" and ch != "\t":
            result = result + ch
    return result


def estimate_tokens(text):
    """粗略估算 Token 数:非 ASCII 字符按 1 个,ASCII 每 4 个字符按 1 个。"""
    wide_count = 0
    ascii_count = 0
    for ch in text:
        if ord(ch) > 127:
            wide_count = wide_count + 1
        else:
            ascii_count = ascii_count + 1
    return wide_count + (ascii_count + 3) // 4


def normalize_vector(vector):
    """把向量缩放成长度 1,这样余弦相似度就等于点积。"""
    total = 0.0
    for x in vector:
        total = total + x * x
    norm = math.sqrt(total)
    if norm == 0:
        return vector
    result = []
    for x in vector:
        result.append(x / norm)
    return result


def dot(vector_a, vector_b):
    """计算两个向量的点积。"""
    total = 0.0
    for i in range(len(vector_a)):
        total = total + vector_a[i] * vector_b[i]
    return total


def cut_json_text(raw_text, open_char, close_char):
    """从模型输出中截取第一个 open_char 到最后一个 close_char 之间的内容。"""
    start = raw_text.find(open_char)
    end = raw_text.rfind(close_char)
    if start == -1 or end == -1 or end <= start:
        return ""
    return raw_text[start:end + 1]


def parse_json_list(raw_text):
    """把模型输出解析成列表,失败返回 None。"""
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


def call_llm(system_prompt, user_prompt, max_tokens):
    """调用本地模型。返回(文本, Token 用量)。失败重试 2 次,最终失败返回空文本。
    用量优先使用服务返回的 usage;没有就用估算值。"""
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=max_tokens)
            content = response.choices[0].message.content
            if content is None:
                content = ""
            content = content.strip()

            prompt_tokens = 0
            completion_tokens = 0
            if response.usage is not None:
                if response.usage.prompt_tokens is not None:
                    prompt_tokens = response.usage.prompt_tokens
                if response.usage.completion_tokens is not None:
                    completion_tokens = response.usage.completion_tokens
            if prompt_tokens == 0:
                prompt_tokens = estimate_tokens(system_prompt + user_prompt)
            if completion_tokens == 0:
                completion_tokens = estimate_tokens(content)
            return content, {"prompt": prompt_tokens, "completion": completion_tokens}
        except Exception as error:
            print("    [LLM 调用失败,第 %d 次] %s" % (attempt + 1, short_text(str(error), 100)))
            time.sleep(1)
    return "", {"prompt": 0, "completion": 0}


# ---------------------------------------------------------------------
# 第 2 部分:语料(47 段材料)与测试问题(24 题)
# 语料是虚构的"星云 Agent 平台"文档。每段材料有编号 P01~P47。
# ---------------------------------------------------------------------

def add_passage(passages, pid, title, version, text):
    """往语料里加一段材料。version 为空字符串表示不区分版本。"""
    passages.append({"id": pid, "title": title, "version": version, "text": text})


def build_corpus():
    """构造语料。除了正确证据,还放入了很多"长得像"的干扰材料。"""
    p = []
    # 错误码(P01~P06)
    add_passage(p, "P01", "错误码 E1003(工具数量超限)", "",
                "单个 Agent 挂载的工具数量超过上限时,注册请求会被拒绝并返回 E1003,需要先卸载不用的工具。")
    add_passage(p, "P02", "错误码 E2017(参数校验失败)", "",
                "工具调用的参数不符合参数模式时返回 E2017,响应中会带上出错的字段名,这类错误不会触发自动重试。")
    add_passage(p, "P03", "错误码 E3042(工具调用超时)", "",
                "工具在超时时间内没有返回结果时报 E3042,该错误属于可重试错误,按重试策略自动重试。")
    add_passage(p, "P04", "错误码 E4031(记忆写入冲突)", "",
                "同一次请求中,同一个键被同时写入了互相矛盾的值时返回 E4031,写入被拒绝,调用方需要合并后重新提交。")
    add_passage(p, "P05", "错误码 E5009(检索服务不可用)", "",
                "向量索引没有加载完成或索引文件损坏时返回 E5009,处理方法是在控制台执行索引重建,并等待状态变为就绪。")
    add_passage(p, "P06", "错误码 E6120(配额耗尽)", "",
                "当月 API 调用配额用完后继续调用会返回 E6120,可以升级套餐或等到下月 1 日配额重置。")
    # 配置项(P07~P10)
    add_passage(p, "P07", "配置项 tool.max_parallel", "",
                "控制单个 Agent 同时并发执行的工具调用数量,默认值为 4,取值范围 1 到 16。")
    add_passage(p, "P08", "配置项 memory.ttl_days", "",
                "控制记忆记录的默认过期天数,默认值为 90 天,设置为 0 表示永不过期。")
    add_passage(p, "P09", "配置项 retry.backoff_base_ms", "",
                "重试的初始等待时间,单位为毫秒,默认 2000,之后每次翻倍。")
    add_passage(p, "P10", "配置项 audit.retention_days", "",
                "控制审计日志的保留天数,取值范围 7 到 365,修改后对新产生的日志立即生效。")
    # 变更单(P11~P12)
    add_passage(p, "P11", "变更单 CHG-2041", "",
                "将工具调用的默认超时时间从 30 秒调整为 60 秒,自 2.0 版起生效,原因是长耗时检索工具经常超时。")
    add_passage(p, "P12", "变更单 CHG-2077", "",
                "将记忆召回的默认条数从 3 条调整为 5 条,自 2.0 版起生效,以提升多信息问题的覆盖率。")
    # 语义题与跨段落题用到的材料(P13~P25)
    add_passage(p, "P13", "会话历史压缩", "",
                "当会话历史超过模型窗口上限时,系统会把较早的对话总结成摘要,只保留最近几轮原文,避免上下文溢出。")
    add_passage(p, "P14", "代码执行沙箱", "",
                "代码执行工具运行在受限容器中,默认禁止访问外部网络,文件系统为只读,只有临时目录可以写入。")
    add_passage(p, "P15", "高风险操作审批", "",
                "涉及删除数据或对外付款的动作不会直接执行,需要管理员确认之后才会继续。")
    add_passage(p, "P16", "流式输出", "",
                "模型生成的内容通过事件流分段返回,调用方可以边接收边展示,首个片段通常在 1 秒内到达。")
    add_passage(p, "P17", "灰度发布", "",
                "新的 Agent 配置先对 10% 的流量生效,观察 30 分钟没有异常再扩大到全部流量。")
    add_passage(p, "P18", "熔断保护", "",
                "同一个工具连续 5 次调用失败后触发熔断,该工具会被暂停 60 秒,期间不再向外部服务发起请求。")
    add_passage(p, "P19", "熔断期间的行为", "",
                "熔断期间的调用不会真正执行,而是直接返回预设的降级结果,并在响应头中标记 X-Degraded 为 true。")
    add_passage(p, "P20", "重试策略", "",
                "可重试错误发生后按指数退避自动重试,每次等待时间翻倍,最多重试 3 次。")
    add_passage(p, "P21", "不会重试的错误", "",
                "参数校验失败和权限不足属于调用方的问题,平台不会自动重试,会直接把错误返回给调用方。")
    add_passage(p, "P22", "记忆冲突处理", "",
                "不同时间写入同一个键的新值时,旧记录会被标记为已替代,但不会被物理删除,保留用于审计和追溯。")
    add_passage(p, "P23", "审计日志保留策略", "",
                "企业版的审计日志默认保留 90 天,标准版保留 30 天,到期后自动清理。")
    add_passage(p, "P24", "网络白名单申请", "",
                "如果沙箱需要访问某个外部地址,需要提交白名单申请,由管理员审批,通过后 10 分钟内生效。")
    add_passage(p, "P25", "发布回滚", "",
                "发布后发现异常,可以一键回滚到上一个稳定版本,回滚操作通常在 2 分钟内完成。")
    # 版本约束题用到的材料(P26~P37):同一事项,1.8 版和 2.0 版数值不同
    add_passage(p, "P26", "工具调用默认超时 | 1.8 版", "1.8",
                "1.8 版中工具调用的默认超时时间为 30 秒,可以在工具配置中调整。")
    add_passage(p, "P27", "工具调用默认超时 | 2.0 版", "2.0",
                "2.0 版中工具调用的默认超时时间为 60 秒,可以在工具配置中调整。")
    add_passage(p, "P28", "单个 Agent 工具数量上限 | 1.8 版", "1.8",
                "1.8 版中单个 Agent 最多可以同时挂载 8 个工具。")
    add_passage(p, "P29", "单个 Agent 工具数量上限 | 2.0 版", "2.0",
                "2.0 版中单个 Agent 最多可以同时挂载 16 个工具。")
    add_passage(p, "P30", "记忆召回默认条数 | 1.8 版", "1.8",
                "1.8 版中记忆召回默认返回相关度最高的 3 条记录。")
    add_passage(p, "P31", "记忆召回默认条数 | 2.0 版", "2.0",
                "2.0 版中记忆召回默认返回相关度最高的 5 条记录。")
    add_passage(p, "P32", "会话空闲超时 | 1.8 版", "1.8",
                "1.8 版中会话空闲 15 分钟后会自动结束。")
    add_passage(p, "P33", "会话空闲超时 | 2.0 版", "2.0",
                "2.0 版中会话空闲 30 分钟后会自动结束。")
    add_passage(p, "P34", "单次请求最大输出 | 1.8 版", "1.8",
                "1.8 版中单次请求最多输出 2048 个 Token。")
    add_passage(p, "P35", "单次请求最大输出 | 2.0 版", "2.0",
                "2.0 版中单次请求最多输出 4096 个 Token。")
    add_passage(p, "P36", "速率限制 | 1.8 版", "1.8",
                "1.8 版中每个工作区每分钟最多允许 60 次请求。")
    add_passage(p, "P37", "速率限制 | 2.0 版", "2.0",
                "2.0 版中每个工作区每分钟最多允许 120 次请求。")
    # 干扰材料(P38~P47)
    add_passage(p, "P38", "工具注册", "", "每个工具需要提供名称、描述和参数模式,注册后才能被 Agent 调用。")
    add_passage(p, "P39", "模型路由", "", "平台可以按任务类型把请求路由到不同的模型,路由规则在控制台配置。")
    add_passage(p, "P40", "提示词模板", "", "提示词模板支持变量占位符,发布前需要通过预览校验。")
    add_passage(p, "P41", "知识库导入", "", "知识库支持上传 PDF、Word 和网页链接,导入后会自动切分并建立索引。")
    add_passage(p, "P42", "权限角色", "", "平台内置管理员、开发者和只读三种角色,角色权限不能自定义。")
    add_passage(p, "P43", "数据加密", "", "传输过程使用 TLS 1.2 以上协议,存储的数据使用 AES-256 加密。")
    add_passage(p, "P44", "监控告警", "", "平台提供调用量、错误率和延迟三类监控指标,可以配置邮件和 Webhook 告警。")
    add_passage(p, "P45", "SDK 安装", "", "Python SDK 通过 pip 安装,支持 Python 3.9 及以上版本。")
    add_passage(p, "P46", "数据备份", "", "控制台每天凌晨自动备份配置数据,备份保留 14 天。")
    add_passage(p, "P47", "账单查看", "", "账单每月 5 日生成,可以在控制台的费用页面下载。")
    return p


def make_question(qid, qtype, text, gold, expect, forbid):
    """创建一个测试问题。
    gold   :正确证据的材料编号列表(跨段落题有 2 个)。
    expect :答案必须满足的事实组。每组是若干"同义写法",答案里出现任意一个即算满足该组;所有组都满足才算答对。
    forbid :答案中不应出现的内容(例如旧版本的数值),出现则判错。"""
    return {"id": qid, "type": qtype, "text": text, "gold": gold, "expect": expect, "forbid": forbid}


def build_questions():
    """构造 24 个测试问题,并标注正确证据与期望答案。"""
    q = []
    # 精确编号(6)
    q.append(make_question("X-1", "精确编号", "E4031 是什么错误?", ["P04"], [["冲突"]], []))
    q.append(make_question("X-2", "精确编号", "E5009 应该怎么处理?", ["P05"], [["重建"]], []))
    q.append(make_question("X-3", "精确编号", "tool.max_parallel 的默认值是多少?", ["P07"], [["4"]], []))
    q.append(make_question("X-4", "精确编号", "变更单 CHG-2041 改了什么?", ["P11"], [["30"], ["60"]], []))
    q.append(make_question("X-5", "精确编号", "memory.ttl_days 控制什么?", ["P08"], [["过期"], ["90"]], []))
    q.append(make_question("X-6", "精确编号", "E6120 出现的原因是什么?", ["P06"], [["配额"]], []))
    # 语义表达(6)
    q.append(make_question("S-1", "语义表达", "对话太长放不下的时候,系统怎么处理前面的内容?", ["P13"], [["摘要", "总结"]], []))
    q.append(make_question("S-2", "语义表达", "某个外部服务一直出问题时,怎样避免平台不停地去请求它?", ["P18"], [["熔断", "暂停"]], []))
    q.append(make_question("S-3", "语义表达", "Agent 跑用户提交的代码时,怎么防止它乱联网、乱改文件?",
                           ["P14"], [["禁止", "不能访问", "无法访问"], ["只读"]], []))
    q.append(make_question("S-4", "语义表达", "危险的操作需要人来点头确认吗?", ["P15"], [["管理员", "确认"]], []))
    q.append(make_question("S-5", "语义表达", "回答可以边生成边显示给用户吗?", ["P16"], [["事件流", "流式", "分段"]], []))
    q.append(make_question("S-6", "语义表达", "上线新配置时怎样降低风险?", ["P17"], [["10%"], ["30 分钟", "30分钟"]], []))
    # 跨段落(6):答案需要两段材料
    q.append(make_question("C-1", "跨段落", "工具调用失败后怎么重试,哪些错误不会被重试?", ["P20", "P21"],
                           [["退避", "翻倍"], ["参数校验", "权限"]], []))
    q.append(make_question("C-2", "跨段落", "记忆键在不同时间被写入新值时旧记录怎么处理,同一次请求里写入矛盾的值又会报什么错误码?",
                           ["P22", "P04"], [["标记", "替代"], ["E4031"]], []))
    q.append(make_question("C-3", "跨段落", "企业版审计日志保留多久,保留期由哪个配置项调整?", ["P23", "P10"],
                           [["90"], ["audit.retention_days"]], []))
    q.append(make_question("C-4", "跨段落", "代码执行工具默认能不能访问外网,需要访问时怎么申请?", ["P14", "P24"],
                           [["禁止", "不能", "无法"], ["白名单"]], []))
    q.append(make_question("C-5", "跨段落", "熔断触发后调用会怎样,多久之后恢复?", ["P18", "P19"],
                           [["降级"], ["60 秒", "60秒"]], []))
    q.append(make_question("C-6", "跨段落", "灰度发布先放多少流量,出了问题怎么回滚?", ["P17", "P25"],
                           [["10%"], ["回滚"]], []))
    # 版本约束(6)
    q.append(make_question("V-1", "版本约束", "1.8 版的工具调用默认超时是多少秒?", ["P26"], [["30"]], ["60 秒", "60秒"]))
    q.append(make_question("V-2", "版本约束", "2.0 版单个 Agent 最多可以挂多少个工具?", ["P29"], [["16"]], ["8 个", "8个"]))
    q.append(make_question("V-3", "版本约束", "1.8 版记忆召回默认返回几条?", ["P30"], [["3 条", "3条"]], ["5 条", "5条"]))
    q.append(make_question("V-4", "版本约束", "2.0 版会话空闲多久会自动结束?", ["P33"], [["30 分钟", "30分钟"]], ["15 分钟", "15分钟"]))
    q.append(make_question("V-5", "版本约束", "1.8 版单次请求最多输出多少 Token?", ["P34"], [["2048"]], ["4096"]))
    q.append(make_question("V-6", "版本约束", "2.0 版每分钟最多允许多少次请求?", ["P37"], [["120"]], ["60 次", "60次"]))
    return q


# ---------------------------------------------------------------------
# 第 3 部分:三个检索组件(BM25、向量、RRF 融合)
# ---------------------------------------------------------------------

def passage_index_text(passage):
    """用于建索引的文本:标题 + 正文(标题里带有版本信息)。"""
    return passage["title"] + "\n" + passage["text"]


def is_word_char(ch):
    """是否属于"英文词"字符:ASCII 字母、数字,以及 _ . - 。"""
    if ord(ch) < 128:
        if ch.isalnum() or ch == "_" or ch == "." or ch == "-":
            return True
    return False


def add_word_tokens(word, tokens):
    """把一个英文词加入词元列表。含字母且带分隔符的词(如 tool.max_parallel)会同时加入整体和各部分。"""
    word = word.strip("._-")
    if word == "":
        return
    tokens.append(word)
    has_letter = False
    has_separator = False
    for ch in word:
        if ch.isalpha():
            has_letter = True
        if ch == "." or ch == "_" or ch == "-":
            has_separator = True
    if has_letter and has_separator:
        unified = word.replace(".", "_").replace("-", "_")
        for part in unified.split("_"):
            if part != "" and part != word:
                tokens.append(part)


def add_cjk_tokens(run, tokens):
    """把一段连续的中文拆成"字二元组"(只有一个字时保留单字)。"""
    if len(run) == 1:
        tokens.append(run)
        return
    for i in range(len(run) - 1):
        tokens.append(run[i:i + 2])


def tokenize(text):
    """分词:英文词/编号/数字按整体保留,中文按相邻两字切分,标点和空格作为分隔。"""
    tokens = []
    word = ""
    cjk_run = ""
    for ch in text.lower() + " ":
        if is_word_char(ch):
            if cjk_run != "":
                add_cjk_tokens(cjk_run, tokens)
                cjk_run = ""
            word = word + ch
        elif ord(ch) > 127 and ch.isalnum():
            if word != "":
                add_word_tokens(word, tokens)
                word = ""
            cjk_run = cjk_run + ch
        else:
            if word != "":
                add_word_tokens(word, tokens)
                word = ""
            if cjk_run != "":
                add_cjk_tokens(cjk_run, tokens)
                cjk_run = ""
    return tokens


class BM25Index:
    """关键词检索:BM25 打分。词频越高、词越稀有、文档越短,得分越高。"""

    def __init__(self, passages):
        self.doc_tf = []      # 每个文档的词频表
        self.doc_len = []     # 每个文档的词元数
        self.df = {}          # 每个词出现在多少个文档里
        total_len = 0
        for passage in passages:
            tokens = tokenize(passage_index_text(passage))
            tf = {}
            for token in tokens:
                if token in tf:
                    tf[token] = tf[token] + 1
                else:
                    tf[token] = 1
            for token in tf:
                if token in self.df:
                    self.df[token] = self.df[token] + 1
                else:
                    self.df[token] = 1
            self.doc_tf.append(tf)
            self.doc_len.append(len(tokens))
            total_len = total_len + len(tokens)
        self.doc_count = len(passages)
        self.avg_len = total_len / float(self.doc_count)

    def search(self, query, top_n):
        """返回得分大于 0 的前 top_n 个文档下标(按得分从高到低)。可能不足 top_n 个。"""
        unique_tokens = []
        seen = {}
        for token in tokenize(query):
            if token not in seen:
                seen[token] = True
                unique_tokens.append(token)

        scored = []
        for doc_index in range(self.doc_count):
            tf = self.doc_tf[doc_index]
            score = 0.0
            for token in unique_tokens:
                if token not in tf:
                    continue
                df = self.df[token]
                idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                freq = tf[token]
                norm = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * self.doc_len[doc_index] / self.avg_len)
                score = score + idf * freq * (BM25_K1 + 1.0) / norm
            if score > 0:
                scored.append([score, -doc_index])
        scored = sorted(scored, reverse=True)

        result = []
        for pair in scored[:top_n]:
            result.append(-pair[1])
        return result


class LocalHashEmbedder:
    """本地哈希嵌入:把文本拆成字符二元组并哈希到固定维度。不依赖服务,结果稳定。
    注意:它本质上仍是"字面相似",语义泛化能力远不如真正的嵌入模型。"""

    def __init__(self):
        self.name = "本地哈希嵌入(字面相似,语义能力弱)"
        self.dim = LOCAL_DIM
        self.stop_chars = "的了是在和与或及对为吗呢吧请我你么什"

    def gram_index(self, gram):
        """把字符片段哈希成向量下标。"""
        value = 0
        for ch in gram:
            value = (value * 131 + ord(ch)) % self.dim
        return value

    def embed_one(self, text):
        """把一段文本变成向量。"""
        chars = []
        for ch in text.lower():
            if ch.isalnum():
                chars.append(ch)
        vector = [0.0] * self.dim
        for i in range(len(chars)):
            ch = chars[i]
            if ch not in self.stop_chars:
                vector[self.gram_index(ch)] += 0.3
            if i + 1 < len(chars):
                next_ch = chars[i + 1]
                if ch not in self.stop_chars and next_ch not in self.stop_chars:
                    vector[self.gram_index(ch + next_ch)] += 1.0
        return normalize_vector(vector)

    def embed_texts(self, texts):
        """批量嵌入。"""
        vectors = []
        for text in texts:
            vectors.append(self.embed_one(text))
        return vectors


class ApiEmbedder:
    """通过 OpenAI SDK 调用 llama.cpp 的 /v1/embeddings。服务需要以 --embeddings 方式启动。"""

    def __init__(self):
        self.name = "服务嵌入(llama.cpp)"
        self.dim = 0

    def embed_texts(self, texts):
        """分批调用嵌入接口,返回归一化后的向量。"""
        vectors = []
        start = 0
        while start < len(texts):
            batch = []
            for text in texts[start:start + API_BATCH]:
                batch.append(text[:1500])
            response = client.embeddings.create(model=MODEL_NAME, input=batch)
            for item in response.data:
                vectors.append(normalize_vector(item.embedding))
            start = start + API_BATCH
        if len(vectors) > 0:
            self.dim = len(vectors[0])
        return vectors

    def check(self):
        """检查接口是否可用。"""
        try:
            vectors = self.embed_texts(["测试文本"])
            if len(vectors) != 1 or len(vectors[0]) == 0:
                return False
            return isinstance(vectors[0][0], float)
        except Exception as error:
            print("  [服务嵌入不可用] %s" % short_text(str(error), 120))
            return False


def choose_embedder():
    """选择向量检索使用的嵌入模型(整个实验中固定不变)。"""
    if EMBED_MODE != "local":
        print("检测本地服务的嵌入接口...")
        api_embedder = ApiEmbedder()
        if api_embedder.check():
            print("  可用,向量维度 %d" % api_embedder.dim)
            return api_embedder
        print("  不可用。提示:请以 llama-server --embeddings --pooling mean 方式启动。")
        if EMBED_MODE == "api":
            return None
    return LocalHashEmbedder()


def vector_search(vectors, query_vector, top_n):
    """向量检索:计算余弦相似度,返回前 top_n 个文档下标。"""
    scored = []
    for i in range(len(vectors)):
        scored.append([dot(query_vector, vectors[i]), -i])
    scored = sorted(scored, reverse=True)
    result = []
    for pair in scored[:top_n]:
        result.append(-pair[1])
    return result


def rrf_fuse(rankings, top_n):
    """RRF 倒数排名融合:每个文档的得分 = 各路排名的 1/(RRF_K + 名次) 之和。
    不需要对不同检索器的分数做归一化,只用名次,简单稳健。"""
    scores = {}
    for ranking in rankings:
        for position in range(len(ranking)):
            doc_index = ranking[position]
            add = 1.0 / (RRF_K + position + 1)
            if doc_index in scores:
                scores[doc_index] = scores[doc_index] + add
            else:
                scores[doc_index] = add
    scored = []
    for doc_index in scores:
        scored.append([scores[doc_index], -doc_index])
    scored = sorted(scored, reverse=True)
    result = []
    for pair in scored[:top_n]:
        result.append(-pair[1])
    return result


# ---------------------------------------------------------------------
# 第 4 部分:重排序、答案生成、忠实度评审(都使用本地 LLM)
# ---------------------------------------------------------------------

RERANK_SYSTEM = (
    "你是检索结果重排序器。给定一个问题和若干候选材料,请按'对回答该问题的帮助程度'从高到低排序。"
    "如果回答问题需要多条材料,请把这些材料都排在前面。"
    "只输出 JSON 数组,元素是候选编号,例如 [3,1,2]。不要输出其他文字。"
)

GENERATE_SYSTEM = (
    "你是知识库问答助手。只能依据给定材料回答,材料里没有的内容不要编造。"
    "回答要简洁。每个事实后面用方括号标注所依据的材料编号,例如 [P12];用到多条材料就分别标注。"
    "如果材料不足以回答,只回答'材料不足'。"
)

JUDGE_SYSTEM = (
    "你是严格的事实核查员。请判断【回答】中的每一条事实陈述是否都能在【材料】中找到依据。"
    "材料之外的内容、编造的数字或编号,都算没有依据。"
    '只输出 JSON,格式为 {"faithful": true} 或 {"faithful": false}。'
)


def llm_rerank(question_text, candidates):
    """让 LLM 对候选材料重新排序(列表式重排,一次调用排完所有候选)。
    返回(新的候选顺序下标列表, Token 用量, 是否成功解析)。解析失败则保持原顺序。"""
    lines = []
    for i in range(len(candidates)):
        lines.append("[%d] %s:%s" % (i + 1, candidates[i]["title"], candidates[i]["text"]))
    user_prompt = "问题:" + question_text + "\n\n候选材料:\n" + "\n".join(lines) + "\n\n请输出排序后的编号 JSON 数组。"
    text, usage = call_llm(RERANK_SYSTEM, user_prompt, 600)
    order = parse_json_list(text)

    new_order = []
    used = {}
    parsed_ok = False
    if order is not None:
        for value in order:
            if isinstance(value, int) and 1 <= value <= len(candidates) and value not in used:
                used[value] = True
                new_order.append(value - 1)
                parsed_ok = True
    # 模型没有列出的候选,按原顺序补在后面
    for i in range(len(candidates)):
        if (i + 1) not in used:
            new_order.append(i)
    return new_order, usage, parsed_ok


def format_context(passages):
    """把材料格式化成提示词里的"材料区"。"""
    if len(passages) == 0:
        return "(没有检索到材料)"
    lines = []
    for passage in passages:
        lines.append("[%s] %s:%s" % (passage["id"], passage["title"], passage["text"]))
    return "\n".join(lines)


def generate_answer(question_text, context_passages):
    """让 LLM 依据材料回答问题。返回(答案文本, Token 用量)。"""
    user_prompt = "材料:\n" + format_context(context_passages) + "\n\n问题:" + question_text
    return call_llm(GENERATE_SYSTEM, user_prompt, 800)


def judge_faithful(context_passages, answer):
    """让 LLM 判断答案是否忠实于材料。返回 1 / 0;无法判断返回 None。"""
    user_prompt = "【材料】\n" + format_context(context_passages) + "\n\n【回答】\n" + answer
    text, usage = call_llm(JUDGE_SYSTEM, user_prompt, 300)
    json_text = cut_json_text(text, "{", "}")
    if json_text == "":
        return None
    try:
        data = json.loads(json_text)
    except Exception:
        return None
    value = data.get("faithful")
    if value is True:
        return 1
    if value is False:
        return 0
    return None


# ---------------------------------------------------------------------
# 第 5 部分:指标计算
# ---------------------------------------------------------------------

def recall_at_k(ranked_ids, gold, k):
    """Recall@K:前 K 个结果里找到了多少比例的正确证据。"""
    found = 0
    for gold_id in gold:
        if gold_id in ranked_ids[:k]:
            found = found + 1
    return found / float(len(gold))


def first_gold_rank(ranked_ids, gold):
    """第一个正确证据的名次(从 1 开始),没找到返回 0。"""
    for i in range(len(ranked_ids)):
        if ranked_ids[i] in gold:
            return i + 1
    return 0


def ndcg_at_k(ranked_ids, gold, k):
    """nDCG@K:正确证据排得越靠前,得分越高。相关度只有 0/1。"""
    dcg = 0.0
    for i in range(min(k, len(ranked_ids))):
        if ranked_ids[i] in gold:
            dcg = dcg + 1.0 / math.log2(i + 2)
    ideal = 0.0
    for i in range(min(k, len(gold))):
        ideal = ideal + 1.0 / math.log2(i + 2)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def check_answer(answer, question):
    """答案正确性(规则判断):每个事实组至少出现一种写法,且不出现禁止内容。"""
    clean_answer = remove_spaces(answer).lower()
    for group in question["expect"]:
        group_ok = False
        for option in group:
            if remove_spaces(option).lower() in clean_answer:
                group_ok = True
        if not group_ok:
            return 0
    for bad in question["forbid"]:
        if remove_spaces(bad).lower() in clean_answer:
            return 0
    return 1


def remove_citations(answer):
    """去掉答案里方括号包住的引用标记,例如 [P12]。"""
    result = ""
    inside = False
    for ch in answer:
        if ch == "[":
            inside = True
        elif ch == "]":
            inside = False
        elif not inside:
            result = result + ch
    return result


def extract_citations(answer):
    """提取答案中引用的材料编号(形如 [P12] 或 [P12, P13])。"""
    cites = []
    position = 0
    while True:
        left = answer.find("[", position)
        if left == -1:
            break
        right = answer.find("]", left)
        if right == -1:
            break
        content = answer[left + 1:right]
        content = content.replace(",", " ").replace(",", " ").replace("、", " ").replace(";", " ")
        for token in content.split():
            token = token.strip()
            if len(token) > 1 and token[0] == "P" and token[1:].isdigit():
                if token not in cites:
                    cites.append(token)
        position = right + 1
    return cites


def citation_precision(cites, gold):
    """引用正确率:答案引用的编号里,有多少比例确实是正确证据。没有引用则记 0。"""
    if len(cites) == 0:
        return 0.0
    right = 0
    for cite in cites:
        if cite in gold:
            right = right + 1
    return right / float(len(cites))


def numbers_supported(answer, context_text, question_text):
    """数值有据率(规则判断):答案里每个含数字的词(如 3.12、E4031、10)
    都必须出现在材料或问题里。全部有据返回 1,否则返回 0。这是对忠实度的一个粗略补充。"""
    body = remove_spaces(remove_citations(answer))
    support = remove_spaces(context_text + question_text)
    buffer = ""
    tokens = []
    for ch in body + " ":
        if ord(ch) < 128 and (ch.isalnum() or ch == "."):
            buffer = buffer + ch
        else:
            if buffer != "":
                tokens.append(buffer.strip("."))
                buffer = ""
    for token in tokens:
        has_digit = False
        for ch in token:
            if ch.isdigit():
                has_digit = True
        if has_digit and token not in support:
            return 0
    return 1


# ---------------------------------------------------------------------
# 第 6 部分:运行实验
# ---------------------------------------------------------------------

def retrieve_all_strategies(question, corpus, bm25, vectors, embedder, llm_ready):
    """对一个问题运行全部检索策略,返回每个策略的"排序结果、延迟、检索阶段 Token 用量"。
    延迟计时说明:
        关键词延迟 = BM25 检索时间
        向量延迟   = 问题嵌入 + 相似度计算时间
        混合延迟   = 关键词 + 向量 + 融合时间(两路按顺序执行,不并行)
        重排延迟   = 混合延迟 + LLM 重排时间"""
    results = {}
    empty_usage = {"prompt": 0, "completion": 0}

    start = time.time()
    keyword_all = bm25.search(question["text"], FUSE_N)
    keyword_ms = (time.time() - start) * 1000.0

    start = time.time()
    query_vector = embedder.embed_texts([question["text"]])[0]
    vector_all = vector_search(vectors, query_vector, FUSE_N)
    vector_ms = (time.time() - start) * 1000.0

    start = time.time()
    hybrid_top = rrf_fuse([keyword_all, vector_all], TOP_K_FINAL)
    fuse_ms = (time.time() - start) * 1000.0

    results["keyword"] = {"indexes": keyword_all[:TOP_K_FINAL], "latency_ms": keyword_ms, "usage": empty_usage}
    results["vector"] = {"indexes": vector_all[:TOP_K_FINAL], "latency_ms": vector_ms, "usage": empty_usage}
    results["hybrid"] = {"indexes": hybrid_top, "latency_ms": keyword_ms + vector_ms + fuse_ms, "usage": empty_usage}

    if llm_ready:
        candidates = []
        for doc_index in hybrid_top:
            candidates.append(corpus[doc_index])
        start = time.time()
        new_order, usage, parsed_ok = llm_rerank(question["text"], candidates)
        rerank_ms = (time.time() - start) * 1000.0
        reranked = []
        for position in new_order:
            reranked.append(hybrid_top[position])
        results["rerank"] = {"indexes": reranked, "latency_ms": keyword_ms + vector_ms + fuse_ms + rerank_ms,
                             "usage": usage, "rerank_parsed": parsed_ok}
    return results


def score_one(question, corpus, ranked_indexes, latency_ms, retrieval_usage, llm_ready):
    """对一个问题、一个策略的检索结果打分;如果 LLM 可用,再生成答案并评估答案质量。"""
    ranked_ids = []
    for doc_index in ranked_indexes:
        ranked_ids.append(corpus[doc_index]["id"])

    record = {
        "qid": question["id"], "type": question["type"], "ranked": ranked_ids,
        "rec1": recall_at_k(ranked_ids, question["gold"], 1),
        "rec3": recall_at_k(ranked_ids, question["gold"], 3),
        "rec5": recall_at_k(ranked_ids, question["gold"], 5),
        "rec10": recall_at_k(ranked_ids, question["gold"], 10),
        "mrr": 0.0, "ndcg": ndcg_at_k(ranked_ids, question["gold"], TOP_K_FINAL),
        "first_rank": first_gold_rank(ranked_ids, question["gold"]),
        "latency_ms": latency_ms, "cand_count": len(ranked_ids),
        "answer": "", "correct": None, "faithful": None, "numbers_ok": None,
        "cite_prec": None, "cites": [], "prompt_tokens": 0, "completion_tokens": 0,
        "tokens": 0, "cost": 0.0,
    }
    if record["first_rank"] > 0:
        record["mrr"] = 1.0 / record["first_rank"]

    prompt_tokens = retrieval_usage["prompt"]
    completion_tokens = retrieval_usage["completion"]

    if llm_ready:
        context_passages = []
        for doc_index in ranked_indexes[:K_CONTEXT]:
            context_passages.append(corpus[doc_index])
        answer, usage = generate_answer(question["text"], context_passages)
        prompt_tokens = prompt_tokens + usage["prompt"]
        completion_tokens = completion_tokens + usage["completion"]

        record["answer"] = answer
        record["correct"] = check_answer(answer, question)
        cites = extract_citations(answer)
        record["cites"] = cites
        record["cite_prec"] = citation_precision(cites, question["gold"])
        context_text = format_context(context_passages)
        record["numbers_ok"] = numbers_supported(answer, context_text, question["text"])

        # 忠实度:回答"材料不足"没有做任何事实陈述,直接算忠实;否则请 LLM 评审(评审用量不计入查询成本)
        if "材料不足" in answer and len(answer) < 20:
            record["faithful"] = 1
        else:
            record["faithful"] = judge_faithful(context_passages, answer)

    record["prompt_tokens"] = prompt_tokens
    record["completion_tokens"] = completion_tokens
    record["tokens"] = prompt_tokens + completion_tokens
    record["cost"] = prompt_tokens / 1000.0 * PRICE_INPUT_PER_1K + completion_tokens / 1000.0 * PRICE_OUTPUT_PER_1K
    return record


def run_experiment(corpus, questions, bm25, vectors, embedder, llm_ready):
    """对全部问题、全部策略跑一遍,返回 {策略键: [每题记录]}。"""
    records = {}
    for strategy_key, strategy_name in STRATEGIES:
        records[strategy_key] = []

    for question in questions:
        print("  处理 %s [%s] %s" % (question["id"], question["type"], question["text"]))
        retrieved = retrieve_all_strategies(question, corpus, bm25, vectors, embedder, llm_ready)
        for strategy_key, strategy_name in STRATEGIES:
            if strategy_key not in retrieved:
                continue
            item = retrieved[strategy_key]
            record = score_one(question, corpus, item["indexes"], item["latency_ms"], item["usage"], llm_ready)
            if strategy_key == "rerank":
                record["rerank_parsed"] = item["rerank_parsed"]
            records[strategy_key].append(record)
    return records


# ---------------------------------------------------------------------
# 第 7 部分:汇总与打印
# ---------------------------------------------------------------------

def mean_of(records, field, question_type):
    """计算某个字段的平均值。question_type 为空字符串表示全部题目。值为 None 的记录会被跳过。"""
    total = 0.0
    count = 0
    for record in records:
        if question_type != "" and record["type"] != question_type:
            continue
        value = record[field]
        if value is None:
            continue
        total = total + value
        count = count + 1
    if count == 0:
        return None
    return total / count


def active_strategies(records):
    """返回本次实际运行了的策略列表(没有 LLM 时没有重排序)。"""
    result = []
    for strategy_key, strategy_name in STRATEGIES:
        if len(records[strategy_key]) > 0:
            result.append([strategy_key, strategy_name])
    return result


def print_setup(corpus, questions, embedder, llm_ready):
    """打印实验设置。"""
    print_title("【实验设置】")
    print("语料:%d 段材料(其中 %d 段是 1.8/2.0 版本成对材料,另有干扰材料)" % (len(corpus), 12))
    print("问题:%d 个" % len(questions))
    for question_type in QUESTION_TYPES:
        count = 0
        for question in questions:
            if question["type"] == question_type:
                count = count + 1
        print("  %s:%d 题" % (question_type, count))
    print("")
    print("固定条件:")
    print("  候选数量      :每个策略最终输出 Top-%d;混合检索内部每路取前 %d 名参与融合" % (TOP_K_FINAL, FUSE_N))
    print("  生成所用材料  :每个策略各取自己的前 %d 条,使用同一模型、同一提示词" % K_CONTEXT)
    print("  向量检索嵌入  :%s" % embedder.name)
    if llm_ready:
        print("  LLM           :%s(重排序、答案生成、忠实度评审)" % MODEL_NAME)
    else:
        print("  LLM           :未启用,只评估检索指标(跳过重排序和答案质量指标)")
    print("  成本折算      :输入 %.4f 元/千Token,输出 %.4f 元/千Token(示例单价,可在脚本顶部修改)" % (PRICE_INPUT_PER_1K, PRICE_OUTPUT_PER_1K))


def print_retrieval_table(records):
    """表 1:检索质量。"""
    print("")
    print("表 1:检索质量(全部 24 题的平均值)")
    rows = []
    for strategy_key, strategy_name in active_strategies(records):
        r = records[strategy_key]
        rows.append([strategy_name, percent(mean_of(r, "rec1", "")), percent(mean_of(r, "rec3", "")),
                     percent(mean_of(r, "rec5", "")), percent(mean_of(r, "rec10", "")),
                     three(mean_of(r, "mrr", "")), three(mean_of(r, "ndcg", ""))])
    print_table(["策略", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "nDCG@10"],
                rows, [20, 10, 10, 10, 11, 8, 10])


def print_efficiency_table(records):
    """表 2:检索延迟、候选数量、单次查询成本。"""
    print("")
    print("表 2:效率与成本(每次查询平均值;Token 与成本包含重排序和答案生成,不含忠实度评审)")
    rows = []
    for strategy_key, strategy_name in active_strategies(records):
        r = records[strategy_key]
        rows.append([strategy_name, "%.1f" % mean_of(r, "cand_count", ""), "%.1f" % mean_of(r, "latency_ms", ""),
                     "%.0f" % mean_of(r, "prompt_tokens", ""), "%.0f" % mean_of(r, "completion_tokens", ""),
                     "%.5f" % mean_of(r, "cost", "")])
    print_table(["策略", "平均候选数", "检索延迟ms", "输入Token", "输出Token", "成本(元)"],
                rows, [20, 12, 12, 12, 12, 10])


def print_generation_table(records):
    """表 3:最终答案质量。"""
    print("")
    print("表 3:最终答案质量(全部 24 题的平均值)")
    rows = []
    for strategy_key, strategy_name in active_strategies(records):
        r = records[strategy_key]
        rows.append([strategy_name, percent(mean_of(r, "correct", "")), percent(mean_of(r, "faithful", "")),
                     percent(mean_of(r, "numbers_ok", "")), percent(mean_of(r, "cite_prec", ""))])
    print_table(["策略", "答案正确率", "答案忠实度", "数值有据率", "引用正确率"], rows, [20, 12, 12, 12, 12])
    print("  说明:正确率=规则判断(必需事实都出现且无旧版数值);忠实度=LLM 评审;")
    print("        数值有据率=答案里的数字/编号都出现在材料中;引用正确率=答案引用的编号里属于正确证据的比例。")


def print_type_table(records, title, field, formatter):
    """按问题类型拆分:行是题型,列是策略。"""
    print("")
    print(title)
    strategies = active_strategies(records)
    headers = ["题型"]
    widths = [12]
    for item in strategies:
        headers.append(item[1])
        widths.append(18)
    rows = []
    for question_type in QUESTION_TYPES:
        row = [question_type]
        for item in strategies:
            value = mean_of(records[item[0]], field, question_type)
            if formatter == "percent":
                row.append(percent(value))
            else:
                row.append(three(value))
        rows.append(row)
    print_table(headers, rows, widths)


def print_rank_grid(records, questions):
    """逐题明细:第一个正确证据的名次,以及答案是否正确。"""
    print("")
    print("逐题明细。名次=第一个正确证据排第几(-表示前 %d 名内没有);答案列 Y=答对 N=答错。" % TOP_K_FINAL)
    print("列名:K=关键词  V=向量  H=混合  R=混合+重排")
    headers = ["题号", "题型", "名次K", "名次V", "名次H", "名次R", "答案K", "答案V", "答案H", "答案R"]
    widths = [7, 10, 7, 7, 7, 7, 7, 7, 7, 7]
    keys = ["keyword", "vector", "hybrid", "rerank"]
    rows = []
    for q_index in range(len(questions)):
        row = [questions[q_index]["id"], questions[q_index]["type"]]
        for key in keys:
            if len(records[key]) == 0:
                row.append("n/a")
                continue
            rank = records[key][q_index]["first_rank"]
            if rank == 0:
                row.append("-")
            else:
                row.append(str(rank))
        for key in keys:
            if len(records[key]) == 0 or records[key][q_index]["correct"] is None:
                row.append("n/a")
                continue
            if records[key][q_index]["correct"] == 1:
                row.append("Y")
            else:
                row.append("N")
        rows.append(row)
    print_table(headers, rows, widths)


def print_rerank_effect(records, questions):
    """观察重排序相对混合检索的变化:哪些题变好、变差、没变。"""
    if len(records["rerank"]) == 0:
        return
    print("")
    print("重排序效果:对比'混合检索'与'混合+重排序'(Top-10 候选相同,只有顺序不同)")
    better = []
    worse = []
    same = 0
    for q_index in range(len(questions)):
        before = records["hybrid"][q_index]["mrr"]
        after = records["rerank"][q_index]["mrr"]
        if after > before:
            better.append(questions[q_index]["id"])
        elif after < before:
            worse.append(questions[q_index]["id"])
        else:
            same = same + 1
    print("  MRR 变好的题:%d 个 %s" % (len(better), " ".join(better)))
    print("  MRR 变差的题:%d 个 %s" % (len(worse), " ".join(worse)))
    print("  没有变化的题:%d 个" % same)

    failed = 0
    for record in records["rerank"]:
        if not record.get("rerank_parsed", True):
            failed = failed + 1
    print("  重排序输出无法解析、退回原顺序的题:%d 个" % failed)

    hybrid_correct = mean_of(records["hybrid"], "correct", "")
    rerank_correct = mean_of(records["rerank"], "correct", "")
    print("  答案正确率:混合检索 %s -> 混合+重排序 %s" % (percent(hybrid_correct), percent(rerank_correct)))
    hybrid_latency = mean_of(records["hybrid"], "latency_ms", "")
    rerank_latency = mean_of(records["rerank"], "latency_ms", "")
    print("  平均检索延迟:%.1f ms -> %.1f ms(重排序额外增加 %.1f ms)" % (hybrid_latency, rerank_latency, rerank_latency - hybrid_latency))


def print_case_studies(records, questions, corpus):
    """每种题型展示一道题:各策略的 Top-3、答案和判定,帮助理解数字背后的原因。"""
    print("")
    print("典型案例(每种题型各一题;Top-3 里带 * 的是正确证据):")
    shown = {}
    for q_index in range(len(questions)):
        question = questions[q_index]
        if question["type"] in shown:
            continue
        shown[question["type"]] = True
        print("")
        print("  [%s] %s %s  正确证据:%s" % (question["type"], question["id"], question["text"], " ".join(question["gold"])))
        for strategy_key, strategy_name in active_strategies(records):
            record = records[strategy_key][q_index]
            top3 = []
            for pid in record["ranked"][:3]:
                if pid in question["gold"]:
                    top3.append(pid + "*")
                else:
                    top3.append(pid)
            line = "    %s Top3:%s" % (pad(strategy_name, 18), " ".join(top3))
            print(line)
            if record["answer"] != "":
                if record["correct"] == 1:
                    mark = "答对"
                else:
                    mark = "答错"
                print("      答案(%s):%s" % (mark, short_text(record["answer"], 70)))


def find_best_by_type(records, field, higher_is_better):
    """按题型找出某指标最好的策略(并列时都列出)。"""
    lines = []
    for question_type in QUESTION_TYPES:
        best_value = None
        best_names = []
        for strategy_key, strategy_name in active_strategies(records):
            value = mean_of(records[strategy_key], field, question_type)
            if value is None:
                continue
            if best_value is None:
                best_value = value
                best_names = [strategy_name]
            elif (higher_is_better and value > best_value + 1e-9) or ((not higher_is_better) and value < best_value - 1e-9):
                best_value = value
                best_names = [strategy_name]
            elif abs(value - best_value) <= 1e-9:
                best_names.append(strategy_name)
        lines.append("  %s:%s" % (pad(question_type, 10), "、".join(best_names)))
    return lines


def print_interpretation(records, embedder):
    """打印结论与解读。"""
    print_title("【结果解读】")
    print("说明:题量较小(24 题、47 段材料),数值有波动,更适合作为机制演示;请以上方表格为准。")
    print("")
    print("按题型看,MRR 最高的策略:")
    for line in find_best_by_type(records, "mrr", True):
        print(line)
    print("")
    print("一般规律(请对照你的实际结果验证):")
    print("1. 精确编号:关键词检索通常最稳,因为错误码、配置项名称必须字面匹配;")
    print("   向量检索容易把'E4031'与'E4xxx'其他错误码混在一起。")
    print("2. 语义表达:提问与原文几乎没有共同词,关键词检索会明显吃亏;向量检索的表现取决于嵌入模型的语义能力。")
    if embedder.name.startswith("本地哈希"):
        print("   (本次使用的是本地哈希嵌入,它只能做字面相似,所以向量检索在语义题上的优势不会充分体现;")
        print("    如果让 llama.cpp 以 --embeddings 方式提供嵌入,再运行一次,对比会更有说明性。)")
    print("3. 跨段落:答案需要两段材料,Recall@K 取的是'找到的证据比例',只找到一段就只能得到一半;")
    print("   混合检索把两路结果合并,通常比单一路径更容易凑齐两段证据。")
    print("4. 版本约束:'1.8 版'和'2.0 版'的材料几乎一模一样,只差版本号和数值;")
    print("   向量检索很难区分版本,关键词检索能利用'1.8'这个词;这类问题更稳妥的做法是在检索时增加元数据过滤。")
    print("5. 重排序:它只改变前 10 个候选的顺序,不会增加召回(Recall@10 与混合检索相同);")
    print("   收益体现在 Recall@1/3、MRR、nDCG,以及送入生成器的前 3 条材料质量;代价是多一次 LLM 调用的延迟和 Token。")
    print("   如果候选里本来就没有正确证据,重排序无能为力;候选质量决定了重排序的上限。")
    print("6. 答案质量:检索是上游,答案正确率、忠实度和引用正确率受它影响,但也受生成模型本身是否照材料作答影响。")
    print("   所以评估 RAG 时,应分别看检索指标和答案指标,才能定位问题出在哪一层。")


def save_results(records, questions, embedder, llm_ready):
    """把全部结果保存成 JSON,方便写书时引用。"""
    summary = {}
    for strategy_key, strategy_name in active_strategies(records):
        r = records[strategy_key]
        item = {"overall": {}, "by_type": {}}
        fields = ["rec1", "rec3", "rec5", "rec10", "mrr", "ndcg", "latency_ms", "cand_count",
                  "prompt_tokens", "completion_tokens", "cost", "correct", "faithful", "numbers_ok", "cite_prec"]
        for field in fields:
            item["overall"][field] = mean_of(r, field, "")
        for question_type in QUESTION_TYPES:
            item["by_type"][question_type] = {}
            for field in fields:
                item["by_type"][question_type][field] = mean_of(r, field, question_type)
        summary[strategy_key] = item

    output = {
        "config": {"top_k_final": TOP_K_FINAL, "fuse_n": FUSE_N, "k_context": K_CONTEXT, "rrf_k": RRF_K,
                   "embedder": embedder.name, "llm_enabled": llm_ready, "model": MODEL_NAME},
        "questions": questions, "summary": summary, "records": records,
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print("")
    print("完整结果已保存到:%s" % RESULT_FILE)


# ---------------------------------------------------------------------
# 第 8 部分:主流程
# ---------------------------------------------------------------------

def check_llm_ready():
    """确认本地 LLM 可用。"""
    if not USE_LLM:
        return False
    text, usage = call_llm("你是助手。", "请只回复两个字母:OK", 200)
    return text != ""


def check_gold_ids(corpus, questions):
    """检查问题标注的正确证据编号都存在于语料中,避免标注笔误。"""
    ids = {}
    for passage in corpus:
        ids[passage["id"]] = True
    for question in questions:
        for gold_id in question["gold"]:
            if gold_id not in ids:
                print("[标注错误] 问题 %s 的正确证据 %s 不在语料中" % (question["id"], gold_id))
                raise SystemExit(1)


def main():
    print_title("实验:关键词、向量与混合检索(及重排序)对比")
    print("模型服务:%s  模型名:%s" % (BASE_URL, MODEL_NAME))

    corpus = build_corpus()
    questions = build_questions()
    check_gold_ids(corpus, questions)

    embedder = choose_embedder()
    if embedder is None:
        print("没有可用的嵌入模型,实验结束。")
        return
    llm_ready = check_llm_ready()
    if USE_LLM and not llm_ready:
        print("无法连接本地 LLM,将只评估检索指标。")

    print_setup(corpus, questions, embedder, llm_ready)

    print_title("【建立索引】")
    start = time.time()
    bm25 = BM25Index(corpus)
    print("BM25 索引:%.1f ms" % ((time.time() - start) * 1000.0))
    texts = []
    for passage in corpus:
        texts.append(passage_index_text(passage))
    start = time.time()
    try:
        vectors = embedder.embed_texts(texts)
    except Exception as error:
        print("服务嵌入在建索引时失败,改用本地哈希嵌入:%s" % short_text(str(error), 100))
        embedder = LocalHashEmbedder()
        vectors = embedder.embed_texts(texts)
    print("向量索引(%s):%.1f ms,共 %d 条向量" % (embedder.name, (time.time() - start) * 1000.0, len(vectors)))

    print_title("【运行实验】")
    records = run_experiment(corpus, questions, bm25, vectors, embedder, llm_ready)

    print_title("【总体结果】")
    print_retrieval_table(records)
    print_efficiency_table(records)
    if llm_ready:
        print_generation_table(records)

    print_title("【按问题类型拆分】")
    print_type_table(records, "Recall@3(前 3 条里找到的正确证据比例)", "rec3", "percent")
    print_type_table(records, "MRR", "mrr", "three")
    print_type_table(records, "nDCG@10", "ndcg", "three")
    if llm_ready:
        print_type_table(records, "答案正确率", "correct", "percent")
        print_type_table(records, "引用正确率", "cite_prec", "percent")

    print_title("【逐题明细与重排序效果】")
    print_rank_grid(records, questions)
    print_rerank_effect(records, questions)
    print_case_studies(records, questions, corpus)

    print_interpretation(records, embedder)
    save_results(records, questions, embedder, llm_ready)


if __name__ == "__main__":
    main()
