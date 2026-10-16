# -*- coding: utf-8 -*-
"""
实验:切分策略与知识组织对检索的影响
=====================================================================
实验目的:
    观察"固定长度切分、结构化切分、父子块切分"在不同文档类型
    (技术文档、FAQ、含表格的制度文档)上的表现,
    证明知识库的效果不仅取决于嵌入模型。

实验设计:
    - 固定嵌入模型、检索器(余弦相似度 Top-N)和测试问题,只改变切分策略。
    - 每个问题都标注了"正确来源片段"(在原文中的字符区间),与切分方式无关。
    - 然后加入标题、页码、版本、产品线等元数据,比较三种情况:
        无元数据   :只对片段正文做嵌入
        元数据入文本:把"文档/章节/版本/产品线/页码"拼在片段前面再做嵌入
        入文本+过滤 :在上一条基础上,再按问题给出的版本和产品线过滤候选片段

观测指标:
    Recall@K、MRR、正确片段完整率、重复片段比例、索引规模、预处理成本、
    以及表格问题的"单位 / 表头 / 适用条件"是否被保留。

嵌入模型:
    优先使用你本地 llama.cpp 提供的 /v1/embeddings(服务需要以 --embeddings 方式启动)。
    如果该接口不可用,自动退回到"本地字符二元组哈希嵌入",保证实验始终能跑完。
    当两种嵌入都可用时,会同时运行,用来对比"换嵌入模型"与"换切分策略"各自的影响。

运行方式:
    pip install openai
    python chunking_experiment.py

可用环境变量(可选):
    LLM_BASE_URL  默认 http://127.0.0.1:8080/v1
    LLM_API_KEY   默认 sk-no-key-required
    LLM_MODEL     默认 gemma-4
    EMBED_MODE    auto(默认,可用就两种都跑)/ local(只跑本地哈希嵌入)/ api(只跑服务嵌入)
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

RESULT_FILE = "chunking_experiment_result.json"

# 切分参数(单位:字符)
FIXED_SIZE = 150        # 固定长度切分:每块长度
FIXED_OVERLAP = 30      # 固定长度切分:相邻块重叠长度
STRUCT_MAX = 400        # 结构化切分:普通段落合并后的最大长度
MIN_PIECE = 6           # 父子块切分:太短的子块直接丢弃

# 检索与评估参数
TOP_N = 5               # 每次检索取前 5 个结果
K_MAIN = 3              # "实际交给大模型阅读"的前 3 个结果
DUP_OVERLAP_RATIO = 0.2 # 与更靠前的结果重叠超过自身长度的 20%,算重复片段
LOCAL_DIM = 1024        # 本地哈希嵌入的维度
API_BATCH = 8           # 调用嵌入接口时,每批文本数量

# 三种切分策略、三种元数据模式(键, 显示名)
STRATEGIES = [
    ("fixed", "固定长度切分"),
    ("structured", "结构化切分"),
    ("parent_child", "父子块切分"),
]
MODES = [
    ("none", "无元数据"),
    ("meta_text", "元数据入文本"),
    ("meta_filter", "入文本+过滤"),
]

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
    """打印一个简单的文本表格。headers 是表头列表,rows 是行列表,widths 是每列宽度。"""
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
    """把小数格式化成百分比文本。"""
    return "%.1f%%" % (value * 100)


def three(value):
    """保留三位小数。"""
    return "%.3f" % value


def estimate_tokens(text):
    """粗略估算 Token 数:非 ASCII 字符按 1 个 Token,ASCII 每 4 个字符按 1 个 Token。
    只用于不同策略之间的横向对比。"""
    wide_count = 0
    ascii_count = 0
    for ch in text:
        if ord(ch) > 127:
            wide_count = wide_count + 1
        else:
            ascii_count = ascii_count + 1
    return wide_count + (ascii_count + 3) // 4


def overlap_len(start_a, end_a, start_b, end_b):
    """计算两个字符区间的重叠长度,没有重叠返回 0。"""
    left = start_a
    if start_b > left:
        left = start_b
    right = end_a
    if end_b < right:
        right = end_b
    if right > left:
        return right - left
    return 0


def short_text(text, limit):
    """把多行文本压成一行并截断,方便预览。"""
    one_line = text.replace("\n", " / ")
    if len(one_line) > limit:
        return one_line[:limit] + "..."
    return one_line


def normalize_vector(vector):
    """把向量缩放成长度为 1,这样余弦相似度就等于点积。"""
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


# ---------------------------------------------------------------------
# 第 2 部分:嵌入模型(整个实验中保持固定)
# ---------------------------------------------------------------------

class LocalHashEmbedder:
    """本地哈希嵌入:把文本拆成"字符二元组",哈希到固定维度的向量。
    不依赖任何服务,同样的文本永远得到同样的向量。适合做对照实验的稳定基线。"""

    def __init__(self):
        self.name = "本地哈希嵌入"
        self.dim = LOCAL_DIM
        self.stop_chars = "的了是在和与或及对为吗呢吧请我你么什"

    def gram_index(self, gram):
        """把一个字符片段哈希成向量下标(不用 Python 内置 hash,保证每次运行一致)。"""
        value = 0
        for ch in gram:
            value = (value * 131 + ord(ch)) % self.dim
        return value

    def embed_one(self, text):
        """把一段文本变成一个向量。"""
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
    """通过 OpenAI SDK 调用本地 llama.cpp 的 /v1/embeddings 接口。
    llama.cpp 需要以 --embeddings 方式启动(例如加 --pooling mean),否则该接口不可用。"""

    def __init__(self):
        self.name = "服务嵌入(llama.cpp)"
        self.dim = 0

    def embed_texts(self, texts):
        """分批调用嵌入接口,返回归一化后的向量列表。"""
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
        """检查接口是否可用:能返回一维浮点数向量才算可用。"""
        try:
            vectors = self.embed_texts(["测试文本"])
            if len(vectors) != 1:
                return False
            if len(vectors[0]) == 0:
                return False
            return isinstance(vectors[0][0], float)
        except Exception as error:
            print("  [服务嵌入不可用] %s" % short_text(str(error), 120))
            return False


# ---------------------------------------------------------------------
# 第 3 部分:实验文档与测试问题
# 文档故意做成"多版本、多产品线"的样子:
#   - 技术手册有 1.0 和 2.0 两个版本,结构相同、数字不同
#   - FAQ 有标准版和企业版两份,问题相同、答案不同
#   - 制度文档含三张表格,并有一份 1.0 旧版
# 这样,不加元数据时,检索很容易找到"看起来对、其实是别的版本"的片段。
# ---------------------------------------------------------------------

def tech_sentences(v):
    """生成技术手册里会被用作正确答案的句子。文档和问题标注共用这些句子,避免写错。"""
    result = {}
    result["tools"] = "单个 Agent 最多可以同时挂载 %s 个工具。超过上限时,注册请求会被拒绝并返回错误码 E1003。" % v["max_tools"]
    result["timeout"] = "工具调用的默认超时时间为 %s 秒,可以在工具配置中调整,最大值为 %s 秒。" % (v["timeout"], v["max_timeout"])
    result["retry"] = "调用失败后,平台默认最多自动重试 %s 次,每次重试的等待时间按指数退避递增,初始间隔为 %s 秒。" % (v["retries"], v["interval"])
    result["retry_scope"] = "只有网络错误和 5xx 响应会触发重试,参数校验错误不会重试。"
    result["supersede"] = "当同一个键写入了不同的值时,旧记录会被标记为已替代,不会被物理删除,以便审计。"
    result["topk"] = "记忆召回默认返回相关度最高的 %s 条记录,可以通过 top_k 参数调整,取值范围为 1 到 %s。" % (v["topk"], v["topk_max"])
    result["log"] = "所有工具调用都会写入审计日志,日志默认保留 %s 天。" % v["log_days"]
    return result


TECH_V2 = {"max_tools": "16", "timeout": "60", "max_timeout": "300", "retries": "3",
           "interval": "2", "topk": "5", "topk_max": "20", "log_days": "30"}
TECH_V1 = {"max_tools": "8", "timeout": "30", "max_timeout": "120", "retries": "2",
           "interval": "1", "topk": "3", "topk_max": "10", "log_days": "15"}

FAQ_QUESTIONS = ["Q: 支持多少个并发会话?", "Q: 对话数据会保留多久?",
                 "Q: 支持单点登录吗?", "Q: 出了问题如何联系技术支持?"]
FAQ_STD = {
    "conc": "A: 标准版每个工作区最多支持 5 个并发会话,超出的请求会排队等待。",
    "keep": "A: 标准版的对话数据保留 7 天,到期后自动删除,不能恢复。",
    "sso": "A: 标准版不支持单点登录,只能使用邮箱和密码登录。",
    "support": "A: 标准版通过工单系统提交问题,工作日 24 小时内回复。",
}
FAQ_ENT = {
    "conc": "A: 企业版每个工作区最多支持 50 个并发会话,可以按需申请提升上限。",
    "keep": "A: 企业版的对话数据保留 90 天,到期后自动归档,可以导出。",
    "sso": "A: 企业版支持单点登录,可以使用 SAML 2.0 或 OIDC 对接公司身份系统。",
    "support": "A: 企业版配备专属客户经理,7x24 小时提供支持,紧急问题 1 小时内响应。",
}

# 制度文档 2.0 版(2026 版)中的表格组成部分
POLICY_V2 = {
    "cap1": "【表 1】各套餐月度调用配额(单位:万次/月)",
    "hdr1": "| 项目 | 标准版 | 企业版 |",
    "row_base": "| 基础配额 | 10 | 100 |",
    "row_burst": "| 突发上限 | 15 | 200 |",
    "cond1": "适用条件:突发上限仅在每月前 15 天内可用,且需要工作区已完成实名认证。",
    "cap2": "【表 2】超额部分的计费单价(单位:元/千次)",
    "hdr2": "| 当月累计调用量 | 标准版 | 企业版 |",
    "row_a": "| 超出基础配额 0~5 万次 | 8 | 5 |",
    "row_b": "| 超出基础配额 5 万次以上 | 6 | 3 |",
    "cond2": "适用条件:单价为含税价格,按超额部分分段累进计算;包年合同客户适用合同约定价格,不适用本表。",
    "cap3": "【表 3】退款比例(单位:%)",
    "hdr3": "| 申请时间 | 退款比例 |",
    "row_r1": "| 购买后 7 天内 | 100 |",
    "row_r2": "| 购买后 8~30 天 | 50 |",
    "cond3": "适用条件:已产生调用量的订单不支持退款,退款按原路返回。",
    "carry": "配额按自然月统计,每月 1 日 0 点重置,未用完的配额不能结转到下个月。",
}
# 制度文档 1.0 版(2025 版,旧版)
POLICY_V1 = {
    "cap1": "【表 1】各套餐月度调用配额(单位:万次/月)",
    "hdr1": "| 项目 | 标准版 | 企业版 |",
    "row_base": "| 基础配额 | 5 | 50 |",
    "row_burst": "| 突发上限 | 8 | 100 |",
    "cond1": "适用条件:突发上限不限制使用时间,无需实名认证。",
    "carry": "配额按自然月统计,每月 1 日 0 点重置,未用完的配额可以结转一个月。",
}


def make_doc(doc_id, title, doc_type, version, product, pages):
    """把页面列表拼成完整文档,并记录每一页在全文中的起始位置(用于页码元数据)。"""
    text = ""
    page_starts = []
    for i in range(len(pages)):
        if i > 0:
            text = text + "\n\n"
        page_starts.append(len(text))
        text = text + pages[i]
    doc = {
        "doc_id": doc_id, "title": title, "doc_type": doc_type,
        "version": version, "product": product,
        "pages": pages, "text": text, "page_starts": page_starts,
    }
    return doc


def build_tech_doc(doc_id, version, values):
    """生成一份技术手册(3 页)。"""
    s = tech_sentences(values)
    page1 = ("# 星云 Agent 平台技术手册\n\n## 工具调用\n\n"
             "Agent 通过工具注册表调用外部工具。每个工具需要提供名称、描述和参数模式,平台会在调用前校验参数是否符合模式。\n\n"
             + s["tools"] + "\n\n## 超时与重试\n\n" + s["timeout"])
    page2 = (s["retry"] + s["retry_scope"] + "\n\n## 记忆服务\n\n"
             "记忆服务把对话中的稳定信息保存为结构化记录。每条记录包含键、值、来源会话和过期时间。\n\n"
             + s["supersede"])
    page3 = s["topk"] + "\n\n## 日志与审计\n\n" + s["log"]
    return make_doc(doc_id, "星云 Agent 平台技术手册", "技术文档", version, "全部", [page1, page2, page3])


def build_faq_doc(doc_id, product, answers):
    """生成一份 FAQ(2 页)。每个问答是一个自然块。"""
    title = "星云 Agent 平台常见问题(%s)" % product
    page1 = ("# " + title + "\n\n## 账号与并发\n\n"
             + FAQ_QUESTIONS[0] + "\n" + answers["conc"] + "\n\n"
             + FAQ_QUESTIONS[1] + "\n" + answers["keep"])
    page2 = ("## 安全与接入\n\n"
             + FAQ_QUESTIONS[2] + "\n" + answers["sso"] + "\n\n"
             + FAQ_QUESTIONS[3] + "\n" + answers["support"])
    return make_doc(doc_id, title, "FAQ", "2.0", product, [page1, page2])


def build_policy_v2_doc():
    """生成含三张表格的制度文档 2.0 版(2 页)。"""
    p = POLICY_V2
    sep = "|---|---|---|"
    sep2 = "|---|---|"
    page1 = ("# API 调用计费与配额制度(2026 版)\n\n## 总则\n\n"
             "为规范星云 Agent 平台的 API 调用计费和配额管理,制定本制度。本制度适用于所有通过平台 API 发起调用的工作区。\n\n"
             + p["carry"] + "\n\n## 月度配额\n\n"
             + p["cap1"] + "\n" + p["hdr1"] + "\n" + sep + "\n" + p["row_base"] + "\n" + p["row_burst"] + "\n" + p["cond1"])
    page2 = ("## 超额计费\n\n"
             + p["cap2"] + "\n" + p["hdr2"] + "\n" + sep + "\n" + p["row_a"] + "\n" + p["row_b"] + "\n" + p["cond2"]
             + "\n\n## 退款规则\n\n"
             + p["cap3"] + "\n" + p["hdr3"] + "\n" + sep2 + "\n" + p["row_r1"] + "\n" + p["row_r2"] + "\n" + p["cond3"])
    return make_doc("P2", "API 调用计费与配额制度(2026 版)", "制度表格", "2.0", "全部", [page1, page2])


def build_policy_v1_doc():
    """生成制度文档 1.0 旧版(1 页),只有配额表。"""
    p = POLICY_V1
    sep = "|---|---|---|"
    page1 = ("# API 调用计费与配额制度(2025 版)\n\n## 总则\n\n"
             "为规范星云 Agent 平台的 API 调用计费和配额管理,制定本制度。本制度适用于所有通过平台 API 发起调用的工作区。\n\n"
             + p["carry"] + "\n\n## 月度配额\n\n"
             + p["cap1"] + "\n" + p["hdr1"] + "\n" + sep + "\n" + p["row_base"] + "\n" + p["row_burst"] + "\n" + p["cond1"])
    return make_doc("P1", "API 调用计费与配额制度(2025 版)", "制度表格", "1.0", "全部", [page1])


def build_documents():
    """生成全部实验文档。"""
    docs = []
    docs.append(build_tech_doc("T2", "2.0", TECH_V2))
    docs.append(build_tech_doc("T1", "1.0", TECH_V1))
    docs.append(build_faq_doc("F1", "标准版", FAQ_STD))
    docs.append(build_faq_doc("F2", "企业版", FAQ_ENT))
    docs.append(build_policy_v2_doc())
    docs.append(build_policy_v1_doc())
    return docs


def make_question_spec(qid, doc_type, text, doc_id, version, product, evidence, table_checks):
    """创建一个问题的原始描述。
    evidence:正确来源片段(原文字符串列表)。第 0 个是"关键片段"(答案所在),其余是"支撑片段"。
    version / product:问题所处的上下文,用于元数据过滤(None 表示不过滤)。
    table_checks:表格问题需要保留的 单位 / 表头 / 适用条件 关键词(没有则为 None)。"""
    return {
        "id": qid, "doc_type": doc_type, "text": text, "doc_id": doc_id,
        "version": version, "product": product,
        "evidence": evidence, "table_checks": table_checks,
    }


def build_question_specs():
    """列出全部测试问题及其正确来源标注。"""
    t2 = tech_sentences(TECH_V2)
    t1 = tech_sentences(TECH_V1)
    p2 = POLICY_V2
    p1 = POLICY_V1
    specs = []

    # 技术文档问题(8 个),其中 1 个问旧版本
    specs.append(make_question_spec("T-1", "技术文档", "单个 Agent 最多能挂载多少个工具?", "T2", "2.0", None, [t2["tools"]], None))
    specs.append(make_question_spec("T-2", "技术文档", "工具调用不指定时多久会超时?", "T2", "2.0", None, [t2["timeout"]], None))
    specs.append(make_question_spec("T-3", "技术文档", "失败重试的退避间隔从多少秒开始?", "T2", "2.0", None, [t2["retry"]], None))
    specs.append(make_question_spec("T-4", "技术文档", "旧版本里记忆召回默认返回几条?", "T1", "1.0", None, [t1["topk"]], None))
    specs.append(make_question_spec("T-5", "技术文档", "审计日志默认保存多久?", "T2", "2.0", None, [t2["log"]], None))
    specs.append(make_question_spec("T-6", "技术文档", "同一个记忆键被写入新值后,旧的记录会怎么处理?", "T2", "2.0", None, [t2["supersede"]], None))
    specs.append(make_question_spec("T-7", "技术文档", "工具超时最多能调到多少秒?", "T2", "2.0", None, [t2["timeout"]], None))
    specs.append(make_question_spec("T-8", "技术文档", "哪些错误会触发自动重试?", "T2", "2.0", None, [t2["retry_scope"]], None))

    # FAQ 问题(4 个),需要产品线上下文
    specs.append(make_question_spec("F-1", "FAQ", "标准版同时能开多少个会话?", "F1", "2.0", "标准版", [FAQ_STD["conc"]], None))
    specs.append(make_question_spec("F-2", "FAQ", "企业版的聊天记录能留多久?", "F2", "2.0", "企业版", [FAQ_ENT["keep"]], None))
    specs.append(make_question_spec("F-3", "FAQ", "我们能不能用公司账号统一登录?", "F2", "2.0", "企业版", [FAQ_ENT["sso"]], None))
    specs.append(make_question_spec("F-4", "FAQ", "标准版出了问题多久能得到回复?", "F1", "2.0", "标准版", [FAQ_STD["support"]], None))

    # 制度表格问题(5 个):前 4 个是表格问题,带 单位/表头/条件 检查;最后 1 个是正文问题
    specs.append(make_question_spec(
        "P-1", "制度表格", "企业版超出基础配额 5 万次以上的单价是多少?", "P2", "2.0", None,
        [p2["row_b"], p2["cap2"], p2["hdr2"], p2["cond2"]],
        {"unit": "元/千次", "header": "标准版 | 企业版", "condition": "含税价格"}))
    specs.append(make_question_spec(
        "P-2", "制度表格", "标准版的突发上限是多少,什么情况下能用?", "P2", "2.0", None,
        [p2["row_burst"], p2["cap1"], p2["hdr1"], p2["cond1"]],
        {"unit": "万次/月", "header": "标准版 | 企业版", "condition": "实名认证"}))
    specs.append(make_question_spec(
        "P-3", "制度表格", "购买后 20 天申请退款能退多少?", "P2", "2.0", None,
        [p2["row_r2"], p2["cap3"], p2["hdr3"], p2["cond3"]],
        {"unit": "单位:%", "header": "申请时间", "condition": "已产生调用量"}))
    specs.append(make_question_spec(
        "P-4", "制度表格", "旧版制度下,企业版的基础配额是多少?", "P1", "1.0", None,
        [p1["row_base"], p1["cap1"], p1["hdr1"], p1["cond1"]],
        {"unit": "万次/月", "header": "标准版 | 企业版", "condition": "不限制使用时间"}))
    specs.append(make_question_spec(
        "P-5", "制度表格", "配额没用完能留到下个月吗?", "P2", "2.0", None, [p2["carry"]], None))
    return specs


def build_questions(docs):
    """给每个问题计算"正确来源片段"在原文中的字符区间,并检查标注是否有效。"""
    doc_map = {}
    for doc in docs:
        doc_map[doc["doc_id"]] = doc

    questions = []
    for spec in build_question_specs():
        doc = doc_map[spec["doc_id"]]
        spans = []
        for evidence in spec["evidence"]:
            count = doc["text"].count(evidence)
            if count != 1:
                print("[标注错误] 问题 %s 的证据在文档 %s 中出现 %d 次:%s" % (spec["id"], doc["doc_id"], count, evidence))
                raise SystemExit(1)
            start = doc["text"].find(evidence)
            spans.append([start, start + len(evidence)])
        spec["spans"] = spans
        questions.append(spec)
    return questions


# ---------------------------------------------------------------------
# 第 4 部分:文档解析(标题、块)和三种切分策略
# 每个切分策略输出一组"索引项":
#   embed_text :用来做嵌入的文本
#   unit       :检索命中后返回给大模型的片段(含原文区间和元数据)
# 固定长度/结构化:embed_text 就是 unit 的正文。
# 父子块:embed_text 是"子块",unit 是它所属的"父块"。
# ---------------------------------------------------------------------

def close_block(current, blocks):
    """如果当前有未结束的块,就把它放进块列表。返回 None 表示已清空。"""
    if current is not None:
        blocks.append(current)
    return None


def parse_blocks(text):
    """把文档解析成"标题列表"和"块列表"。
    块的种类:
        table:以 "|" 或 "【表" 开头的连续行(表题、表头、数据行、适用条件视为一个整体)
        qa   :以 "Q:" 开头的问答对
        text :普通段落
    每个块记录它在全文中的起止位置和所属章节标题路径。"""
    lines = text.split("\n")
    blocks = []
    headings = []
    heading_stack = []
    current = None
    offset = 0

    for line in lines:
        line_start = offset
        line_end = offset + len(line)
        offset = line_end + 1
        stripped = line.strip()

        if stripped.startswith("#"):
            current = close_block(current, blocks)
            level = 0
            while level < len(stripped) and stripped[level] == "#":
                level = level + 1
            title = stripped[level:].strip()
            heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)
            headings.append({"offset": line_start, "path": " > ".join(heading_stack)})
            continue

        if stripped == "":
            current = close_block(current, blocks)
            continue

        if current is None:
            kind = "text"
            if stripped.startswith("|") or stripped.startswith("【表"):
                kind = "table"
            elif stripped.startswith("Q:"):
                kind = "qa"
            current = {"kind": kind, "start": line_start, "end": line_end,
                       "heading": " > ".join(heading_stack)}
        else:
            current["end"] = line_end

    current = close_block(current, blocks)
    return blocks, headings


def find_heading_path(headings, position):
    """查找某个位置之前最近的一个标题(章节路径)。"""
    path = ""
    for heading in headings:
        if heading["offset"] <= position:
            path = heading["path"]
    return path


def find_page(doc, position):
    """查找某个位置所在的页码(从 1 开始)。"""
    page = 1
    for i in range(len(doc["page_starts"])):
        if doc["page_starts"][i] <= position:
            page = i + 1
    return page


def make_unit(doc, start, end, heading_path, unit_id):
    """创建一个"片段":包含正文、原文区间和元数据(标题、页码、版本、产品线)。"""
    unit = {
        "unit_id": unit_id, "doc_id": doc["doc_id"],
        "start": start, "end": end, "text": doc["text"][start:end],
        "heading": heading_path, "page": find_page(doc, start),
        "title": doc["title"], "version": doc["version"], "product": doc["product"],
    }
    return unit


def build_fixed_items(doc):
    """策略 1:固定长度切分。不看文档结构,每 FIXED_SIZE 个字符切一刀,相邻块重叠 FIXED_OVERLAP。"""
    blocks, headings = parse_blocks(doc["text"])
    text = doc["text"]
    items = []
    start = 0
    number = 0
    while start < len(text):
        end = start + FIXED_SIZE
        if end > len(text):
            end = len(text)
        unit = make_unit(doc, start, end, find_heading_path(headings, start), "%s-F%d" % (doc["doc_id"], number))
        items.append({"embed_text": unit["text"], "unit": unit})
        number = number + 1
        if end >= len(text):
            break
        start = end - FIXED_OVERLAP
    return items


def build_structured_units(doc):
    """结构化切分的核心:返回结构化片段(也是父子块策略里的"父块")。
    规则:
        - 以标题划分章节,章节标题不放进正文,而是作为元数据。
        - 问答对、表格各自成为一个完整片段(表格带着表题、表头、适用条件)。
        - 同一章节内的普通段落合并,直到超过 STRUCT_MAX。
    注意:本实验的表格都不超过 STRUCT_MAX,所以表格不需要再拆分。"""
    blocks, headings = parse_blocks(doc["text"])
    units = []
    pending = None
    number = 0

    for block in blocks:
        if block["kind"] == "text":
            if pending is None:
                pending = {"start": block["start"], "end": block["end"], "heading": block["heading"]}
            elif pending["heading"] == block["heading"] and (block["end"] - pending["start"]) <= STRUCT_MAX:
                pending["end"] = block["end"]
            else:
                units.append(make_unit(doc, pending["start"], pending["end"], pending["heading"], "%s-S%d" % (doc["doc_id"], number)))
                number = number + 1
                pending = {"start": block["start"], "end": block["end"], "heading": block["heading"]}
        else:
            if pending is not None:
                units.append(make_unit(doc, pending["start"], pending["end"], pending["heading"], "%s-S%d" % (doc["doc_id"], number)))
                number = number + 1
                pending = None
            units.append(make_unit(doc, block["start"], block["end"], block["heading"], "%s-S%d" % (doc["doc_id"], number)))
            number = number + 1

    if pending is not None:
        units.append(make_unit(doc, pending["start"], pending["end"], pending["heading"], "%s-S%d" % (doc["doc_id"], number)))
    return units


def build_structured_items(doc):
    """策略 2:结构化切分。每个结构化片段直接作为索引项。"""
    items = []
    for unit in build_structured_units(doc):
        items.append({"embed_text": unit["text"], "unit": unit})
    return items


def is_separator_text(text):
    """判断一段文字是不是表格分隔线(例如 |---|---|)。"""
    for ch in text.strip():
        if ch != "|" and ch != "-" and ch != ":" and ch != " ":
            return False
    return True


def split_pieces(text):
    """把父块拆成"子块":按换行和句号切开。返回(起点, 终点)列表(相对父块)。
    表格会被拆成一行一个子块,问答会被拆成问题和答案各一个子块。"""
    pieces = []
    piece_start = 0
    for i in range(len(text)):
        ch = text[i]
        if ch == "\n" or ch == "。":
            pieces.append([piece_start, i + 1])
            piece_start = i + 1
    if piece_start < len(text):
        pieces.append([piece_start, len(text)])

    kept = []
    for piece in pieces:
        content = text[piece[0]:piece[1]].strip()
        if len(content) < MIN_PIECE:
            continue
        if is_separator_text(content):
            continue
        kept.append(piece)
    return kept


def build_parent_child_items(doc):
    """策略 3:父子块切分。用小的子块做嵌入(便于精确匹配),命中后返回完整的父块(保证上下文完整)。
    父块 = 结构化片段;子块 = 父块内部的句子或表格行。"""
    items = []
    for parent in build_structured_units(doc):
        for piece in split_pieces(parent["text"]):
            child_text = parent["text"][piece[0]:piece[1]].strip()
            items.append({"embed_text": child_text, "unit": parent})
    return items


def chunk_all(strategy_key, docs):
    """对全部文档执行某个切分策略,返回索引项列表。"""
    items = []
    for doc in docs:
        if strategy_key == "fixed":
            items.extend(build_fixed_items(doc))
        elif strategy_key == "structured":
            items.extend(build_structured_items(doc))
        else:
            items.extend(build_parent_child_items(doc))
    return items


# ---------------------------------------------------------------------
# 第 5 部分:建立索引与检索
# ---------------------------------------------------------------------

def build_embed_text(item, use_meta):
    """生成真正送去嵌入的文本。use_meta 为真时,在正文前拼上元数据前缀。"""
    if not use_meta:
        return item["embed_text"]
    unit = item["unit"]
    prefix = "[文档:%s | 章节:%s | 版本:%s | 产品线:%s | 第%d页]\n" % (
        unit["title"], unit["heading"], unit["version"], unit["product"], unit["page"])
    return prefix + item["embed_text"]


def build_index(embedder, items, use_meta):
    """对所有索引项做嵌入,建立向量索引,并记录索引规模与预处理成本。"""
    texts = []
    total_chars = 0
    for item in items:
        text = build_embed_text(item, use_meta)
        texts.append(text)
        total_chars = total_chars + len(text)

    start_time = time.time()
    vectors = embedder.embed_texts(texts)
    embed_seconds = time.time() - start_time

    entries = []
    unique_units = {}
    for i in range(len(items)):
        entries.append({"vec": vectors[i], "unit": items[i]["unit"]})
        unique_units[items[i]["unit"]["unit_id"]] = items[i]["unit"]

    stored_chars = 0
    for unit_id in unique_units:
        stored_chars = stored_chars + len(unique_units[unit_id]["text"])

    dim = 0
    if len(vectors) > 0:
        dim = len(vectors[0])
    cost = {
        "vector_count": len(entries),
        "unit_count": len(unique_units),
        "embed_chars": total_chars,
        "stored_chars": stored_chars,
        "vector_kb": len(entries) * dim * 4 / 1024.0,
        "embed_seconds": embed_seconds,
        "chunk_ms": 0.0,
    }
    return {"entries": entries, "cost": cost}


def passes_filter(unit, question):
    """元数据过滤:版本必须一致;如果问题指定了产品线,片段必须属于该产品线或"全部"。"""
    if question["version"] is not None:
        if unit["version"] != question["version"]:
            return False
    if question["product"] is not None:
        if unit["product"] != question["product"] and unit["product"] != "全部":
            return False
    return True


def retrieve_units(index, query_vector, question, use_filter):
    """检索:对候选片段计算余弦相似度,取前 TOP_N 个,返回它们对应的片段(可能有重复的父块)。"""
    scored = []
    for i in range(len(index["entries"])):
        entry = index["entries"][i]
        if use_filter and not passes_filter(entry["unit"], question):
            continue
        scored.append([dot(query_vector, entry["vec"]), i])
    scored = sorted(scored, reverse=True)

    units = []
    for pair in scored[:TOP_N]:
        units.append(index["entries"][pair[1]]["unit"])
    return units


# ---------------------------------------------------------------------
# 第 6 部分:评估
# ---------------------------------------------------------------------

def dedupe_units(units):
    """去掉区间完全相同的片段(父子块里多个子块会指向同一个父块),保留首次出现的位置。"""
    seen = {}
    result = []
    for unit in units:
        key = "%s:%d:%d" % (unit["doc_id"], unit["start"], unit["end"])
        if key not in seen:
            seen[key] = True
            result.append(unit)
    return result


def count_duplicates(raw_units):
    """统计原始 Top-N 中的"重复片段"个数。
    一个片段如果满足下面任一条件,就算重复:
        1. 文字与更靠前的某个结果完全相同(例如不同版本里的同一句话);
        2. 与更靠前的结果重叠,且重叠部分超过自身长度的 DUP_OVERLAP_RATIO。"""
    duplicate_count = 0
    for i in range(len(raw_units)):
        current = raw_units[i]
        is_duplicate = False
        for j in range(i):
            earlier = raw_units[j]
            if earlier["text"].strip() == current["text"].strip():
                is_duplicate = True
            elif earlier["doc_id"] == current["doc_id"]:
                overlap = overlap_len(current["start"], current["end"], earlier["start"], earlier["end"])
                length = current["end"] - current["start"]
                if length > 0 and overlap / float(length) >= DUP_OVERLAP_RATIO:
                    is_duplicate = True
        if is_duplicate:
            duplicate_count = duplicate_count + 1
    return duplicate_count


def unit_hits_key(unit, question):
    """片段是否"命中":来自正确文档,且与关键证据片段有重叠。"""
    if unit["doc_id"] != question["doc_id"]:
        return False
    key_span = question["spans"][0]
    return overlap_len(unit["start"], unit["end"], key_span[0], key_span[1]) > 0


def unit_coverage(unit, question):
    """片段覆盖了多少比例的正确来源(所有证据片段的字符总数为分母)。"""
    if unit["doc_id"] != question["doc_id"]:
        return 0.0
    total = 0
    covered = 0
    for span in question["spans"]:
        total = total + (span[1] - span[0])
        covered = covered + overlap_len(unit["start"], unit["end"], span[0], span[1])
    if total == 0:
        return 0.0
    return covered / float(total)


def check_table_keep(unit, table_checks):
    """检查返回的片段里,单位、表头、适用条件是否都还在。"""
    result = {}
    result["unit"] = table_checks["unit"] in unit["text"]
    result["header"] = table_checks["header"] in unit["text"]
    result["condition"] = table_checks["condition"] in unit["text"]
    return result


def evaluate_mode(index, questions, query_vectors, use_filter):
    """在一个索引上跑完全部问题,计算各项指标。"""
    per_question = []
    sums = {"r1": 0.0, "r3": 0.0, "r5": 0.0, "mrr": 0.0, "complete": 0.0,
            "dup": 0.0, "tokens": 0.0}
    by_type = {}
    table_total = 0
    table_unit = 0
    table_header = 0
    table_condition = 0
    table_all = 0

    for q_index in range(len(questions)):
        question = questions[q_index]
        raw_units = retrieve_units(index, query_vectors[q_index], question, use_filter)
        ranked = dedupe_units(raw_units)

        # 命中名次:第一个命中关键证据的片段的位置(从 1 开始),0 表示没命中
        hit_rank = 0
        for i in range(len(ranked)):
            if unit_hits_key(ranked[i], question):
                hit_rank = i + 1
                break

        r1 = 0.0
        r3 = 0.0
        r5 = 0.0
        mrr = 0.0
        if hit_rank > 0:
            mrr = 1.0 / hit_rank
            r5 = 1.0
            if hit_rank <= 3:
                r3 = 1.0
            if hit_rank <= 1:
                r1 = 1.0

        # 完整率:前 K_MAIN 个结果里,单个片段对正确来源的最大覆盖比例
        complete = 0.0
        tokens = 0
        for unit in ranked[:K_MAIN]:
            coverage = unit_coverage(unit, question)
            if coverage > complete:
                complete = coverage
            tokens = tokens + estimate_tokens(unit["text"])

        # 重复片段比例
        dup_ratio = 0.0
        if len(raw_units) > 0:
            dup_ratio = count_duplicates(raw_units) / float(len(raw_units))

        sums["r1"] += r1
        sums["r3"] += r3
        sums["r5"] += r5
        sums["mrr"] += mrr
        sums["complete"] += complete
        sums["dup"] += dup_ratio
        sums["tokens"] += tokens

        doc_type = question["doc_type"]
        if doc_type not in by_type:
            by_type[doc_type] = {"count": 0, "r3": 0.0, "mrr": 0.0, "complete": 0.0}
        by_type[doc_type]["count"] += 1
        by_type[doc_type]["r3"] += r3
        by_type[doc_type]["mrr"] += mrr
        by_type[doc_type]["complete"] += complete

        # 表格问题:检查"前 K_MAIN 个结果中第一个命中片段"是否保留了单位、表头、适用条件
        table_flags = None
        if question["table_checks"] is not None:
            table_total = table_total + 1
            table_flags = {"unit": False, "header": False, "condition": False}
            if hit_rank > 0 and hit_rank <= K_MAIN:
                table_flags = check_table_keep(ranked[hit_rank - 1], question["table_checks"])
            if table_flags["unit"]:
                table_unit = table_unit + 1
            if table_flags["header"]:
                table_header = table_header + 1
            if table_flags["condition"]:
                table_condition = table_condition + 1
            if table_flags["unit"] and table_flags["header"] and table_flags["condition"]:
                table_all = table_all + 1

        top_hit_text = ""
        if hit_rank > 0:
            top_hit_text = ranked[hit_rank - 1]["text"]
        per_question.append({
            "id": question["id"], "doc_type": doc_type, "hit_rank": hit_rank,
            "complete": complete, "table_flags": table_flags,
            "top_hit_text": top_hit_text,
            "top_units": short_list_of_units(ranked[:K_MAIN]),
        })

    count = float(len(questions))
    type_result = {}
    for doc_type in by_type:
        item = by_type[doc_type]
        type_result[doc_type] = {
            "r3": item["r3"] / item["count"], "mrr": item["mrr"] / item["count"],
            "complete": item["complete"] / item["count"],
        }

    table_rate = {"unit": 0.0, "header": 0.0, "condition": 0.0, "all": 0.0}
    if table_total > 0:
        table_rate["unit"] = table_unit / float(table_total)
        table_rate["header"] = table_header / float(table_total)
        table_rate["condition"] = table_condition / float(table_total)
        table_rate["all"] = table_all / float(table_total)

    metrics = {
        "recall1": sums["r1"] / count, "recall3": sums["r3"] / count, "recall5": sums["r5"] / count,
        "mrr": sums["mrr"] / count, "complete": sums["complete"] / count,
        "dup": sums["dup"] / count, "tokens": sums["tokens"] / count,
        "by_type": type_result, "table": table_rate, "per_question": per_question,
    }
    return metrics


def short_list_of_units(units):
    """把片段列表压缩成便于保存的简要信息。"""
    result = []
    for unit in units:
        result.append({"unit_id": unit["unit_id"], "page": unit["page"],
                       "heading": unit["heading"], "preview": short_text(unit["text"], 60)})
    return result


# ---------------------------------------------------------------------
# 第 7 部分:运行实验
# ---------------------------------------------------------------------

def run_embedder(embedder, docs, questions):
    """用一个固定的嵌入模型,跑完 3 种切分 × 3 种元数据模式。
    注意:"元数据入文本"和"入文本+过滤"共用同一个索引,区别只是检索时是否过滤。"""
    run = {"embedder": embedder.name, "results": {}, "costs": {}}

    # 问题向量只算一次(所有策略共用),不计入预处理成本
    question_texts = []
    for question in questions:
        question_texts.append(question["text"])
    query_vectors = embedder.embed_texts(question_texts)

    for strategy_key, strategy_name in STRATEGIES:
        print("  策略:%s" % strategy_name)
        start_time = time.time()
        items = chunk_all(strategy_key, docs)
        chunk_ms = (time.time() - start_time) * 1000.0

        run["results"][strategy_key] = {}
        run["costs"][strategy_key] = {}

        print("    构建索引(无元数据),共 %d 个向量..." % len(items))
        index_plain = build_index(embedder, items, False)
        index_plain["cost"]["chunk_ms"] = chunk_ms
        run["costs"][strategy_key]["none"] = index_plain["cost"]
        run["results"][strategy_key]["none"] = evaluate_mode(index_plain, questions, query_vectors, False)

        print("    构建索引(元数据入文本)...")
        index_meta = build_index(embedder, items, True)
        index_meta["cost"]["chunk_ms"] = chunk_ms
        run["costs"][strategy_key]["meta"] = index_meta["cost"]
        run["results"][strategy_key]["meta_text"] = evaluate_mode(index_meta, questions, query_vectors, False)
        run["results"][strategy_key]["meta_filter"] = evaluate_mode(index_meta, questions, query_vectors, True)
    return run


def choose_embedders():
    """按 EMBED_MODE 选择要运行的嵌入模型,列表第一个是"主嵌入模型"(用于详细报告)。"""
    embedders = []
    if EMBED_MODE != "local":
        print("检测本地服务的嵌入接口...")
        api_embedder = ApiEmbedder()
        if api_embedder.check():
            print("  可用,向量维度 %d" % api_embedder.dim)
            embedders.append(api_embedder)
        else:
            print("  不可用。提示:请以 llama-server --embeddings --pooling mean 方式启动模型。")
            if EMBED_MODE == "api":
                return []
    if EMBED_MODE != "api":
        embedders.append(LocalHashEmbedder())
    return embedders


# ---------------------------------------------------------------------
# 第 8 部分:打印报告
# ---------------------------------------------------------------------

def print_setup(docs, questions):
    """打印实验设置:文档、问题、切分参数。"""
    print_title("【实验设置】")
    print("切分参数:固定长度 %d 字符(重叠 %d);结构化段落合并上限 %d;父子块的子块按句/行拆分" % (FIXED_SIZE, FIXED_OVERLAP, STRUCT_MAX))
    print("检索:余弦相似度 Top-%d;Recall/MRR 在 Top-%d 内计算;完整率与注入 Token 按前 %d 个结果计算" % (TOP_N, TOP_N, K_MAIN))
    print("")
    print("实验文档:")
    rows = []
    for doc in docs:
        rows.append([doc["doc_id"], doc["doc_type"], doc["version"], doc["product"],
                     str(len(doc["pages"])), str(len(doc["text"])), doc["title"]])
    print_table(["编号", "类型", "版本", "产品线", "页数", "字符数", "标题"], rows, [6, 10, 6, 8, 6, 8, 40])
    print("")
    print("测试问题:共 %d 个" % len(questions))
    for question in questions:
        filter_text = "版本=%s" % question["version"]
        if question["product"] is not None:
            filter_text = filter_text + ",产品线=" + question["product"]
        print("  %s [%s] %s  (正确来源:%s;过滤条件:%s)" % (
            question["id"], question["doc_type"], question["text"], question["doc_id"], filter_text))


def print_overall_table(run, mode_key, title):
    """打印某个元数据模式下,三种切分策略的总体指标。"""
    print("")
    print(title)
    rows = []
    for strategy_key, strategy_name in STRATEGIES:
        m = run["results"][strategy_key][mode_key]
        rows.append([strategy_name, percent(m["recall1"]), percent(m["recall3"]), percent(m["recall5"]),
                     three(m["mrr"]), percent(m["complete"]), percent(m["dup"]), "%.0f" % m["tokens"]])
    print_table(["策略", "Recall@1", "Recall@3", "Recall@5", "MRR", "片段完整率", "重复片段比", "注入Token"],
                rows, [16, 10, 10, 10, 8, 12, 12, 10])


def print_by_type_table(run, mode_key):
    """按文档类型拆开,看每种切分策略在哪类文档上更好或更差。"""
    print("")
    print("按文档类型拆分(格式:Recall@3 / MRR / 完整率):")
    doc_types = ["技术文档", "FAQ", "制度表格"]
    rows = []
    for doc_type in doc_types:
        row = [doc_type]
        for strategy_key, strategy_name in STRATEGIES:
            item = run["results"][strategy_key][mode_key]["by_type"][doc_type]
            row.append("%s / %s / %s" % (percent(item["r3"]), three(item["mrr"]), percent(item["complete"])))
        rows.append(row)
    headers = ["文档类型"]
    for strategy_key, strategy_name in STRATEGIES:
        headers.append(strategy_name)
    print_table(headers, rows, [12, 30, 30, 30])


def print_table_check(run, questions):
    """打印表格问题的"单位/表头/适用条件"保留情况,并展示命中片段原文供人工检查。"""
    print("")
    print("表格问题自动检查(无元数据模式;对=单位/表头/适用条件 是否都出现在命中片段里):")
    rows = []
    for strategy_key, strategy_name in STRATEGIES:
        t = run["results"][strategy_key]["none"]["table"]
        rows.append([strategy_name, percent(t["unit"]), percent(t["header"]), percent(t["condition"]), percent(t["all"])])
    print_table(["策略", "单位保留", "表头保留", "适用条件保留", "三项都保留"], rows, [16, 12, 12, 14, 12])

    print("")
    print("逐题明细(单位/表头/条件:Y=保留,N=丢失,-=没有命中):")
    for question in questions:
        if question["table_checks"] is None:
            continue
        line = "  %s %s" % (question["id"], question["text"])
        print(line)
        for strategy_key, strategy_name in STRATEGIES:
            for item in run["results"][strategy_key]["none"]["per_question"]:
                if item["id"] != question["id"]:
                    continue
                flags = item["table_flags"]
                if item["hit_rank"] == 0 or item["hit_rank"] > K_MAIN:
                    mark = "-/-/-"
                else:
                    mark = "%s/%s/%s" % (yes_no(flags["unit"]), yes_no(flags["header"]), yes_no(flags["condition"]))
                print("      %s:%s" % (pad(strategy_name, 14), mark))

    print("")
    print("人工检查示例:问题 P-1 在三种策略下,命中片段的原文如下(请对照检查单位、表头、适用条件):")
    for strategy_key, strategy_name in STRATEGIES:
        for item in run["results"][strategy_key]["none"]["per_question"]:
            if item["id"] == "P-1":
                print("  --- %s ---" % strategy_name)
                if item["top_hit_text"] == "":
                    print("  (没有命中)")
                else:
                    for text_line in item["top_hit_text"].strip().split("\n"):
                        print("  | " + text_line)


def yes_no(flag):
    """把布尔值显示为 Y / N。"""
    if flag:
        return "Y"
    return "N"


def print_rank_grid(run, questions):
    """打印每个问题在 9 种组合下的命中名次,方便看哪里错了。"""
    print("")
    print("逐题命中名次(数字=正确片段排第几,-=前 %d 名内没找到)。" % TOP_N)
    print("列名:F=固定长度  S=结构化  P=父子块;-N=无元数据  -T=元数据入文本  -F=入文本+过滤")
    headers = ["问题"]
    codes = []
    for strategy_key, strategy_name in STRATEGIES:
        letter = "F"
        if strategy_key == "structured":
            letter = "S"
        if strategy_key == "parent_child":
            letter = "P"
        codes.append([strategy_key, "none", letter + "-N"])
        codes.append([strategy_key, "meta_text", letter + "-T"])
        codes.append([strategy_key, "meta_filter", letter + "-F"])
    for code in codes:
        headers.append(code[2])
    widths = [8]
    for code in codes:
        widths.append(6)

    rows = []
    for q_index in range(len(questions)):
        row = [questions[q_index]["id"]]
        for code in codes:
            rank = run["results"][code[0]][code[1]]["per_question"][q_index]["hit_rank"]
            if rank == 0:
                row.append("-")
            else:
                row.append(str(rank))
        rows.append(row)
    print_table(headers, rows, widths)


def print_metadata_effect(run):
    """比较每种策略在三种元数据模式下的变化。"""
    print("")
    print("元数据的影响(同一策略、同一嵌入模型下对比):")
    rows = []
    for strategy_key, strategy_name in STRATEGIES:
        for mode_key, mode_name in MODES:
            m = run["results"][strategy_key][mode_key]
            rows.append([strategy_name, mode_name, percent(m["recall1"]), percent(m["recall3"]),
                         three(m["mrr"]), percent(m["complete"]), percent(m["dup"])])
    print_table(["策略", "元数据模式", "Recall@1", "Recall@3", "MRR", "完整率", "重复比"],
                rows, [16, 16, 10, 10, 8, 10, 10])


def print_cost_table(run):
    """打印索引规模与预处理成本。"""
    print("")
    print("索引规模与预处理成本(无元数据的索引;括号内为加了元数据前缀后的数值):")
    rows = []
    for strategy_key, strategy_name in STRATEGIES:
        plain = run["costs"][strategy_key]["none"]
        meta = run["costs"][strategy_key]["meta"]
        rows.append([
            strategy_name,
            str(plain["vector_count"]),
            str(plain["unit_count"]),
            "%d" % plain["stored_chars"],
            "%d(%d)" % (plain["embed_chars"], meta["embed_chars"]),
            "%.1f" % plain["vector_kb"],
            "%.1f" % plain["chunk_ms"],
            "%.2f(%.2f)" % (plain["embed_seconds"], meta["embed_seconds"]),
        ])
    print_table(["策略", "向量数", "片段数", "存储字符", "嵌入字符", "向量KB", "切分ms", "嵌入耗时s"],
                rows, [16, 8, 8, 10, 16, 9, 9, 16])


def spread(values):
    """极差:最大值减最小值。"""
    low = values[0]
    high = values[0]
    for value in values:
        if value < low:
            low = value
        if value > high:
            high = value
    return high - low


def print_cross_embedder(all_runs):
    """多个嵌入模型都可用时,比较"换嵌入模型"和"换切分策略"各自带来的变化。"""
    print_title("【换嵌入模型 vs 换切分策略】")
    headers = ["策略 / 元数据模式"]
    widths = [30]
    for run in all_runs:
        headers.append("MRR:" + run["embedder"])
        widths.append(30)
    rows = []
    for strategy_key, strategy_name in STRATEGIES:
        for mode_key, mode_name in MODES:
            row = [strategy_name + " / " + mode_name]
            for run in all_runs:
                row.append(three(run["results"][strategy_key][mode_key]["mrr"]))
            rows.append(row)
    print_table(headers, rows, widths)

    print("")
    print("变化幅度(MRR 的极差,越大说明该因素影响越大):")
    for run in all_runs:
        values = []
        for strategy_key, strategy_name in STRATEGIES:
            values.append(run["results"][strategy_key]["none"]["mrr"])
        print("  嵌入模型固定为[%s],只换切分策略:MRR 变化 %s" % (run["embedder"], three(spread(values))))
    for strategy_key, strategy_name in STRATEGIES:
        values = []
        for run in all_runs:
            values.append(run["results"][strategy_key]["none"]["mrr"])
        print("  切分策略固定为[%s],只换嵌入模型:MRR 变化 %s" % (strategy_name, three(spread(values))))


def print_interpretation(primary):
    """打印结论与解读,其中的数字来自主嵌入模型的实际结果。"""
    print_title("【结果解读】")
    print("说明:以下数字来自主嵌入模型[%s];不同模型和提示词下具体数值会不同,请以上方表格为准。" % primary["embedder"])
    print("")

    none_values = []
    for strategy_key, strategy_name in STRATEGIES:
        none_values.append(primary["results"][strategy_key]["none"]["mrr"])
    print("1. 只换切分策略(嵌入模型、检索器、问题都不变),MRR 的变化幅度为 %s。" % three(spread(none_values)))
    print("   这说明:知识库效果不只取决于嵌入模型,切分方式本身就是一个独立的变量。")
    print("")
    print("2. 固定长度切分:实现最简单、预处理最快,但会在句子中间、表格中间切断。")
    print("   完整率偏低说明正确来源常被切成两半;表格的表题(单位)、表头、适用条件容易和数据行分离。")
    print("   相邻块重叠还会让重复片段比例上升。")
    print("3. 结构化切分:按标题、问答对、表格整体切分,正确片段更完整,表格的单位/表头/条件基本都在同一片段里。")
    print("   代价是需要写解析规则,并且片段长短不一;文档格式不规范时规则会失效。")
    print("4. 父子块切分:用小的子块做精确匹配,命中后返回完整父块,既保证匹配精度又保证上下文完整。")
    print("   代价是索引里向量更多(规模和嵌入成本更高),并且 Top-N 名额常被同一个父块的多个子块占用(重复比例高),")
    print("   实际能看到的不同父块变少;在表格里一行一个子块时尤其明显。")
    print("5. 元数据:把标题、版本、产品线拼进嵌入文本,能帮助区分'结构相同、数字不同'的多版本、多产品线内容;")
    print("   再按版本和产品线过滤,可以直接排除不属于当前上下文的片段,减少旧版本和别的产品线内容混入。")
    print("   页码主要作用是可追溯:命中片段能告诉用户'来自哪份文档第几页',对检索排序本身帮助很小。")
    print("")
    print("工程建议:先固定一个够用的嵌入模型,再把时间花在文档解析、切分规则、元数据设计和过滤条件上,")
    print("通常比反复更换嵌入模型的收益更稳定。")


def save_results(all_runs, docs, questions):
    """把全部结果保存成 JSON,方便写书时引用。"""
    doc_info = []
    for doc in docs:
        doc_info.append({"doc_id": doc["doc_id"], "type": doc["doc_type"], "version": doc["version"],
                         "product": doc["product"], "pages": len(doc["pages"]), "chars": len(doc["text"])})
    question_info = []
    for question in questions:
        question_info.append({"id": question["id"], "type": question["doc_type"], "text": question["text"],
                              "doc_id": question["doc_id"], "spans": question["spans"],
                              "version": question["version"], "product": question["product"]})
    output = {
        "config": {"fixed_size": FIXED_SIZE, "fixed_overlap": FIXED_OVERLAP, "struct_max": STRUCT_MAX,
                   "top_n": TOP_N, "k_main": K_MAIN},
        "documents": doc_info, "questions": question_info, "runs": all_runs,
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print("")
    print("完整结果已保存到:%s" % RESULT_FILE)


# ---------------------------------------------------------------------
# 第 9 部分:主流程
# ---------------------------------------------------------------------

def main():
    print_title("实验:切分策略与知识组织对检索的影响")
    print("模型服务:%s  模型名:%s  EMBED_MODE=%s" % (BASE_URL, MODEL_NAME, EMBED_MODE))

    docs = build_documents()
    questions = build_questions(docs)
    print_setup(docs, questions)

    print_title("【运行实验】")
    embedders = choose_embedders()
    if len(embedders) == 0:
        print("没有可用的嵌入模型,实验结束。")
        return

    all_runs = []
    for embedder in embedders:
        print("")
        print("嵌入模型:%s" % embedder.name)
        try:
            run = run_embedder(embedder, docs, questions)
            all_runs.append(run)
        except Exception as error:
            print("  [运行失败,跳过该嵌入模型] %s" % short_text(str(error), 160))

    if len(all_runs) == 0:
        print("所有嵌入模型都运行失败,实验结束。")
        return

    primary = all_runs[0]
    print_title("【主嵌入模型的结果】" + primary["embedder"])
    print_overall_table(primary, "none", "表 1:只改变切分策略(无元数据)")
    print_by_type_table(primary, "none")
    print_table_check(primary, questions)
    print_metadata_effect(primary)
    print_rank_grid(primary, questions)
    print_cost_table(primary)

    if len(all_runs) > 1:
        print_cross_embedder(all_runs)
    else:
        print("")
        print("(只有一个嵌入模型可用,跳过'换嵌入模型'的对比。)")

    print_interpretation(primary)
    save_results(all_runs, docs, questions)


if __name__ == "__main__":
    main()
