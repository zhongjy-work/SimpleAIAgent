"""纯 Python 的 BM25 工具检索（零依赖）。

业界常见的三类"工具搜索"实现：
  1. 正则 / 关键词匹配（最简单，模型自己写 pattern）
  2. BM25 词频检索（本文件采用，适合名称 + 描述的短文本）
  3. 向量检索（embedding），语义更强，但需要额外的 embedding 服务

检索的文档 = 工具名（加权）+ 描述 + 参数名 + hint（同义词/中文关键词）。
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """英文按单词（简单去复数），中文按单字 + 二元组切分。"""
    text = text.lower().replace("_", " ")
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group()
        if "一" <= s[0] <= "鿿":
            tokens += list(s)
            tokens += [s[i:i + 2] for i in range(len(s) - 1)]
        else:
            if len(s) > 3 and s.endswith("s"):
                s = s[:-1]
            tokens.append(s)
    return tokens


class ToolIndex:
    def __init__(self, tools: list[dict], k1: float = 1.5, b: float = 0.75):
        self.tools = tools
        self.k1, self.b = k1, b
        self.docs: list[Counter] = []
        for t in tools:
            props = " ".join(t["input_schema"].get("properties", {}).keys())
            text = " ".join([
                (t["name"] + " ") * 3,          # 名称加权
                t["description"],
                props,
                t.get("hint", ""),
            ])
            self.docs.append(Counter(tokenize(text)))
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        q = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for w in q:
                if w not in doc:
                    continue
                tf = doc[w]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                score += self.idf[w] * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.tools[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
