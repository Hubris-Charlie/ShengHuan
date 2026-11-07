"""合规闸门：禁语拦截（流式）+ 必说话术覆盖核对。

两个关键设计
------------
1. **流式 hold window**：LLM 是逐 token 出的，不能等整段生成完再检查（会毁掉时延），
   也不能逐字放行（禁语跨 chunk 就漏了）。做法是永远扣留最后 K 个字符，
   K = 最长禁语长度 - 1。这样任何命中的禁语一定完整落在已收到文本里。
2. **归一化后再匹配**：全角/半角、数字、标点、空格都会被改写。
   归一化会改变长度，所以必须维护 norm→原文 的下标映射，
   否则扣留窗口的字符数会对不上原文。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 归一化

_FULL2HALF = {ord(c): ord(c) - 0xFEE0 for c in "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"}
_CN_DIGIT = {"零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
             "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_PUNCT = set(" \t\r\n,，。.、;；:：!！?？\"'“”‘’()（）[]【】<>《》-—_~～…·/\\|+*&^%$#@`")
_KEEP = set("0123456789abcdefghijklmnopqrstuvwxyz")


def normalize_with_map(s: str) -> tuple[str, list[int]]:
    """返回 (归一化文本, idx_map)。

    idx_map[i] = 归一化后第 i 个字符来自原文的哪个下标。
    """
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(s):
        c = ch.translate(_FULL2HALF)
        if c in _PUNCT:
            continue
        c = _CN_DIGIT.get(c, c)
        c = c.lower()
        if c in _KEEP or "\u4e00" <= c <= "\u9fff":
            out.append(c)
            idx.append(i)
        # 其它字符（emoji、生僻符号）直接丢弃
    return "".join(out), idx


def normalize(s: str) -> str:
    return normalize_with_map(s)[0]


# ---------------------------------------------------------------- Aho-Corasick

class AhoCorasick:
    """多模式串匹配，用于一次扫出所有禁语。"""

    def __init__(self, patterns: list[str]):
        self.goto: list[dict[str, int]] = [{}]
        self.fail: list[int] = [0]
        self.out: list[list[str]] = [[]]
        for p in patterns:
            if p:
                self._add(p)
        self._build()

    def _add(self, p: str) -> None:
        node = 0
        for ch in p:
            nxt = self.goto[node].get(ch)
            if nxt is None:
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
                nxt = len(self.goto) - 1
                self.goto[node][ch] = nxt
            node = nxt
        self.out[node].append(p)

    def _build(self) -> None:
        from collections import deque

        q: deque[int] = deque()
        for child in self.goto[0].values():
            q.append(child)
        while q:
            r = q.popleft()
            for ch, u in self.goto[r].items():
                q.append(u)
                f = self.fail[r]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[u] = self.goto[f].get(ch, 0) if ch in self.goto[f] else 0
                self.out[u] = self.out[u] + self.out[self.fail[u]]

    def search(self, text: str) -> list[tuple[int, str]]:
        """返回 [(结束下标(含), 命中模式)]。"""
        hits: list[tuple[int, str]] = []
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for pat in self.out[node]:
                hits.append((i, pat))
        return hits


# ---------------------------------------------------------------- 流式禁语拦截

@dataclass
class GuardHit:
    kind: str            # "banned" | "mandatory"
    detail: str
    text: str = ""


class StreamingGuard:
    """边生成边拦。feed() 返回可以安全交给 TTS 的文本。"""

    def __init__(self, banned_phrases: list[str]):
        self.banned_raw = [b for b in banned_phrases if b]
        self.banned_norm = [normalize(b) for b in self.banned_raw if normalize(b)]
        self.ac = AhoCorasick(self.banned_norm)
        # 归一化后 -> 原始禁语。质检报告必须显示人话（"一定通过"），
        # 不能显示归一化结果（"1定通过"），否则报告读不懂。
        self._orig: dict[str, str] = {}
        for b in self.banned_raw:
            nb = normalize(b)
            if nb:
                self._orig[nb] = b
        self.hold = max((len(b) for b in self.banned_norm), default=1) - 1
        self._pending = ""      # 已收到、尚未放行的原文
        self._released = ""     # 已放行的原文
        self.hits: list[GuardHit] = []

    def feed(self, chunk: str) -> tuple[str, GuardHit | None]:
        """喂入新生成的文本块。

        返回 (本次可放行文本, 命中信息)。命中后调用方应立即停止生成并改说兜底话术。
        """
        self._pending += chunk
        norm, idx = normalize_with_map(self._pending)

        hits = self.ac.search(norm)
        if hits:
            end_norm, pat = hits[0]
            end_src = idx[end_norm] if end_norm < len(idx) else len(self._pending) - 1
            hit = GuardHit(kind="banned", detail=self._orig.get(pat, pat), text=self._pending[: end_src + 1])
            self.hits.append(hit)
            # 命中即判定整段不可放行
            return "", hit

        if len(norm) <= self.hold:
            # 全部扣留，但要把原文长度换算过去
            keep_src = idx[0] if idx else len(self._pending)
            release = self._pending[:keep_src]
            self._pending = self._pending[keep_src:]
        else:
            cut_norm = len(norm) - self.hold
            cut_src = idx[cut_norm] if cut_norm < len(idx) else len(self._pending)
            release = self._pending[:cut_src]
            self._pending = self._pending[cut_src:]

        self._released += release
        return release, None

    def flush(self) -> tuple[str, GuardHit | None]:
        """生成结束，放行剩余（再做一次完整检查）。"""
        rest = self._pending
        self._pending = ""
        norm, idx = normalize_with_map(rest)
        hits = self.ac.search(norm)
        if hits:
            end_norm, pat = hits[0]
            end_src = idx[end_norm] if end_norm < len(idx) else len(rest) - 1
            hit = GuardHit(kind="banned", detail=self._orig.get(pat, pat), text=rest[: end_src + 1])
            self.hits.append(hit)
            return "", hit
        self._released += rest
        return rest, None

    @property
    def released(self) -> str:
        return self._released


# ---------------------------------------------------------------- 必说话术覆盖

@dataclass
class MandatorySpec:
    """一条必说话术。

    groups 里每一组都必须至少出现一个关键词，全部组命中才算说了这句。
    requires_state：只有通话进入过该状态时，这条话术才纳入考核
    （例如"转人工告知"只在真的转人工时才要求）。
    """

    uid: str
    label: str
    groups: list[list[str]]
    requires_state: str | None = None

    def matched(self, normalized_text: str) -> bool:
        for group in self.groups:
            if not any(normalize(k) in normalized_text for k in group):
                return False
        return True


class MandatoryChecker:
    def __init__(self, specs: list[MandatorySpec]):
        self.specs = specs
        self._seen = ""
        self.visited: set[str] = set()

    def observe(self, agent_text: str) -> None:
        self._seen += agent_text

    def visit(self, state: str) -> None:
        self.visited.add(state)

    def applicable(self) -> list[MandatorySpec]:
        return [s for s in self.specs if s.requires_state is None or s.requires_state in self.visited]

    def coverage(self) -> dict[str, bool]:
        norm = normalize(self._seen)
        return {s.uid: s.matched(norm) for s in self.applicable()}

    def missing(self) -> list[MandatorySpec]:
        cov = self.coverage()
        return [s for s in self.applicable() if not cov[s.uid]]


def load_banned(path_json: dict) -> tuple[list[str], list[MandatorySpec]]:
    banned = list(path_json.get("banned", []))
    specs = [
        MandatorySpec(
            uid=m["uid"],
            label=m.get("label", m["uid"]),
            groups=m["groups"],
            requires_state=(m.get("requires_state") or None),
        )
        for m in path_json.get("mandatory", [])
    ]
    return banned, specs
