"""
aic_core.query.parser
=====================
Query parsing, noise stripping, query type detection, and Core/Context decomposition.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    KIS = "KIS"
    TRAKE = "TRAKE"
    QA = "QA"


@dataclass
class DecomposedQuery:
    original_query: str
    task_type: TaskType
    clean_query: str
    core_anchor: str
    context_query: str
    sub_queries: List[Tuple[str, float]] = field(default_factory=list)  # [(query_text, weight)]
    negative_terms: List[str] = field(default_factory=list)
    temporal_events: List[str] = field(default_factory=list)


class QueryDecomposer:
    """
    Decomposes queries to eliminate distractor noise and extract core anchors.
    Supports Vietnamese and English query formulations common in AIC competitions.
    """

    # Common competition meta-phrases and video fluff that dilute CLIP vector
    STOP_PHRASES = [
        # Vietnamese fluff
        r"đoạn video(?: này)?(?: mô tả| ghi lại| quay| cho thấy| có)?(?: cảnh)?",
        r"thước phim(?: này)?(?: mô tả| ghi lại| quay| cho thấy)?(?: cảnh)?",
        r"hình ảnh(?: cho thấy| ghi lại| trong video)?",
        r"video quay cảnh",
        r"video có cảnh",
        r"tìm kiếm video",
        r"tìm đoạn video",
        r"tìm kiếm cảnh",
        r"tìm cảnh",
        r"hãy tìm video",
        r"trong video(?: có)?",
        r"trong khung cảnh(?: có)?",
        r"ở góc quay(?: từ trên xuống| cận cảnh| toàn cảnh)?",
        r"góc máy(?: từ trên xuống| toàn cảnh| cận cảnh)?",
        r"khung hình(?: xuất hiện| có)?",
        # English fluff
        r"a video(?: showing| depicting| of)?",
        r"footage(?: showing| depicting| of)?",
        r"the video shows",
        r"find a video of",
        r"camera angle from (?:above|below)",
        r"scene showing",
    ]

    # Task detection regexes
    TRAKE_PATTERNS = [
        r"e\d+\s*:",
        r"sự kiện \d+",
        r"khoảnh khắc.*?(?:đầu tiên|bắt đầu|kết thúc|tiếp theo)",
        r"(?:trước tiên|đầu tiên).*?sau đó.*?cuối cùng",
        r"phase \d+",
        r"stage \d+",
    ]

    QA_PATTERNS = [
        r"\?$",
        r"ai\s+(?:là|đang|đã)",
        r"cái gì\b",
        r"ở đâu\b",
        r"như thế nào\b",
        r"bao nhiêu\b",
        r"mấy\s+(?:người|con|cái|chiếc)",
        r"màu gì\b",
        r"thời gian nào\b",
        r"^what\b",
        r"^who\b",
        r"^where\b",
        r"^how many\b",
        r"^when\b",
    ]

    # Context indicators
    CONTEXT_PREPOSITIONS = [
        r"trong một (?:căn phòng|ngôi nhà|khu vườn|nhà máy|nhà xưởng|bếp|lớp học|công viên|quán ăn)",
        r"ở (?:trong phòng|ngoài trời|dưới nước|trên trời|bờ sông|bãi biển|vỉa hè|đường phố)",
        r"xung quanh có",
        r"phía sau có",
        r"bên cạnh có",
        r"trên nền",
        r"bối cảnh là",
        r"thời tiết",
    ]

    def __init__(self):
        self.stop_regexes = [re.compile(p, re.IGNORECASE) for p in self.STOP_PHRASES]

    def detect_task_type(self, query: str) -> TaskType:
        """Detect whether query is TRAKE, QA, or KIS."""
        q_lower = query.lower()
        for p in self.TRAKE_PATTERNS:
            if re.search(p, q_lower):
                return TaskType.TRAKE

        for p in self.QA_PATTERNS:
            if re.search(p, q_lower):
                return TaskType.QA

        return TaskType.KIS

    def strip_noise(self, query: str) -> str:
        """Removes filler stop-phrases and normalizes whitespace."""
        clean = query
        for rgx in self.stop_regexes:
            clean = rgx.sub(" ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @classmethod
    def decompose(cls, raw_query: str) -> DecomposedQuery:
        """
        Main decomposition entry point:
        1. Identifies TaskType
        2. Strips filler fluff
        3. Separates Core Anchor (subject + key action) and Context
        4. Produces weighted sub-queries
        """
        self = cls() if isinstance(cls, type) else cls
        task_type = self.detect_task_type(raw_query)
        clean_q = self.strip_noise(raw_query)

        # Handle TRAKE events if present
        temporal_events = []
        if task_type == TaskType.TRAKE:
            # Split by events or sequential markers
            event_splits = re.split(r"(?:E\d+:|sau đó|tiếp theo|cuối cùng)", clean_q, flags=re.IGNORECASE)
            temporal_events = [ev.strip() for ev in event_splits if len(ev.strip()) > 3]

        # Context separation heuristic
        context_parts = []
        remaining_q = clean_q

        for ctx_pat in self.CONTEXT_PREPOSITIONS:
            match = re.search(ctx_pat, remaining_q, flags=re.IGNORECASE)
            if match:
                context_parts.append(match.group(0))
                remaining_q = remaining_q.replace(match.group(0), " ")

        remaining_q = re.sub(r"\s+", " ", remaining_q).strip(" ,;.")
        context_query = " ".join(context_parts).strip()

        # Core anchor is the remaining action/subject
        core_anchor = remaining_q if remaining_q else clean_q

        # Sub-queries formulation:
        # 1. Core anchor (weight: 0.7) - avoids background drift
        # 2. Clean full query (weight: 0.3) - retains context confirmation
        sub_queries = []
        if core_anchor and core_anchor.lower() != clean_q.lower():
            sub_queries.append((core_anchor, 0.7))
            sub_queries.append((clean_q, 0.3))
        else:
            sub_queries.append((clean_q, 1.0))

        if context_query:
            # Low-weight contextual query if needed
            sub_queries.append((context_query, 0.2))

        return DecomposedQuery(
            original_query=raw_query,
            task_type=task_type,
            clean_query=clean_q,
            core_anchor=core_anchor,
            context_query=context_query,
            sub_queries=sub_queries,
            temporal_events=temporal_events,
        )
