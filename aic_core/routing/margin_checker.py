"""
aic_core.routing.margin_checker
===============================
Ambiguity detection and execution routing (Colab Automation vs Local HITL).
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from aic_core.retrieval.base import SearchItem
from aic_core.query.parser import TaskType


class ExecutionRoute(str, Enum):
    COLAB_AUTO = "COLAB_AUTO"
    LOCAL_HITL = "LOCAL_HITL"


@dataclass
class RoutingDecision:
    route: ExecutionRoute
    is_ambiguous: bool
    margin: float
    confidence_score: float
    reason: str


class MarginRouter:
    """
    Evaluates candidate list distribution to determine if the query is ambiguous,
    and decides whether to run auto-submission on Colab or escalate to Local HITL.
    """

    def __init__(
        self,
        min_margin_threshold: float = 0.015,
        min_top1_score: float = 0.22,
    ):
        self.min_margin_threshold = min_margin_threshold
        self.min_top1_score = min_top1_score

    def evaluate(
        self,
        task_type: TaskType,
        items: List[SearchItem],
        has_temporal_events: bool = False,
    ) -> RoutingDecision:
        """
        Evaluates retrieval output:
        1. TRAKE / QA always default to LOCAL_HITL.
        2. KIS evaluates score gap between rank 1 and rank 2:
           - Large gap (clear winner) -> COLAB_AUTO
           - Small gap (flat distribution / ambiguous / noise) -> LOCAL_HITL
        """
        # TRAKE / QA always require human reasoning or local sequence inspection
        if task_type in [TaskType.TRAKE, TaskType.QA] or has_temporal_events:
            return RoutingDecision(
                route=ExecutionRoute.LOCAL_HITL,
                is_ambiguous=False,
                margin=0.0,
                confidence_score=items[0].score if items else 0.0,
                reason=f"Task type {task_type.value} requires local HITL verification.",
            )

        if not items:
            return RoutingDecision(
                route=ExecutionRoute.LOCAL_HITL,
                is_ambiguous=True,
                margin=0.0,
                confidence_score=0.0,
                reason="No candidate results returned.",
            )

        if len(items) == 1:
            return RoutingDecision(
                route=ExecutionRoute.COLAB_AUTO,
                is_ambiguous=False,
                margin=1.0,
                confidence_score=items[0].score,
                reason="Single candidate found.",
            )

        score_1 = items[0].score
        score_2 = items[1].score
        margin = score_1 - score_2

        # Relative margin = margin / max(score_1, 1e-6)
        rel_margin = margin / max(abs(score_1), 1e-6)

        is_ambiguous = (rel_margin < self.min_margin_threshold) or (score_1 < self.min_top1_score)

        if is_ambiguous:
            reason = (
                f"Low confidence margin (score_1={score_1:.4f}, score_2={score_2:.4f}, "
                f"rel_margin={rel_margin:.4f} < {self.min_margin_threshold}). "
                "High probability of distractor noise or ambiguity."
            )
            return RoutingDecision(
                route=ExecutionRoute.LOCAL_HITL,
                is_ambiguous=True,
                margin=float(margin),
                confidence_score=float(score_1),
                reason=reason,
            )
        else:
            reason = (
                f"High confidence distinct match (score_1={score_1:.4f}, "
                f"rel_margin={rel_margin:.4f}). Safe for automated submission."
            )
            return RoutingDecision(
                route=ExecutionRoute.COLAB_AUTO,
                is_ambiguous=False,
                margin=float(margin),
                confidence_score=float(score_1),
                reason=reason,
            )

    @classmethod
    def calculate_margin(cls, scores: List[float]) -> dict:
        """Helper to quickly calculate margin from a list of float scores."""
        if not scores:
            return {"margin": 0.0, "rel_margin": 0.0, "is_ambiguous": True}
        if len(scores) == 1:
            return {"margin": 1.0, "rel_margin": 1.0, "is_ambiguous": False}
        s1, s2 = float(scores[0]), float(scores[1])
        m = s1 - s2
        rel = m / max(abs(s1), 1e-6)
        is_ambiguous = (rel < 0.02) or (s1 < 0.22)
        return {"margin": float(m), "rel_margin": float(rel), "is_ambiguous": is_ambiguous}


# Alias for backward compatibility
MarginChecker = MarginRouter

