"""
Unit tests for Margin Router and Rerank Budget Combiner
"""

import tempfile
from pathlib import Path
import pytest
from aic_core.retrieval.base import SearchItem
from aic_core.query.parser import TaskType
from aic_core.routing.margin_checker import MarginRouter, ExecutionRoute
from aic_core.rerank.combiner import RerankBudgetCombiner


def make_item(video_name: str, frame_id: int, score: float, rank: int = 1) -> SearchItem:
    return SearchItem(
        video_name=video_name,
        frame_id=frame_id,
        score=score,
        rank=rank,
    )


def test_margin_router_confident_kis():
    router = MarginRouter(min_margin_threshold=0.05, min_top1_score=0.2)
    # Clear gap between rank 1 and rank 2
    items = [
        make_item("V01", 100, 0.45, 1),
        make_item("V02", 200, 0.30, 2),
    ]
    decision = router.evaluate(TaskType.KIS, items)
    assert decision.route == ExecutionRoute.COLAB_AUTO
    assert not decision.is_ambiguous


def test_margin_router_ambiguous_kis():
    router = MarginRouter(min_margin_threshold=0.05, min_top1_score=0.2)
    # Very flat distribution (difference 0.001)
    items = [
        make_item("V01", 100, 0.350, 1),
        make_item("V02", 200, 0.349, 2),
    ]
    decision = router.evaluate(TaskType.KIS, items)
    assert decision.route == ExecutionRoute.LOCAL_HITL
    assert decision.is_ambiguous


def test_margin_router_trake_always_local():
    router = MarginRouter()
    items = [
        make_item("V01", 100, 0.99, 1),
        make_item("V02", 200, 0.10, 2),
    ]
    decision = router.evaluate(TaskType.TRAKE, items)
    assert decision.route == ExecutionRoute.LOCAL_HITL


def test_rerank_budget_combiner():
    combiner = RerankBudgetCombiner(rerank_budget=3, total_output=5)

    # 10 raw items
    raw_pool = [make_item("V01", i * 10, 0.9 - i * 0.05, i + 1) for i in range(10)]

    # Select top 3 for reranking
    selected_for_rerank = combiner.select_for_rerank(raw_pool)
    assert len(selected_for_rerank) == 3

    # Pretend reranker inverted order of the 3
    reranked = [selected_for_rerank[2], selected_for_rerank[0], selected_for_rerank[1]]

    combined = combiner.combine(reranked, raw_pool)
    assert len(combined) == 5

    # First 3 must follow reranked order
    assert combined[0].frame_id == selected_for_rerank[2].frame_id
    assert combined[1].frame_id == selected_for_rerank[0].frame_id
    assert combined[2].frame_id == selected_for_rerank[1].frame_id

    # Check 1-based ranks
    for r, it in enumerate(combined):
        assert it.rank == r + 1

    # Test CSV export
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "submission.csv"
        combiner.export_csv(combined, query_id="query_1", output_file=str(csv_file))
        assert csv_file.exists()
        lines = csv_file.read_text().strip().split("\n")
        assert len(lines) == 5
