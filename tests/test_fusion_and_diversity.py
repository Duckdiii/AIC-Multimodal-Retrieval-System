"""
Unit tests for RRF Fusion and Video Diversification
"""

import pytest
from aic_core.retrieval.base import SearchItem
from aic_core.fusion.rrf import RrfFusion
from aic_core.fusion.diversity import VideoDiversifier


def make_item(video_name: str, frame_id: int, score: float, rank: int = 1) -> SearchItem:
    return SearchItem(
        video_name=video_name,
        frame_id=frame_id,
        score=score,
        rank=rank,
    )


def test_rrf_fusion():
    list_a = [
        make_item("V01", 100, 0.9, 1),
        make_item("V02", 200, 0.85, 2),
        make_item("V03", 300, 0.8, 3),
    ]
    list_b = [
        make_item("V02", 200, 0.95, 1),  # V02 appears rank 1 here and rank 2 in list_a
        make_item("V01", 100, 0.7, 2),
        make_item("V04", 400, 0.6, 3),
    ]

    rrf = RrfFusion(k=60)
    fused = rrf.fuse([(list_a, 0.7), (list_b, 0.3)], top_k=5)

    assert len(fused) == 4
    # Check that V01 and V02 are top ranks
    top_video_ids = [it.video_name for it in fused[:2]]
    assert "V01" in top_video_ids
    assert "V02" in top_video_ids
    assert fused[0].rank == 1
    assert fused[1].rank == 2


def test_video_diversifier_strict_cap():
    # 10 items all from V01
    items = [make_item("V01", i * 50, 0.9 - i * 0.01, i + 1) for i in range(10)]
    # 5 items from V02
    items.extend([make_item("V02", i * 50, 0.8 - i * 0.01, i + 1) for i in range(5)])

    diversifier = VideoDiversifier(max_per_video=3, min_frame_gap=25, strategy="strict_cap")
    diversified = diversifier.diversify(items, top_k=10)

    # Count per video
    counts = {}
    for it in diversified:
        counts[it.video_name] = counts.get(it.video_name, 0) + 1

    assert counts.get("V01", 0) <= 3
    assert counts.get("V02", 0) <= 3
