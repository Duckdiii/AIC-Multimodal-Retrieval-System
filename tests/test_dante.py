"""
Unit tests for DANTE Temporal Sequence Alignment
"""

import pytest
from aic_core.retrieval.base import SearchItem
from aic_core.temporal.dante import DanteTemporalAligner


def make_item(video_name: str, frame_id: int, score: float) -> SearchItem:
    return SearchItem(
        video_name=video_name,
        frame_id=frame_id,
        score=score,
    )


def test_dante_alignment():
    # Event 1: candidates in Video A and Video B
    event_1 = [
        make_item("Video_A", 100, 0.9),
        make_item("Video_B", 500, 0.8),
    ]

    # Event 2: candidates in Video A (one before E1, one after E1)
    event_2 = [
        make_item("Video_A", 50, 0.95),   # Invalid: before E1 (50 < 100)
        make_item("Video_A", 200, 0.85),  # Valid: after E1 (200 > 100)
        make_item("Video_B", 300, 0.9),   # Invalid for Video B: 300 < 500
    ]

    # Event 3: candidates in Video A
    event_3 = [
        make_item("Video_A", 350, 0.88),  # Valid: 350 > 200
    ]

    dante = DanteTemporalAligner(min_event_gap=10, max_event_gap=1000)
    matches = dante.align_sequence([event_1, event_2, event_3])

    assert len(matches) == 1
    m = matches[0]
    assert m.video_name == "Video_A"
    assert m.event_frames == [100, 200, 350]
    assert m.event_frames[0] < m.event_frames[1] < m.event_frames[2]
