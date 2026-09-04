"""
Unit tests for Query Decomposition and Noise Stripping
"""

import pytest
from aic_core.query.parser import QueryDecomposer, TaskType


@pytest.fixture
def decomposer():
    return QueryDecomposer()


def test_strip_vietnamese_noise(decomposer):
    raw = "Đoạn video mô tả cảnh người đàn ông cưỡi ngựa trên thảo nguyên xanh"
    clean = decomposer.strip_noise(raw)
    assert "người đàn ông cưỡi ngựa trên thảo nguyên xanh" in clean
    assert "đoạn video" not in clean.lower()


def test_core_and_context_separation(decomposer):
    raw = "Đoạn video quay cảnh cô gái mặc áo vàng đang gọt dứa trong một căn phòng có chiếc bàn gỗ"
    decomp = decomposer.decompose(raw)

    assert decomp.task_type == TaskType.KIS
    assert "gọt dứa" in decomp.core_anchor
    assert len(decomp.sub_queries) >= 1
    # Check that weights sum up properly or are reasonable
    core_weight = next(w for q, w in decomp.sub_queries if q == decomp.core_anchor)
    assert core_weight >= 0.7


def test_trake_detection(decomposer):
    raw = "E1: đổ bột vào chảo sau đó E2: lật mặt bánh chín vàng"
    decomp = decomposer.decompose(raw)

    assert decomp.task_type == TaskType.TRAKE
    assert len(decomp.temporal_events) >= 2


def test_qa_detection(decomposer):
    raw = "Người đàn ông áo xanh đang cầm vật thể gì trên tay?"
    decomp = decomposer.decompose(raw)

    assert decomp.task_type == TaskType.QA
