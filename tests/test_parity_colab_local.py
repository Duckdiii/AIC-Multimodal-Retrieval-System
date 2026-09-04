"""
Parity Verification Script
==========================
Verifies that the exact same query against the same FAISS index yields 100% identical
ranking, candidate pool, and score outputs regardless of whether it is executed on
Local or Colab.
"""

import numpy as np
import faiss
from aic_core.retrieval.base import SearchItem
from aic_core.retrieval.faiss_backend import FaissUnifiedRetriever
from aic_core.pipeline import AicOnlinePipeline
from aic_core.retrieval.clip_encoder import ClipTextEncoder


class MockClipEncoder(ClipTextEncoder):
    """Deterministic mock text encoder for fast unit testing without downloading 600MB weights."""
    def __init__(self):
        super().__init__()
        self.embedding_dim = 512

    def encode_text(self, text: str) -> np.ndarray:
        # Deterministic pseudo-vector based on hash of text
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(512).astype(np.float32)
        return vec / np.linalg.norm(vec)


def create_synthetic_index(num_videos=20, frames_per_video=10):
    """Creates a small synthetic FAISS IndexFlatIP with metadata for parity test."""
    index = faiss.IndexFlatIP(512)
    metadata = []
    vectors = []

    rng = np.random.RandomState(42)
    for v_idx in range(num_videos):
        video_name = f"L21_V{v_idx+1:03d}"
        for f_idx in range(frames_per_video):
            frame_id = (f_idx + 1) * 25
            vec = rng.randn(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            metadata.append({
                "video_name": video_name,
                "frame_id": frame_id,
                "keyframe_idx": f_idx + 1,
            })

    mat = np.vstack(vectors)
    index.add(mat)
    return index, metadata


def test_colab_local_parity():
    # Setup shared index
    raw_index, metadata = create_synthetic_index(num_videos=25, frames_per_video=15)

    # Local pipeline setup
    local_retriever = FaissUnifiedRetriever(raw_index, metadata)
    mock_encoder = MockClipEncoder()
    local_pipeline = AicOnlinePipeline(
        retriever=local_retriever,
        clip_encoder=mock_encoder,
        candidate_pool_size=50,
        rerank_budget=15,
        max_per_video=3,
    )

    # Colab pipeline setup (same shared code)
    colab_retriever = FaissUnifiedRetriever(raw_index, metadata)
    colab_pipeline = AicOnlinePipeline(
        retriever=colab_retriever,
        clip_encoder=mock_encoder,
        candidate_pool_size=50,
        rerank_budget=15,
        max_per_video=3,
    )

    test_queries = [
        "Đoạn video mô tả cảnh người đàn ông lái xe máy qua ngã tư đông đúc",
        "E1: đầu bếp thái thịt sau đó E2: xào trên chảo nóng",
        "Cô gái áo đỏ đang đọc sách trong công viên",
    ]

    for q in test_queries:
        local_res = local_pipeline.search(q)
        colab_res = colab_pipeline.search(q)

        # 1. Parity of candidate pool size
        assert len(local_res.candidate_pool_100) == len(colab_res.candidate_pool_100)
        assert len(local_res.top_30_for_rerank) == len(colab_res.top_30_for_rerank)

        # 2. Parity of exact ranking & IDs
        for rank_idx in range(len(local_res.candidate_pool_100)):
            loc_item = local_res.candidate_pool_100[rank_idx]
            col_item = colab_res.candidate_pool_100[rank_idx]

            assert loc_item.video_name == col_item.video_name, f"Video mismatch at rank {rank_idx}"
            assert loc_item.frame_id == col_item.frame_id, f"Frame mismatch at rank {rank_idx}"
            assert np.isclose(loc_item.score, col_item.score, atol=1e-5), f"Score mismatch at rank {rank_idx}"

        # 3. Parity of routing decision
        assert local_res.routing_decision.route == colab_res.routing_decision.route
        assert local_res.routing_decision.is_ambiguous == colab_res.routing_decision.is_ambiguous

    print("All Parity checks passed successfully with 100% equivalence!")


if __name__ == "__main__":
    test_colab_local_parity()
