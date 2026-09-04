"""
demo_pipeline.py
================
Demonstration of the unified aic_core online retrieval pipeline.
Simulates end-to-end flow:
Query -> Decomposition -> Retrieval -> RRF -> Video Grouping -> Margin & Routing -> 30/100 Export.
"""

import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import faiss
from aic_core.pipeline import AicOnlinePipeline
from aic_core.retrieval.faiss_backend import FaissUnifiedRetriever
from aic_core.retrieval.clip_encoder import ClipTextEncoder


def run_demo():
    print("=== AIC ONLINE PIPELINE DEMO ===")

    # 1. Create a synthetic index for instant demonstration
    print("\n[Step 1] Initializing FAISS Index (Synthetic 20 videos, 200 keyframes)...")
    index = faiss.IndexFlatIP(512)
    metadata = []
    vectors = []
    rng = np.random.RandomState(42)

    for v in range(1, 21):
        v_name = f"L21_V{v:03d}"
        for f in range(1, 11):
            frame_id = f * 50
            vec = rng.randn(512).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vectors.append(vec)
            metadata.append({
                "video_name": v_name,
                "frame_id": frame_id,
                "keyframe_idx": f,
            })
    index.add(np.vstack(vectors))
    retriever = FaissUnifiedRetriever(index, metadata)

    # 2. Initialize Pipeline (with lightweight mock encoder for fast local demo)
    class FastDemoEncoder(ClipTextEncoder):
        def encode_text(self, text: str) -> np.ndarray:
            r = np.random.RandomState(abs(hash(text)) % (2**31))
            v = r.randn(512).astype(np.float32)
            return v / np.linalg.norm(v)

    pipeline = AicOnlinePipeline(
        retriever=retriever,
        clip_encoder=FastDemoEncoder(),
        candidate_pool_size=100,
        rerank_budget=30,
        max_per_video=3,
        diversify_strategy="balanced",
    )

    # 3. Test queries
    test_queries = [
        # Query 1: KIS with filler noise
        "Đoạn video mô tả cảnh cô gái mặc áo vàng đang gọt quả dứa trong một căn phòng có chiếc bàn gỗ",
        # Query 2: TRAKE query
        "E1: đầu bếp đổ bột vào chảo sau đó E2: lật mặt bánh chín vàng",
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n" + "=" * 60)
        print(f"[Query {idx}]: {q}")
        result = pipeline.search(q)

        decomp = result.decomposed_query
        print(f"\n--- [1. Query Understanding & Noise Stripping] ---")
        print(f"Detected Task Type : {decomp.task_type.value}")
        print(f"Clean Query        : {decomp.clean_query}")
        print(f"Core Anchor        : {decomp.core_anchor}")
        print(f"Context Query      : {decomp.context_query}")
        print(f"Sub-queries        : {decomp.sub_queries}")

        route = result.routing_decision
        print(f"\n--- [2. Ambiguity & Execution Routing] ---")
        print(f"Recommended Route  : {route.route.value}")
        print(f"Is Ambiguous       : {route.is_ambiguous}")
        print(f"Margin (Rank1-2)   : {route.margin:.4f}")
        print(f"Decision Reason    : {route.reason}")

        print(f"\n--- [3. Candidate Pool & Rerank Budget] ---")
        print(f"Total Output Pool  : {len(result.candidate_pool_100)} candidates")
        print(f"Top-30 for Rerank  : {len(result.top_30_for_rerank)} candidates")
        print(f"Top 5 Frames Preview:")
        for it in result.candidate_pool_100[:5]:
            print(f"  Rank #{it.rank}: {it.video_name} - Frame {it.frame_id} (Score: {it.score:.4f})")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
