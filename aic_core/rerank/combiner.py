"""
aic_core.rerank.combiner
========================
Manages the 30/100 candidate budget:
- Selects top 30 for VLM reranking
- Combines 30 reranked items with 70 original RRF items
- Generates competition-compliant submission CSV for KIS, QA, and TRAKE (with jitter rows).
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from aic_core.retrieval.base import SearchItem


class RerankBudgetCombiner:
    """
    Implements the 30/100 candidate strategy:
    Top 30 are sent to VLM reranker, remaining 70 preserve original RRF order.
    Also handles TRAKE jitter rows to optimize R@k metrics.
    """

    def __init__(self, rerank_budget: int = 30, total_output: int = 100):
        self.rerank_budget = rerank_budget
        self.total_output = total_output

    def select_for_rerank(self, raw_candidates: List[SearchItem]) -> List[SearchItem]:
        """Selects top-N candidates from the pool for reranking."""
        return raw_candidates[: self.rerank_budget]

    def combine(
        self,
        reranked_top: List[SearchItem],
        raw_pool: List[SearchItem],
    ) -> List[SearchItem]:
        """
        Merges reranked items with unreranked items from raw_pool:
        - 1 to len(reranked_top): Reranked order (e.g. 1-30)
        - Rest: Preserves original order from raw_pool (e.g. 31-100)
        - Deduplication guarantees uniqueness of (video_name, frame_id).
        """
        combined: List[SearchItem] = []
        seen_keys: Set[Tuple[str, int]] = set()

        # 1. Add reranked items first
        for item in reranked_top:
            key = (item.video_name, item.frame_id)
            if key not in seen_keys:
                seen_keys.add(key)
                combined.append(item)

        # 2. Fill remaining slots from raw_pool
        for item in raw_pool:
            if len(combined) >= self.total_output:
                break
            key = (item.video_name, item.frame_id)
            if key not in seen_keys:
                seen_keys.add(key)
                combined.append(item)

        # 3. Assign final ranks 1 to len(combined)
        for rank_idx, item in enumerate(combined):
            item.rank = rank_idx + 1

        return combined

    @staticmethod
    def generate_trake_jitter_rows(
        video_id: str,
        frame_ids: List[int],
        max_rows: int = 100,
        offsets: Optional[List[int]] = None,
    ) -> List[List[Any]]:
        """
        Generates shifted multi-frame rows to hedge temporal alignment errors for TRAKE.
        Follows official AIC competition practice:
        Row 1: [video_id, f1, f2, ...] (offset=0, most confident)
        Subsequent rows: [video_id, f1+d, f2+d, ...]
        """
        if offsets is None:
            offsets = [0, -3, 3, -6, 6, -9, 9, -12, 12, -15, 15, -20, 20, -25, 25, -30, 30]

        rows = []
        seen = set()

        for offset in offsets:
            if len(rows) >= max_rows:
                break
            shifted = []
            for fid in frame_ids:
                if fid is None:
                    shifted.append(0)
                else:
                    shifted.append(max(0, int(fid) + offset))
            key = tuple(shifted)
            if key in seen:
                continue
            seen.add(key)
            rows.append([video_id] + shifted)

        return rows

    def export_csv(
        self,
        items: List[SearchItem],
        query_id: str,
        output_file: str,
        include_header: bool = False,
    ) -> str:
        """
        Exports standard KIS / QA submission CSV.
        Format: video_name, frame_id
        """
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if include_header:
                writer.writerow(["query_id", "video_name", "frame_id", "rank"])

            for item in items[: self.total_output]:
                if include_header:
                    writer.writerow([query_id, item.video_name, item.frame_id, item.rank])
                else:
                    writer.writerow([item.video_name, item.frame_id])

        return str(out_path)

    def export_trake_csv(
        self,
        video_id: str,
        frame_ids: List[int],
        output_file: str,
        max_rows: int = 100,
    ) -> str:
        """
        Exports TRAKE multi-frame submission CSV with jitter offsets.
        Format: video_id, frame_e1, frame_e2, ...
        """
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self.generate_trake_jitter_rows(video_id, frame_ids, max_rows=max_rows)
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        return str(out_path)
