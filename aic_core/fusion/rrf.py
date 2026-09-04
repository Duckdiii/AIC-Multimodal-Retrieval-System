"""
aic_core.fusion.rrf
===================
Reciprocal Rank Fusion (RRF) for multi-query and multi-modal ranking.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from aic_core.retrieval.base import SearchItem


class RrfFusion:
    """
    Combines ranked candidate lists using weighted Reciprocal Rank Fusion.
    Formula: score(item) = sum_i [ w_i / (k + rank_i(item)) ]
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        ranked_lists: List[Tuple[List[SearchItem], float]],
        top_k: int = 100,
    ) -> List[SearchItem]:
        """
        Fuses multiple ranked lists with corresponding weights.
        ranked_lists: List of (items, weight)
        Returns: Top-k SearchItems with updated RRF scores and ranks.
        """
        if not ranked_lists:
            return []

        if len(ranked_lists) == 1 and ranked_lists[0][1] == 1.0:
            # Single list without fusion needed
            items = ranked_lists[0][0][:top_k]
            for r, it in enumerate(items):
                it.rank = r + 1
            return items

        fused_scores: Dict[Tuple[str, int], float] = defaultdict(float)
        item_registry: Dict[Tuple[str, int], SearchItem] = {}

        for item_list, weight in ranked_lists:
            if weight <= 0.0 or not item_list:
                continue

            for rank_idx, item in enumerate(item_list):
                key = (item.video_name, item.frame_id)
                rank = rank_idx + 1  # 1-indexed
                rrf_contrib = weight / (self.k + rank)
                fused_scores[key] += rrf_contrib

                if key not in item_registry:
                    item_registry[key] = item
                else:
                    # Retain the highest raw similarity or most complete metadata
                    existing = item_registry[key]
                    if item.score > existing.score:
                        item_registry[key].score = item.score
                    if item.image_path and not existing.image_path:
                        item_registry[key].image_path = item.image_path

        # Sort by fused score descending
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

        final_items: List[SearchItem] = []
        for rank, key in enumerate(sorted_keys[:top_k]):
            base_item = item_registry[key]
            new_item = SearchItem(
                video_name=base_item.video_name,
                frame_id=base_item.frame_id,
                score=float(fused_scores[key]),
                rank=rank + 1,
                keyframe_idx=base_item.keyframe_idx,
                image_path=base_item.image_path,
                metadata={**base_item.metadata, "rrf_score": float(fused_scores[key])},
            )
            final_items.append(new_item)

        return final_items
