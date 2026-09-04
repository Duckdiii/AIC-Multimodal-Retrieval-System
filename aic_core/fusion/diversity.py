"""
aic_core.fusion.diversity
=========================
Video-level result grouping and diversification to eliminate single-video dominance.
"""

from typing import List, Dict, Optional
from collections import defaultdict

from aic_core.retrieval.base import SearchItem


class VideoDiversifier:
    """
    Prevents single-video dominance by grouping and diversifying frames across distinct videos.
    """

    def __init__(
        self,
        max_per_video: int = 5,
        min_frame_gap: int = 25,
        strategy: str = "balanced",  # "strict_cap", "balanced", or "decay"
        decay_factor: float = 0.85,
    ):
        self.max_per_video = max_per_video
        self.min_frame_gap = min_frame_gap
        self.strategy = strategy
        self.decay_factor = decay_factor

    def diversify(
        self,
        items: List[SearchItem],
        top_k: int = 100,
    ) -> List[SearchItem]:
        """
        Applies video-level diversification to the candidate list.
        """
        if not items:
            return []

        if self.strategy == "strict_cap":
            return self._apply_strict_cap(items, top_k)
        elif self.strategy == "decay":
            return self._apply_score_decay(items, top_k)
        else:
            return self._apply_balanced(items, top_k)

    def _apply_strict_cap(self, items: List[SearchItem], top_k: int) -> List[SearchItem]:
        video_counts = defaultdict(int)
        video_frames = defaultdict(list)
        selected = []

        for item in items:
            v_name = item.video_name
            f_id = item.frame_id

            if video_counts[v_name] >= self.max_per_video:
                continue

            # Check temporal frame gap within the same video
            is_too_close = False
            for prev_f_id in video_frames[v_name]:
                if abs(prev_f_id - f_id) < self.min_frame_gap:
                    is_too_close = True
                    break

            if is_too_close:
                continue

            video_counts[v_name] += 1
            video_frames[v_name].append(f_id)
            selected.append(item)

            if len(selected) >= top_k:
                break

        # Re-assign ranks
        for r, it in enumerate(selected):
            it.rank = r + 1
        return selected

    def _apply_score_decay(self, items: List[SearchItem], top_k: int) -> List[SearchItem]:
        video_counts = defaultdict(int)
        video_frames = defaultdict(list)
        reranked_pool = []

        for item in items:
            v_name = item.video_name
            f_id = item.frame_id
            cnt = video_counts[v_name]

            # Soft penalty per subsequent frame from the same video
            adjusted_score = item.score * (self.decay_factor ** cnt)

            # Extra penalty if too close temporally
            for prev_f in video_frames[v_name]:
                if abs(prev_f - f_id) < self.min_frame_gap:
                    adjusted_score *= 0.5
                    break

            video_counts[v_name] += 1
            video_frames[v_name].append(f_id)

            decayed_item = SearchItem(
                video_name=item.video_name,
                frame_id=item.frame_id,
                score=float(adjusted_score),
                rank=0,
                keyframe_idx=item.keyframe_idx,
                image_path=item.image_path,
                metadata={**item.metadata, "raw_score": item.score},
            )
            reranked_pool.append(decayed_item)

        # Sort by adjusted score descending
        reranked_pool.sort(key=lambda x: x.score, reverse=True)
        selected = reranked_pool[:top_k]
        for r, it in enumerate(selected):
            it.rank = r + 1
        return selected

    def _apply_balanced(self, items: List[SearchItem], top_k: int) -> List[SearchItem]:
        # Group items by video
        grouped = defaultdict(list)
        for item in items:
            grouped[item.video_name].append(item)

        # Sort videos by highest scoring frame
        video_order = sorted(grouped.keys(), key=lambda v: grouped[v][0].score, reverse=True)

        selected = []
        # Round robin across top videos
        for turn in range(self.max_per_video):
            for v_name in video_order:
                if turn < len(grouped[v_name]):
                    cand = grouped[v_name][turn]
                    # Check gap with already selected frames from this video
                    prev_frames = [s.frame_id for s in selected if s.video_name == v_name]
                    if any(abs(pf - cand.frame_id) < self.min_frame_gap for pf in prev_frames):
                        continue
                    selected.append(cand)
                    if len(selected) >= top_k:
                        break
            if len(selected) >= top_k:
                break

        # Re-assign ranks
        for r, it in enumerate(selected):
            it.rank = r + 1
        return selected
