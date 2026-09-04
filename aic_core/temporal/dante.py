"""
aic_core.temporal.dante
=======================
DANTE (Dynamic Alignment for Narrative and Temporal Events) algorithm.
Solves sequential TRAKE event matching using Dynamic Programming.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from aic_core.retrieval.base import SearchItem


@dataclass
class TrakeSequenceMatch:
    video_name: str
    event_frames: List[int]  # [frame_E1, frame_E2, ...]
    event_scores: List[float]
    total_score: float
    rank: int = 0


class DanteTemporalAligner:
    """
    Dynamic Programming algorithm to find optimal chronologically ordered keyframes
    satisfying: frame_id(E_1) < frame_id(E_2) < ... < frame_id(E_m) within the same video.
    """

    def __init__(self, min_event_gap: int = 15, max_event_gap: int = 5000):
        self.min_event_gap = min_event_gap
        self.max_event_gap = max_event_gap

    def align_sequence(
        self,
        event_candidate_lists: List[List[SearchItem]],
        top_k_videos: int = 30,
    ) -> List[TrakeSequenceMatch]:
        """
        event_candidate_lists: List of candidate SearchItems for each event E_1, E_2, ..., E_m.
        Returns: Top aligned video sequences sorted by cumulative score.
        """
        if not event_candidate_lists:
            return []

        num_events = len(event_candidate_lists)
        if num_events == 1:
            # Single event: degenerate case
            matches = []
            for rank, item in enumerate(event_candidate_lists[0][:top_k_videos]):
                matches.append(
                    TrakeSequenceMatch(
                        video_name=item.video_name,
                        event_frames=[item.frame_id],
                        event_scores=[item.score],
                        total_score=item.score,
                        rank=rank + 1,
                    )
                )
            return matches

        # Group candidates by video for each event
        # video -> event_idx -> list of (frame_id, score)
        video_events: Dict[str, Dict[int, List[Tuple[int, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for event_idx, cands in enumerate(event_candidate_lists):
            for item in cands:
                video_events[item.video_name][event_idx].append((item.frame_id, item.score))

        matches: List[TrakeSequenceMatch] = []

        for video_name, ev_dict in video_events.items():
            # Check if video has candidates for all events
            if len(ev_dict) < num_events:
                continue

            # Sort candidate frames per event by frame_id
            for ev_idx in range(num_events):
                ev_dict[ev_idx].sort(key=lambda x: x[0])

            # Dynamic Programming DP table
            # dp[ev_idx][cand_idx] = (max_score, backpointer_cand_idx)
            dp: List[List[Tuple[float, int]]] = []

            # Base case: event 0
            dp.append([(score, -1) for (_, score) in ev_dict[0]])

            # Transitions for subsequent events
            possible = True
            for ev_idx in range(1, num_events):
                curr_cands = ev_dict[ev_idx]
                prev_cands = ev_dict[ev_idx - 1]
                prev_dp = dp[ev_idx - 1]

                layer_dp = []
                for curr_f, curr_s in curr_cands:
                    best_prev_score = -1e9
                    best_prev_idx = -1

                    for prev_idx, (prev_f, _) in enumerate(prev_cands):
                        gap = curr_f - prev_f
                        if self.min_event_gap <= gap <= self.max_event_gap:
                            prev_cum_score = prev_dp[prev_idx][0]
                            if prev_cum_score > best_prev_score:
                                best_prev_score = prev_cum_score
                                best_prev_idx = prev_idx

                    if best_prev_idx != -1:
                        layer_dp.append((best_prev_score + curr_s, best_prev_idx))
                    else:
                        layer_dp.append((-1e9, -1))

                if not any(entry[1] != -1 for entry in layer_dp):
                    possible = False
                    break

                dp.append(layer_dp)

            if not possible or not dp[-1]:
                continue

            # Find best ending candidate in last event
            last_layer = dp[-1]
            best_last_idx = int(np.argmax([entry[0] for entry in last_layer]))
            best_total_score = last_layer[best_last_idx][0]

            if best_total_score < -1e8:
                continue

            # Backtrack to reconstruct sequence
            seq_frames = [0] * num_events
            seq_scores = [0.0] * num_events

            curr_idx = best_last_idx
            for ev_idx in range(num_events - 1, -1, -1):
                f_id, sc = ev_dict[ev_idx][curr_idx]
                seq_frames[ev_idx] = f_id
                seq_scores[ev_idx] = sc
                curr_idx = dp[ev_idx][curr_idx][1]

            matches.append(
                TrakeSequenceMatch(
                    video_name=video_name,
                    event_frames=seq_frames,
                    event_scores=seq_scores,
                    total_score=float(best_total_score),
                )
            )

        # Sort matches by total score descending
        matches.sort(key=lambda m: m.total_score, reverse=True)
        for rank, match in enumerate(matches[:top_k_videos]):
            match.rank = rank + 1

        return matches[:top_k_videos]
