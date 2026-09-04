"""
aic_core.retrieval.base
=======================
Base data structures and interfaces for video retrieval.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class SearchItem:
    """Standardized retrieved item across all search backends."""
    video_name: str
    frame_id: int
    score: float
    rank: int = 0
    keyframe_idx: Optional[int] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "frame_id": self.frame_id,
            "score": self.score,
            "rank": self.rank,
            "keyframe_idx": self.keyframe_idx,
            "image_path": self.image_path,
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    """Abstract interface for all retrieval backends."""

    @abstractmethod
    def search(self, query_features: np.ndarray, top_k: int = 100) -> List[SearchItem]:
        pass
