"""
aic_core.retrieval.faiss_backend
================================
FAISS and Unified Index (.rvdb) retrieval backend.
Provides uniform SearchItem output from static precomputed indexes.
Natively compatible with both Colab Parquet datasets and Local .rvdb archives.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import faiss

from aic_core.retrieval.base import BaseRetriever, SearchItem

logger = logging.getLogger(__name__)


class FaissUnifiedRetriever(BaseRetriever):
    """
    Retriever capable of searching:
    1. Direct Colab extracted dataset (clip_faiss.index + clip_mapping.parquet + map_keyframes_index.parquet)
    2. A single unified .rvdb file (using unified_index / h5py)
    3. A standard .faiss index file alongside metadata JSON/CSV
    4. In-memory FAISS IndexFlatIP
    """

    def __init__(
        self,
        index_or_path: Optional[Union[str, faiss.Index]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        self.index = None
        self.metadata = []
        self.is_loaded = False

        if index_or_path is not None:
            if isinstance(index_or_path, str):
                self.load_index(index_or_path, metadata_list)
            elif isinstance(index_or_path, faiss.Index):
                self.index = index_or_path
                self.metadata = metadata_list or []
                self.is_loaded = True

    @classmethod
    def from_colab_extracted(cls, extracted_root: str):
        """
        Natively loads directly from Colab extracted dataset folder:
        - clip_index/clip_faiss.index
        - clip_index/clip_mapping.parquet
        - map_keyframes/map_keyframes_index.parquet
        Resolves local_frame_idx (0-based) -> n (1-based) -> real frame_idx.
        """
        import pandas as pd

        root = Path(extracted_root)
        faiss_path = root / "clip_index" / "clip_faiss.index"
        mapping_path = root / "clip_index" / "clip_mapping.parquet"
        keyframe_map_path = root / "map_keyframes" / "map_keyframes_index.parquet"

        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

        logger.info(f"Loading FAISS from {faiss_path}...")
        index = faiss.read_index(str(faiss_path))

        clip_mapping = pd.read_parquet(mapping_path) if mapping_path.exists() else None
        map_df = pd.read_parquet(keyframe_map_path) if keyframe_map_path.exists() else None

        # Build O(1) frame lookup map: (video_id, n) -> frame_idx
        frame_lookup = {}
        if map_df is not None:
            for _, r in map_df.iterrows():
                frame_lookup[(str(r["video_id"]), int(r["n"]))] = int(r["frame_idx"])

        metadata = []
        if clip_mapping is not None:
            for _, r in clip_mapping.iterrows():
                vid = str(r["video_id"])
                local_idx = int(r["local_frame_idx"])
                # Crucial Colab schema fix: n == local_frame_idx + 1
                real_frame_id = frame_lookup.get((vid, local_idx + 1), local_idx)
                metadata.append({
                    "video_name": vid,
                    "video_id": vid,
                    "local_frame_idx": local_idx,
                    "frame_id": real_frame_id,
                    "frame_idx": real_frame_id,
                })

        logger.info(f"Loaded {len(metadata)} keyframe mappings from {extracted_root}")
        return cls(index, metadata)

    def load_index(self, path_str: str, metadata_list: Optional[List[Dict[str, Any]]] = None):
        """Loads index from .rvdb or .faiss file or extracted directory."""
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Index file or directory not found: {path_str}")

        if path.is_dir():
            # Check if it's extracted_root
            if (path / "clip_index" / "clip_faiss.index").exists():
                retriever = self.from_colab_extracted(str(path))
                self.index = retriever.index
                self.metadata = retriever.metadata
                self.is_loaded = True
                return

        if path.suffix.lower() == ".rvdb":
            self._load_rvdb(str(path))
        elif path.suffix.lower() in [".faiss", ".index", ".bin"]:
            self._load_faiss(str(path), metadata_list)
        else:
            self._load_faiss(str(path), metadata_list)

    def _load_rvdb(self, rvdb_path: str):
        """Loads from .rvdb unified archive."""
        try:
            from unified_index import UnifiedIndex
            ui = UnifiedIndex()
            success = ui.load_index(rvdb_path)
            if not success:
                raise RuntimeError(f"Failed to load .rvdb file: {rvdb_path}")

            self.index = ui.faiss_index
            self.metadata = ui.metadata
            self.is_loaded = True
            logger.info(f"Loaded .rvdb index successfully: {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading .rvdb with unified_index: {e}")
            raise

    def _load_faiss(self, faiss_path: str, metadata_list: Optional[List[Dict[str, Any]]] = None):
        """Loads raw .faiss file."""
        self.index = faiss.read_index(faiss_path)
        self.metadata = metadata_list or []
        self.is_loaded = True
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")

    def search(self, query_features: np.ndarray, top_k: int = 100) -> List[SearchItem]:
        """
        Executes dense vector search against FAISS index.
        query_features: 1D or 2D array of shape [512] or [1, 512].
        """
        if not self.is_loaded or self.index is None:
            raise RuntimeError("Retriever index is not loaded!")

        if query_features.ndim == 1:
            q_mat = np.ascontiguousarray(query_features.reshape(1, -1).astype(np.float32))
        else:
            q_mat = np.ascontiguousarray(query_features.astype(np.float32))

        # Ensure L2 normalization for cosine similarity with IndexFlatIP
        norm = np.linalg.norm(q_mat, axis=-1, keepdims=True)
        if norm[0, 0] > 1e-8:
            q_mat = q_mat / norm

        scores, indices = self.index.search(q_mat, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0:
                continue

            meta = {}
            if self.metadata and idx < len(self.metadata):
                meta = self.metadata[idx]

            # Extract video_name and frame_id safely
            video_name = meta.get("video_name") or meta.get("video_id") or meta.get("folder_name") or f"video_{idx}"
            frame_id = meta.get("frame_id") or meta.get("frame_idx") or idx
            keyframe_idx = meta.get("local_frame_idx") or meta.get("keyframe_idx")
            image_path = meta.get("image_path")

            item = SearchItem(
                video_name=str(video_name),
                frame_id=int(frame_id),
                score=float(score),
                rank=rank + 1,
                keyframe_idx=keyframe_idx,
                image_path=image_path,
                metadata=meta,
            )
            results.append(item)

        return results
