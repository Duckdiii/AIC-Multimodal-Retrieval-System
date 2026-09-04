"""
aic_core.retrieval.clip_encoder
===============================
Lightweight, thread-safe CLIP Text Encoder with cache, device auto-detection,
and vector arithmetic (weighted combo and negative prompt subtraction).
"""

import threading
from typing import List, Optional, Union
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer


class ClipTextEncoder:
    """
    Encodes text queries to normalized CLIP embeddings.
    Designed for fast local execution (CPU or GPU).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_size: int = 1000,
    ):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None
        self.cache_size = cache_size
        self._cache = {}
        self._lock = threading.RLock()
        self.embedding_dim = 512

    def _ensure_loaded(self):
        """Lazy loader for model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            with self._lock:
                if self.model is None:
                    self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                    self.model = CLIPModel.from_pretrained(self.model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    # Check projection dimension
                    if hasattr(self.model, "projection_dim"):
                        self.embedding_dim = self.model.projection_dim

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encodes a single text string into a 1D normalized numpy array (e.g. float32, 512).
        Uses memory cache for fast repeat searches.
        """
        text = text.strip()
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        cache_key = f"{self.model_name}:{text}"
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key].copy()

        self._ensure_loaded()

        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            vec = text_features.cpu().numpy()[0].astype(np.float32)

        with self._lock:
            if len(self._cache) >= self.cache_size:
                # Evict one
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = vec

        return vec.copy()

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into a 2D numpy array [N, D]."""
        results = [self.encode_text(t) for t in texts]
        return np.vstack(results)

    @staticmethod
    def combine_vectors(
        vectors_and_weights: List[tuple[np.ndarray, float]],
        negative_vectors: Optional[List[tuple[np.ndarray, float]]] = None,
    ) -> np.ndarray:
        """
        Performs weighted vector addition and negative vector subtraction,
        then L2 renormalizes the resulting vector.
        """
        if not vectors_and_weights:
            raise ValueError("vectors_and_weights cannot be empty")

        total = np.zeros_like(vectors_and_weights[0][0], dtype=np.float32)
        for vec, weight in vectors_and_weights:
            total += vec.astype(np.float32) * weight

        if negative_vectors:
            for neg_vec, neg_weight in negative_vectors:
                total -= neg_vec.astype(np.float32) * neg_weight

        norm = np.linalg.norm(total)
        if norm > 1e-8:
            total /= norm
        return total
