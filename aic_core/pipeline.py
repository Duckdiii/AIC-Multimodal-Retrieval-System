"""
aic_core.pipeline
=================
Unified Online Retrieval Pipeline orchestrating:
Query Decomposition -> Vector Encoding -> FAISS Dense Search ->
Multi-Query RRF Fusion -> Video Diversification -> Dante (if TRAKE) ->
Margin Evaluation -> 30/100 Candidate Budgeting.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

from aic_core.query.parser import QueryDecomposer, DecomposedQuery, TaskType
from aic_core.retrieval.base import BaseRetriever, SearchItem
from aic_core.retrieval.clip_encoder import ClipTextEncoder
from aic_core.fusion.rrf import RrfFusion
from aic_core.fusion.diversity import VideoDiversifier
from aic_core.temporal.dante import DanteTemporalAligner, TrakeSequenceMatch
from aic_core.routing.margin_checker import MarginRouter, RoutingDecision, ExecutionRoute
from aic_core.rerank.combiner import RerankBudgetCombiner

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Standardized output of the end-to-end pipeline."""
    query_text: str
    decomposed_query: DecomposedQuery
    candidate_pool_100: List[SearchItem]
    top_30_for_rerank: List[SearchItem]
    routing_decision: RoutingDecision
    trake_matches: Optional[List[TrakeSequenceMatch]] = None


class AicOnlinePipeline:
    """
    Unified Pipeline that runs equivalently in both Local and Colab environments.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        clip_encoder: Optional[ClipTextEncoder] = None,
        candidate_pool_size: int = 100,
        rerank_budget: int = 30,
        max_per_video: int = 5,
        diversify_strategy: str = "balanced",
    ):
        self.retriever = retriever
        self.clip_encoder = clip_encoder or ClipTextEncoder()
        self.decomposer = QueryDecomposer()
        self.rrf = RrfFusion(k=60)
        self.diversifier = VideoDiversifier(
            max_per_video=max_per_video,
            strategy=diversify_strategy,
        )
        self.dante = DanteTemporalAligner()
        self.router = MarginRouter()
        self.combiner = RerankBudgetCombiner(
            rerank_budget=rerank_budget,
            total_output=candidate_pool_size,
        )
        self.candidate_pool_size = candidate_pool_size

    def search(
        self,
        query_text: str,
        negative_keywords: Optional[List[str]] = None,
    ) -> PipelineResult:
        """
        Executes end-to-end online search pipeline:
        1. Decomposes query & strips filler noise.
        2. Encodes sub-queries with CLIP.
        3. Retrieves candidates for each sub-query.
        4. Fuses candidates using RRF.
        5. Diversifies video representation.
        6. Runs DANTE if query has sequential TRAKE events.
        7. Evaluates margin for ambiguity.
        8. Packages 100 candidates and selects top 30 for reranker.
        """
        # 1. Decompose Query
        decomposed = self.decomposer.decompose(query_text)
        if negative_keywords:
            decomposed.negative_terms.extend(negative_keywords)

        # 2. Multi-query search and RRF fusion
        ranked_lists_for_fusion: List[Tuple[List[SearchItem], float]] = []

        # If TRAKE with detected events, search event by event for DANTE
        trake_matches = None
        if decomposed.task_type == TaskType.TRAKE and len(decomposed.temporal_events) > 1:
            event_candidates = []
            for ev_text in decomposed.temporal_events:
                ev_vec = self.clip_encoder.encode_text(ev_text)
                cands = self.retriever.search(ev_vec, top_k=self.candidate_pool_size * 2)
                event_candidates.append(cands)

            trake_matches = self.dante.align_sequence(
                event_candidates,
                top_k_videos=self.candidate_pool_size,
            )

        # Standard Multi-Query Search (Core vs Context vs Full)
        neg_vecs = []
        if decomposed.negative_terms:
            neg_vecs = [(self.clip_encoder.encode_text(nt), 0.3) for nt in decomposed.negative_terms]

        for sub_q, weight in decomposed.sub_queries:
            sub_vec = self.clip_encoder.encode_text(sub_q)
            if neg_vecs:
                sub_vec = self.clip_encoder.combine_vectors([(sub_vec, 1.0)], negative_vectors=neg_vecs)

            # Retrieve top candidates
            raw_cands = self.retriever.search(sub_vec, top_k=self.candidate_pool_size * 2)
            ranked_lists_for_fusion.append((raw_cands, weight))

        # 4. RRF Fusion across sub-queries
        fused_items = self.rrf.fuse(ranked_lists_for_fusion, top_k=self.candidate_pool_size * 2)

        # 5. Video-level Diversification
        diversified_pool = self.diversifier.diversify(
            fused_items,
            top_k=self.candidate_pool_size,
        )

        # 6. Ambiguity & Margin Evaluation
        has_temporal = decomposed.task_type == TaskType.TRAKE
        routing_decision = self.router.evaluate(
            task_type=decomposed.task_type,
            items=diversified_pool,
            has_temporal_events=has_temporal,
        )

        # 7. Select Top 30 for Reranking
        top_30 = self.combiner.select_for_rerank(diversified_pool)

        return PipelineResult(
            query_text=query_text,
            decomposed_query=decomposed,
            candidate_pool_100=diversified_pool,
            top_30_for_rerank=top_30,
            routing_decision=routing_decision,
            trake_matches=trake_matches,
        )
