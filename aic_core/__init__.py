"""
aic_core - Unified Online Video Retrieval Core Package
======================================================
Shared module for AIC competition workflow across Colab and Local systems.
"""

from aic_core.query.parser import TaskType, DecomposedQuery, QueryDecomposer
from aic_core.retrieval.base import BaseRetriever, SearchItem
from aic_core.retrieval.clip_encoder import ClipTextEncoder
from aic_core.retrieval.faiss_backend import FaissUnifiedRetriever
from aic_core.fusion.rrf import RrfFusion
from aic_core.fusion.diversity import VideoDiversifier
from aic_core.temporal.dante import DanteTemporalAligner, TrakeSequenceMatch
from aic_core.routing.margin_checker import ExecutionRoute, RoutingDecision, MarginRouter
from aic_core.rerank.combiner import RerankBudgetCombiner
from aic_core.pipeline import AicOnlinePipeline, PipelineResult

__all__ = [
    "TaskType",
    "DecomposedQuery",
    "QueryDecomposer",
    "BaseRetriever",
    "SearchItem",
    "ClipTextEncoder",
    "FaissUnifiedRetriever",
    "RrfFusion",
    "VideoDiversifier",
    "DanteTemporalAligner",
    "TrakeSequenceMatch",
    "ExecutionRoute",
    "RoutingDecision",
    "MarginRouter",
    "RerankBudgetCombiner",
    "AicOnlinePipeline",
    "PipelineResult",
]
