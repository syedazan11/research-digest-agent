"""Pluggable retrievers. Each one scores papers against a query; HybridRanker combines them."""

from digest.retrieval.base import Retriever
from digest.retrieval.bm25 import BM25Retriever
from digest.retrieval.hybrid import HybridRanker, RankedPaper

__all__ = ["Retriever", "BM25Retriever", "HybridRanker", "RankedPaper"]
