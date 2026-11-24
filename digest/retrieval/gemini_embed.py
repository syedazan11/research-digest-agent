"""Semantic relevance with Gemini embeddings, cached in SQLite.

Catches papers that use different words for your topic (which BM25 misses).
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Sequence

import numpy as np

from digest.db import Database
from digest.models import Paper

log = logging.getLogger(__name__)

RETRYABLE = {429, 500, 502, 503, 504}  # rate limited or server hiccup -> wait and retry


def _unit(v: np.ndarray) -> np.ndarray:
    # gemini-embedding-001 only returns normalized vectors at 3072 dims;
    # at 768 we must normalize ourselves before cosine similarity.
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class GeminiEmbeddingRetriever:
    name = "embedding"

    def __init__(
        self,
        db: Database,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        batch_size: int = 50,
        client=None,  # injectable so tests can pass a fake, no network needed
    ):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client
        self.db = db
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.api_calls = 0  # handy for showing "cache saved N calls" in your README

    # --- API ---
    def _call(self, texts: list[str], task_type: str):
        from google.genai import errors, types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimensions)
        for attempt in range(4):
            try:
                self.api_calls += 1
                return self.client.models.embed_content(model=self.model, contents=texts, config=config)
            except errors.APIError as e:
                if e.code not in RETRYABLE or attempt == 3:
                    raise
                wait = 5 * 2**attempt  # 5s, 10s, 20s
                log.warning("Gemini API %s, retrying in %ss", e.code, wait)
                time.sleep(wait)

    def _embed(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._call(texts[i : i + self.batch_size], task_type)
            vectors.extend(np.asarray(e.values, dtype=np.float32) for e in resp.embeddings)
        return vectors

    # --- cached embedding ---
    def embed_papers(self, papers: Sequence[Paper]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        missing: list[Paper] = []
        for p in papers:
            v = self.db.get_embedding(p.id, self.model, self.dimensions)
            if v is None:
                missing.append(p)
            else:
                out[p.id] = v
        if missing:
            log.info("Embedding %d papers (%d from cache)", len(missing), len(out))
            for p, v in zip(missing, self._embed([p.text for p in missing], "RETRIEVAL_DOCUMENT")):
                self.db.put_embedding(p.id, self.model, self.dimensions, v)
                out[p.id] = v
        return out

    def embed_query(self, query: str) -> np.ndarray:
        key = "query:" + hashlib.sha1(query.encode("utf-8")).hexdigest()
        v = self.db.get_embedding(key, self.model, self.dimensions)
        if v is None:
            v = self._embed([query], "RETRIEVAL_QUERY")[0]
            self.db.put_embedding(key, self.model, self.dimensions, v)
        return v

    # --- Retriever interface ---
    def score(self, query: str, papers: Sequence[Paper]) -> list[float]:
        if not papers:
            return []
        q = _unit(self.embed_query(query))
        vecs = self.embed_papers(papers)
        return [float(np.dot(q, _unit(vecs[p.id]))) for p in papers]
