"""Tiny tokenizer for BM25. Kept simple on purpose: lowercase words, no stopwords."""

from __future__ import annotations

import re

# Hyphens split words, so "retrieval-augmented" matches a query saying "retrieval augmented".
_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an and are as at be but by can for from has have in into is it its of on or our
    that the their then there these this those to was we were which while with without
    via using use used based new show shows propose proposed paper approach method methods
    """.split()
)


def tokenize(text: str) -> list[str]:
    """'Retrieval-Augmented Generation for QA' -> ['retrieval', 'augmented', 'generation', 'qa']"""
    return [t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]
