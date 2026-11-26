"""Metrics against hand-computed values."""

import math

import pytest

from digest.metrics import dcg_at_k, ndcg_at_k, precision_at_k, recall_at_k

# a and c are relevant and ranked; f is relevant but missing from the ranking; b, d, e are not relevant
REL = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 0, "f": 1}
RANKING = ["a", "b", "c", "d", "e"]


def test_precision():
    assert precision_at_k(RANKING, REL, 1) == 1.0
    assert precision_at_k(RANKING, REL, 2) == 0.5
    assert precision_at_k(RANKING, REL, 5) == 0.4
    assert precision_at_k(["a"], REL, 5) == 0.2  # fewer than k ranked: missing slots count as misses
    with pytest.raises(ValueError):
        precision_at_k(RANKING, REL, 0)


def test_recall():
    assert recall_at_k(RANKING, REL, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RANKING, REL, 3) == pytest.approx(2 / 3)  # f was never retrieved
    assert recall_at_k(RANKING, {"a": 0}, 3) == 0.0  # no relevant papers at all


def test_ndcg_hand_computed():
    # gains [1, 0, 1] -> DCG = 1/log2(2) + 0 + 1/log2(4) = 1.5
    # ideal [1, 1, 1] -> 1 + 1/log2(3) + 1/log2(4) = 2.1309...
    assert dcg_at_k([1, 0, 1], 3) == pytest.approx(1.5)
    expected = 1.5 / (1 + 1 / math.log2(3) + 0.5)
    assert ndcg_at_k(RANKING, REL, 3) == pytest.approx(expected)
    assert expected == pytest.approx(0.70392, abs=1e-5)


def test_ndcg_bounds():
    assert ndcg_at_k(["a", "c", "f", "b"], REL, 10) == pytest.approx(1.0)  # perfect order
    assert ndcg_at_k(["b", "d"], REL, 2) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 5) == 0.0  # nothing relevant: defined as 0, not a crash
    # one relevant paper at rank 2 instead of 1: 1/log2(3)
    assert ndcg_at_k(["b", "a"], {"a": 1, "b": 0}, 10) == pytest.approx(1 / math.log2(3))
