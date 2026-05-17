"""tests/test_ensemble.py — hypothesis ensemble weight assignment."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ix_world_model.hypotheses.ensemble import (
    EnsembleConfig,
    HypothesisEnsemble,
)


@pytest.fixture
def t0():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def test_ensemble_initial_weights_uniform(t0):
    ens = HypothesisEnsemble("BTC/USDT", ["a", "b", "c"])
    w = ens.weights(t0)
    for v in w.weights.values():
        assert abs(v - 1.0 / 3) < 1e-9


def test_ensemble_concentrates_on_winner(t0):
    ens = HypothesisEnsemble("BTC/USDT", ["winner", "loser"],
                              config=EnsembleConfig(score_scale=10.0))
    # Winner gets steady positive PnL, loser gets negative
    for _ in range(50):
        ens.report_pnl({"winner": Decimal("5"), "loser": Decimal("-2")})
    w = ens.weights(t0)
    assert w.leader == "winner"
    assert w.weights["winner"] > w.weights["loser"]


def test_ensemble_rejects_unknown_hypothesis(t0):
    ens = HypothesisEnsemble("BTC/USDT", ["a", "b"])
    with pytest.raises(ValueError, match="unknown hypothesis"):
        ens.report_pnl({"c": Decimal("1")})


def test_ensemble_treats_unreported_as_zero_pnl(t0):
    """A hypothesis that doesn't trade should be treated as 0 PnL, not skipped."""
    ens = HypothesisEnsemble("BTC/USDT", ["a", "b"],
                              config=EnsembleConfig(score_scale=10.0))
    for _ in range(20):
        ens.report_pnl({"a": Decimal("3")})  # b implicitly 0
    w = ens.weights(t0)
    assert w.leader == "a"
    # b is still in the ensemble at min_weight, but a dominates
    assert w.weights["a"] > w.weights["b"]


def test_ensemble_rejects_empty_or_duplicate_names():
    with pytest.raises(ValueError, match="empty"):
        HypothesisEnsemble("BTC/USDT", [])
    with pytest.raises(ValueError, match="duplicates"):
        HypothesisEnsemble("BTC/USDT", ["a", "a"])


def test_ensemble_min_weight_floor_active(t0):
    """A hypothesis with consistently terrible PnL should still get min_weight (post-normalization)."""
    ens = HypothesisEnsemble(
        "BTC/USDT", ["winner", "loser"],
        config=EnsembleConfig(score_scale=1.0, min_weight=0.05),
    )
    # Catastrophic difference
    for _ in range(200):
        ens.report_pnl({"winner": Decimal("100"), "loser": Decimal("-100")})
    w = ens.weights(t0)
    assert w.weights["loser"] >= 0.05 - 1e-9  # floor holds after normalization
    assert abs(sum(w.weights.values()) - 1.0) < 1e-9  # still normalized


def test_ensemble_disagreement_drops_as_weights_concentrate(t0):
    """High initial disagreement (uniform weights) -> low after concentration."""
    import math
    ens = HypothesisEnsemble("BTC/USDT", ["a", "b", "c"],
                              config=EnsembleConfig(score_scale=10.0, min_weight=0.0))
    initial = ens.weights(t0)
    initial_dis = initial.disagreement_nats
    assert abs(initial_dis - math.log(3)) < 0.01  # uniform -> log(N)

    for _ in range(100):
        ens.report_pnl({"a": Decimal("10"), "b": Decimal("-1"), "c": Decimal("-1")})
    final = ens.weights(t0)
    assert final.disagreement_nats < initial_dis


def test_ensemble_window_truncates_old_pnl(t0):
    """Verify the rolling-window mechanism actually evicts old samples."""
    ens = HypothesisEnsemble(
        "BTC/USDT", ["a", "b"],
        config=EnsembleConfig(pnl_window=10, score_scale=1.0, min_weight=0.0),
    )
    # Phase 1: a wins big
    for _ in range(50):
        ens.report_pnl({"a": Decimal("10"), "b": Decimal("-10")})
    # Phase 2: b wins big — older samples should fall off
    for _ in range(20):
        ens.report_pnl({"a": Decimal("-10"), "b": Decimal("10")})
    w = ens.weights(t0)
    # Window is 10 most-recent, so the regime change should have flipped leader
    assert w.leader == "b"
