"""tests/test_contracts.py — Phase 1 emitted contracts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ix_world_model.contracts.events import (
    HypothesisWeights,
    MarketSnapshot,
    Regime,
    RegimePosterior,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


# ── RegimePosterior ──────────────────────────────────────────────────────

def test_regime_posterior_valid(now):
    rp = RegimePosterior(
        symbol="BTC/USDT", window_end=now,
        probabilities={
            Regime.TREND_UP: 0.4, Regime.TREND_DOWN: 0.1,
            Regime.MEAN_REVERT: 0.3, Regime.VOLATILE: 0.15, Regime.BROKEN: 0.05,
        },
        observations_used=100,
    )
    assert rp.mode == Regime.TREND_UP
    assert abs(rp.mode_probability - 0.4) < 1e-9
    assert rp.entropy_nats > 0


def test_regime_posterior_rejects_non_simplex(now):
    with pytest.raises(ValidationError, match="sum to 1"):
        RegimePosterior(
            symbol="BTC/USDT", window_end=now,
            probabilities={
                Regime.TREND_UP: 0.5, Regime.TREND_DOWN: 0.1,
                Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.1, Regime.BROKEN: 0.1,
            },
            observations_used=10,
        )


def test_regime_posterior_rejects_missing_regime(now):
    with pytest.raises(ValidationError, match="missing probability"):
        RegimePosterior(
            symbol="BTC/USDT", window_end=now,
            probabilities={Regime.TREND_UP: 0.5, Regime.TREND_DOWN: 0.5},
            observations_used=10,
        )


def test_regime_posterior_rejects_negative_probability(now):
    with pytest.raises(ValidationError, match="out of"):
        RegimePosterior(
            symbol="BTC/USDT", window_end=now,
            probabilities={
                Regime.TREND_UP: 1.2, Regime.TREND_DOWN: -0.2,
                Regime.MEAN_REVERT: 0.0, Regime.VOLATILE: 0.0, Regime.BROKEN: 0.0,
            },
            observations_used=10,
        )


def test_regime_posterior_entropy_uniform_is_log5(now):
    import math
    p = 1.0 / 5
    rp = RegimePosterior(
        symbol="BTC/USDT", window_end=now,
        probabilities={r: p for r in Regime},
        observations_used=10,
    )
    assert abs(rp.entropy_nats - math.log(5)) < 1e-9


# ── HypothesisWeights ────────────────────────────────────────────────────

def test_hypothesis_weights_valid(now):
    hw = HypothesisWeights(
        symbol="BTC/USDT", window_end=now,
        weights={"trend": 0.6, "meanrev": 0.3, "carry": 0.1},
        realized_pnl_window={
            "trend": Decimal("50.5"),
            "meanrev": Decimal("10.0"),
            "carry": Decimal("-5.0"),
        },
        observations_used=42,
    )
    assert hw.leader == "trend"
    assert abs(hw.leader_weight - 0.6) < 1e-9


def test_hypothesis_weights_rejects_missing_pnl(now):
    with pytest.raises(ValidationError, match="missing realized PnL"):
        HypothesisWeights(
            symbol="BTC/USDT", window_end=now,
            weights={"trend": 0.5, "meanrev": 0.5},
            realized_pnl_window={"trend": Decimal("0")},
            observations_used=10,
        )


# ── MarketSnapshot ───────────────────────────────────────────────────────

def _rp(symbol, ts):
    return RegimePosterior(
        symbol=symbol, window_end=ts,
        probabilities={
            Regime.TREND_UP: 0.4, Regime.TREND_DOWN: 0.1,
            Regime.MEAN_REVERT: 0.3, Regime.VOLATILE: 0.15, Regime.BROKEN: 0.05,
        },
        observations_used=10,
    )


def _hw(symbol, ts):
    return HypothesisWeights(
        symbol=symbol, window_end=ts,
        weights={"a": 0.5, "b": 0.5},
        realized_pnl_window={"a": Decimal("0"), "b": Decimal("0")},
        observations_used=10,
    )


def test_snapshot_valid(now):
    snap = MarketSnapshot(
        symbol="BTC/USDT", window_end=now,
        regime=_rp("BTC/USDT", now), hypotheses=_hw("BTC/USDT", now),
    )
    assert snap.symbol == "BTC/USDT"


def test_snapshot_rejects_misaligned_timestamps(now):
    later = now + timedelta(seconds=5)
    with pytest.raises(ValidationError, match="misaligned"):
        MarketSnapshot(
            symbol="BTC/USDT", window_end=now,
            regime=_rp("BTC/USDT", now), hypotheses=_hw("BTC/USDT", later),
        )


def test_snapshot_rejects_symbol_mismatch(now):
    with pytest.raises(ValidationError, match="does not match"):
        MarketSnapshot(
            symbol="BTC/USDT", window_end=now,
            regime=_rp("ETH/USDT", now), hypotheses=_hw("BTC/USDT", now),
        )
