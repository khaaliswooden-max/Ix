"""tests/test_features.py — feature extraction from tick stream."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ix_world_model.regimes.features import (
    FeatureConfig,
    FeatureExtractor,
)


@pytest.fixture
def t0():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def test_extractor_returns_none_on_first_tick(t0):
    fx = FeatureExtractor("BTC/USDT")
    assert fx.observe_tick(t0, Decimal("65000"), Decimal("3.0")) is None


def test_extractor_cold_start_protection(t0):
    """Should return None until enough returns have accumulated."""
    fx = FeatureExtractor("BTC/USDT")
    out = []
    for i in range(5):
        ts = t0 + timedelta(seconds=i)
        fv = fx.observe_tick(ts, Decimal("65000") + Decimal(i), Decimal("3.0"))
        out.append(fv)
    # First tick: None (no prior mid). Ticks 2-3: still None (n<4 returns).
    # Tick 4+: should yield features.
    assert out[0] is None
    assert out[-1] is not None
    assert out[-1].observations_used >= 4


def test_extractor_detects_positive_autocorrelation(t0):
    """
    A series of returns with same-sign deviations from the mean
    (e.g. positive returns clustered, then negative returns clustered)
    yields positive lag-1 autocorrelation.
    """
    fx = FeatureExtractor("BTC/USDT", FeatureConfig(autocorr_window=40))
    base = Decimal("65000")
    last_fv = None
    # First 30 ticks: all up. Last 30 ticks: all down.
    # Each block contributes same-sign deviations from the mean.
    for i in range(60):
        ts = t0 + timedelta(seconds=i)
        if i < 30:
            base = base * Decimal("1.005")
        else:
            base = base * Decimal("0.995")
        last_fv = fx.observe_tick(ts, base, Decimal("3.0")) or last_fv
    assert last_fv is not None
    # Most consecutive returns are same-signed within the window -> positive autocorr
    assert last_fv.autocorr_lag1 > 0


def test_extractor_detects_negative_autocorrelation(t0):
    """Alternating signs should yield negative autocorr_lag1."""
    fx = FeatureExtractor("BTC/USDT", FeatureConfig(autocorr_window=32))
    last_fv = None
    base = Decimal("65000")
    direction = 1
    for i in range(60):
        ts = t0 + timedelta(seconds=i)
        mult = Decimal("1.01") if direction > 0 else Decimal("0.99")
        base = base * mult
        direction *= -1
        last_fv = fx.observe_tick(ts, base, Decimal("3.0")) or last_fv
    assert last_fv is not None
    assert last_fv.autocorr_lag1 < 0


def test_extractor_disagreement_pass_through(t0):
    fx = FeatureExtractor("BTC/USDT")
    fx.observe_disagreement(Decimal("5.0"))
    fx.observe_disagreement(Decimal("10.0"))
    # Prime with some ticks
    base = Decimal("65000")
    last_fv = None
    for i in range(10):
        ts = t0 + timedelta(seconds=i)
        last_fv = fx.observe_tick(ts, base + Decimal(i), Decimal("3.0")) or last_fv
    assert last_fv is not None
    # Disagreement intensity reflects the two observations we pushed
    assert last_fv.disagreement_intensity_bps == pytest.approx(7.5)


def test_extractor_realized_vol_positive(t0):
    fx = FeatureExtractor("BTC/USDT")
    import random
    random.seed(42)
    base = Decimal("65000")
    last_fv = None
    for i in range(100):
        ts = t0 + timedelta(seconds=i)
        # Random ±0.5% moves
        mult = Decimal(str(1.0 + 0.005 * random.gauss(0, 1)))
        base = base * mult
        last_fv = fx.observe_tick(ts, base, Decimal("3.0")) or last_fv
    assert last_fv is not None
    assert last_fv.realized_vol_annualized > 0
