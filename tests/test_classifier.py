"""tests/test_classifier.py — Bayesian regime classifier behavior."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ix_world_model.contracts.events import Regime
from ix_world_model.regimes.classifier import (
    ClassifierConfig,
    DEFAULT_PROFILES,
    RegimeClassifier,
)
from ix_world_model.regimes.features import FeatureVector


@pytest.fixture
def t0():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def _fv(t0, **overrides):
    """Construct a default FeatureVector with optional overrides."""
    base = dict(
        symbol="BTC/USDT", timestamp=t0,
        log_return=0.001, realized_vol_annualized=0.40,
        autocorr_lag1=0.15, spread_bps_mean=3.0,
        disagreement_intensity_bps=2.0, observations_used=100,
    )
    base.update(overrides)
    return FeatureVector(**base)


def test_classifier_first_observation_uniform_prior(t0):
    c = RegimeClassifier("BTC/USDT")
    # Initially, all Dirichlet counts equal -> roughly uniform posterior
    counts = c.counts
    assert all(abs(counts[r] - counts[Regime.TREND_UP]) < 1e-9 for r in Regime)


def test_classifier_concentrates_on_matching_regime_with_repeated_evidence(t0):
    """
    Feed a feature vector that matches TREND_UP's profile many times.
    The posterior should concentrate on TREND_UP.
    """
    c = RegimeClassifier("BTC/USDT")
    fv = _fv(
        t0,
        realized_vol_annualized=0.40, autocorr_lag1=0.15,
        spread_bps_mean=3.0, disagreement_intensity_bps=2.0,
    )
    last = None
    for i in range(200):
        last = c.observe(FeatureVector(
            symbol=fv.symbol, timestamp=t0 + timedelta(seconds=i),
            log_return=fv.log_return,
            realized_vol_annualized=fv.realized_vol_annualized,
            autocorr_lag1=fv.autocorr_lag1,
            spread_bps_mean=fv.spread_bps_mean,
            disagreement_intensity_bps=fv.disagreement_intensity_bps,
            observations_used=fv.observations_used,
        ))
    assert last is not None
    assert last.mode == Regime.TREND_UP
    assert last.mode_probability > 0.5


def test_classifier_detects_broken_regime(t0):
    """Extreme vol + extreme spread + extreme disagreement -> BROKEN."""
    c = RegimeClassifier("BTC/USDT")
    last = None
    for i in range(100):
        fv = _fv(
            t0 + timedelta(seconds=i),
            realized_vol_annualized=2.5,
            spread_bps_mean=40.0,
            disagreement_intensity_bps=40.0,
        )
        last = c.observe(fv)
    assert last is not None
    assert last.mode == Regime.BROKEN


def test_classifier_switches_when_evidence_changes(t0):
    """Start in TREND_UP, then switch evidence to VOLATILE; classifier follows."""
    c = RegimeClassifier(
        "BTC/USDT",
        config=ClassifierConfig(prior_strength=5.0, decay=0.95, min_count=0.05),
    )
    # Trend-up phase
    for i in range(100):
        c.observe(_fv(
            t0 + timedelta(seconds=i),
            realized_vol_annualized=0.40, autocorr_lag1=0.15,
            spread_bps_mean=3.0, disagreement_intensity_bps=2.0,
        ))
    p1 = c.observe(_fv(
        t0 + timedelta(seconds=100),
        realized_vol_annualized=0.40, autocorr_lag1=0.15,
        spread_bps_mean=3.0, disagreement_intensity_bps=2.0,
    ))
    assert p1.mode == Regime.TREND_UP

    # Volatile phase
    last = None
    for i in range(100, 250):
        last = c.observe(_fv(
            t0 + timedelta(seconds=i),
            realized_vol_annualized=1.0, autocorr_lag1=0.0,
            spread_bps_mean=8.0, disagreement_intensity_bps=6.0,
        ))
    assert last is not None
    assert last.mode == Regime.VOLATILE


def test_classifier_emits_valid_posterior(t0):
    c = RegimeClassifier("BTC/USDT")
    p = c.observe(_fv(t0))
    # Validation already happens inside RegimePosterior; just confirm output.
    total = sum(p.probabilities.values())
    assert abs(total - 1.0) < 1e-6
    assert p.observations_used == 1


def test_classifier_mode_run_length_tracks_persistence(t0):
    c = RegimeClassifier("BTC/USDT")
    last = None
    # Repeat the same evidence; mode should stabilize and run_length grow
    for i in range(50):
        last = c.observe(_fv(
            t0 + timedelta(seconds=i),
            realized_vol_annualized=0.40, autocorr_lag1=0.15,
            spread_bps_mean=3.0, disagreement_intensity_bps=2.0,
        ))
    assert last is not None
    assert last.half_life_observations is not None
    assert last.half_life_observations > 1
