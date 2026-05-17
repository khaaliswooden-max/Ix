"""tests/test_orchestrator.py — full Phase 2 wire-up and feedback loop."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ix_world_model.contracts.events import (
    HypothesisWeights,
    MarketSnapshot,
    Regime,
    RegimePosterior,
)

from ix_strategy_synthesis.contracts.events import ActionKind
from ix_strategy_synthesis.hypotheses.specialists import default_roster_names
from ix_strategy_synthesis.synthesis.orchestrator import (
    FillSimulator,
    FillSimulatorConfig,
    StrategySynthesizer,
    SynthesizerConfig,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def _snap(now, regime_probs):
    rp = RegimePosterior(
        symbol="BTC/USDT", window_end=now,
        probabilities={r: regime_probs.get(r, 0.0) for r in Regime},
        observations_used=100,
    )
    hw = HypothesisWeights(
        symbol="BTC/USDT", window_end=now,
        weights={n: 1.0 / 5 for n in default_roster_names()},
        realized_pnl_window={n: Decimal(0) for n in default_roster_names()},
        observations_used=100,
    )
    return MarketSnapshot(symbol="BTC/USDT", window_end=now, regime=rp, hypotheses=hw)


def test_synthesizer_emits_long_in_trend_up(now):
    snap = _snap(now, {Regime.TREND_UP: 0.7, Regime.TREND_DOWN: 0.05,
                       Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.1, Regime.BROKEN: 0.05})
    synth = StrategySynthesizer(SynthesizerConfig(symbol="BTC/USDT"))
    request, report = synth.step(snap)
    assert request.chosen_proposal.action == ActionKind.LONG
    assert request.chosen_proposal.hypothesis == "trend_follower"
    assert report is None  # no fill simulator configured


def test_synthesizer_emits_risk_off_in_broken(now):
    snap = _snap(now, {Regime.TREND_UP: 0.1, Regime.TREND_DOWN: 0.1,
                       Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.2, Regime.BROKEN: 0.5})
    synth = StrategySynthesizer(SynthesizerConfig(symbol="BTC/USDT"))
    request, _ = synth.step(snap)
    assert request.chosen_proposal.action == ActionKind.RISK_OFF
    assert request.chosen_proposal.hypothesis == "liquidation_avoider"


def test_synthesizer_produces_feedback_report_when_simulator_attached(now):
    snap = _snap(now, {Regime.TREND_UP: 0.7, Regime.TREND_DOWN: 0.05,
                       Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.1, Regime.BROKEN: 0.05})
    synth = StrategySynthesizer(SynthesizerConfig(
        symbol="BTC/USDT",
        fill_simulator=FillSimulator(FillSimulatorConfig(seed=42)),
    ))
    _, report = synth.step(snap)
    assert report is not None
    assert report.source == "stub"
    # All five hypotheses must report something (0 for non-acting)
    for name in default_roster_names():
        assert name in report.pnl_by_hypothesis


def test_synthesizer_rejects_wrong_symbol(now):
    snap = _snap(now, {Regime.TREND_UP: 0.7, Regime.TREND_DOWN: 0.05,
                       Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.1, Regime.BROKEN: 0.05})
    synth = StrategySynthesizer(SynthesizerConfig(symbol="ETH/USDT"))
    with pytest.raises(ValueError, match="snapshot symbol"):
        synth.step(snap)


def test_feedback_loop_window_tracks_steps(now):
    snap1 = _snap(now, {Regime.TREND_UP: 0.7, Regime.TREND_DOWN: 0.05,
                        Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.1, Regime.BROKEN: 0.05})
    snap2 = _snap(now + timedelta(minutes=1),
                  {Regime.TREND_UP: 0.7, Regime.TREND_DOWN: 0.05,
                   Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.1, Regime.BROKEN: 0.05})
    synth = StrategySynthesizer(SynthesizerConfig(
        symbol="BTC/USDT",
        fill_simulator=FillSimulator(FillSimulatorConfig(seed=42)),
    ))
    _, r1 = synth.step(snap1)
    _, r2 = synth.step(snap2)
    # First report's window is degenerate (start == end at first step)
    assert r1.window_start == snap1.window_end
    assert r1.window_end == snap1.window_end
    # Second report's window spans from previous to current
    assert r2.window_start == snap1.window_end
    assert r2.window_end == snap2.window_end


def test_hypothesis_names_property_matches_default_roster():
    synth = StrategySynthesizer(SynthesizerConfig(symbol="BTC/USDT"))
    assert synth.hypothesis_names == default_roster_names()


def test_trend_follower_pnl_positive_when_correct_in_trend_up(now):
    """End-to-end: trend follower goes long in TREND_UP -> simulator says correct -> positive PnL."""
    snap = _snap(now, {Regime.TREND_UP: 0.8, Regime.TREND_DOWN: 0.05,
                       Regime.MEAN_REVERT: 0.05, Regime.VOLATILE: 0.05, Regime.BROKEN: 0.05})
    synth = StrategySynthesizer(SynthesizerConfig(
        symbol="BTC/USDT",
        fill_simulator=FillSimulator(FillSimulatorConfig(noise_std=0.0, seed=42)),
    ))
    _, report = synth.step(snap)
    assert report.pnl_by_hypothesis["trend_follower"] > 0
