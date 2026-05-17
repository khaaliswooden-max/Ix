"""tests/test_planner.py — MCTS planner with complexity penalty."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ix_world_model.contracts.events import (
    HypothesisWeights,
    MarketSnapshot,
    Regime,
    RegimePosterior,
)

from ix_strategy_synthesis.contracts.events import ActionKind, ActionProposal
from ix_strategy_synthesis.planner.complexity_penalty import ComplexityPenaltyConfig
from ix_strategy_synthesis.planner.constraints import ConstraintConfig
from ix_strategy_synthesis.planner.mcts import MCTSPlanner, PlannerConfig


@pytest.fixture
def now():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def _snap(now, regime_probs=None):
    probs = regime_probs or {Regime.TREND_UP: 0.7, Regime.TREND_DOWN: 0.05,
                              Regime.MEAN_REVERT: 0.1, Regime.VOLATILE: 0.1, Regime.BROKEN: 0.05}
    rp = RegimePosterior(
        symbol="BTC/USDT", window_end=now,
        probabilities={r: probs.get(r, 0.0) for r in Regime},
        observations_used=100,
    )
    hw = HypothesisWeights(
        symbol="BTC/USDT", window_end=now,
        weights={"a": 1.0},
        realized_pnl_window={"a": Decimal(0)},
        observations_used=100,
    )
    return MarketSnapshot(symbol="BTC/USDT", window_end=now, regime=rp, hypotheses=hw)


def _prop(now, hyp, action, size, conf, bits):
    return ActionProposal(
        hypothesis=hyp, symbol="BTC/USDT", window_end=now,
        action=action, size_fraction=Decimal(str(size)),
        confidence=conf, description_length_bits=bits, rationale="t",
    )


def test_planner_picks_highest_penalized_value(now):
    snap = _snap(now)
    # Two LONG proposals at same conf and size, different complexity
    simple = _prop(now, "simple", ActionKind.LONG, "0.1", 0.8, bits=4.0)
    complex_ = _prop(now, "complex", ActionKind.LONG, "0.1", 0.8, bits=10.0)
    planner = MCTSPlanner()
    request = planner.choose(snap, [simple, complex_])
    assert request.chosen_proposal == simple


def test_planner_chooses_complex_when_value_justifies(now):
    """A complex strategy with much higher confidence/size can still win."""
    snap = _snap(now)
    weak_simple = _prop(now, "simple", ActionKind.LONG, "0.05", 0.2, bits=4.0)
    strong_complex = _prop(now, "complex", ActionKind.LONG, "0.2", 0.9, bits=10.0)
    planner = MCTSPlanner()
    request = planner.choose(snap, [weak_simple, strong_complex])
    assert request.chosen_proposal == strong_complex


def test_planner_falls_back_to_hold_when_all_below_floor(now):
    snap = _snap(now)
    # High complexity, low value -> penalty drives value negative
    bad = _prop(now, "complex", ActionKind.LONG, "0.01", 0.05, bits=20.0)
    planner = MCTSPlanner(
        complexity_config=ComplexityPenaltyConfig(coefficient=1.0, min_value_to_overcome_penalty=0.0),
        constraint_config=ConstraintConfig(epsilon_floor=0.01),
    )
    request = planner.choose(snap, [bad])
    assert request.chosen_proposal.action == ActionKind.HOLD


def test_planner_risk_off_dominates_long(now):
    snap = _snap(now)
    long = _prop(now, "trend", ActionKind.LONG, "0.2", 0.9, bits=4.0)
    risk_off = _prop(now, "avoider", ActionKind.RISK_OFF, "0", 0.3, bits=7.0)
    planner = MCTSPlanner()
    request = planner.choose(snap, [long, risk_off])
    assert request.chosen_proposal.action == ActionKind.RISK_OFF


def test_planner_rejects_empty_proposals(now):
    snap = _snap(now)
    with pytest.raises(ValueError, match="cannot be empty"):
        MCTSPlanner().choose(snap, [])


def test_planner_rejects_symbol_mismatch(now):
    snap = _snap(now)
    p = ActionProposal(
        hypothesis="x", symbol="ETH/USDT", window_end=now,
        action=ActionKind.LONG, size_fraction=Decimal("0.1"),
        confidence=0.5, description_length_bits=4.0, rationale="t",
    )
    with pytest.raises(ValueError, match="proposal symbol"):
        MCTSPlanner().choose(snap, [p])


def test_planner_request_includes_all_candidates(now):
    snap = _snap(now)
    long = _prop(now, "trend", ActionKind.LONG, "0.1", 0.8, bits=4.0)
    hold = _prop(now, "vol", ActionKind.HOLD, "0", 0.0, bits=6.0)
    planner = MCTSPlanner()
    request = planner.choose(snap, [long, hold])
    assert len(request.candidate_proposals) == 2
    assert long in request.candidate_proposals
    assert hold in request.candidate_proposals


def test_planner_records_regime_context(now):
    snap = _snap(now)
    long = _prop(now, "trend", ActionKind.LONG, "0.1", 0.8, bits=4.0)
    request = MCTSPlanner().choose(snap, [long])
    assert request.regime_at_decision == "trend_up"
    assert request.regime_mode_probability == pytest.approx(0.7)


def test_penalty_amount_reported_correctly(now):
    """The complexity_penalty_applied should equal coefficient × bits of chosen."""
    snap = _snap(now)
    # size 0.2 × conf 0.9 × 10 = raw 1.8; at coefficient 0.1 × 4 bits = 0.4 penalty
    # penalized = 1.4, well above floor, so chosen survives without fallback.
    p = _prop(now, "trend", ActionKind.LONG, "0.2", 0.9, bits=4.0)
    planner = MCTSPlanner(complexity_config=ComplexityPenaltyConfig(coefficient=0.1))
    request = planner.choose(snap, [p])
    assert request.chosen_proposal == p
    assert request.complexity_penalty_applied == pytest.approx(0.4)  # 0.1 × 4
