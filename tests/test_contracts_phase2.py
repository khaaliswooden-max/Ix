"""tests/test_contracts.py — Phase 2 emitted contracts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ix_strategy_synthesis.contracts.events import (
    ActionKind,
    ActionProposal,
    ExecutionRequest,
    RealizationReport,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


# ── ActionProposal ───────────────────────────────────────────────────────

def _long_prop(now, **overrides):
    base = dict(
        hypothesis="x", symbol="BTC/USDT", window_end=now,
        action=ActionKind.LONG, size_fraction=Decimal("0.1"),
        confidence=0.5, description_length_bits=4.0, rationale="t",
    )
    base.update(overrides)
    return ActionProposal(**base)


def test_long_requires_positive_size(now):
    with pytest.raises(ValidationError, match="LONG action requires size_fraction > 0"):
        _long_prop(now, size_fraction=Decimal("0"))


def test_short_requires_negative_size(now):
    with pytest.raises(ValidationError, match="SHORT action requires size_fraction < 0"):
        ActionProposal(
            hypothesis="x", symbol="BTC/USDT", window_end=now,
            action=ActionKind.SHORT, size_fraction=Decimal("0.1"),
            confidence=0.5, description_length_bits=4.0, rationale="t",
        )


def test_hold_requires_zero_size(now):
    with pytest.raises(ValidationError, match="hold action requires size_fraction == 0"):
        ActionProposal(
            hypothesis="x", symbol="BTC/USDT", window_end=now,
            action=ActionKind.HOLD, size_fraction=Decimal("0.1"),
            confidence=0.0, description_length_bits=1.0, rationale="t",
        )


def test_risk_off_requires_zero_size(now):
    with pytest.raises(ValidationError, match="risk_off action requires size_fraction == 0"):
        ActionProposal(
            hypothesis="x", symbol="BTC/USDT", window_end=now,
            action=ActionKind.RISK_OFF, size_fraction=Decimal("0.1"),
            confidence=0.5, description_length_bits=4.0, rationale="t",
        )


def test_size_bounded_to_unit_interval(now):
    with pytest.raises(ValidationError):
        _long_prop(now, size_fraction=Decimal("1.5"))


def test_description_length_bits_minimum(now):
    with pytest.raises(ValidationError):
        _long_prop(now, description_length_bits=0.5)


def test_confidence_bounded(now):
    with pytest.raises(ValidationError):
        _long_prop(now, confidence=1.5)


def test_proposal_is_frozen(now):
    p = _long_prop(now)
    with pytest.raises(ValidationError):
        p.size_fraction = Decimal("0.2")  # type: ignore


# ── ExecutionRequest ─────────────────────────────────────────────────────

def test_request_requires_chosen_in_candidates(now):
    a = _long_prop(now)
    b = _long_prop(now, hypothesis="other")
    # `a` is not in candidates [b]
    with pytest.raises(ValidationError, match="chosen_proposal must be one of"):
        ExecutionRequest(
            symbol="BTC/USDT", window_end=now,
            chosen_proposal=a, candidate_proposals=(b,),
            planner_value=1.0, complexity_penalty_applied=0.5,
            regime_at_decision="trend_up", regime_mode_probability=0.7,
        )


def test_request_symbol_must_match(now):
    a = _long_prop(now)
    with pytest.raises(ValidationError, match="does not match"):
        ExecutionRequest(
            symbol="ETH/USDT", window_end=now,
            chosen_proposal=a, candidate_proposals=(a,),
            planner_value=1.0, complexity_penalty_applied=0.5,
            regime_at_decision="trend_up", regime_mode_probability=0.7,
        )


def test_request_valid(now):
    a = _long_prop(now)
    req = ExecutionRequest(
        symbol="BTC/USDT", window_end=now,
        chosen_proposal=a, candidate_proposals=(a,),
        planner_value=1.0, complexity_penalty_applied=0.5,
        regime_at_decision="trend_up", regime_mode_probability=0.7,
    )
    assert req.chosen_proposal == a


# ── RealizationReport ────────────────────────────────────────────────────

def test_report_rejects_empty_pnl_map(now):
    with pytest.raises(ValidationError, match="cannot be empty"):
        RealizationReport(
            symbol="BTC/USDT",
            window_start=now, window_end=now,
            pnl_by_hypothesis={}, source="stub",
        )


def test_report_rejects_inverted_window(now):
    with pytest.raises(ValidationError, match="window_end must be >="):
        RealizationReport(
            symbol="BTC/USDT",
            window_start=now + timedelta(seconds=10), window_end=now,
            pnl_by_hypothesis={"a": Decimal("0")}, source="stub",
        )


def test_report_valid(now):
    r = RealizationReport(
        symbol="BTC/USDT",
        window_start=now, window_end=now + timedelta(seconds=60),
        pnl_by_hypothesis={"trend_follower": Decimal("10.5"), "mean_reverter": Decimal("-2")},
        source="stub",
    )
    assert r.pnl_by_hypothesis["trend_follower"] == Decimal("10.5")
