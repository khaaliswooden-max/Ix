"""tests/test_constraints.py — hard-constraint filtering."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ix_strategy_synthesis.contracts.events import ActionKind, ActionProposal
from ix_strategy_synthesis.planner.constraints import (
    ConstraintConfig,
    filter_and_clip,
)


@pytest.fixture
def now():
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)


def _p(now, hyp, action, size, conf, bits=4.0):
    return ActionProposal(
        hypothesis=hyp, symbol="BTC/USDT", window_end=now,
        action=action, size_fraction=Decimal(str(size)),
        confidence=conf, description_length_bits=bits, rationale="t",
    )


def test_risk_off_dominates_other_proposals(now):
    long = _p(now, "trend", ActionKind.LONG, "0.1", 0.8)
    risk = _p(now, "avoider", ActionKind.RISK_OFF, "0", 0.7)
    out = filter_and_clip([long, risk])
    assert len(out) == 1
    assert out[0].action == ActionKind.RISK_OFF


def test_highest_confidence_risk_off_wins(now):
    risk_lo = _p(now, "a", ActionKind.RISK_OFF, "0", 0.3)
    risk_hi = _p(now, "b", ActionKind.RISK_OFF, "0", 0.9)
    out = filter_and_clip([risk_lo, risk_hi])
    assert len(out) == 1
    assert out[0].confidence == 0.9
    assert out[0].hypothesis == "b"


def test_epsilon_floor_rejects_low_confidence_actions(now):
    low = _p(now, "trend", ActionKind.LONG, "0.1", 0.01)
    out = filter_and_clip([low], ConstraintConfig(epsilon_floor=0.05))
    assert out[0].action == ActionKind.HOLD
    assert "rejected" in out[0].rationale


def test_size_is_clipped_to_max(now):
    big = _p(now, "trend", ActionKind.LONG, "0.8", 0.9)
    out = filter_and_clip([big], ConstraintConfig(max_size_fraction=Decimal("0.25")))
    assert out[0].size_fraction == Decimal("0.25")
    assert out[0].action == ActionKind.LONG


def test_size_clipping_preserves_sign(now):
    big_short = _p(now, "trend", ActionKind.SHORT, "-0.8", 0.9)
    out = filter_and_clip([big_short], ConstraintConfig(max_size_fraction=Decimal("0.25")))
    assert out[0].size_fraction == Decimal("-0.25")
    assert out[0].action == ActionKind.SHORT


def test_hold_proposals_pass_through_floor(now):
    """HOLD proposals have confidence 0 by design; they should not be rejected."""
    hold = _p(now, "x", ActionKind.HOLD, "0", 0.0)
    out = filter_and_clip([hold], ConstraintConfig(epsilon_floor=0.05))
    assert out[0].action == ActionKind.HOLD
