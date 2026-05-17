"""
examples/run_world_model_demo.py
─────────────────────────────────
Offline demo: drive synthetic regime transitions through the world model
and watch it detect and report them.

We generate three phases:
  1. Trending up: positive drift, autocorrelated returns
  2. Volatile: high variance, no autocorrelation
  3. Mean-reverting: alternating signed returns

For each phase we check that the world model's regime posterior
concentrates on the expected regime. Also exercises the hypothesis
ensemble by reporting different per-hypothesis PnL profiles.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ix_substrate.contracts.events import FeedKind, TickEvent

from ix_world_model.contracts.events import Regime
from ix_world_model.inference.orchestrator import (
    WorldModel,
    WorldModelConfig,
)
from ix_world_model.regimes.features import FeatureConfig


def _tick(symbol: str, mid: Decimal, ts: datetime) -> TickEvent:
    return TickEvent(
        feed_kind=FeedKind.CEX, venue="binance", symbol=symbol,
        venue_ts=ts, ingress_ts=ts,
        bid_price=mid * Decimal("0.9999"), bid_size=Decimal("1"),
        ask_price=mid * Decimal("1.0001"), ask_size=Decimal("1"),
    )


def main() -> None:
    random.seed(0)
    # Default profiles are calibrated for minute-cadence. We synthesize at
    # minute cadence here. See ADR-0003 and classifier.py for cadence notes.
    minutes_per_year = 365.0 * 24.0 * 60.0
    wm = WorldModel(WorldModelConfig(
        symbol="BTC/USDT",
        hypothesis_names=("trend_follower", "mean_reverter", "vol_harvester"),
        feature_config=FeatureConfig(annualization_factor=minutes_per_year),
    ))

    t = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    base = Decimal("65000")

    print("Phase 1: TREND_UP (400 1-minute ticks)")
    print("-" * 60)
    last_ret = 0.0
    for i in range(400):
        t += timedelta(minutes=1)
        # Strong positive autocorrelation: 0.7 weight on prior return
        # → autocorr ~0.4, vol ~30% annualized, drift positive.
        noise = 4e-4 * random.gauss(0, 1)
        ret = 0.7 * last_ret + 1e-4 + noise
        last_ret = ret
        base = base * Decimal(str(1.0 + ret))
        wm.observe_tick(_tick("BTC/USDT", base, t))
        wm.report_hypothesis_pnl({
            "trend_follower": Decimal("5"),
            "mean_reverter": Decimal("-2"),
            "vol_harvester": Decimal("0"),
        })

    snap = wm.snapshot()
    print(f"  mode: {snap.regime.mode.value}  P={snap.regime.mode_probability:.3f}")
    print(f"  realized vol annualized: "
          f"{wm._features._vol.std * (minutes_per_year ** 0.5):.3f}")
    print(f"  autocorr lag-1: {wm._features._autocorr.lag1:.4f}")
    print(f"  hypothesis leader: {snap.hypotheses.leader} "
          f"(weight {snap.hypotheses.leader_weight:.3f})")
    print(f"  ensemble disagreement: {snap.hypotheses.disagreement_nats:.3f} nats")

    print("\nPhase 2: VOLATILE (400 1-minute ticks)")
    print("-" * 60)
    for i in range(400):
        t += timedelta(minutes=1)
        # ~100% annualized vol, no drift
        ret = 1.4e-3 * random.gauss(0, 1)
        base = base * Decimal(str(1.0 + ret))
        wm.observe_tick(_tick("BTC/USDT", base, t))
        wm.report_hypothesis_pnl({
            "trend_follower": Decimal("-3"),
            "mean_reverter": Decimal("-1"),
            "vol_harvester": Decimal("4"),
        })

    snap = wm.snapshot()
    print(f"  mode: {snap.regime.mode.value}  P={snap.regime.mode_probability:.3f}")
    print(f"  realized vol annualized: "
          f"{wm._features._vol.std * (minutes_per_year ** 0.5):.3f}")
    print(f"  autocorr lag-1: {wm._features._autocorr.lag1:.4f}")
    print(f"  hypothesis leader: {snap.hypotheses.leader} "
          f"(weight {snap.hypotheses.leader_weight:.3f})")
    print(f"  ensemble disagreement: {snap.hypotheses.disagreement_nats:.3f} nats")

    print("\nPhase 3: MEAN_REVERT (400 1-minute ticks)")
    print("-" * 60)
    last_ret = 0.0
    for i in range(400):
        t += timedelta(minutes=1)
        # ~30% annualized vol with strong negative autocorrelation
        noise = 4e-4 * random.gauss(0, 1)
        ret = -0.5 * last_ret + noise
        last_ret = ret
        base = base * Decimal(str(1.0 + ret))
        wm.observe_tick(_tick("BTC/USDT", base, t))
        wm.report_hypothesis_pnl({
            "trend_follower": Decimal("-2"),
            "mean_reverter": Decimal("6"),
            "vol_harvester": Decimal("0"),
        })

    snap = wm.snapshot()
    print(f"  mode: {snap.regime.mode.value}  P={snap.regime.mode_probability:.3f}")
    print(f"  realized vol annualized: "
          f"{wm._features._vol.std * (minutes_per_year ** 0.5):.3f}")
    print(f"  autocorr lag-1: {wm._features._autocorr.lag1:.4f}")
    print(f"  hypothesis leader: {snap.hypotheses.leader} "
          f"(weight {snap.hypotheses.leader_weight:.3f})")
    print(f"  ensemble disagreement: {snap.hypotheses.disagreement_nats:.3f} nats")

    print("\nDone.")


if __name__ == "__main__":
    main()
