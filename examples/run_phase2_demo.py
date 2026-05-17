"""
examples/run_phase2_demo.py
────────────────────────────
End-to-end demo: drive synthetic ticks through Phase 0+1+2 and watch
the agent choose actions per regime, with the complexity penalty
visibly affecting selection.

For each of three synthetic regimes (TREND_UP, VOLATILE, MEAN_REVERT),
we:
  1. Feed the world model 400 minute-cadence ticks.
  2. At each tick, snapshot the world model and ask Phase 2 to choose
     an action.
  3. Report fill-simulator PnL back to Phase 1's ensemble.
  4. Summarize at the end of each phase.

No Phase 0 substrate runs here — we synthesize ticks directly and feed
them to the world model. The full Phase 0+1+2 integration test
(real CEX feeds) is left as a manual exercise.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ix_substrate.contracts.events import FeedKind, TickEvent

from ix_world_model.inference.orchestrator import (
    WorldModel,
    WorldModelConfig,
)
from ix_world_model.regimes.features import FeatureConfig

from ix_strategy_synthesis.hypotheses.specialists import default_roster_names
from ix_strategy_synthesis.synthesis.orchestrator import (
    FillSimulator,
    FillSimulatorConfig,
    StrategySynthesizer,
    SynthesizerConfig,
)


def _tick(symbol: str, mid: Decimal, ts: datetime) -> TickEvent:
    return TickEvent(
        feed_kind=FeedKind.CEX, venue="binance", symbol=symbol,
        venue_ts=ts, ingress_ts=ts,
        bid_price=mid * Decimal("0.9999"), bid_size=Decimal("1"),
        ask_price=mid * Decimal("1.0001"), ask_size=Decimal("1"),
    )


def run_phase(
    label: str,
    n_ticks: int,
    return_generator,
    wm: WorldModel,
    synth: StrategySynthesizer,
    start_ts: datetime,
    start_price: Decimal,
):
    print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    t = start_ts
    price = start_price
    action_counts: dict[str, int] = {}
    cumulative_pnl: dict[str, Decimal] = {n: Decimal(0) for n in default_roster_names()}

    for i in range(n_ticks):
        t += timedelta(minutes=1)
        ret = return_generator(i)
        price = price * Decimal(str(1.0 + ret))
        wm.observe_tick(_tick("BTC/USDT", price, t))

        snapshot = wm.snapshot()
        if snapshot is None:
            continue
        request, report = synth.step(snapshot)
        if report is not None:
            for name, pnl in report.pnl_by_hypothesis.items():
                cumulative_pnl[name] = cumulative_pnl[name] + pnl
            wm.report_hypothesis_pnl(report.pnl_by_hypothesis)

        chosen_hyp = request.chosen_proposal.hypothesis
        action_counts[chosen_hyp] = action_counts.get(chosen_hyp, 0) + 1

    snap = wm.snapshot()
    print(f"  detected regime: {snap.regime.mode.value} "
          f"(P={snap.regime.mode_probability:.3f})")
    print(f"  action selection (planner chose):")
    for hyp, n in sorted(action_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {hyp:>22}: {n:>4} ticks ({100*n/sum(action_counts.values()):>5.1f}%)")
    print(f"  cumulative simulated PnL per hypothesis:")
    for name in default_roster_names():
        pnl = cumulative_pnl[name]
        print(f"    {name:>22}: {float(pnl):>+10.2f}")
    print(f"  Phase 1 ensemble leader: {snap.hypotheses.leader} "
          f"(weight {snap.hypotheses.leader_weight:.3f})")

    return t, price


def main() -> None:
    random.seed(0)
    minutes_per_year = 365.0 * 24.0 * 60.0

    wm = WorldModel(WorldModelConfig(
        symbol="BTC/USDT",
        hypothesis_names=default_roster_names(),
        feature_config=FeatureConfig(annualization_factor=minutes_per_year),
    ))
    synth = StrategySynthesizer(SynthesizerConfig(
        symbol="BTC/USDT",
        fill_simulator=FillSimulator(FillSimulatorConfig(seed=7)),
    ))

    t = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    price = Decimal("65000")

    # ── Phase A: TREND_UP ──────────────────────────────────────────
    last_ret = [0.0]

    def trend_up_returns(i: int) -> float:
        noise = 4e-4 * random.gauss(0, 1)
        ret = 0.7 * last_ret[0] + 1e-4 + noise
        last_ret[0] = ret
        return ret

    t, price = run_phase(
        "Phase A: TREND_UP — expect trend_follower to dominate",
        400, trend_up_returns, wm, synth, t, price,
    )

    # ── Phase B: VOLATILE ──────────────────────────────────────────
    def volatile_returns(i: int) -> float:
        return 1.4e-3 * random.gauss(0, 1)

    t, price = run_phase(
        "Phase B: VOLATILE — expect vol_harvester to be most-frequent but PnL flat",
        400, volatile_returns, wm, synth, t, price,
    )

    # ── Phase C: MEAN_REVERT ──────────────────────────────────────
    last_ret2 = [0.0]

    def mean_revert_returns(i: int) -> float:
        noise = 4e-4 * random.gauss(0, 1)
        ret = -0.5 * last_ret2[0] + noise
        last_ret2[0] = ret
        return ret

    t, price = run_phase(
        "Phase C: MEAN_REVERT — expect mean_reverter to dominate",
        400, mean_revert_returns, wm, synth, t, price,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
