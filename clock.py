"""
ix_substrate.cli
─────────────────
`ix-substrate run --config configs/substrate.example.yaml`

Brings up all configured feeds concurrently, aligns ticks pairwise, runs
the disagreement detector, persists both ticks and disagreement events.

This is intentionally simple: no supervision tree, no graceful-restart,
no metrics endpoint. Those belong in v0.2 once the contract surface
stabilizes.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from decimal import Decimal
from itertools import combinations
from pathlib import Path

from .config import SubstrateConfig
from .contracts.events import TickEvent
from .feeds.cex import CexFeed
from .reconciliation.alignment import AlignConfig, align_two
from .reconciliation.clock import ClockSkewTracker
from .reconciliation.disagreement import (
    DisagreementConfig,
    PriceDisagreementDetector,
)
from .storage.duckdb_store import DuckDBStore


log = logging.getLogger("ix_substrate")


async def _run_feed(feed: CexFeed, queue: asyncio.Queue[TickEvent]) -> None:
    async with feed:
        async for tick in feed.stream():
            await queue.put(tick)


async def _consume(
    queue: asyncio.Queue[TickEvent],
    store: DuckDBStore,
    detector: PriceDisagreementDetector,
    skew: ClockSkewTracker,
    cfg: SubstrateConfig,
    stop: asyncio.Event,
) -> None:
    by_venue: dict[str, TickEvent] = {}
    tick_batch: list[TickEvent] = []

    while not stop.is_set():
        try:
            tick = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if tick_batch:
                store.write_ticks(tick_batch)
                tick_batch.clear()
            continue

        tick_batch.append(tick)
        skew.observe(tick)
        prev = by_venue.get(tick.venue)
        by_venue[tick.venue] = tick

        # Naïve pairwise comparison: every new tick vs the most recent
        # tick of every other venue on the same symbol.
        for other_venue, other_tick in by_venue.items():
            if other_venue == tick.venue:
                continue
            if other_tick.symbol != tick.symbol:
                continue
            event = detector.observe(tick, other_tick)
            if event is not None:
                store.write_disagreements([event])
                log.info(
                    "disagreement: %s %s vs %s = %s bps half_life=%s",
                    tick.symbol,
                    event.venues[0], event.venues[1],
                    event.magnitude, event.half_life_ms,
                )

        if len(tick_batch) >= 32:
            store.write_ticks(tick_batch)
            tick_batch.clear()

    if tick_batch:
        store.write_ticks(tick_batch)


async def _main(cfg: SubstrateConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    queue: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=4096)
    feeds = [
        CexFeed(venue=f.venue, symbol=f.symbol, poll_ms=f.poll_ms)
        for f in cfg.feeds
        if f.kind == "cex"
    ]
    if len(feeds) < 2:
        raise SystemExit("substrate needs at least two feeds for disagreement work")

    detector = PriceDisagreementDetector(DisagreementConfig(
        threshold_bps=Decimal(str(cfg.threshold_bps)),
        expected_band_multiplier=Decimal(str(cfg.expected_band_multiplier)),
    ))
    skew = ClockSkewTracker()
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    with DuckDBStore(cfg.db_path) as store:
        feed_tasks = [asyncio.create_task(_run_feed(f, queue)) for f in feeds]
        consumer = asyncio.create_task(
            _consume(queue, store, detector, skew, cfg, stop)
        )
        await stop.wait()
        for t in feed_tasks:
            t.cancel()
        await asyncio.gather(*feed_tasks, return_exceptions=True)
        await consumer


def main() -> None:
    ap = argparse.ArgumentParser(prog="ix-substrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="Run the substrate")
    runp.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    if args.cmd == "run":
        cfg = SubstrateConfig.load(args.config)
        asyncio.run(_main(cfg))


if __name__ == "__main__":
    main()
