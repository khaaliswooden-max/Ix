"""
ix_substrate.feeds.onchain
───────────────────────────
EXTENSION POINT — Phase 0.3.

Raw on-chain events (large transfers, contract calls, mempool tx) are
different from DexFeed: they are not price ticks but events. The
substrate emits them via a separate event contract (OnchainEvent, to be
specified in v0.2) rather than TickEvent.

Reason for separation: the World Model (Phase 1) consumes ticks via the
Bayesian regime classifier; on-chain events feed the orderflow/liquidity
posterior, which is architecturally distinct.
"""
from __future__ import annotations
