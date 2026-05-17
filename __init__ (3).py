"""
ix_substrate.feeds.dex
───────────────────────
EXTENSION POINT — Phase 0.2.

DEX feeds (Uniswap V3, Curve, Raydium, etc.) require:
  - web3.py / ethers.py per-chain client
  - pool address registry per (symbol, chain)
  - tick-to-price conversion (Uniswap V3 uses sqrtPriceX96 ticks)
  - gas-aware effective-price computation (real fill ≠ pool mid for
    non-trivial size; downstream Phase 3 cares about this)

DESIGN NOTE
───────────
DEX disagreement vs CEX is the highest-value class of disagreement we
expect to detect: it reflects bridging cost, MEV extraction, and
cross-venue liquidity stress simultaneously. Implement BEFORE additional
CEX venues.

For v0.1 we ship the protocol and a deliberately broken implementation
that raises so callers know to plug in real chain reads.
"""
from __future__ import annotations

from typing import AsyncIterator

from ..contracts.events import TickEvent


class DexFeed:
    """Placeholder. See module docstring for implementation plan."""

    def __init__(self, venue: str, symbol: str, chain: str, pool_address: str):
        self.venue = venue
        self.symbols = (symbol,)
        self.chain = chain
        self.pool_address = pool_address

    async def __aenter__(self) -> "DexFeed":
        raise NotImplementedError(
            "DexFeed is a Phase 0.2 extension point. "
            "Implementation plan: web3.py async client → pool sqrtPriceX96 "
            "reads → tick conversion → TickEvent emission. See module docstring."
        )

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def stream(self) -> AsyncIterator[TickEvent]:
        raise NotImplementedError
        yield  # pragma: no cover  (makes this a generator)
