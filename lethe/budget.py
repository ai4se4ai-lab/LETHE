"""Exact provider-usage accounting with a hard dollar cap."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PriceConfig
from .schemas import Usage


@dataclass
class BudgetMeter:
    cap_usd: float
    prices: PriceConfig
    spent_usd: float = 0.0

    def cost(self, usage: Usage) -> float:
        p = self.prices
        return (
            usage.input_tokens * p.input_per_mtok
            + usage.output_tokens * p.output_per_mtok
            + usage.reasoning_tokens * p.reasoning_per_mtok
            + usage.cache_read_tokens * p.cache_read_per_mtok
            + usage.cache_write_tokens * p.cache_write_per_mtok
        ) / 1_000_000

    def charge(self, usage: Usage) -> float:
        amount = self.cost(usage)
        self.spent_usd += amount
        return amount

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.cap_usd
