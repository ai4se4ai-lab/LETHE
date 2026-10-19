"""Immutable, validated run configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AgentConfig(FrozenModel):
    n: int = Field(default=4, ge=1, le=32)
    provider: Literal["anthropic", "openai_compat", "mock"] = "mock"
    model: str = "mock-scripted"
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    max_tool_calls_per_turn: int = Field(default=40, ge=1)


class ReducerConfig(FrozenModel):
    share_peers: bool = True
    memory: Literal["candidate_only", "candidate_plus_facts", "none"] = (
        "candidate_plus_facts"
    )
    show_history: Literal["own_all", "own_best_only"] = "own_all"
    views: tuple[str, ...] | None = None
    granularity_schedule: tuple[Literal["method", "statement"], ...] = (
        "method",
        "statement",
    )


class ForgetConfig(FrozenModel):
    policy: Literal["never", "fixed", "plateau"] = "fixed"
    rounds: int = Field(default=4, ge=1)
    plateau_window: int = Field(default=8, ge=2)
    plateau_epsilon: float = Field(default=0.001, ge=0.0)
    jaccard_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class JudgeConfig(FrozenModel):
    signals: tuple[Literal["fixprobe", "patch_validation", "spectrum_prior"], ...] = (
        "fixprobe",
        "patch_validation",
        "spectrum_prior",
    )
    fix_weight: float = Field(default=0.70, ge=0.0)
    progress_weight: float = Field(default=0.25, ge=0.0)
    spectrum_weight: float = Field(default=0.05, ge=0.0)
    max_mutants_per_element: int = Field(default=60, ge=1)
    mutant_timeout_s: int = Field(default=30, ge=1)
    top_k: int = Field(default=10, ge=1, le=10)


class PriceConfig(FrozenModel):
    input_per_mtok: float = Field(default=0.0, ge=0.0)
    output_per_mtok: float = Field(default=0.0, ge=0.0)
    reasoning_per_mtok: float = Field(default=0.0, ge=0.0)
    cache_read_per_mtok: float = Field(default=0.0, ge=0.0)
    cache_write_per_mtok: float = Field(default=0.0, ge=0.0)


class RunSection(FrozenModel):
    name: str = "lethe"
    budget_usd: float = Field(default=2.0, gt=0.0)
    seed: int = 1


class RunConfig(FrozenModel):
    run: RunSection = Field(default_factory=RunSection)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    reducer: ReducerConfig = Field(default_factory=ReducerConfig)
    forget: ForgetConfig = Field(default_factory=ForgetConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    prices: PriceConfig = Field(default_factory=PriceConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.model_validate(raw)
