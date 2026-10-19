"""Versioned schemas shared across processes and persisted artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION


class Hypothesis(Artifact):
    rank: int = Field(ge=1)
    element: str = Field(min_length=1)
    granularity: Literal["method", "statement"]
    rationale: str = Field(default="", max_length=500)


class Candidate(Artifact):
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=10)
    notes: str = ""

    @model_validator(mode="after")
    def validate_ranking(self) -> "Candidate":
        ranks = [h.rank for h in self.hypotheses]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("hypothesis ranks must be contiguous and start at 1")
        elements = [h.element for h in self.hypotheses]
        if len(elements) != len(set(elements)):
            raise ValueError("candidate contains duplicate elements")
        return self


class TestTrace(Artifact):
    id: str
    outcome: Literal["pass", "fail", "error", "skipped"]
    duration_s: float = Field(default=0.0, ge=0.0)
    failure_message: str = ""
    stack_trace: str = ""


class TraceModel(Artifact):
    tests: list[TestTrace] = Field(default_factory=list)
    covers: dict[str, set[str]] = Field(default_factory=dict)
    failing_slice: set[str] = Field(default_factory=set)


class Fact(Artifact):
    element_id: str
    kind: Literal["exonerated", "confirmed", "inconclusive"]
    evidence_ref: str
    source_agent: int | None = None
    round: int = Field(default=0, ge=0)
    mutants_run: int = Field(default=0, ge=0)


class Usage(Artifact):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)


class Completion(Artifact):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str
    latency_s: float = Field(default=0.0, ge=0.0)
    stop_reason: str = ""
