import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ModelCall:
    """
    This class tracks model calls.
    It separates kind: generation (regular model or calculator_tool tool call) and embedding (direct RAG call or search_docs tool call).

    The input_tokens and output_tokens are the number of tokens used for the model call.
    """
    kind: str
    model: str
    input_tokens: int
    output_tokens: int = 0  # embeddings don't have output tokens


@dataclass
class UsageCollector:
    calls: list[ModelCall] = field(default_factory=list)

    def record(self, kind: str, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        self.calls.append(ModelCall(kind, model, input_tokens, output_tokens))

    @property
    def total_input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)


# the ambient corrent collector
_current: contextvars.ContextVar[UsageCollector | None] = contextvars.ContextVar(
    "usage_collector",
    default=None,
)


def current_collector() -> UsageCollector | None:
    return _current.get()


@contextmanager
def collect_usage() -> Iterator[UsageCollector]:
    collector = UsageCollector()
    token = _current.set(collector)
    try:
        yield collector
    finally:
        _current.reset(token)  # clean teardown - critical for correctness


# OpenAI pricing per 1M tokens
PRICING = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}


def estimate_cost(collector: UsageCollector) -> float:
    total = 0.0
    for call in collector.calls:
        pricing = PRICING.get(call.model)
        if pricing:
            total += (call.input_tokens * pricing["input"] +
                      call.output_tokens * pricing["output"]) / 1_000_000
    return total
