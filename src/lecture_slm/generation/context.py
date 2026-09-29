"""Conservative context-tier selection from assembled request text."""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextSelection:
    estimated_input_tokens: int
    required_tokens_with_margin: int
    selected_context: int


def estimate_tokens_from_characters(text: str) -> int:
    """Estimate tokens conservatively as one token per four input characters."""

    return math.ceil(len(text) / 4)


def select_context_tier(
    assembled_input: str,
    context_tiers: Sequence[int],
    *,
    safety_margin: float,
    output_reserve_tokens: int = 0,
) -> ContextSelection:
    """Select the smallest configured tier that fits estimated input with margin."""

    if not context_tiers:
        raise ValueError("at least one context tier is required")
    if safety_margin < 1.0:
        raise ValueError("context safety margin must be at least 1.0")
    if output_reserve_tokens < 0:
        raise ValueError("output reserve must not be negative")
    sorted_tiers = sorted(set(context_tiers))
    if any(tier <= 0 for tier in sorted_tiers):
        raise ValueError("context tiers must be positive")
    estimated_tokens = estimate_tokens_from_characters(assembled_input)
    required_tokens = math.ceil(estimated_tokens * safety_margin) + output_reserve_tokens
    for tier in sorted_tiers:
        if tier >= required_tokens:
            return ContextSelection(estimated_tokens, required_tokens, tier)
    raise ValueError(
        f"estimated input needs {required_tokens} tokens including margin, "
        f"but configured tiers stop at {max(sorted_tiers)}"
    )
