"""Shared stage timing and output-limit interpretation helpers."""


def detect_output_limit(
    *,
    stop_reason: str | None,
    generated_tokens: int | None,
    output_budget: int,
) -> bool | None:
    """Identify a likely cap stop; preserve unknown when Ollama omits evidence."""

    if stop_reason is not None and stop_reason.lower() in {
        "length",
        "max_tokens",
        "max_token",
        "token_limit",
        "limit",
    }:
        return True
    if generated_tokens is not None:
        return generated_tokens >= output_budget
    if stop_reason is not None and stop_reason.lower() in {"stop", "end_turn", "eos"}:
        return False
    return None
