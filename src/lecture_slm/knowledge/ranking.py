"""Deterministic reciprocal-rank fusion."""


def reciprocal_rank_fusion(
    lexical_ranks: dict[str, int],
    semantic_ranks: dict[str, int],
    *,
    constant: int = 60,
) -> dict[str, float]:
    """Fuse ranked lists without combining their incomparable raw score scales."""

    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    fused: dict[str, float] = {}
    for rank_map in (lexical_ranks, semantic_ranks):
        for chunk_id, rank in rank_map.items():
            if rank <= 0:
                raise ValueError("retrieval ranks must be positive")
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (constant + rank)
    return fused
