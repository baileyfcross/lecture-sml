"""Convert retrieval results into the existing generation SourceMaterial contract."""

import re
from collections import Counter, defaultdict

from lecture_slm.generation.context import estimate_tokens_from_characters
from lecture_slm.generation.models import SourceMaterial
from lecture_slm.knowledge.models import RetrievalMatch, RetrievalResult
from lecture_slm.knowledge.roles import ChunkRole


class KnowledgeContextAssembler:
    """Select and merge retrieved text while respecting a source-token budget."""

    def assemble(
        self, result: RetrievalResult, *, source_context_budget: int
    ) -> list[SourceMaterial]:
        if source_context_budget <= 0:
            raise ValueError("source_context_budget must be positive")
        eligible_roles = {
            str(role)
            for role in result.diagnostics.get("eligible_passage_roles", [ChunkRole.CONTENT.value])
        }
        excluded_roles = Counter(
            match.role.value for match in result.matches if match.role.value not in eligible_roles
        )
        eligible_matches = [match for match in result.matches if match.role.value in eligible_roles]
        groups = self._ordered_groups(eligible_matches)
        output: list[SourceMaterial] = []
        used_tokens = 0
        selected_chunk_ids: list[str] = []
        excluded_budget: list[str] = []
        for group in groups:
            text = _merge_overlap([match.text for match in group])
            if not text.strip():
                continue
            section = group[0].section or (
                f"Page {group[0].page_number}" if group[0].page_number else None
            )
            rendered = "\n".join(
                value for value in (group[0].source_title, section or "", text) if value
            )
            token_count = estimate_tokens_from_characters(rendered)
            if used_tokens + token_count > source_context_budget:
                excluded_budget.extend(match.chunk_id for match in group)
                continue
            used_tokens += token_count
            first = group[0]
            selected_chunk_ids.extend(match.chunk_id for match in group)
            output.append(
                SourceMaterial(
                    source_id=first.source_id,
                    title=first.source_title,
                    section=section,
                    text=text,
                    metadata={
                        "source_origin": "retrieved_knowledge",
                        "chunk_role": first.role.value,
                        "chunk_ids": [match.chunk_id for match in group],
                        "relative_path": first.source_path,
                        "page_numbers": list(
                            dict.fromkeys(
                                match.page_number
                                for match in group
                                if match.page_number is not None
                            )
                        ),
                        "slide_numbers": list(
                            dict.fromkeys(
                                match.slide_number
                                for match in group
                                if match.slide_number is not None
                            )
                        ),
                        "section_path": first.section_path,
                        "source_version": first.provenance.get("source_version"),
                        "retrieval": [
                            {
                                "lexical_rank": match.lexical_rank,
                                "lexical_score": match.lexical_score,
                                "semantic_rank": match.semantic_rank,
                                "semantic_score": match.semantic_score,
                                "fused_rank": match.fused_rank,
                                "fused_score": match.fused_score,
                                "rerank_rank": match.rerank_rank,
                                "neighbor_of": match.neighbor_of,
                            }
                            for match in group
                        ],
                    },
                )
            )
        result.diagnostics["context_assembly"] = {
            "candidate_chunks": len(result.matches),
            "eligible_chunks": len(eligible_matches),
            "excluded_by_role": dict(excluded_roles),
            "selected_chunk_ids": selected_chunk_ids,
            "selected_chunks": len(selected_chunk_ids),
            "source_material_count": len(output),
            "estimated_tokens": used_tokens,
            "configured_budget": source_context_budget,
            "unused_budget": max(0, source_context_budget - used_tokens),
            "excluded_by_budget": excluded_budget,
        }
        return output

    @staticmethod
    def _ordered_groups(matches: list[RetrievalMatch]) -> list[list[RetrievalMatch]]:
        by_source_section: dict[
            tuple[str, str, str, int | None, int | None, str], list[RetrievalMatch]
        ] = defaultdict(list)
        for match in matches:
            key = (
                match.source_id,
                match.source_path,
                " > ".join(match.section_path),
                match.page_number,
                match.slide_number,
                match.role.value,
            )
            by_source_section[key].append(match)
        groups: list[list[RetrievalMatch]] = []
        for entries in by_source_section.values():
            entries.sort(
                key=lambda match: (
                    int(match.metadata["chunk_index"]),
                    match.rerank_rank or match.fused_rank,
                )
            )
            contiguous: list[RetrievalMatch] = []
            previous_index: int | None = None
            for match in entries:
                index = int(match.metadata["chunk_index"])
                if previous_index is not None and index != previous_index + 1:
                    if contiguous:
                        groups.append(contiguous)
                    contiguous = []
                contiguous.append(match)
                previous_index = index
            if contiguous:
                groups.append(contiguous)
        groups.sort(key=lambda group: min(match.rerank_rank or match.fused_rank for match in group))
        return groups


def _merge_overlap(texts: list[str]) -> str:
    if not texts:
        return ""
    combined = texts[0]
    for text in texts[1:]:
        current_words = list(re.finditer(r"\S+", combined))
        next_words = list(re.finditer(r"\S+", text))
        overlap = 0
        upper = min(len(current_words), len(next_words))
        for size in range(1, upper + 1):
            if [word.group() for word in current_words[-size:]] == [
                word.group() for word in next_words[:size]
            ]:
                overlap = size
        if overlap:
            remainder = text[next_words[overlap - 1].end() :].lstrip(" \t")
            combined += remainder
        else:
            combined = f"{combined.rstrip()}\n\n{text.lstrip()}"
    return combined
