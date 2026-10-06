"""Versioned prompts for grounding review and bounded artifact revision."""

from lecture_slm.generation.models import (
    GenerationRequest,
    GroundingClaimInput,
    GroundingReview,
)
from lecture_slm.generation.prompts.base import PromptPackage, json_block

GROUNDING_REVIEW_PROMPT_VERSION = "grounding-review-v4"
GROUNDING_REVISION_PROMPT_VERSION = "grounding-revision-v2"

GROUNDING_REVIEW_SYSTEM_PROMPT = (
    "You are a source-grounding reviewer. The supplied sources are the only factual evidence "
    "for this review. Do not decide whether a claim is generally true; decide whether the supplied "
    "sources state or reasonably entail it. Do not use pretrained knowledge as evidence or use "
    "external sources. Claims already confirmed by deterministic source matching have been "
    "removed from this review request. Review only the unresolved claims listed below and return "
    "exactly one decision for each claim_id. Do not add claims, omit IDs, duplicate IDs, or use "
    "claim text as the review key. For each remaining claim, decide whether the supplied sources "
    "reasonably support or entail it. General model knowledge is not evidence. Include framing, "
    "motivation, "
    "historical, causal, application, significance, importance, prevalence, foundational status, "
    "and broader-field claims. Mark a claim supported only when supplied sources state or "
    "reasonably entail it; otherwise mark it unsupported. Every supported claim must include one "
    "or more exact supporting_excerpts copied from its cited source_ref values and cite those "
    "source_refs. Do not invent or paraphrase evidence excerpts: the application verifies them "
    "against the cited source. Reasonable entailment is allowed when the source evidence is "
    "explicit and sufficient. Distinguish a pedagogical claim that is only a stipulated "
    "hypothetical setup or formal example from factual narrator assertions; classify only the "
    "former as pedagogical. Do not call real-world facts pedagogical. Be strict about inflated "
    "framing such as essential, foundational, primary motivation, historical causation, and "
    "broader-field significance: support the full wording, not merely a related capability. Give "
    "a concise reason for every decision. Return compact JSON using the schema and source_ref "
    "values, with no extra commentary."
)

GROUNDING_REVISION_SYSTEM_PROMPT = (
    "Revise the supplied complete artifact to address the grounding review. Make the smallest "
    "necessary changes. Remove or narrow unsupported factual claims identified by the reviewer; "
    "preserve supported content, useful pedagogical examples unless they were specifically "
    "flagged, overall structure, and the requested audience and style. Do not introduce "
    "replacement facts from pretrained knowledge or add new unsupported claims. Do not discuss "
    "the review process. Return the complete revised artifact only, without a patch or commentary."
)


def grounding_source_ref_map(request: GenerationRequest) -> dict[str, str]:
    return {
        f"S{index}": source.source_id
        for index, source in enumerate(request.source_material, start=1)
    }


def _grounding_sources(request: GenerationRequest) -> list[dict[str, str | None]]:
    return [
        {
            "source_ref": source_ref,
            "title": source.title,
            "section": source.section,
            "text": source.text,
        }
        for source_ref, source in zip(
            grounding_source_ref_map(request),
            request.source_material,
            strict=True,
        )
    ]


def build_grounding_review_prompt(
    request: GenerationRequest,
    claims: list[GroundingClaimInput],
) -> PromptPackage:
    sections = [
        json_block(
            "Original user instruction",
            {"instruction": request.instruction, "task": request.task.value},
        ),
        json_block(
            "Authoritative supplied sources (cite source_ref values)", _grounding_sources(request)
        ),
        json_block("Unresolved factual claims to adjudicate", claims),
    ]
    return PromptPackage(
        version=GROUNDING_REVIEW_PROMPT_VERSION,
        system_message=GROUNDING_REVIEW_SYSTEM_PROMPT,
        user_message="\n\n".join(sections),
    )


def build_grounding_revision_prompt(
    request: GenerationRequest,
    artifact: str,
    feedback: GroundingReview,
) -> PromptPackage:
    findings = {
        "unsupported_excerpts": list(dict.fromkeys(issue.excerpt for issue in feedback.issues)),
        "revision_instructions": feedback.revision_instructions,
    }
    sections = [
        json_block(
            "Original user instruction",
            {"instruction": request.instruction, "task": request.task.value},
        ),
        json_block(
            "Authoritative supplied sources (cite source_ref values)", _grounding_sources(request)
        ),
        json_block("Grounding review findings", findings),
        f"## Complete original artifact to revise\n```\n{artifact}\n```",
    ]
    return PromptPackage(
        version=GROUNDING_REVISION_PROMPT_VERSION,
        system_message=GROUNDING_REVISION_SYSTEM_PROMPT,
        user_message="\n\n".join(sections),
    )
