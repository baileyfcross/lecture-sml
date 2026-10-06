"""Versioned prompts for grounding review and bounded artifact revision."""

from lecture_slm.generation.grounding_evidence import build_evidence_ledger
from lecture_slm.generation.models import (
    GenerationRequest,
    GroundingClaimInput,
    GroundingReview,
)
from lecture_slm.generation.prompts.base import PromptPackage, json_block

GROUNDING_REVIEW_PROMPT_VERSION = "grounding-review-v5"
GROUNDING_REVISION_PROMPT_VERSION = "grounding-revision-v2"

GROUNDING_REVIEW_SYSTEM_PROMPT = (
    "You are a source-grounding reviewer. The deterministic evidence ledger is the complete "
    "factual evidence available for this review. Do not use pretrained knowledge or external "
    "sources as evidence. Do not decide whether a claim is generally true; decide whether the "
    "evidence ledger states or reasonably entails it. Claims already confirmed by deterministic "
    "direct matching have been removed from this review request. Review only the unresolved "
    "claims listed below and return exactly one decision for each claim_id. Do not add claims, "
    "omit IDs, duplicate IDs, or use claim text as the review key. For each remaining claim, "
    "classify it as supported, pedagogical, or unsupported. A supported claim must include one "
    "or more evidence_ids that actually support or reasonably entail the claim. Reasonable "
    "paraphrase does not need to match source wording exactly. Do not require exact lexical "
    "identity. Do not inflate a technical statement into importance, significance, historical "
    "cause, foundational status, broader application, or general necessity unless the evidence "
    "establishes that stronger claim. Pedagogical claims may remain pedagogical when they are "
    "stipulated examples or illustrative setups. General model knowledge is not evidence. Return "
    "compact JSON only.\n\n"
    "Example 1:\n"
    "Evidence:\n"
    "[E001] Variables remain free until a universal or existential quantifier binds them.\n"
    "[E002] This lets logic express general claims.\n"
    "Claim: Quantifiers bind free variables so that general claims can be expressed.\n"
    "Correct: supported with evidence_ids [E001, E002].\n\n"
    "Example 2:\n"
    "Evidence:\n"
    "[E001] Predicate logic extends propositional logic.\n"
    "Claim: Predicate logic is essential to modern computer science.\n"
    "Correct: unsupported.\n\n"
    "Example 3:\n"
    "Claim: Let P(x) mean 'x is wise.'\n"
    "Correct: pedagogical."
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


def _evidence_ledger(request: GenerationRequest) -> list[dict[str, str | int]]:
    return [
        {
            "evidence_id": span.evidence_id,
            "source_id": span.source_id,
            "source_title": span.source_title,
            "text": span.text,
            "order": span.order,
        }
        for span in build_evidence_ledger(request)
    ]


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
            "Deterministic evidence ledger (cite evidence_id values)", _evidence_ledger(request)
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
