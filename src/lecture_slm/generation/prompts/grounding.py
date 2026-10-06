"""Versioned prompts for grounding review and bounded artifact revision."""

from lecture_slm.generation.models import (
    GenerationRequest,
    GroundingReview,
)
from lecture_slm.generation.prompts.base import PromptPackage, json_block

GROUNDING_REVIEW_PROMPT_VERSION = "grounding-review-v3"
GROUNDING_REVISION_PROMPT_VERSION = "grounding-revision-v2"

GROUNDING_REVIEW_SYSTEM_PROMPT = (
    "You are a source-grounding reviewer. The supplied sources are the only factual evidence "
    "for this review. Do not decide whether a claim is generally true; decide whether the supplied "
    "sources state or reasonably entail it. Do not use pretrained knowledge as evidence or use "
    "external sources. Create an exhaustive claim ledger with exactly one assessment for every "
    "factual sentence. Use that sentence verbatim as the excerpt; do not combine sentences or "
    "omit a factual sentence. After drafting the ledger, scan the complete artifact again from "
    "beginning to end and verify each factual sentence is present. Explicitly check every sentence "
    "in the final paragraph, including the last sentence. Include definitions and claims in "
    "introductions, conclusions, "
    "transitions, and examples presented as true. Include framing, motivation, "
    "historical, causal, application, significance, importance, prevalence, foundational status, "
    "and broader-field claims. Mark a sentence supported only when a supplied source states or "
    "reasonably entails it; otherwise mark it unsupported. Each supported sentence must cite "
    "source ID(s). Do not infer support because a claim is familiar or likely true. "
    "Reasonable paraphrase and entailment count as support. "
    "Distinguish a hypothetical pedagogical setup, stipulated domain or predicate definition, "
    "formal example, or object-language formula being analyzed from a narrator's factual "
    "assertion. Such illustrative constructions do not require source support unless the artifact "
    "generalizes them as real-world facts. Do not flag them merely because the source does not "
    "contain the example. Do not require source support for practice questions, checks for "
    "understanding, or analogies unless they assert real-world facts. Still include a ledger entry "
    "for each other sentence that makes a factual assertion. A pass requires all factual "
    "sentences to be "
    "listed as supported and no issues. Every unsupported excerpt must exactly match an issue "
    "excerpt. Flag every material unsupported factual sentence and give concise revision "
    "instructions. Use the schema's short key names, source_ref values, and compact JSON on one "
    "line; return no extra commentary."
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
    artifact: str,
) -> PromptPackage:
    sections = [
        json_block(
            "Original user instruction",
            {"instruction": request.instruction, "task": request.task.value},
        ),
        json_block(
            "Authoritative supplied sources (cite source_ref values)", _grounding_sources(request)
        ),
        f"## Candidate artifact\n```\n{artifact}\n```",
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
