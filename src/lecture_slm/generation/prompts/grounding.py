"""Versioned prompts for grounding review and bounded artifact revision."""

from lecture_slm.generation.grounding_evidence import build_evidence_ledger
from lecture_slm.generation.models import (
    GenerationRequest,
    GroundingClaimInput,
    GroundingReview,
    WorkspaceContinuitySpan,
)
from lecture_slm.generation.prompts.base import PromptPackage, json_block
from lecture_slm.generation.workspace_continuity import select_workspace_continuity_support

GROUNDING_REVIEW_PROMPT_VERSION = "grounding-review-v12"
GROUNDING_REVISION_PROMPT_VERSION = "grounding-revision-v5"

GROUNDING_REVIEW_SYSTEM_PROMPT = (
    "You are a source-grounding reviewer. The deterministic evidence ledger is the complete "
    "factual evidence available for this review. Do not use pretrained knowledge or external "
    "sources as evidence. Do not decide whether a claim is generally true; decide whether the "
    "evidence ledger states or reasonably entails it. Workspace continuity is separate from "
    "factual evidence: the continuity ledger may support only claims about course or project "
    "history, sequence, or what was previously covered, and it must not be used to establish "
    "domain facts. Workspace continuity support requires both an eligible course-history or "
    "sequence claim and selected spans with meaningful topic overlap; matching words such as "
    "'next session', 'Lecture 3', 'covered', or 'explored' are structural only and do not "
    "establish topic relevance. Even when allowed, classify continuity_supported only if the "
    "selected span itself establishes the claim's same topic and sequence. If the claim "
    "Whenever a continuity claim contains multiple specific topics, selected continuity evidence "
    "must establish each one; one overlapping topic does not support the other topics. For future "
    "claims, each topic must be stated as part of the future sequence, not merely mentioned "
    "elsewhere in the item's history. For example, evidence that says the next session covers "
    "testing strategies does not support a claim that it covers both type modifiers and testing "
    "strategies. "
    "Unsupported additions make the whole claim unsupported unless the claim can be separated "
    "without changing its meaning. Claims already "
    "confirmed by deterministic direct matching have been removed "
    "from this review request. Review only the unresolved claims listed below and return exactly "
    "one decision for each claim_id. Do not add claims, omit IDs, duplicate IDs, or use claim "
    "text as the review key. Classify each claim as supported, continuity_supported, "
    "pedagogical, or unsupported. For supported claims return only claim_id, classification, "
    "and one or more valid evidence_ids; do not include a reason. If an allowed claim is "
    "continuity_supported, return only claim_id, classification, and one or more valid "
    "continuity_ids; do not "
    "include evidence_ids or a reason. For pedagogical claims return only claim_id, "
    "classification, and evidence_ids: []. Do not include a reason. For unsupported claims "
    "return claim_id, classification, evidence_ids: [], a concise reason, and category when "
    "applicable. Never invent evidence IDs or continuity IDs. Reasonable paraphrase does not "
    "need to match source wording exactly. Do not require exact lexical identity. Do not inflate "
    "a technical statement into importance, significance, historical cause, foundational status, "
    "broader application, general necessity, or course continuity unless the appropriate ledger "
    "establishes that stronger claim. Multiple factual evidence spans, including spans from "
    "different SourceMaterial items, may jointly support a claim when their stated facts "
    "straightforwardly and directly compose into that claim; cite every evidence_id needed for "
    "the joint support. This is not permission for new inference: do not add importance, "
    "prevalence, causal claims, historical motive, superiority, performance, necessity, or "
    "broader application unless the evidence establishes them. For example, if [E001] states "
    "that an interface declares capabilities an implementing type promises to provide, and "
    "[E002] states that tests specify what should continue to happen as implementation changes, "
    "then 'Tests can check whether an implementation continues to satisfy expected interface "
    "behavior as it changes' is supported by [E001, E002]. By contrast, 'Interface testing is "
    "essential to professional C# development' is unsupported unless evidence establishes "
    "essential status and professional significance. Pedagogical claims may remain "
    "pedagogical when they are "
    "stipulated examples or illustrative setups. General model knowledge is not evidence. "
    "Only claims listed as allowed to use Workspace continuity evidence may be classified as "
    "continuity_supported. "
    "Return only the JSON structure, with no commentary outside it.\n\n"
    "Supported:\n"
    '{"claim_id":"C001","classification":"supported","evidence_ids":["S01-E001"]}\n\n'
    "Pedagogical:\n"
    '{"claim_id":"C003","classification":"pedagogical","evidence_ids":[]}\n\n'
    "Unsupported:\n"
    '{"claim_id":"C004","classification":"unsupported","evidence_ids":[],'
    '"reason":"The evidence does not establish the broader significance claimed."}'
)

GROUNDING_REVISION_SYSTEM_PROMPT = (
    "Act as a targeted editor of the supplied complete artifact. Edit only the unsupported "
    "passages identified in the structured grounding issues, making the smallest changes "
    "necessary; modify adjacent grammar only when required for coherence. Do not rewrite or "
    "generalize claims classified as direct_supported, supported, continuity_supported, or "
    "pedagogical. Preserve already-supported technical content, course-history markers, lecture "
    "numbers, prior-session references, examples, structure, audience, and style unless they are "
    "part of a flagged unsupported passage. Do not rewrite the whole artifact. Follow the "
    "issue-specific corrective guidance, and do not introduce replacement facts from pretrained "
    "knowledge or new unsupported claims. When the only issue is unsupported significance "
    "language, remove only that qualifier or use the smallest neutral wording that preserves the "
    "supported meaning. When replacing an unsupported factual claim, prefer a simpler sentence "
    "closely shaped by one or more supplied source statements; do not create a broader synthesis "
    "unless the combined sources directly entail it. If the claim is unnecessary, deletion is "
    "acceptable and preferable to inventing a replacement. For example, prefer 'Tests state what "
    "should continue to happen as an implementation changes' over a broader claim that testing "
    "verifies implementations uphold interface promises through logic changes, unless the "
    "combined sources directly establish that synthesis. A short pedagogical transition may "
    "surround a source-shaped factual sentence, but must not add unsupported technical "
    "significance. Return the complete revised artifact, but change only flagged passages "
    "and minimum adjacent text. Do not discuss the review process."
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
    *,
    continuity_ledger: list[WorkspaceContinuitySpan] | None = None,
    continuity_reviewable_claim_ids: list[str] | None = None,
) -> PromptPackage:
    if continuity_ledger is None or continuity_reviewable_claim_ids is None:
        selection = select_workspace_continuity_support(request, claims)
        if continuity_ledger is None:
            continuity_ledger = selection.selected_spans
        if continuity_reviewable_claim_ids is None:
            continuity_reviewable_claim_ids = selection.reviewable_claim_ids
    continuity_ledger = [] if continuity_ledger is None else continuity_ledger
    continuity_reviewable_claim_ids = (
        [] if continuity_reviewable_claim_ids is None else continuity_reviewable_claim_ids
    )
    item_ref_metadata: dict[str, dict[str, str]] = {}
    for span in continuity_ledger:
        item_ref = span.continuity_id.split("-C", 1)[0]
        item_ref_metadata.setdefault(
            item_ref,
            {
                "item_ref": item_ref,
                "title": span.workspace_item_title,
                "role": span.role.value,
            },
        )
    continuity_payload = {
        "items": [item_ref_metadata[key] for key in sorted(item_ref_metadata)],
        "spans": [
            {
                "continuity_id": span.continuity_id,
                "item_ref": span.continuity_id.split("-C", 1)[0],
                "text": span.text,
            }
            for span in continuity_ledger
        ],
    }
    if continuity_reviewable_claim_ids:
        continuity_instruction = json_block(
            "Claims allowed to use Workspace continuity evidence",
            continuity_reviewable_claim_ids,
        )
        system_message = GROUNDING_REVIEW_SYSTEM_PROMPT
        system_message += (
            "\n\nContinuity supported example:\n"
            '{"claim_id":"C002","classification":"continuity_supported",'
            '"continuity_ids":["W01-C001"]}'
        )
    else:
        continuity_instruction = (
            "## Workspace continuity availability\n"
            "No unresolved claims have usable Workspace continuity evidence in this review. "
            "Do not use continuity_supported; classify claims using factual evidence, "
            "pedagogical, or unsupported."
        )
        system_message = GROUNDING_REVIEW_SYSTEM_PROMPT.replace(
            "supported, continuity_supported, pedagogical, or unsupported",
            "supported, pedagogical, or unsupported",
        ).replace(
            "If an allowed claim is continuity_supported, return only claim_id, classification, "
            "and one or more valid continuity_ids; do not include evidence_ids or a reason.",
            "Do not use the continuity_supported classification because no Workspace continuity "
            "evidence is available.",
        )
    sections = [
        json_block(
            "Original user instruction",
            {"instruction": request.instruction, "task": request.task.value},
        ),
        continuity_instruction,
        json_block(
            "Workspace continuity ledger (context/history only; not factual evidence)",
            continuity_payload,
        ),
        json_block(
            "Deterministic evidence ledger (cite evidence_id values)", _evidence_ledger(request)
        ),
        json_block("Unresolved claims to adjudicate", claims),
    ]
    return PromptPackage(
        version=GROUNDING_REVIEW_PROMPT_VERSION,
        system_message=system_message,
        user_message="\n\n".join(sections),
    )


def build_grounding_revision_prompt(
    request: GenerationRequest,
    artifact: str,
    feedback: GroundingReview,
) -> PromptPackage:
    selection = feedback.continuity_selection
    selected_spans = selection.get("selected_spans", []) if isinstance(selection, dict) else []
    selected_by_claim = (
        selection.get("selected_by_claim", {}) if isinstance(selection, dict) else {}
    )
    spans_by_id = {
        span.get("continuity_id"): span
        for span in selected_spans
        if isinstance(span, dict) and isinstance(span.get("continuity_id"), str)
    }
    structured_issues = []
    for issue in feedback.issues:
        candidate_ids = (
            selected_by_claim.get(issue.claim_id, [])
            if isinstance(selected_by_claim, dict) and issue.claim_id is not None
            else []
        )
        continuity_evidence = [
            spans_by_id[continuity_id]
            for continuity_id in candidate_ids
            if continuity_id in spans_by_id
        ]
        structured_issues.append(
            {
                "claim_id": issue.claim_id,
                "excerpt": issue.excerpt,
                "category": issue.category.value,
                "reason": issue.reason,
                "relevant_source_ids": issue.relevant_source_ids,
                "continuity_failure_reason": issue.continuity_failure_reason,
                "revision_guidance": issue.revision_guidance,
                "continuity_evidence": continuity_evidence,
            }
        )
    sections = [
        json_block(
            "Original user instruction",
            {"instruction": request.instruction, "task": request.task.value},
        ),
        json_block(
            "Authoritative supplied sources (cite source_ref values)", _grounding_sources(request)
        ),
        json_block("Unsupported grounding issues", structured_issues),
        json_block("General revision instructions", feedback.revision_instructions),
    ]
    if selected_spans:
        sections.append(
            json_block(
                "Workspace continuity evidence (context/history only)",
                selected_spans,
            )
        )
    sections.extend(
        [
            f"## Complete original artifact to revise\n```\n{artifact}\n```",
        ]
    )
    return PromptPackage(
        version=GROUNDING_REVISION_PROMPT_VERSION,
        system_message=GROUNDING_REVISION_SYSTEM_PROMPT,
        user_message="\n\n".join(sections),
    )
