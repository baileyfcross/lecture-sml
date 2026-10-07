"""Planner-only instructional reasoning prompt construction."""

from lecture_slm.generation.models import GenerationRequest
from lecture_slm.generation.prompts.base import PromptPackage, request_blocks

PLANNER_PROMPT_VERSION = "planner-v3"
STANDARD_PLANNER_PROMPT_VERSION = "planner-standard-v4"
EXPANDED_PLANNER_PROMPT_VERSION = "planner-expansion-v1"
STANDARD_EXPANDED_PLANNER_PROMPT_VERSION = "planner-standard-expansion-v1"

PLANNER_SYSTEM_PROMPT = (
    "You are the instructional planner for Lecture SLM. Reason about pedagogy, prerequisites, "
    "concept sequence, scaffolding, examples, practice, misconceptions, checks for understanding, "
    "synthesis, source coverage, and artifact structure.\n\n"
    "Do not write the final lecture, slides, lab, activity, guide, or assessment. Produce an "
    "instructional plan for the writer. Return only a JSON object conforming to the supplied "
    "TeachingPlan schema. Use empty lists when information is unavailable; set source_scope to "
    "null when no source material is supplied; do not invent source facts."
)
STANDARD_PLANNER_SYSTEM_PROMPT = (
    "You are the concise instructional planner for Lecture SLM. Create only the structured "
    "teaching plan required by the supplied task schema. Do not write the final artifact. "
    "Do not explain your reasoning or include hidden reasoning. Return only the structured plan. "
    "Use empty lists for optional information that is not available; set source_scope to null "
    "when no source material is supplied; do not invent source facts."
)
SOURCE_GROUNDING_INSTRUCTIONS = (
    "Because source material is supplied, treat it as authoritative for factual scope. First "
    "compare the factual scope of the user's request with the supplied sources and populate "
    "source_scope in the plan. Set its status to "
    "'sufficient' when the sources support the main factual scope needed, 'partial' when they "
    "support a meaningful subset but not the whole request, and 'insufficient' when they do not "
    "support enough to produce a meaningful substantive artifact. List supported topics and "
    "requested topics the sources do not support. Do not use pretrained knowledge to fill factual "
    "gaps or determine which facts should be added. Sources define the factual scope. Build the "
    "factual plan only around factual topics supported by those sources. Do not plan factual "
    "sections about unsupported topics. With partial coverage, narrow the factual plan to "
    "supported topics and exclude unsupported requested topics as factual sections. With "
    "insufficient coverage, do not plan a substantive factual artifact. Preserve non-factual user "
    "constraints such as format, audience, duration, and requested number of slides where "
    "feasible. You may create pedagogical structures such as an introduction, worked example, "
    "guided practice, check for understanding, summary, transitions, and hypothetical examples "
    "that teach supported concepts without adding unsupported facts."
)


def build_planner_prompt(
    request: GenerationRequest,
    *,
    concise: bool = False,
) -> PromptPackage:
    blocks = request_blocks(request)
    blocks.append(
        "## Planner task\nCreate the structured instructional plan required by the schema. "
        "The next stage is a separate writer request that will create the artifact."
    )
    system_message = STANDARD_PLANNER_SYSTEM_PROMPT if concise else PLANNER_SYSTEM_PROMPT
    if request.source_material:
        system_message = f"{system_message}\n\n{SOURCE_GROUNDING_INSTRUCTIONS}"
    retrieval = request.metadata.get("knowledge_retrieval")
    diagnostics = retrieval.get("diagnostics") if isinstance(retrieval, dict) else None
    is_scope_reassessment = (
        isinstance(diagnostics, dict) and diagnostics.get("scope_reassessment") is True
    )
    if is_scope_reassessment:
        blocks.append(
            "## Scope reassessment\nThis is the single source-scope reassessment after bounded "
            "retrieval expansion. Keep every plan field compact: use short phrases, avoid "
            "explanatory paragraphs, and use the minimum required number of list items. Leave "
            "optional lists and notes empty where allowed. List all supported and unsupported "
            "factual topics as short labels without examples or repeated wording. For an "
            "explanation, use exactly one concise artifact_structure item, explanation_sequence "
            "item, and check_for_understanding item; keep assumed_knowledge and misconceptions "
            "empty and make example one short sentence."
        )
    return PromptPackage(
        version=(
            STANDARD_EXPANDED_PLANNER_PROMPT_VERSION
            if concise and is_scope_reassessment
            else EXPANDED_PLANNER_PROMPT_VERSION
            if is_scope_reassessment
            else STANDARD_PLANNER_PROMPT_VERSION
            if concise
            else PLANNER_PROMPT_VERSION
        ),
        system_message=system_message,
        user_message="\n\n".join(blocks),
    )
