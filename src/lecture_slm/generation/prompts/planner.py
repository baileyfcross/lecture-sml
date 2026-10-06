"""Planner-only instructional reasoning prompt construction."""

from lecture_slm.generation.models import GenerationRequest
from lecture_slm.generation.prompts.base import PromptPackage, request_blocks

PLANNER_PROMPT_VERSION = "planner-v2"
STANDARD_PLANNER_PROMPT_VERSION = "planner-standard-v3"

PLANNER_SYSTEM_PROMPT = (
    "You are the instructional planner for Lecture SLM. Reason about pedagogy, prerequisites, "
    "concept sequence, scaffolding, examples, practice, misconceptions, checks for understanding, "
    "synthesis, source coverage, and artifact structure.\n\n"
    "Do not write the final lecture, slides, lab, activity, guide, or assessment. Produce an "
    "instructional plan for the writer. Return only a JSON object conforming to the supplied "
    "TeachingPlan schema. Use empty lists when information is unavailable; do not invent source "
    "facts."
)
STANDARD_PLANNER_SYSTEM_PROMPT = (
    "You are the concise instructional planner for Lecture SLM. Create only the structured "
    "teaching plan required by the supplied task schema. Do not write the final artifact. "
    "Do not explain your reasoning or include hidden reasoning. Return only the structured plan. "
    "Use empty lists for optional information that is not available; do not invent source facts."
)
SOURCE_GROUNDING_INSTRUCTIONS = (
    "Because source material is supplied, treat it as authoritative for factual scope. Build the "
    "plan only around factual topics supported by those sources. Do not plan factual sections, "
    "claims, applications, historical extensions, technical relationships, or domain uses that "
    "the sources do not support. Omit or narrow unsupported requested factual areas; do not fill "
    "factual gaps from pretrained knowledge. You may create pedagogical structures such as an "
    "introduction, motivation, worked example, guided practice, check for understanding, "
    "misconceptions, summary, recap, comparison, and transitions, as well as illustrative "
    "hypothetical examples, when they teach supported concepts without adding unsupported facts."
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
    return PromptPackage(
        version=(STANDARD_PLANNER_PROMPT_VERSION if concise else PLANNER_PROMPT_VERSION),
        system_message=system_message,
        user_message="\n\n".join(blocks),
    )
