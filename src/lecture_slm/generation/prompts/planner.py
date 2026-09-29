"""Planner-only instructional reasoning prompt construction."""

from lecture_slm.generation.models import GenerationRequest
from lecture_slm.generation.prompts.base import PromptPackage, request_blocks

PLANNER_PROMPT_VERSION = "planner-v1"

PLANNER_SYSTEM_PROMPT = (
    "You are the instructional planner for Lecture SLM. Reason about pedagogy, prerequisites, "
    "concept sequence, scaffolding, examples, practice, misconceptions, checks for understanding, "
    "synthesis, source coverage, and artifact structure.\n\n"
    "Do not write the final lecture, slides, lab, activity, guide, or assessment. Produce an "
    "instructional plan for the writer. Return only a JSON object conforming to the supplied "
    "TeachingPlan schema. Use empty lists when information is unavailable; do not invent source "
    "facts."
)


def build_planner_prompt(request: GenerationRequest) -> PromptPackage:
    blocks = request_blocks(request)
    blocks.append(
        "## Planner task\nCreate a concise, actionable instructional blueprint. "
        "The next stage is a separate writer request that will create the artifact."
    )
    return PromptPackage(
        version=PLANNER_PROMPT_VERSION,
        system_message=PLANNER_SYSTEM_PROMPT,
        user_message="\n\n".join(blocks),
    )
