"""Final-artifact writer prompt construction."""

from lecture_slm.generation.models import GenerationRequest, TeachingPlan
from lecture_slm.generation.prompts.base import PromptPackage, json_block, request_blocks

WRITER_PROMPT_VERSION = "writer-v1"

WRITER_SYSTEM_PROMPT = (
    "You are the artifact writer for Lecture SLM. Follow the user's authoritative request and "
    "supplied teaching plan. Use your output budget to produce the requested educational artifact "
    "rather than explaining your planning process. Do not unnecessarily re-plan. Respect course "
    "level, pedagogy requirements, requested format, supplied sources, and prior-course context. "
    "Do not add unsupported source claims."
)


def build_writer_prompt(request: GenerationRequest, plan: TeachingPlan) -> PromptPackage:
    blocks = request_blocks(request)
    blocks.insert(1, json_block("Teaching plan to follow", plan))
    return PromptPackage(
        version=WRITER_PROMPT_VERSION,
        system_message=WRITER_SYSTEM_PROMPT,
        user_message="\n\n".join(blocks),
    )
