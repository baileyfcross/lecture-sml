"""Planner-only instructional reasoning prompt construction."""

import json

from lecture_slm.generation.models import GenerationRequest, TaskTeachingPlan
from lecture_slm.generation.prompts.base import PromptPackage, request_blocks

PLANNER_PROMPT_VERSION = "planner-v4"
STANDARD_PLANNER_PROMPT_VERSION = "planner-standard-v5"
PLANNER_REASSESSMENT_PROMPT_VERSION = "planner-reassessment-v3"

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
    "that teach supported concepts without adding unsupported facts. In source_scope, "
    "supported_topics lists facts established by SourceMaterial; unsupported_requested_topics "
    "lists only missing facts "
    "needed for the user's actual request; core_unsupported_topics is the indispensable subset; "
    "copy core entries exactly from unsupported_requested_topics and mark a topic core only if "
    "the artifact would be inadequate without it; "
    "optional_unsupported_topics lists useful but unnecessary enrichment. Core topics may be "
    "foundational even when not literally named, while explicitly requested major topics are "
    "requested and generally core. Optional entries must not appear in "
    "unsupported_requested_topics. "
    "Do not broaden requested scope from retrieved sources or optional teaching ideas. Keep topic "
    "strings concise, self-contained, factual, and useful as retrieval queries. Optional gaps do "
    "not "
    "make scope partial or insufficient."
)


def build_planner_prompt(request: GenerationRequest, *, concise: bool = False) -> PromptPackage:
    blocks = request_blocks(request)
    blocks.append(
        "## Planner task\nCreate the structured instructional plan required by the schema. "
        "The next stage is a separate writer request that will create the artifact."
    )
    system_message = STANDARD_PLANNER_SYSTEM_PROMPT if concise else PLANNER_SYSTEM_PROMPT
    if request.source_material:
        system_message = f"{system_message}\n\n{SOURCE_GROUNDING_INSTRUCTIONS}"
    return PromptPackage(
        version=STANDARD_PLANNER_PROMPT_VERSION if concise else PLANNER_PROMPT_VERSION,
        system_message=system_message,
        user_message="\n\n".join(blocks),
    )


PLANNER_REASSESSMENT_SYSTEM_PROMPT = (
    "You are reassessing an already-valid instructional plan after one bounded retrieval "
    "expansion. Do not re-plan from scratch. Preserve its pedagogy, sequence, structure, "
    "examples, prerequisites, and constraints unless newly supplied factual sources require a "
    "change. Reassess source coverage using the expanded sources, update factual portions only "
    "as needed, and add a previously unsupported requested topic only when these authoritative "
    "sources support it. Do not invent factual support. Ignore irrelevant retrieved topics. "
    "Only SourceMaterial may establish domain facts, including explicit caller-provided factual "
    "sources, Workspace Reference material, and retrieved global knowledge. Workspace Context "
    "and History may establish course sequence, prior topics, student progress, and instructional "
    "continuity, but they are not factual evidence for technical claims and must not fill factual "
    "gaps identified by source_scope. Context and History must not establish syntax, API behavior, "
    "algorithm behavior, technical definitions, historical facts, performance characteristics, "
    "mechanisms, or implementation details unless explicitly promoted to Reference and supplied "
    "as SourceMaterial. Keep unsupported requested topics unsupported when expanded "
    "SourceMaterial does not establish them. Do not claim that course continuity supplies "
    "missing factual details. In source_scope, keep supported_topics grounded in SourceMaterial; "
    "unsupported_requested_topics contains only missing facts needed for the user's actual "
    "request; "
    "core_unsupported_topics contains only exact entries from unsupported_requested_topics that "
    "are indispensable; optional_unsupported_topics is useful but unnecessary enrichment. Preserve "
    "foundational core topics even if not literally named; explicitly requested major topics are "
    "requested and generally core. Optional entries must not appear in "
    "unsupported_requested_topics. "
    "Do not broaden scope from retrieved sources or optional teaching ideas. Keep topic strings "
    "concise, self-contained, factual, and useful as retrieval queries. Optional gaps never make "
    "scope partial or insufficient. "
    "Return the complete updated task-specific TeachingPlan JSON only. Do not explain your "
    "reasoning."
)


def build_planner_reassessment_prompt(
    request: GenerationRequest,
    initial_plan: TaskTeachingPlan,
) -> PromptPackage:
    scope = initial_plan.source_scope
    initial_status = "not available" if scope is None else scope.status.value
    unsupported_topics = [] if scope is None else scope.unsupported_requested_topics
    compact_plan = json.dumps(
        initial_plan.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    blocks = request_blocks(request)
    blocks.extend(
        [
            (f"## Existing validated teaching plan\n```json\n{compact_plan}\n```"),
            (
                "## Initial source-scope assessment\n"
                f"- Status: {initial_status}\n"
                f"- Previously unsupported requested topics: "
                f"{json.dumps(unsupported_topics, ensure_ascii=False)}\n"
                "Additional retrieval has now occurred. Reassess whether the expanded factual "
                "sources change coverage."
            ),
            (
                "## Reassessment task\nPreserve the existing plan and update only what the "
                "expanded evidence justifies. Retrieved documents do not redefine the requested "
                "scope. Return the complete updated TeachingPlan required by the task schema, "
                "not a patch or source-scope object."
            ),
        ]
    )
    return PromptPackage(
        version=PLANNER_REASSESSMENT_PROMPT_VERSION,
        system_message=PLANNER_REASSESSMENT_SYSTEM_PROMPT,
        user_message="\n\n".join(blocks),
    )
