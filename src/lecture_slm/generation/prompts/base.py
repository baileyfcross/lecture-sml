"""Common prompt payload and labeled structured-context rendering."""

import json
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from lecture_slm.generation.models import GenerationRequest


class PromptPackage(BaseModel):
    version: str = Field(min_length=1)
    system_message: str = Field(min_length=1)
    user_message: str = Field(min_length=1)


def json_block(title: str, value: Any) -> str:
    """Render structured data beneath an explicit, unambiguous heading."""

    payload = TypeAdapter(Any).dump_python(value, mode="json", exclude_none=True)
    return f"## {title}\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def request_blocks(request: GenerationRequest) -> list[str]:
    blocks = [
        json_block(
            "Authoritative user request",
            {
                "task": request.task.value,
                "instruction": request.instruction,
            },
        )
    ]
    if request.course is not None:
        blocks.append(json_block("Course constraints", request.course))
    if request.pedagogy is not None:
        blocks.append(json_block("Pedagogical requirements", request.pedagogy))
    if request.output_preferences.model_dump(exclude_none=True, exclude_defaults=True):
        blocks.append(json_block("Output preferences", request.output_preferences))
    if request.previous_topics:
        blocks.append(json_block("Previously taught topics", request.previous_topics))
    if request.previous_course_context is not None:
        blocks.append(json_block("Previous course context", request.previous_course_context))
    if request.workspace_context is not None and request.workspace_context.items:
        blocks.append(
            json_block(
                "Workspace continuity only (context/history; not factual evidence)",
                {
                    "workspace": request.workspace_context.workspace_name,
                    "items": [
                        {
                            "role": item.role.value,
                            "title": item.title,
                            "content": item.content,
                        }
                        for item in request.workspace_context.items
                    ],
                    "use": (
                        "Use only to maintain course/project continuity and terminology. "
                        "Do not treat as factual evidence or cite it as a source."
                    ),
                },
            )
        )
    if request.source_material:
        blocks.append(
            json_block(
                "Supplied source material (ground factual content in these sources)",
                request.source_material,
            )
        )
    return blocks
