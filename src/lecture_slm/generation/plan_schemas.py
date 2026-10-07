"""Explicit task-to-plan-schema registry used by structured Planner requests."""

from typing import cast

from pydantic import BaseModel

from lecture_slm.generation.models import (
    ActivityPlan,
    AssessmentPlan,
    ExplanationPlan,
    InstructorGuidePlan,
    LabPlan,
    LecturePlan,
    SlidesPlan,
    TaskTeachingPlan,
)
from lecture_slm.schemas.dataset import TaskType

type PlanSchema = type[BaseModel]

PLAN_SCHEMA_BY_TASK: dict[TaskType, PlanSchema] = {
    TaskType.EXPLANATION: ExplanationPlan,
    TaskType.LECTURE: LecturePlan,
    TaskType.SLIDES: SlidesPlan,
    TaskType.LAB: LabPlan,
    TaskType.ACTIVITY: ActivityPlan,
    TaskType.INSTRUCTOR_GUIDE: InstructorGuidePlan,
    TaskType.ASSESSMENT: AssessmentPlan,
    TaskType.HOMEWORK: AssessmentPlan,
}


def plan_schema_for_task(task: TaskType) -> PlanSchema:
    """Return the one canonical plan schema registered for an educational task."""

    try:
        return PLAN_SCHEMA_BY_TASK[task]
    except KeyError as error:
        raise ValueError(
            f"No teaching-plan schema is registered for task '{task.value}'"
        ) from error


def validate_plan_for_task(
    task: TaskType,
    payload: str,
    *,
    require_source_scope: bool = False,
) -> TaskTeachingPlan:
    """Validate JSON using the task-specific plan schema without filling omissions."""

    schema = plan_schema_for_task(task)
    plan = cast(TaskTeachingPlan, schema.model_validate_json(payload))
    if plan.task is not task:
        raise ValueError(f"Planner returned task '{plan.task.value}', expected '{task.value}'")
    if require_source_scope and plan.source_scope is None:
        raise ValueError("Sourced planner response must include a source-scope assessment")
    return plan
