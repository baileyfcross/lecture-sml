"""Generate one artifact through the configured planner/writer workflow."""

import argparse
import sys
from pathlib import Path

from lecture_slm.config.loader import (
    load_course_config,
    load_model_config,
    load_pedagogy_config,
)
from lecture_slm.generation.models import (
    GenerationProfileName,
    GenerationRequest,
    OutputPreferences,
    PreviousCourseContext,
    ProgressEvent,
    SourceMaterial,
)
from lecture_slm.generation.persistence import create_generation_run_directory, save_generation_run
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.router import GenerationRouter
from lecture_slm.schemas.dataset import TaskType


def _progress(event: ProgressEvent) -> None:
    detail = f"[{event.stage.value}] {event.message} ({event.elapsed_seconds:.1f}s elapsed)"
    if event.estimate_seconds_remaining is not None:
        detail += f"; estimated stage time {event.estimate_seconds_remaining:.0f}s (approx.)"
    if event.generated_tokens is not None:
        detail += f"; {event.generated_tokens} tokens"
    if event.tokens_per_second is not None:
        detail += f" at {event.tokens_per_second:.2f} tokens/s"
    if event.estimate_is_approximate and event.estimate_seconds_remaining is not None:
        detail += " [estimate, not a deadline]"
    print(detail, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=[task.value for task in TaskType])
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in GenerationProfileName],
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/qwen35-9b.yaml"))
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/profiles.yaml"),
    )
    parser.add_argument("--course", type=Path)
    parser.add_argument("--pedagogy", type=Path)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--source-title")
    parser.add_argument("--previous-topic", action="append", default=[])
    parser.add_argument("--previous-lecture-summary")
    parser.add_argument("--previous-lab-summary")
    parser.add_argument("--known-term", action="append", default=[])
    parser.add_argument("--not-yet-taught", action="append", default=[])
    parser.add_argument("--duration-minutes", type=int)
    parser.add_argument("--output-format")
    parser.add_argument("--tone")
    parser.add_argument("--constraint", action="append", default=[])
    parser.add_argument("--save-run", action="store_true")
    args = parser.parse_args()

    try:
        model_config = load_model_config(args.model_config)
        generation_profiles = load_generation_profiles(args.generation_config)
        profile = GenerationProfileName(args.profile or generation_profiles.default_profile.value)
        course = load_course_config(args.course) if args.course else None
        pedagogy = load_pedagogy_config(args.pedagogy) if args.pedagogy else None
        source_material = []
        if args.source_file:
            source_material.append(
                SourceMaterial(
                    source_id=args.source_id or args.source_file.stem,
                    title=args.source_title or args.source_file.name,
                    text=args.source_file.read_text(encoding="utf-8"),
                )
            )
        previous_context = None
        if (
            args.previous_lecture_summary
            or args.previous_lab_summary
            or args.known_term
            or args.not_yet_taught
        ):
            previous_context = PreviousCourseContext(
                recently_taught_topics=args.previous_topic,
                previous_lecture_summary=args.previous_lecture_summary,
                previous_lab_summary=args.previous_lab_summary,
                known_terminology=args.known_term,
                not_yet_taught=args.not_yet_taught,
            )
        request = GenerationRequest(
            task=TaskType(args.task),
            profile=profile,
            instruction=args.instruction,
            course=course,
            pedagogy=pedagogy,
            source_material=source_material,
            previous_topics=args.previous_topic,
            previous_course_context=previous_context,
            output_preferences=OutputPreferences(
                format=args.output_format,
                tone=args.tone,
                duration_minutes=args.duration_minutes,
                constraints=args.constraint,
            ),
        )
        router = GenerationRouter(
            model_config=model_config,
            profiles=generation_profiles,
        )
        result = router.route(request, on_progress=_progress)
        if args.save_run:
            run_directory = create_generation_run_directory(
                Path("artifacts/generations"),
                request.request_id,
            )
            save_generation_run(run_directory, request, result)
            print(f"Saved generation run: {run_directory}")
    except (OSError, ValueError) as exception:
        print(f"Generation failed before execution: {exception}", file=sys.stderr)
        return 1

    print(f"Model: {result.model}")
    print(f"Profile: {result.profile.value}; task: {result.task.value}")
    print(f"Status: {result.status.value}; total: {result.timing.duration_seconds:.1f}s")
    for stage_name, record in (
        ("Planner", result.planner_result),
        ("Writer", result.writer_result),
    ):
        if record is not None and record.timing is not None:
            print(
                f"{stage_name}: {record.timing.duration_seconds:.1f}s, "
                f"context={record.timing.selected_context}, "
                f"input~{record.timing.estimated_input_tokens} tokens, "
                f"generated={record.timing.generated_tokens}, "
                f"{record.timing.tokens_per_second or 0.0:.2f} tokens/s, "
                f"stop={record.timing.stop_reason}"
            )
    if result.final_output:
        print("\n" + result.final_output)
    if result.errors:
        for error in result.errors:
            print(f"Error: {error}", file=sys.stderr)
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
