import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lecture_slm.config.loader import load_model_config
from lecture_slm.evaluation.evaluator import CompletionStatus, EvaluationResult
from lecture_slm.evaluation.profile import load_baseline_evaluation_profile
from lecture_slm.evaluation.runner import execute_evaluation
from lecture_slm.evaluation.structure import inspect_response_structure
from lecture_slm.inference.ollama_client import ChatResponse
from lecture_slm.schemas.dataset import TaskType

ROOT = Path(__file__).parents[1]
PROFILE_PATH = ROOT / "configs/evaluation/qwen35-9b-baseline.yaml"
PROMPTS_PATH = ROOT / "evals/prompts/baseline.jsonl"


class CappedClient:
    def __init__(self) -> None:
        self.request: dict[str, object] = {}

    def ensure_model_available(self, model: str) -> None:
        return None

    def chat(
        self,
        *,
        model: str,
        user_message: str,
        system_message: str | None = None,
        context: str | None = None,
        options: dict[str, str | int | float | bool] | None = None,
        think: bool | None = None,
        keep_alive: str | int | None = None,
        allow_empty_content: bool = False,
    ) -> ChatResponse:
        self.request = {
            "model": model,
            "options": options,
            "think": think,
        }
        return ChatResponse(
            model=model,
            content="Lecture outline:\n1. Objectives\n2. Worked example:\n",
            prompt_tokens=900,
            completion_tokens=2048,
            total_duration_ns=720_000_000_000,
            prompt_eval_duration_ns=4_000_000_000,
            eval_duration_ns=705_000_000_000,
            completion_reason="length",
        )


def test_baseline_profile_loads_canonical_task_budgets() -> None:
    profile = load_baseline_evaluation_profile(PROFILE_PATH)
    assert profile.model == "qwen3.5:9b"
    assert profile.context_length == 4096
    assert profile.think is False
    assert profile.temperature == 0.5
    assert profile.top_p == 0.9
    assert profile.seed == 3407
    assert profile.request_timeout_seconds == 900
    assert profile.task_output_tokens == {
        TaskType.EXPLANATION: 512,
        TaskType.SLIDES: 1024,
        TaskType.ACTIVITY: 1024,
        TaskType.INSTRUCTOR_GUIDE: 1024,
        TaskType.ASSESSMENT: 1024,
        TaskType.HOMEWORK: 1024,
        TaskType.LAB: 1536,
        TaskType.LECTURE: 2048,
    }
    assert profile.output_tokens_for(TaskType.EXPLANATION, ["prerequisite-reasoning"]) == 512
    assert profile.output_tokens_for(TaskType.LECTURE, []) == 2048
    assert profile.output_tokens_for(TaskType.ACTIVITY, []) == 1024


def test_profile_task_budget_falls_back_for_unconfigured_task() -> None:
    profile = load_baseline_evaluation_profile(PROFILE_PATH)
    assert profile.output_tokens_for(TaskType.EXPLANATION, []) == 512
    assert profile.output_tokens_for(TaskType.HOMEWORK, []) == 1024


def test_invalid_profile_budget_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "id: invalid\nversion: '1'\nmodel: model-x\ncontext_length: 4096\n"
        "think: false\ntemperature: 0.5\ntop_p: 0.9\nseed: 1\n"
        "request_timeout_seconds: 30\nmax_output_tokens: 2048\n"
        "task_output_tokens:\n  lecture: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="must be positive"):
        load_baseline_evaluation_profile(path)


def test_profile_precedence_does_not_mutate_model_defaults_and_marks_truncation(
    tmp_path: Path,
) -> None:
    config = load_model_config(
        ROOT / "configs/models/qwen35-9b.yaml",
        environ={"OLLAMA_HOST": "http://profile-test.example:11434"},
    )
    original_defaults = config.model_dump(mode="json")
    client = CappedClient()
    run_dir = tmp_path / "profile-validation"

    execute_evaluation(
        prompts_path=PROMPTS_PATH,
        run_dir=run_dir,
        model_config=config,
        project_root=ROOT,
        prompt_ids=["lecture-70-min-intro"],
        evaluation_profile=load_baseline_evaluation_profile(PROFILE_PATH),
        evaluation_profile_path=PROFILE_PATH,
        run_kind="profile_validation",
        client=client,  # type: ignore[arg-type]
    )

    assert config.model_dump(mode="json") == original_defaults
    assert client.request["think"] is False
    request_options = client.request["options"]
    assert request_options["num_ctx"] == 4096
    assert request_options["num_predict"] == 2048
    assert request_options["temperature"] == 0.5
    assert request_options["top_p"] == 0.9
    assert request_options["seed"] == 3407

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "profile_validation"
    assert manifest["model_configuration"]["inference"]["context_length"] == 32768
    assert manifest["model_configuration"]["inference"]["think"] is True
    assert manifest["model_configuration"]["inference"]["max_output_tokens"] == 2048
    assert manifest["evaluation_profile"]["context_length"] == 4096
    assert manifest["evaluation_profile"]["think"] is False
    assert (
        manifest["prompt_generation_configurations"]["lecture-70-min-intro"]["num_predict"] == 2048
    )

    result = EvaluationResult.model_validate_json(
        (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert result.completion_status is CompletionStatus.COMPLETED
    assert result.generation_configuration.num_ctx == 4096
    assert result.generation_configuration.think is False
    assert result.generation_configuration.num_predict == 2048
    assert result.completion_reason == "length"
    assert result.output_limit_reached is True
    assert result.potentially_truncated is True
    assert result.structural_complete is False


def test_profile_model_must_match_model_config(tmp_path: Path) -> None:
    profile = load_baseline_evaluation_profile(PROFILE_PATH).model_copy(
        update={"model": "another-model"}
    )
    config = load_model_config(ROOT / "configs/models/qwen35-9b.yaml", environ={})
    with pytest.raises(ValueError, match="targets 'another-model'"):
        execute_evaluation(
            prompts_path=PROMPTS_PATH,
            run_dir=tmp_path / "mismatch",
            model_config=config,
            project_root=ROOT,
            prompt_ids=["lecture-70-min-intro"],
            evaluation_profile=profile,
        )


def test_structural_checks_are_observations_not_quality_scores() -> None:
    complete, observations = inspect_response_structure(
        task=TaskType.SLIDES,
        instruction="Create multiple slides.",
        response="## Slide 1\nTitle\n## Slide 2\nExample.",
        potentially_truncated=False,
    )
    assert complete is True
    assert observations == []

    complete, observations = inspect_response_structure(
        task=TaskType.LAB,
        instruction="Include a checkpoint and extension.",
        response="Lab instructions:\n1. Start with this step:",
        potentially_truncated=True,
    )
    assert complete is False
    assert any("output limit" in observation for observation in observations)
    assert any("checkpoint" in observation for observation in observations)
    assert any("extension" in observation for observation in observations)
