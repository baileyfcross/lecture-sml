import json
from pathlib import Path

import pytest

from lecture_slm.config.loader import ModelConfig, load_model_config
from lecture_slm.evaluation.evaluator import CompletionStatus, EvaluationResult
from lecture_slm.evaluation.prompts import load_evaluation_prompts
from lecture_slm.evaluation.runner import create_run_directory, execute_evaluation
from lecture_slm.inference.ollama_client import ChatResponse, OllamaTimeoutError

ROOT = Path(__file__).parents[1]
PROMPTS = ROOT / "evals/prompts/baseline.jsonl"


class FakeOllamaClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.ensure_calls = 0
        self.chat_calls = 0

    def ensure_model_available(self, model: str) -> None:
        self.ensure_calls += 1

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
    ) -> ChatResponse:
        self.chat_calls += 1
        if self.fail:
            raise OllamaTimeoutError("mock generation timeout")
        return ChatResponse(
            model=model,
            content="A concise mock educational response.",
            prompt_tokens=15,
            completion_tokens=7,
            total_duration_ns=1_000_000_000,
            load_duration_ns=10_000_000,
            prompt_eval_duration_ns=600_000_000,
            eval_duration_ns=390_000_000,
        )


def model_config() -> ModelConfig:
    return load_model_config(
        ROOT / "configs/models/qwen35-9b.yaml",
        environ={"OLLAMA_HOST": "http://private.example:11434"},
    )


def test_run_directory_creation_prevents_overwrite(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path, "run-one")
    assert first.is_dir()
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "run-one")


def test_explicit_think_override_is_separate_from_model_config(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "think-override")
    execute_evaluation(
        prompts_path=PROMPTS,
        run_dir=run_dir,
        model_config=model_config(),
        project_root=ROOT,
        limit=1,
        think_override=False,
        max_output_tokens_override=512,
        client=FakeOllamaClient(),  # type: ignore[arg-type]
    )
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    result = EvaluationResult.model_validate_json(
        (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert manifest["model_configuration"]["inference"]["think"] is True
    assert manifest["model_configuration"]["inference"]["max_output_tokens"] == 2048
    assert manifest["generation_configuration"]["think"] is False
    assert manifest["generation_configuration"]["num_predict"] == 512
    assert result.generation_configuration.think is False
    assert result.generation_configuration.num_predict == 512


def test_resume_skips_successes_and_rerun_appends_attempts(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "resume-test")
    first_client = FakeOllamaClient()
    first_summary = execute_evaluation(
        prompts_path=PROMPTS,
        run_dir=run_dir,
        model_config=model_config(),
        project_root=ROOT,
        limit=2,
        client=first_client,  # type: ignore[arg-type]
    )
    assert first_summary.completed_count == 2
    assert first_client.chat_calls == 2

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["model_configuration"]["inference"]["host"] == "[redacted]"
    assert "private.example" not in (run_dir / "run.json").read_text(encoding="utf-8")
    assert set(manifest["profile_config_sha256"]) == {
        "configs/courses/example-course.yaml",
        "configs/pedagogy/default.yaml",
    }
    assert {"run.json", "responses.jsonl", "review.jsonl", "summary.json"}.issubset(
        {path.name for path in run_dir.iterdir()}
    )

    resumed_client = FakeOllamaClient()
    resumed_summary = execute_evaluation(
        prompts_path=PROMPTS,
        run_dir=run_dir,
        model_config=model_config(),
        project_root=ROOT,
        limit=2,
        client=resumed_client,  # type: ignore[arg-type]
    )
    assert resumed_summary.skipped_count == 2
    assert resumed_client.ensure_calls == 0
    assert resumed_client.chat_calls == 0

    rerun_client = FakeOllamaClient()
    rerun_summary = execute_evaluation(
        prompts_path=PROMPTS,
        run_dir=run_dir,
        model_config=model_config(),
        project_root=ROOT,
        limit=2,
        rerun=True,
        client=rerun_client,  # type: ignore[arg-type]
    )
    assert rerun_summary.skipped_count == 0
    records = [
        EvaluationResult.model_validate_json(line)
        for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert {record.attempt for record in records[-2:]} == {2}


def test_failed_generation_is_saved_and_retried(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "failure-test")
    failed = execute_evaluation(
        prompts_path=PROMPTS,
        run_dir=run_dir,
        model_config=model_config(),
        project_root=ROOT,
        limit=1,
        client=FakeOllamaClient(fail=True),  # type: ignore[arg-type]
    )
    assert failed.failed_count == 1
    first_record = EvaluationResult.model_validate_json(
        (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_record.completion_status is CompletionStatus.FAILED
    assert first_record.error_type == "OllamaTimeoutError"

    retried = execute_evaluation(
        prompts_path=PROMPTS,
        run_dir=run_dir,
        model_config=model_config(),
        project_root=ROOT,
        limit=1,
        client=FakeOllamaClient(),  # type: ignore[arg-type]
    )
    assert retried.completed_count == 1
    assert retried.failed_count == 0
    records = [
        EvaluationResult.model_validate_json(line)
        for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record.attempt for record in records] == [1, 2]
    assert records[1].timing.prompt_tokens == 15


def test_prompt_set_has_twenty_five_unique_cases() -> None:
    prompts = load_evaluation_prompts(PROMPTS)
    assert len(prompts) == 26
    assert len({prompt.id for prompt in prompts}) == 26
