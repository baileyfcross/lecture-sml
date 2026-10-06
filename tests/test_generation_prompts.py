from lecture_slm.generation.models import (
    ExplanationPlan,
    GenerationProfileName,
    GenerationRequest,
    SourceMaterial,
)
from lecture_slm.generation.prompts.planner import (
    PLANNER_PROMPT_VERSION,
    STANDARD_PLANNER_PROMPT_VERSION,
    STANDARD_PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
)
from lecture_slm.generation.prompts.planner import (
    SOURCE_GROUNDING_INSTRUCTIONS as PLANNER_SOURCE_GROUNDING_INSTRUCTIONS,
)
from lecture_slm.generation.prompts.writer import (
    SOURCE_GROUNDING_INSTRUCTIONS,
    WRITER_PROMPT_VERSION,
    WRITER_SYSTEM_PROMPT,
    build_writer_prompt,
)
from lecture_slm.schemas.dataset import TaskType


def _request(
    profile: GenerationProfileName = GenerationProfileName.STANDARD,
    *,
    with_source: bool = True,
) -> GenerationRequest:
    return GenerationRequest(
        task=TaskType.EXPLANATION,
        profile=profile,
        instruction="Explain predicates.",
        source_material=(
            [
                SourceMaterial(
                    source_id="logic-notes",
                    title="Logic notes",
                    text="A predicate becomes a proposition when applied to an entity.",
                )
            ]
            if with_source
            else []
        ),
    )


def _contains_source_policy(prompt_text: str) -> None:
    lowered = prompt_text.lower()
    assert "authoritative for factual content" in lowered
    assert "historical claims, applications, technical claims" in lowered
    assert "do not add factual claims from pretrained knowledge" in lowered
    assert "omit it" in lowered
    assert "examples" in lowered
    assert "analogies" in lowered
    assert "practice and check-for-understanding questions" in lowered
    assert "unsupported real-world or domain-specific claims" in lowered
    assert "framing, motivation, conclusions, significance" in lowered
    assert "importance, foundational status, historical purpose, influence, prevalence" in lowered
    assert "connecting the concept to a broader field" in lowered
    assert "foundational, central, widely used, important to a field" in lowered
    assert "influential, transformative, a bridge between fields" in lowered
    assert "unless the sources support that characterization" in lowered
    assert "supported technical relationship into a broader claim" in lowered
    assert "pedagogical motivation is allowed" in lowered
    assert "helps the learner understand the topic" in lowered
    assert "historical claims such as why a concept was developed" in lowered
    assert "unless the sources say so" in lowered
    assert "predicate logic allows statements about all or some members of a domain" in lowered
    assert "foundational to computer science" in lowered
    assert "bridges logic and computation" in lowered
    assert "widely used in ai" in lowered
    assert "developed primarily to solve" in lowered
    assert "the primary motivation for" in lowered
    assert "the main reason" in lowered
    assert "bridges the gap between logical reasoning and computational processes" in lowered
    assert "keep motivation and conclusions to the specific distinctions" in lowered
    assert "do not infer a concept's historical cause from what it can express" in lowered
    assert "the primary motivation for moving from propositional logic" in lowered
    assert "do not inflate a supported capability into a claim that it is" in lowered
    assert "'essential,'" in lowered
    assert (
        "bridges the gap between abstract logical reasoning and computational procedures" in lowered
    )


def _contains_planner_source_policy(prompt_text: str) -> None:
    lowered = prompt_text.lower()
    assert "authoritative for factual scope" in lowered
    assert "only around factual topics supported by those sources" in lowered
    assert "do not plan factual sections" in lowered
    assert "do not fill factual gaps from pretrained knowledge" in lowered
    assert "worked example" in lowered
    assert "check for understanding" in lowered
    assert "hypothetical examples" in lowered


def test_writer_prompt_applies_source_grounding_policy_when_sources_exist() -> None:
    prompt = build_writer_prompt(_request(), None)

    _contains_source_policy(prompt.system_message)
    assert "Supplied source material" in prompt.user_message
    assert prompt.version == WRITER_PROMPT_VERSION == "writer-v7"


def test_writer_prompt_preserves_normal_behavior_without_sources() -> None:
    prompt = build_writer_prompt(_request(with_source=False), None)

    assert prompt.system_message == WRITER_SYSTEM_PROMPT
    assert SOURCE_GROUNDING_INSTRUCTIONS not in prompt.system_message
    assert "Create the requested artifact directly" in prompt.user_message


def test_writer_prompt_keeps_grounding_policy_with_teaching_plan() -> None:
    plan = ExplanationPlan(
        task=TaskType.EXPLANATION,
        artifact_structure=["concept", "example"],
        concept="Predicates describe properties.",
        assumed_knowledge=[],
        explanation_sequence=["Introduce a predicate", "Apply it to an entity"],
        example='Let P(x) mean "x is wise."',
        misconceptions=[],
        check_for_understanding=["What does P(Socrates) express?"],
    )
    prompt = build_writer_prompt(_request(), plan)

    _contains_source_policy(prompt.system_message)
    assert "Teaching plan to follow" in prompt.user_message
    assert "Let P(x) mean" in prompt.user_message
    assert "Supplied source material" in prompt.user_message


def test_writer_grounding_precedes_unsupported_teaching_plan_sections() -> None:
    plan = ExplanationPlan(
        task=TaskType.EXPLANATION,
        artifact_structure=["concept", "Applications"],
        concept="Predicates describe properties.",
        assumed_knowledge=[],
        explanation_sequence=["Introduce predicates", "Discuss applications"],
        example="Let P(x) mean x is wise.",
        misconceptions=[],
        check_for_understanding=["What does P(Socrates) mean?"],
    )
    prompt = build_writer_prompt(_request(), plan)
    lowered = prompt.system_message.lower()

    assert "source grounding takes precedence over the teaching plan" in lowered
    assert "if the plan requests any factual section" in lowered
    assert "omit or narrow that part of the plan" in lowered
    assert "do not fill unsupported plan requirements using pretrained knowledge" in lowered
    assert "supplied source factual scope, user request and course/pedagogy constraints" in lowered
    assert "preserve supported pedagogical elements requested by the plan" in lowered
    assert "checks for understanding" in lowered
    assert "source grounding takes precedence over the teaching plan" in lowered
    assert "significance" in lowered
    assert '"Applications"' in prompt.user_message


def test_writer_policy_allows_pedagogical_motivation_of_supported_concepts() -> None:
    prompt = build_writer_prompt(_request(), None)
    lowered = prompt.system_message.lower()

    assert "pedagogical motivation is allowed" in lowered
    assert "explain how a source-supported distinction helps the learner" in lowered
    assert "historical claims such as why a concept was developed" in lowered
    assert "require source support" in lowered


def test_quick_writer_prompt_uses_same_source_grounding_policy() -> None:
    prompt = build_writer_prompt(_request(GenerationProfileName.QUICK), None)

    _contains_source_policy(prompt.system_message)
    assert "Supplied source material" in prompt.user_message


def test_sourced_planner_limits_factual_scope_and_allows_pedagogical_structure() -> None:
    for concise, expected_version in (
        (False, PLANNER_PROMPT_VERSION),
        (True, STANDARD_PLANNER_PROMPT_VERSION),
    ):
        prompt = build_planner_prompt(_request(), concise=concise)

        _contains_planner_source_policy(prompt.system_message)
        assert "Supplied source material" in prompt.user_message
        assert prompt.version == expected_version


def test_source_free_planner_keeps_normal_prompt_behavior() -> None:
    prompt = build_planner_prompt(_request(with_source=False), concise=True)

    assert prompt.system_message == STANDARD_PLANNER_SYSTEM_PROMPT
    assert PLANNER_SOURCE_GROUNDING_INSTRUCTIONS not in prompt.system_message
    assert prompt.version == STANDARD_PLANNER_PROMPT_VERSION


def test_standard_sourced_request_has_both_grounding_policies() -> None:
    request = _request(GenerationProfileName.STANDARD)
    planner_prompt = build_planner_prompt(request, concise=True)
    writer_prompt = build_writer_prompt(request, None)

    _contains_planner_source_policy(planner_prompt.system_message)
    assert "source grounding takes precedence over the teaching plan" in (
        writer_prompt.system_message.lower()
    )


def test_source_free_standard_request_has_no_strict_grounding_policies() -> None:
    request = _request(GenerationProfileName.STANDARD, with_source=False)
    planner_prompt = build_planner_prompt(request, concise=True)
    writer_prompt = build_writer_prompt(request, None)

    assert PLANNER_SOURCE_GROUNDING_INSTRUCTIONS not in planner_prompt.system_message
    assert SOURCE_GROUNDING_INSTRUCTIONS not in writer_prompt.system_message
    assert planner_prompt.system_message == STANDARD_PLANNER_SYSTEM_PROMPT
    assert writer_prompt.system_message == WRITER_SYSTEM_PROMPT
