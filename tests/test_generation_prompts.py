from lecture_slm.generation.grounding_evidence import build_evidence_ledger
from lecture_slm.generation.models import (
    ExplanationPlan,
    GenerationProfileName,
    GenerationRequest,
    GroundingClaimAssessment,
    GroundingClaimClassification,
    GroundingClaimInput,
    GroundingDecision,
    GroundingIssue,
    GroundingReview,
    GroundingSupportMethod,
    SourceMaterial,
    SourceScopeAssessment,
    SourceScopeStatus,
    WorkspaceContext,
)
from lecture_slm.generation.prompts.grounding import (
    GROUNDING_REVIEW_PROMPT_VERSION,
    GROUNDING_REVISION_PROMPT_VERSION,
    build_grounding_review_prompt,
    build_grounding_revision_prompt,
)
from lecture_slm.generation.prompts.planner import (
    PLANNER_PROMPT_VERSION,
    PLANNER_REASSESSMENT_PROMPT_VERSION,
    STANDARD_PLANNER_PROMPT_VERSION,
    STANDARD_PLANNER_SYSTEM_PROMPT,
    build_planner_prompt,
    build_planner_reassessment_prompt,
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
from lecture_slm.generation.workspace_continuity import select_workspace_continuity_support
from lecture_slm.schemas.dataset import TaskType
from lecture_slm.workspaces.models import WorkspaceContextItem, WorkspaceItemRole


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
    assert "worked example" in lowered
    assert "check for understanding" in lowered
    assert "hypothetical examples" in lowered
    assert "populate source_scope" in lowered
    assert "'sufficient'" in lowered
    assert "'partial'" in lowered
    assert "'insufficient'" in lowered
    assert "do not plan a substantive factual artifact" in lowered
    assert "do not use pretrained knowledge to fill factual gaps" in lowered
    assert "preserve non-factual user constraints" in lowered


def test_writer_prompt_applies_source_grounding_policy_when_sources_exist() -> None:
    prompt = build_writer_prompt(_request(), None)

    _contains_source_policy(prompt.system_message)
    assert "Supplied source material" in prompt.user_message
    assert prompt.version == WRITER_PROMPT_VERSION == "writer-v11"


def test_writer_prompt_preserves_normal_behavior_without_sources() -> None:
    prompt = build_writer_prompt(_request(with_source=False), None)

    assert prompt.system_message == WRITER_SYSTEM_PROMPT
    assert SOURCE_GROUNDING_INSTRUCTIONS not in prompt.system_message
    assert "Create the requested artifact directly" in prompt.user_message


def test_writer_prompt_keeps_grounding_policy_with_teaching_plan() -> None:
    plan = ExplanationPlan(
        task=TaskType.EXPLANATION,
        artifact_structure=["concept", "example"],
        source_scope=None,
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
    assert prompt.version == "writer-v11"


def test_writer_grounding_precedes_unsupported_teaching_plan_sections() -> None:
    plan = ExplanationPlan(
        task=TaskType.EXPLANATION,
        artifact_structure=["concept", "Applications"],
        source_scope=None,
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
    assert PLANNER_PROMPT_VERSION == "planner-v3"
    assert STANDARD_PLANNER_PROMPT_VERSION == "planner-standard-v4"
    for concise, expected_version in (
        (False, PLANNER_PROMPT_VERSION),
        (True, STANDARD_PLANNER_PROMPT_VERSION),
    ):
        prompt = build_planner_prompt(_request(), concise=concise)

        _contains_planner_source_policy(prompt.system_message)
        assert "Supplied source material" in prompt.user_message
        assert prompt.version == expected_version


def test_planner_reassessment_prompt_preserves_plan_and_updates_scope_conservatively() -> None:
    request = _request()
    initial_plan = ExplanationPlan(
        task=TaskType.EXPLANATION,
        artifact_structure=["concept", "example", "check"],
        source_scope=SourceScopeAssessment(
            status=SourceScopeStatus.PARTIAL,
            supported_topics=["DNS"],
            unsupported_requested_topics=["DNSSEC"],
        ),
        concept="DNS maps names to network addresses.",
        assumed_knowledge=["domain names"],
        explanation_sequence=["name", "lookup", "address"],
        example="example.com resolves to an IP address.",
        misconceptions=["DNS is the website itself."],
        check_for_understanding=["What does DNS return?"],
    )

    prompt = build_planner_reassessment_prompt(request, initial_plan)

    assert prompt.version == PLANNER_REASSESSMENT_PROMPT_VERSION == "planner-reassessment-v1"
    assert "## Existing validated teaching plan" in prompt.user_message
    assert '"concept":"DNS maps names to network addresses."' in prompt.user_message
    assert "Supplied source material" in prompt.user_message
    assert "preserve the existing plan" in prompt.user_message.lower()
    assert "preserve its pedagogy, sequence, structure" in prompt.system_message.lower()
    assert "add a previously unsupported requested topic only when these authoritative sources" in (
        prompt.system_message.lower()
    )
    assert "ignore irrelevant retrieved topics" in prompt.system_message.lower()
    assert "Previously unsupported requested topics" in prompt.user_message
    assert "Additional retrieval has now occurred" in prompt.user_message
    assert "complete updated TeachingPlan" in prompt.user_message
    assert "hidden reasoning" not in prompt.system_message.lower()


def test_source_scope_schema_and_serialization() -> None:
    schema = ExplanationPlan.model_json_schema()
    scope_schema = schema["$defs"]["SourceScopeAssessment"]
    assert "source_scope" in schema["required"]
    status_ref = scope_schema["properties"]["status"]["$ref"]
    assert status_ref.endswith("/SourceScopeStatus")
    assert set(schema["$defs"]["SourceScopeStatus"]["enum"]) == {
        "sufficient",
        "partial",
        "insufficient",
    }
    assert scope_schema["properties"]["supported_topics"]["type"] == "array"
    assert scope_schema["properties"]["unsupported_requested_topics"]["type"] == "array"

    plan = ExplanationPlan(
        task=TaskType.EXPLANATION,
        artifact_structure=["supported topic"],
        source_scope=SourceScopeAssessment(
            status=SourceScopeStatus.PARTIAL,
            supported_topics=["quantum annealing"],
            unsupported_requested_topics=["general quantum-computing fundamentals"],
            scope_note="Available sources cover only specific models.",
        ),
        concept="Quantum annealing",
        assumed_knowledge=[],
        explanation_sequence=["Describe supported operation"],
        example="A source-grounded example",
        misconceptions=[],
        check_for_understanding=["What topic do the sources cover?"],
    )

    assert plan.model_dump(mode="json")["source_scope"]["status"] == "partial"
    assert plan.model_dump(mode="json")["source_scope"]["unsupported_requested_topics"] == [
        "general quantum-computing fundamentals"
    ]


def test_sourced_planner_prompt_requires_partial_scope_assessment() -> None:
    request = GenerationRequest(
        task=TaskType.EXPLANATION,
        instruction="Explain quantum computing.",
        source_material=[
            SourceMaterial(
                source_id="quantum",
                title="Computing models",
                text="Quantum annealing maps optimization problems to physical models.",
            )
        ],
    )
    planner_prompt = build_planner_prompt(request, concise=True)

    assert "source_scope" in planner_prompt.system_message
    assert "partial" in planner_prompt.system_message
    assert "quantum annealing" in planner_prompt.user_message.lower()
    assert "Explain quantum computing." in planner_prompt.user_message


def test_source_free_planner_keeps_normal_prompt_behavior() -> None:
    prompt = build_planner_prompt(_request(with_source=False), concise=True)

    assert prompt.system_message == STANDARD_PLANNER_SYSTEM_PROMPT
    assert "source_scope to null" in prompt.system_message
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


def test_grounding_review_and_revision_prompts_are_versioned_and_structured() -> None:
    request = _request().model_copy(
        update={
            "workspace_context": WorkspaceContext(
                workspace_id="workspace-1",
                workspace_name="Workspace One",
                items=[
                    WorkspaceContextItem(
                        id="item-1",
                        role=WorkspaceItemRole.HISTORY,
                        title="Lecture 3 - Generic Methods",
                        content="We explored type parameters and reusable APIs.",
                    )
                ],
            )
        }
    )
    review_prompt = build_grounding_review_prompt(
        request,
        [GroundingClaimInput(claim_id="C001", text="Predicates describe properties of entities.")],
    )
    review = GroundingReview.model_validate(
        {
            "decision": "revision_required",
            "claim_assessments": [
                GroundingClaimAssessment(
                    claim_id="C001",
                    text="Predicate logic is essential to all computer science.",
                    classification=GroundingClaimClassification.UNSUPPORTED,
                    support_method=GroundingSupportMethod.UNSUPPORTED,
                    reason="The supplied source does not establish this significance claim.",
                    category="unsupported_significance",
                )
            ],
            "evidence_ledger": build_evidence_ledger(request),
            "issues": [
                {
                    "claim_id": "C001",
                    "excerpt": "Predicate logic is essential to all computer science.",
                    "category": "unsupported_significance",
                    "reason": "The supplied source does not establish this significance claim.",
                }
            ],
            "revision_instructions": ["Remove the unsupported significance claim."],
        }
    )
    revision_prompt = build_grounding_revision_prompt(
        request,
        "# Predicates\n\nPredicates describe properties of entities.",
        review,
    )

    assert review_prompt.version == GROUNDING_REVIEW_PROMPT_VERSION == "grounding-review-v12"
    assert "evidence ledger" in review_prompt.system_message.lower()
    assert "deterministic evidence ledger" in review_prompt.system_message
    assert "workspace continuity ledger" in review_prompt.user_message.lower()
    assert "No unresolved claims have usable Workspace continuity evidence" in (
        review_prompt.user_message
    )
    assert '"item_ref": "W01"' not in review_prompt.user_message
    assert '"items": []' in review_prompt.user_message
    assert '"spans": []' in review_prompt.user_message
    assert '"workspace_item_id"' not in review_prompt.user_message
    assert "claim_id" in review_prompt.user_message
    assert "do not include a reason" in review_prompt.system_message.lower()
    assert "concise reason" in review_prompt.system_message.lower()
    assert '"classification":"supported"' in review_prompt.system_message
    assert "reasonable paraphrase" in review_prompt.system_message.lower()
    assert "S01-E001" in review_prompt.user_message
    assert revision_prompt.version == GROUNDING_REVISION_PROMPT_VERSION
    assert review_prompt.version == "grounding-review-v12"
    assert revision_prompt.version == "grounding-revision-v5"
    assert "minimum adjacent text" in revision_prompt.system_message.lower()
    assert "deletion is acceptable" in revision_prompt.system_message.lower()
    assert "closely shaped by one or more supplied source statements" in (
        revision_prompt.system_message.lower()
    )
    assert "do not create a broader synthesis" in revision_prompt.system_message.lower()
    assert "Tests state what should continue to happen as an implementation changes" in (
        revision_prompt.system_message
    )
    assert "smallest neutral wording" in revision_prompt.system_message.lower()
    assert "Unsupported grounding issues" in revision_prompt.user_message
    assert "Predicate logic is essential to all computer science." in revision_prompt.user_message
    assert '"unsupported_issues"' not in revision_prompt.user_message
    assert '"claim_id": "C001"' in revision_prompt.user_message
    assert '"category": "unsupported_significance"' in revision_prompt.user_message
    assert (
        '"reason": "The supplied source does not establish this significance claim."'
        in revision_prompt.user_message
    )
    assert '"revision_guidance"' in revision_prompt.user_message
    assert '"claim_assessments"' not in revision_prompt.user_message
    assert "edit only the unsupported passages" in revision_prompt.system_message.lower()
    assert "do not rewrite or generalize claims classified as direct_supported" in (
        revision_prompt.system_message.lower()
    )
    assert "minimum adjacent text" in revision_prompt.system_message.lower()


def test_empty_continuity_review_does_not_advertise_or_example_continuity_ids() -> None:
    request = _request()
    claims = [
        GroundingClaimInput(
            claim_id="C067",
            text="In our next session, we will explore advanced delegates.",
        )
    ]

    prompt = build_grounding_review_prompt(
        request,
        claims,
        continuity_ledger=[],
        continuity_reviewable_claim_ids=[],
    )

    assert "No unresolved claims have usable Workspace continuity evidence" in (prompt.user_message)
    assert "Do not use continuity_supported" in prompt.user_message
    assert "W01-C001" not in prompt.system_message + prompt.user_message
    assert '"continuity_ids": ["W01-C001"]' not in prompt.system_message + prompt.user_message


def test_reviewable_continuity_prompt_lists_only_claims_with_selected_evidence() -> None:
    request = _request().model_copy(
        update={
            "workspace_context": WorkspaceContext(
                workspace_id="workspace-1",
                workspace_name="Course",
                items=[
                    WorkspaceContextItem(
                        id="roadmap",
                        role=WorkspaceItemRole.CONTEXT,
                        title="Lecture 5 - Advanced Delegates and Lambdas",
                        content="The next lecture covers advanced delegates and lambdas.",
                    )
                ],
            )
        }
    )
    claim = GroundingClaimInput(
        claim_id="C002",
        text="In our next session, we will explore advanced delegates and lambdas.",
    )
    selected = select_workspace_continuity_support(request, [claim])

    prompt = build_grounding_review_prompt(
        request,
        [claim],
        continuity_ledger=selected.selected_spans,
        continuity_reviewable_claim_ids=selected.reviewable_claim_ids,
    )

    assert selected.eligible_claim_ids == ["C002"]
    assert selected.reviewable_claim_ids == ["C002"]
    assert "Claims allowed to use Workspace continuity evidence" in prompt.user_message
    assert "Continuity supported example" in prompt.system_message
    assert selected.selected_spans[0].continuity_id in prompt.user_message


def test_revision_prompt_includes_continuity_specific_corrective_guidance() -> None:
    request = _request()
    excerpt = (
        "In our earlier lectures, we explored how interfaces define contracts that allow "
        "code to depend on capabilities."
    )
    reason = (
        "Workspace continuity may only support claims about course/project history or sequence; "
        "this claim mixes continuity framing with domain factual assertions."
    )
    review = GroundingReview(
        decision=GroundingDecision.REVISION_REQUIRED,
        claims=[
            GroundingClaimAssessment(
                claim_id="C002",
                text=excerpt,
                classification=GroundingClaimClassification.UNSUPPORTED,
                support_method=GroundingSupportMethod.UNSUPPORTED,
                reason=reason,
                category="unsupported_fact",
                continuity_failure_reason="mixed_continuity_domain_claim",
                revision_guidance=(
                    "Remove or separate the course-history framing. Keep only factual statements "
                    "that are independently supported by factual SourceMaterial."
                ),
            )
        ],
        evidence_ledger=build_evidence_ledger(request),
        issues=[
            GroundingIssue(
                claim_id="C002",
                claim=excerpt,
                kind="unsupported_fact",
                why=reason,
                continuity_failure_reason="mixed_continuity_domain_claim",
                revision_guidance=(
                    "Remove or separate the course-history framing. Keep only factual statements "
                    "that are independently supported by factual SourceMaterial."
                ),
            )
        ],
        revision_instructions=["Remove, narrow, or qualify the unsupported claim."],
    )
    prompt = build_grounding_revision_prompt(request, "# Artifact\n\n" + excerpt, review)

    assert '"claim_id": "C002"' in prompt.user_message
    assert '"excerpt":' in prompt.user_message
    assert '"category": "unsupported_fact"' in prompt.user_message
    assert reason in prompt.user_message
    assert '"continuity_failure_reason": "mixed_continuity_domain_claim"' in prompt.user_message
    assert "Remove or separate the course-history framing" in prompt.user_message
    assert "Remove, narrow, or qualify the unsupported claim." in prompt.user_message


def test_writer_future_sequence_safeguard_is_workspace_conditional() -> None:
    request = _request(with_source=False).model_copy(
        update={
            "workspace_context": WorkspaceContext(
                workspace_id="workspace-1",
                workspace_name="Course",
                items=[
                    WorkspaceContextItem(
                        id="history-1",
                        role=WorkspaceItemRole.HISTORY,
                        title="Lecture 3 - Generic Methods",
                        content="The class practiced generic methods.",
                    )
                ],
            )
        }
    )

    prompt = build_writer_prompt(request, None)

    assert WRITER_PROMPT_VERSION == "writer-v11"
    assert "specific future topic" in prompt.system_message
    assert "do not infer, invent, broaden, substitute, chain, or predict" in (prompt.system_message)
    assert "only that same topic" in prompt.system_message
    assert "do not append related but unstated applications" in prompt.system_message.lower()
    assert "testing strategies and async programming" in prompt.system_message.lower()
    assert "a past History item without an explicit future topic does not establish one" in (
        prompt.system_message
    )
    assert "an unrelated future marker elsewhere in the item does not carry over" in (
        prompt.system_message
    )
    assert "Generic pedagogical transitions" in prompt.system_message
    without_workspace = build_writer_prompt(_request(with_source=False), None)
    assert "specific next lecture" not in without_workspace.system_message


def test_writer_future_policy_limits_history_and_prioritizes_context() -> None:
    request = _request(with_source=False).model_copy(
        update={
            "workspace_context": WorkspaceContext(
                workspace_id="workspace-1",
                workspace_name="Course",
                items=[
                    WorkspaceContextItem(
                        id="roadmap",
                        role=WorkspaceItemRole.CONTEXT,
                        title="Course roadmap",
                        content="Lecture 5: Delegates and lambdas.",
                    ),
                    WorkspaceContextItem(
                        id="history-1",
                        role=WorkspaceItemRole.HISTORY,
                        title="Lecture 4 - Generics",
                        content="Next session may cover generic classes or testing.",
                    ),
                ],
            )
        }
    )

    prompt = build_writer_prompt(request, None)

    assert "Use a current Workspace Context roadmap as the primary authority" in (
        prompt.system_message
    )
    assert "only for the topic it explicitly states" in prompt.system_message
    assert "generic classes or testing" in prompt.user_message
    assert "Delegates and lambdas" in prompt.user_message
    assert "delegates and lambdas' or" in prompt.system_message


def test_grounding_reviewer_prompt_allows_direct_multi_span_entailment_safely() -> None:
    request = _request()
    prompt = build_grounding_review_prompt(
        request,
        [
            GroundingClaimInput(
                claim_id="C001",
                text=(
                    "Tests can check whether an implementation continues to satisfy expected "
                    "interface behavior as it changes."
                ),
            )
        ],
    )

    text = prompt.system_message.lower()
    assert "multiple factual evidence spans" in text
    assert "different sourcematerial items" in text
    assert "straightforwardly and directly compose" in text
    assert "one overlapping topic does not support the other topics" in text
    assert "each one" in text
    assert "type modifiers and testing strategies" in text
    assert "[e001, e002]" in text
    assert "interface testing is essential to professional c# development" in text
    assert "unsupported unless evidence establishes" in text
    assert "prevalence, causal claims, historical motive, superiority, performance" in text
