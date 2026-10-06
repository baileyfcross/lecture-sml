"""Legacy reviewer interface and bounded source-grounding review stages."""

import json
import re
import time
from typing import Protocol

from pydantic import ValidationError

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.context import select_context_tier
from lecture_slm.generation.models import (
    GenerationRequest,
    GenerationStatus,
    GroundingClaimAssessment,
    GroundingClaimClassification,
    GroundingClaimInput,
    GroundingDecision,
    GroundingIssue,
    GroundingIssueCategory,
    GroundingReview,
    GroundingReviewerResponse,
    GroundingReviewRecord,
    GroundingSupportMethod,
    ReviewFeedback,
    StageRecord,
    StageTiming,
    TeachingPlan,
)
from lecture_slm.generation.profiles import GenerationProfiles, StageProfile
from lecture_slm.generation.prompts.grounding import (
    GROUNDING_REVIEW_PROMPT_VERSION,
    build_grounding_review_prompt,
    build_grounding_revision_prompt,
    grounding_source_ref_map,
)
from lecture_slm.generation.timing import detect_output_limit
from lecture_slm.inference.ollama_client import ChatResponse, OllamaClient, OllamaError

GROUNDING_REVIEW_OUTPUT_TOKENS = 4096


def _normalize_grounding_review(content: str) -> GroundingReviewerResponse:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Grounding review response must be a JSON object")
    payload = value
    if "claims" not in payload and "claim_assessments" in payload:
        payload = {**payload, "claims": payload["claim_assessments"]}
    return GroundingReviewerResponse.model_validate(payload)


def _sentence_key(text: str) -> str:
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = text.replace("$", "")
    math_symbols = {
        "∀": "forall",
        "∃": "exists",
        "→": "implies",
        "∧": "and",
        "∨": "or",
        "¬": "not",
    }
    latex_symbols = {
        r"\forall": "forall",
        r"\exists": "exists",
        r"\rightarrow": "implies",
        r"\land": "and",
        r"\lor": "or",
        r"\neg": "not",
    }
    for symbol, name in math_symbols.items():
        text = text.replace(symbol, name)
    for command, name in latex_symbols.items():
        text = text.replace(command, name)
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[*`_$]", "", text)
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("‐", "-")
        .replace("‑", "-")
        .replace("‒", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("…", "...")
    )
    text = re.sub(r"""(?<!\w)["']|["'](?!\w)""", "", text)
    text = text.strip().strip("\"'.,;:!?")
    return " ".join(text.split()).casefold()


def _factual_sentence_excerpts(artifact: str) -> list[str]:
    sentences: list[str] = []
    in_fence = False
    ignored_prefixes = (
        "welcome ",
        "we will ",
        "let us ",
        "try ",
        "pause ",
        "consider the statement ",
        "explain ",
        "scenario",
        "for example, if ",
        "for instance, if ",
        "exercise ",
        "hint:",
        "domain",
        "expression",
        "analysis",
        "step-by-step",
        "identify ",
        "combine ",
        "apply ",
        "binding variables:",
    )
    for raw_line in artifact.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if (
            in_fence
            or not line
            or line.startswith("#")
            or re.fullmatch(r"\*{1,2}[^*]+\*{1,2}", line)
        ):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", line)
        line = re.sub(r"\b(?:e\.g|i\.e)\.", lambda match: match.group().replace(".", "<DOT>"), line)
        parts = re.split(r'(?<=[.!?])["”’)]*\s+(?=[A-Z0-9“"($\[])', line)
        for part in parts:
            excerpt = part.replace("<DOT>", ".").strip()
            normalized = _sentence_key(excerpt)
            normalized_for_filter = normalized.lstrip(" (")
            is_symbol_definition = re.match(
                r"^\*{0,2}\$?[A-Za-z]\w*(?:\([^)]*\))?\$?\*{0,2}:",
                excerpt,
            )
            if (
                not normalized
                or "?" in excerpt
                or normalized_for_filter.startswith(ignored_prefixes)
                or (normalized_for_filter.startswith("for example") and "$" in excerpt)
                or is_symbol_definition is not None
            ):
                continue
            if any(character.isalpha() for character in excerpt):
                sentences.append(excerpt)
    return sentences


def _validate_claim_coverage(artifact: str, review: GroundingReview) -> None:
    assessed = [_sentence_key(claim.text) for claim in review.claim_assessments]
    missing = [
        sentence
        for sentence in _factual_sentence_excerpts(artifact)
        if not any(_claim_covers_sentence(claim, _sentence_key(sentence)) for claim in assessed)
    ]
    if missing:
        raise ValueError(
            f"Grounding review omitted {len(missing)} factual sentence(s) from its claim ledger"
        )


def _claim_covers_sentence(claim: str, sentence: str) -> bool:
    return claim == sentence or re.search(rf"(?<!\w){re.escape(sentence)}(?!\w)", claim) is not None


def _direct_match_key(text: str) -> str:
    text = _sentence_key(text)
    text = re.sub(r"(?<!\d)[,:;.!?]+|[,:;.!?]+(?!\d)", " ", text)
    return " ".join(text.split())


def prepare_grounding_claims(
    request: GenerationRequest,
    artifact: str,
) -> tuple[list[GroundingClaimInput], list[GroundingClaimAssessment], list[GroundingClaimInput]]:
    """Assign stable IDs and deterministically resolve obvious source-text matches."""
    refs = grounding_source_ref_map(request)
    sources_by_ref = {
        ref: source for ref, source in zip(refs, request.source_material, strict=True)
    }
    inputs = [
        GroundingClaimInput(claim_id=f"C{index:03d}", text=text)
        for index, text in enumerate(_factual_sentence_excerpts(artifact), start=1)
    ]
    direct: list[GroundingClaimAssessment] = []
    unresolved: list[GroundingClaimInput] = []
    for claim in inputs:
        claim_key = _direct_match_key(claim.text)
        matched_refs = [
            ref
            for ref, source in sources_by_ref.items()
            if _claim_covers_sentence(_direct_match_key(source.text), claim_key)
        ]
        if not matched_refs:
            unresolved.append(claim)
            continue
        direct.append(
            GroundingClaimAssessment(
                claim_id=claim.claim_id,
                text=claim.text,
                classification=GroundingClaimClassification.DIRECT_SUPPORTED,
                support_method=GroundingSupportMethod.NORMALIZED_DIRECT_MATCH,
                source_ids=[sources_by_ref[ref].source_id for ref in matched_refs],
                supporting_excerpts=[claim.text],
                reason="Normalized claim text occurs directly in the supplied source.",
            )
        )
    return inputs, direct, unresolved


def merge_grounding_review(
    all_claims: list[GroundingClaimInput],
    direct_claims: list[GroundingClaimAssessment],
    reviewer_response: GroundingReviewerResponse | None,
    source_ref_map: dict[str, str],
    source_texts: dict[str, str],
) -> GroundingReview:
    """Validate reviewer IDs/evidence and assemble the authoritative complete ledger."""
    unresolved_ids = {claim.claim_id for claim in all_claims} - {
        claim.claim_id for claim in direct_claims
    }
    adjudications = [] if reviewer_response is None else reviewer_response.claim_assessments
    received_ids = [item.claim_id for item in adjudications]
    if len(received_ids) != len(set(received_ids)):
        raise ValueError("Grounding review returned duplicate claim IDs")
    if set(received_ids) != unresolved_ids:
        missing = unresolved_ids - set(received_ids)
        unknown = set(received_ids) - unresolved_ids
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError("Grounding review claim IDs are incomplete: " + "; ".join(details))

    inputs_by_id = {claim.claim_id: claim for claim in all_claims}
    merged = list(direct_claims)
    evidence_failures = 0
    for assessment in adjudications:
        claim = inputs_by_id[assessment.claim_id]
        valid_refs = [
            source_ref for source_ref in assessment.source_ids if source_ref in source_ref_map
        ]
        invalid_refs = [
            source_ref for source_ref in assessment.source_ids if source_ref not in source_ref_map
        ]
        valid_evidence: list[str] = []
        claim_evidence_failures = 0
        if assessment.classification is GroundingClaimClassification.SUPPORTED:
            if not valid_refs:
                claim_evidence_failures += 1
            for excerpt in assessment.supporting_excerpts:
                excerpt_key = _direct_match_key(excerpt)
                if excerpt_key and any(
                    _claim_covers_sentence(_direct_match_key(source_text), excerpt_key)
                    for ref in valid_refs
                    if (source_text := source_texts.get(ref)) is not None
                ):
                    valid_evidence.append(excerpt)
                else:
                    claim_evidence_failures += 1
            if not assessment.supporting_excerpts:
                claim_evidence_failures += 1
        classification = assessment.classification
        reason = assessment.reason
        category = assessment.category
        if invalid_refs:
            claim_evidence_failures += 1
            classification = GroundingClaimClassification.UNSUPPORTED
            reason += " Reviewer cited unknown source references."
        if classification is GroundingClaimClassification.SUPPORTED and claim_evidence_failures:
            classification = GroundingClaimClassification.UNSUPPORTED
            reason += " No supporting excerpt was verified in the cited source material."
        if claim_evidence_failures:
            evidence_failures += 1
        if classification is GroundingClaimClassification.UNSUPPORTED and category is None:
            category = GroundingIssueCategory.UNSUPPORTED_FACT
        method = {
            GroundingClaimClassification.SUPPORTED: GroundingSupportMethod.REVIEWER_ENTAILMENT,
            GroundingClaimClassification.PEDAGOGICAL: GroundingSupportMethod.PEDAGOGICAL,
            GroundingClaimClassification.UNSUPPORTED: GroundingSupportMethod.UNSUPPORTED,
        }[classification]
        mapped_source_ids = [source_ref_map[source_ref] for source_ref in valid_refs]
        merged.append(
            GroundingClaimAssessment(
                claim_id=claim.claim_id,
                text=claim.text,
                classification=classification,
                support_method=method,
                source_ids=mapped_source_ids,
                supporting_excerpts=valid_evidence,
                reason=reason,
                category=category,
            )
        )

    if len(merged) != len(all_claims) or {claim.claim_id for claim in merged} != {
        claim.claim_id for claim in all_claims
    }:
        raise ValueError(
            "Merged grounding ledger does not cover every extracted claim exactly once"
        )
    merged_by_id = {claim.claim_id: claim for claim in merged}
    merged = [merged_by_id[claim.claim_id] for claim in all_claims]

    issues = [
        GroundingIssue(
            claim_id=claim.claim_id,
            claim=claim.text,
            kind=claim.category or GroundingIssueCategory.UNSUPPORTED_FACT,
            why=claim.reason,
            sources=claim.source_ids,
        )
        for claim in merged
        if claim.classification is GroundingClaimClassification.UNSUPPORTED
    ]
    notes = []
    if evidence_failures:
        notes.append(
            f"{evidence_failures} reviewer support decision(s) failed "
            "programmatic evidence validation."
        )
    decision = GroundingDecision.REVISION_REQUIRED if issues else GroundingDecision.PASS
    return GroundingReview(
        decision=decision,
        claims=merged,
        issues=issues,
        fixes=(
            ["Remove, narrow, or qualify each claim without verified source support."]
            if issues
            else []
        ),
        notes=notes,
        evidence_validation_failures=evidence_failures,
        coverage_complete=True,
    )


class GenerationReviewer(Protocol):
    """Future reviewer returns actionable revision guidance, never a quality score."""

    def review(
        self,
        request: GenerationRequest,
        plan: TeachingPlan | None,
        artifact: str,
    ) -> ReviewFeedback: ...


class GroundingStageRunner:
    """Run strict source-grounding reviews and one bounded artifact revision."""

    def __init__(
        self,
        *,
        client: OllamaClient,
        model: str,
        model_config: ModelConfig,
        profiles: GenerationProfiles,
        settings: StageProfile,
    ) -> None:
        self.client = client
        self.model = model
        self.model_config = model_config
        self.profiles = profiles
        self.settings = settings

    def review(
        self,
        request: GenerationRequest,
        artifact: str,
    ) -> GroundingReviewRecord:
        all_claims, direct_claims, unresolved_claims = prepare_grounding_claims(request, artifact)
        source_ref_map = grounding_source_ref_map(request)
        source_texts = {
            ref: source.text
            for ref, source in zip(source_ref_map, request.source_material, strict=True)
        }
        if not unresolved_claims:
            try:
                direct_review = merge_grounding_review(
                    all_claims,
                    direct_claims,
                    None,
                    source_ref_map,
                    source_texts,
                )
                _validate_claim_coverage(artifact, direct_review)
            except (ValidationError, ValueError) as error:
                return GroundingReviewRecord(
                    status=GenerationStatus.FAILED,
                    error_type=type(error).__name__,
                    error_message=self._safe_error(str(error)),
                )
            return GroundingReviewRecord(
                status=GenerationStatus.COMPLETED,
                review=direct_review,
                prompt_version=GROUNDING_REVIEW_PROMPT_VERSION,
            )

        prompt = build_grounding_review_prompt(request, unresolved_claims)
        budget = GROUNDING_REVIEW_OUTPUT_TOKENS
        try:
            selection = select_context_tier(
                f"{prompt.system_message}\n\n{prompt.user_message}",
                self.settings.context_tiers,
                safety_margin=self.profiles.context_safety_margin,
                output_reserve_tokens=budget,
            )
        except ValueError as error:
            return GroundingReviewRecord(
                status=GenerationStatus.FAILED,
                prompt_version=prompt.version,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        started = time.perf_counter()
        response: ChatResponse | None = None
        review: GroundingReview | None = None
        try:
            response = self.client.chat(
                model=self.model,
                system_message=prompt.system_message,
                user_message=prompt.user_message,
                options={
                    "temperature": 0,
                    "top_p": self.model_config.inference.top_p,
                    "seed": self.model_config.inference.seed,
                    "num_ctx": selection.selected_context,
                    "num_predict": budget,
                },
                think=False,
                keep_alive=self.model_config.inference.keep_alive,
                format=GroundingReviewerResponse.model_json_schema(),
            )
            reviewer_response = _normalize_grounding_review(response.content)
            review = merge_grounding_review(
                all_claims,
                direct_claims,
                reviewer_response,
                source_ref_map,
                source_texts,
            )
            _validate_claim_coverage(artifact, review)
        except (OllamaError, ValidationError, ValueError) as error:
            return GroundingReviewRecord(
                status=GenerationStatus.FAILED,
                review=review,
                timing=self._timing(
                    response,
                    started=started,
                    selected_context=selection.selected_context,
                    estimated_input_tokens=selection.estimated_input_tokens,
                    budget=budget,
                ),
                raw_response=None if response is None else response.content,
                prompt_version=prompt.version,
                error_type=type(error).__name__,
                error_message=self._safe_error(str(error)),
            )
        return GroundingReviewRecord(
            status=GenerationStatus.COMPLETED,
            review=review,
            timing=self._timing(
                response,
                started=started,
                selected_context=selection.selected_context,
                estimated_input_tokens=selection.estimated_input_tokens,
                budget=budget,
            ),
            raw_response=response.content,
            prompt_version=prompt.version,
        )

    def revise(
        self,
        request: GenerationRequest,
        artifact: str,
        feedback: GroundingReview,
    ) -> StageRecord:
        prompt = build_grounding_revision_prompt(request, artifact, feedback)
        budget = self.settings.output_budget(request.task)
        try:
            selection = select_context_tier(
                f"{prompt.system_message}\n\n{prompt.user_message}",
                self.settings.context_tiers,
                safety_margin=self.profiles.context_safety_margin,
                output_reserve_tokens=budget,
            )
        except ValueError as error:
            return StageRecord(
                status=GenerationStatus.FAILED,
                prompt_version=prompt.version,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        started = time.perf_counter()
        response: ChatResponse | None = None
        try:
            response = self.client.chat(
                model=self.model,
                system_message=prompt.system_message,
                user_message=prompt.user_message,
                options={
                    "temperature": (
                        self.model_config.inference.temperature
                        if self.settings.temperature is None
                        else self.settings.temperature
                    ),
                    "top_p": self.model_config.inference.top_p,
                    "seed": self.model_config.inference.seed,
                    "num_ctx": selection.selected_context,
                    "num_predict": budget,
                },
                think=False,
                keep_alive=self.model_config.inference.keep_alive,
            )
            if not response.content.strip():
                raise ValueError("Grounding revision returned an empty artifact")
        except (OllamaError, ValueError) as error:
            return StageRecord(
                status=GenerationStatus.FAILED,
                timing=self._timing(
                    response,
                    started=started,
                    selected_context=selection.selected_context,
                    estimated_input_tokens=selection.estimated_input_tokens,
                    budget=budget,
                ),
                raw_response=None if response is None else response.content,
                prompt_version=prompt.version,
                error_type=type(error).__name__,
                error_message=self._safe_error(str(error)),
            )
        return StageRecord(
            status=GenerationStatus.COMPLETED,
            timing=self._timing(
                response,
                started=started,
                selected_context=selection.selected_context,
                estimated_input_tokens=selection.estimated_input_tokens,
                budget=budget,
            ),
            raw_response=response.content,
            prompt_version=prompt.version,
        )

    def _safe_error(self, message: str) -> str:
        return message.replace(self.model_config.inference.host, "[configured Ollama server]")

    @staticmethod
    def _timing(
        response: ChatResponse | None,
        *,
        started: float,
        selected_context: int,
        estimated_input_tokens: int,
        budget: int,
    ) -> StageTiming:
        if response is None:
            return StageTiming(
                duration_seconds=time.perf_counter() - started,
                selected_context=selected_context,
                estimated_input_tokens=estimated_input_tokens,
                output_budget=budget,
            )
        rate = None
        if response.completion_tokens is not None and response.eval_duration_ns:
            rate = response.completion_tokens * 1_000_000_000 / response.eval_duration_ns
        duration = (
            response.total_duration_ns / 1_000_000_000
            if response.total_duration_ns is not None
            else time.perf_counter() - started
        )
        output_limited = detect_output_limit(
            stop_reason=response.completion_reason,
            generated_tokens=response.completion_tokens,
            output_budget=budget,
        )
        return StageTiming(
            duration_seconds=duration,
            prompt_tokens=response.prompt_tokens,
            generated_tokens=response.completion_tokens,
            tokens_per_second=rate,
            stop_reason=response.completion_reason,
            output_limit_reached=output_limited,
            potentially_truncated=output_limited,
            thinking_enabled=False,
            selected_context=selected_context,
            estimated_input_tokens=estimated_input_tokens,
            output_budget=budget,
        )
