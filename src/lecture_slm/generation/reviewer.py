"""Legacy reviewer interface and bounded source-grounding review stages."""

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from lecture_slm.config.loader import ModelConfig
from lecture_slm.generation.context import select_context_tier
from lecture_slm.generation.grounding_evidence import build_evidence_ledger
from lecture_slm.generation.models import (
    EvidenceSpan,
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
    PedagogicalReviewerClaim,
    ReviewFeedback,
    StageRecord,
    StageTiming,
    SupportedReviewerClaim,
    TeachingPlan,
    UnsupportedReviewerClaim,
)
from lecture_slm.generation.profiles import GenerationProfiles, StageProfile
from lecture_slm.generation.prompts.grounding import (
    GROUNDING_REVIEW_PROMPT_VERSION,
    build_grounding_review_prompt,
    build_grounding_revision_prompt,
)
from lecture_slm.generation.timing import detect_output_limit
from lecture_slm.inference.ollama_client import ChatResponse, OllamaClient, OllamaError

GROUNDING_REVIEW_NORMAL_OUTPUT_TOKENS = 4096
GROUNDING_REVIEW_LARGE_OUTPUT_TOKENS = 8192
GROUNDING_REVIEW_LARGE_CLAIM_THRESHOLD = 40


def grounding_review_output_budget(unresolved_claim_count: int) -> int:
    if unresolved_claim_count < 0:
        raise ValueError("unresolved claim count must not be negative")
    if unresolved_claim_count > GROUNDING_REVIEW_LARGE_CLAIM_THRESHOLD:
        return GROUNDING_REVIEW_LARGE_OUTPUT_TOKENS
    return GROUNDING_REVIEW_NORMAL_OUTPUT_TOKENS


@dataclass(frozen=True)
class GroundingReviewPreparation:
    all_claims: list[GroundingClaimInput]
    direct_claims: list[GroundingClaimAssessment]
    unresolved_claims: list[GroundingClaimInput]
    evidence_ledger: list[EvidenceSpan]
    output_budget: int | None

    @property
    def claims_extracted(self) -> int:
        return len(self.all_claims)

    @property
    def direct_supported_claim_count(self) -> int:
        return len(self.direct_claims)

    @property
    def unresolved_claim_count(self) -> int:
        return len(self.unresolved_claims)


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
        line = re.sub(
            r"\b(?:e\.g|i\.e|vs)\.",
            lambda match: match.group().replace(".", "<DOT>"),
            line,
            flags=re.IGNORECASE,
        )
        parts = _split_grounding_sentences(line)
        for part in parts:
            excerpt = part.replace("<DOT>", ".").strip()
            normalized = _sentence_key(excerpt)
            normalized_for_filter = normalized.lstrip(" (")
            is_symbol_definition = re.match(
                r"^\*{0,2}\$?[A-Za-z]\w*(?:\([^)]*\))?\$?\*{0,2}:",
                excerpt,
            )
            is_labeled_relation_prompt = re.match(
                r"^\*[^*\r\n]{1,60}\*:\s*consider the relation\b",
                excerpt,
                flags=re.IGNORECASE,
            )
            if (
                not normalized
                or "?" in excerpt
                or normalized_for_filter.startswith(ignored_prefixes)
                or (normalized_for_filter.startswith("for example") and "$" in excerpt)
                or is_symbol_definition is not None
                or is_labeled_relation_prompt is not None
            ):
                continue
            if any(character.isalpha() for character in excerpt):
                sentences.append(excerpt)
    return sentences


def _split_grounding_sentences(line: str) -> list[str]:
    boundaries = [
        match.start(2)
        for match in re.finditer(
            r'(?<=[.!?])(["”’)]*)(\s+)(?=[A-Z0-9“"($\[])',
            line,
        )
    ]
    parts: list[str] = []
    start = 0
    for boundary in boundaries:
        parts.append(line[start:boundary].rstrip())
        start = boundary
    parts.append(line[start:].strip())
    return parts


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
    text = _strip_presentation_prefix(text)
    text = _sentence_key(text)
    text = re.sub(r"(?<!\d)[,:;.!?]+|[,:;.!?]+(?!\d)", " ", text)
    return " ".join(text.split())


def _strip_presentation_prefix(text: str) -> str:
    """Remove a short emphasized Markdown label without changing factual wording."""
    return re.sub(
        r"^\s*>?\s*(\*\*|__|\*|_)([^*_:\r\n]{1,60})\1\s*:\s*",
        "",
        text,
        count=1,
    )


def _evidence_ids_for_direct_claim(
    claim_text: str,
    request: GenerationRequest,
    evidence_ledger: list[EvidenceSpan],
) -> list[str]:
    claim_key = _direct_match_key(claim_text)
    evidence_by_source: dict[str, list[EvidenceSpan]] = {}
    for span in evidence_ledger:
        evidence_by_source.setdefault(span.source_id, []).append(span)

    matched_ids: list[str] = []
    for source in request.source_material:
        spans = evidence_by_source.get(source.source_id, [])
        span_matches = [
            span.evidence_id
            for span in spans
            if _claim_covers_sentence(_direct_match_key(span.text), claim_key)
        ]
        if span_matches:
            matched_ids.extend(span_matches)
            continue
        if _claim_covers_sentence(_direct_match_key(source.text), claim_key):
            matched_ids.extend(span.evidence_id for span in spans)
    return list(dict.fromkeys(matched_ids))


def prepare_grounding_claims(
    request: GenerationRequest,
    artifact: str,
) -> tuple[list[GroundingClaimInput], list[GroundingClaimAssessment], list[GroundingClaimInput]]:
    """Assign stable IDs and deterministically resolve obvious source-text matches."""
    evidence_ledger = build_evidence_ledger(request)
    return _prepare_grounding_claims(request, artifact, evidence_ledger)


def _prepare_grounding_claims(
    request: GenerationRequest,
    artifact: str,
    evidence_ledger: list[EvidenceSpan],
) -> tuple[list[GroundingClaimInput], list[GroundingClaimAssessment], list[GroundingClaimInput]]:
    inputs = [
        GroundingClaimInput(claim_id=f"C{index:03d}", text=text)
        for index, text in enumerate(_factual_sentence_excerpts(artifact), start=1)
    ]
    direct: list[GroundingClaimAssessment] = []
    unresolved: list[GroundingClaimInput] = []
    for claim in inputs:
        matched_evidence_ids = _evidence_ids_for_direct_claim(claim.text, request, evidence_ledger)
        if not matched_evidence_ids:
            unresolved.append(claim)
            continue
        direct.append(
            GroundingClaimAssessment(
                claim_id=claim.claim_id,
                text=claim.text,
                classification=GroundingClaimClassification.DIRECT_SUPPORTED,
                support_method=GroundingSupportMethod.NORMALIZED_DIRECT_MATCH,
                evidence_ids=matched_evidence_ids,
                reason="Normalized claim text occurs directly in the supplied source.",
            )
        )
    return inputs, direct, unresolved


def merge_grounding_review(
    all_claims: list[GroundingClaimInput],
    direct_claims: list[GroundingClaimAssessment],
    reviewer_response: GroundingReviewerResponse | None,
    evidence_ledger: list[EvidenceSpan],
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
    evidence_by_id = {span.evidence_id: span for span in evidence_ledger}
    merged = list(direct_claims)
    evidence_failures = 0
    for assessment in adjudications:
        claim = inputs_by_id[assessment.claim_id]
        valid_evidence_ids = [
            evidence_id for evidence_id in assessment.evidence_ids if evidence_id in evidence_by_id
        ]
        invalid_evidence_ids = [
            evidence_id
            for evidence_id in assessment.evidence_ids
            if evidence_id not in evidence_by_id
        ]
        claim_evidence_failures = 0
        if assessment.classification is GroundingClaimClassification.SUPPORTED:
            if not valid_evidence_ids:
                claim_evidence_failures += 1
            if invalid_evidence_ids:
                claim_evidence_failures += 1
        classification = assessment.classification
        if isinstance(assessment, SupportedReviewerClaim):
            reason = (
                f"Supported by reviewer-selected evidence {', '.join(assessment.evidence_ids)}."
            )
            category = None
        elif isinstance(assessment, PedagogicalReviewerClaim):
            reason = "Reviewer classified this claim as a pedagogical example or setup."
            category = None
        elif isinstance(assessment, UnsupportedReviewerClaim):
            reason = assessment.reason
            category = assessment.category
        else:
            raise TypeError("Unknown grounding reviewer claim type")
        if invalid_evidence_ids:
            claim_evidence_failures += 1
            classification = GroundingClaimClassification.UNSUPPORTED
            reason += " Reviewer cited unknown evidence IDs."
        if classification is GroundingClaimClassification.SUPPORTED and claim_evidence_failures:
            classification = GroundingClaimClassification.UNSUPPORTED
            reason += " No supporting evidence IDs were verified in the cited source material."
        if claim_evidence_failures:
            evidence_failures += 1
        if classification is GroundingClaimClassification.UNSUPPORTED and category is None:
            category = GroundingIssueCategory.UNSUPPORTED_FACT
        method = {
            GroundingClaimClassification.SUPPORTED: GroundingSupportMethod.REVIEWER_ENTAILMENT,
            GroundingClaimClassification.PEDAGOGICAL: GroundingSupportMethod.PEDAGOGICAL,
            GroundingClaimClassification.UNSUPPORTED: GroundingSupportMethod.UNSUPPORTED,
        }[classification]
        mapped_evidence_ids = valid_evidence_ids
        if classification is GroundingClaimClassification.UNSUPPORTED:
            mapped_evidence_ids = []
        merged.append(
            GroundingClaimAssessment(
                claim_id=claim.claim_id,
                text=claim.text,
                classification=classification,
                support_method=method,
                evidence_ids=mapped_evidence_ids,
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
            sources=[
                evidence_by_id[evidence_id].source_id
                for evidence_id in claim.evidence_ids
                if evidence_id in evidence_by_id
            ],
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
        evidence_ledger=evidence_ledger,
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

    def prepare_review(
        self,
        request: GenerationRequest,
        artifact: str,
    ) -> GroundingReviewPreparation:
        evidence_ledger = build_evidence_ledger(request)
        all_claims, direct_claims, unresolved_claims = _prepare_grounding_claims(
            request,
            artifact,
            evidence_ledger,
        )
        return GroundingReviewPreparation(
            all_claims=all_claims,
            direct_claims=direct_claims,
            unresolved_claims=unresolved_claims,
            evidence_ledger=evidence_ledger,
            output_budget=(
                grounding_review_output_budget(len(unresolved_claims))
                if unresolved_claims
                else None
            ),
        )

    def review(
        self,
        request: GenerationRequest,
        artifact: str,
        preparation: GroundingReviewPreparation | None = None,
    ) -> GroundingReviewRecord:
        prepared = preparation or self.prepare_review(request, artifact)
        all_claims = prepared.all_claims
        direct_claims = prepared.direct_claims
        unresolved_claims = prepared.unresolved_claims
        evidence_ledger = prepared.evidence_ledger
        if not unresolved_claims:
            try:
                direct_review = merge_grounding_review(
                    all_claims,
                    direct_claims,
                    None,
                    evidence_ledger,
                )
                _validate_claim_coverage(artifact, direct_review)
            except (ValidationError, ValueError) as error:
                return GroundingReviewRecord(
                    status=GenerationStatus.FAILED,
                    error_type=type(error).__name__,
                    error_message=self._safe_error(str(error)),
                    claims_extracted=prepared.claims_extracted,
                    direct_supported_claim_count=prepared.direct_supported_claim_count,
                    unresolved_claim_count=prepared.unresolved_claim_count,
                    review_output_budget=prepared.output_budget,
                )
            return GroundingReviewRecord(
                status=GenerationStatus.COMPLETED,
                review=direct_review,
                prompt_version=GROUNDING_REVIEW_PROMPT_VERSION,
                claims_extracted=prepared.claims_extracted,
                direct_supported_claim_count=prepared.direct_supported_claim_count,
                unresolved_claim_count=prepared.unresolved_claim_count,
                review_output_budget=prepared.output_budget,
            )

        prompt = build_grounding_review_prompt(request, unresolved_claims)
        budget = prepared.output_budget
        if budget is None:
            raise RuntimeError("grounding review preparation has no output budget")
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
                claims_extracted=prepared.claims_extracted,
                direct_supported_claim_count=prepared.direct_supported_claim_count,
                unresolved_claim_count=prepared.unresolved_claim_count,
                review_output_budget=prepared.output_budget,
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
                evidence_ledger,
            )
            _validate_claim_coverage(artifact, review)
        except (OllamaError, ValidationError, ValueError) as error:
            truncated = (
                isinstance(error, (ValidationError, ValueError))
                and response is not None
                and response.completion_reason == "length"
                and response.completion_tokens is not None
                and response.completion_tokens >= budget
            )
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
                error_type=("GroundingReviewTruncatedError" if truncated else type(error).__name__),
                error_message=(
                    self._safe_error(
                        "Grounding review output was truncated after reaching its "
                        f"{budget:,}-token generation budget. The incomplete structured review "
                        "could not be validated."
                    )
                    if truncated
                    else self._safe_error(str(error))
                ),
                claims_extracted=prepared.claims_extracted,
                direct_supported_claim_count=prepared.direct_supported_claim_count,
                unresolved_claim_count=prepared.unresolved_claim_count,
                review_output_budget=prepared.output_budget,
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
            claims_extracted=prepared.claims_extracted,
            direct_supported_claim_count=prepared.direct_supported_claim_count,
            unresolved_claim_count=prepared.unresolved_claim_count,
            review_output_budget=prepared.output_budget,
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
