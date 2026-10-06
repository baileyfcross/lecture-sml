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
    GroundingClaimStatus,
    GroundingDecision,
    GroundingIssueCategory,
    GroundingReview,
    GroundingReviewRecord,
    ReviewFeedback,
    StageRecord,
    StageTiming,
    TeachingPlan,
)
from lecture_slm.generation.profiles import GenerationProfiles, StageProfile
from lecture_slm.generation.prompts.grounding import (
    build_grounding_review_prompt,
    build_grounding_revision_prompt,
    grounding_source_ref_map,
)
from lecture_slm.generation.timing import detect_output_limit
from lecture_slm.inference.ollama_client import ChatResponse, OllamaClient, OllamaError

GROUNDING_REVIEW_OUTPUT_TOKENS = 4096


def _normalize_grounding_review(content: str) -> GroundingReview:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Grounding review response must be a JSON object")
    payload = dict(value)
    claims_key = "claims" if "claims" in payload else "claim_assessments"
    fixes_key = "fixes" if "fixes" in payload else "revision_instructions"
    notes_key = "notes" if "notes" in payload else "source_consistency_notes"
    claims = payload.get(claims_key)
    issues = payload.get("issues", [])
    if not isinstance(claims, list) or not isinstance(issues, list):
        raise ValueError("Grounding review must include claim assessments and issues lists")

    def excerpt(item: object) -> str | None:
        if not isinstance(item, dict):
            return None
        text = item.get("claim", item.get("excerpt"))
        return text if isinstance(text, str) else None

    def normalized(text: str) -> str:
        return " ".join(text.split()).casefold()

    unsupported_claims = [
        claim
        for claim in claims
        if isinstance(claim, dict) and claim.get("status") == GroundingClaimStatus.UNSUPPORTED.value
    ]
    issue_excerpts = {normalized(text) for issue in issues if (text := excerpt(issue)) is not None}
    for claim in unsupported_claims:
        text = excerpt(claim)
        if text is not None and normalized(text) not in issue_excerpts:
            issues.append(
                {
                    "claim": text,
                    "kind": GroundingIssueCategory.UNSUPPORTED_FACT.value,
                    "why": (
                        "The reviewer marked this sentence unsupported but omitted issue details; "
                        "it must remain unapproved until grounded."
                    ),
                    "sources": [],
                }
            )
            issue_excerpts.add(normalized(text))

    for issue in list(issues):
        text = excerpt(issue)
        claim_excerpts = {
            normalized(claim_text) for claim in claims if (claim_text := excerpt(claim)) is not None
        }
        if text is not None and normalized(text) not in claim_excerpts:
            claims.append({"claim": text, "status": "unsupported", "sources": []})

    if issues:
        original_decision = payload.get("decision")
        payload["decision"] = GroundingDecision.REVISION_REQUIRED.value
        if original_decision != GroundingDecision.REVISION_REQUIRED.value:
            notes = payload.get(notes_key)
            if not isinstance(notes, list):
                notes = []
                payload[notes_key] = notes
            notes.append(
                "The model marked at least one claim unsupported while returning a pass; "
                "the decision was conservatively changed to revision_required."
            )
        fixes = payload.get(fixes_key)
        if not isinstance(fixes, list) or not fixes:
            payload[fixes_key] = [
                "Remove, narrow, or qualify each sentence unsupported by the supplied sources."
            ]

    payload[claims_key] = claims
    payload["issues"] = issues
    return GroundingReview.model_validate(payload)


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
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
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
    assessed = {_sentence_key(claim.excerpt) for claim in review.claim_assessments}
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
        prompt = build_grounding_review_prompt(request, artifact)
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
                format=GroundingReview.model_json_schema(),
            )
            review = _normalize_grounding_review(response.content)
            _validate_claim_coverage(artifact, review)
            source_ref_map = grounding_source_ref_map(request)
            cited_source_refs = {
                source_id for claim in review.claim_assessments for source_id in claim.source_ids
            } | {source_id for issue in review.issues for source_id in issue.relevant_source_ids}
            unknown_source_refs = cited_source_refs - set(source_ref_map)
            if unknown_source_refs:
                raise ValueError(
                    "Grounding review cited unknown source references: "
                    + ", ".join(sorted(unknown_source_refs))
                )
            review_payload = review.model_dump()
            for claim in review_payload["claim_assessments"]:
                claim["source_ids"] = [source_ref_map[source] for source in claim["source_ids"]]
            for issue in review_payload["issues"]:
                issue["relevant_source_ids"] = [
                    source_ref_map[source] for source in issue["relevant_source_ids"]
                ]
            review = GroundingReview.model_validate(review_payload)
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
