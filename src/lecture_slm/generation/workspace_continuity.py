"""Deterministic workspace continuity gating and compact claim-aware selection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lecture_slm.generation.models import (
    GenerationRequest,
    GroundingClaimInput,
    WorkspaceContinuitySpan,
)
from lecture_slm.workspaces.models import WorkspaceItemRole

MAX_CONTINUITY_SPANS_PER_CLAIM = 6
MAX_CONTINUITY_SPANS_TOTAL = 24

_CONTINUITY_MARKERS = (
    r"\bprevious(?:ly)?\s+(?:sessions?|lectures?|lessons?|labs?|classes?|topics?|materials?)\b",
    r"\b(?:prior|earlier|last)\s+(?:sessions?|lectures?|lessons?|labs?|classes?|topics?|materials?)\b",
    r"\bnext\s+(?:sessions?|lectures?|lessons?|labs?|classes?|topics?|materials?)\b",
    r"\b(?:previous|prior|earlier|last|next)\s+topics?\b",
    r"\b(?:previous|prior|earlier|last|next)\s+lectures?\b",
    r"\b(?:previous|prior|earlier|last|next)\s+labs?\b",
    r"\b(?:previous|prior|earlier|last|next)\s+lessons?\b",
    r"\b(?:already|already\s+worked\s+with|already\s+covered)\b",
    r"\bbuilds?\s+on\b",
    r"\bcourse\s+roadmap\b",
    r"\broadmap\s+(?:is|shows|describes)\b",
    r"\blecture\s+\d+\b",
    r"\blab\s+\d+\b",
    r"\blesson\s+\d+\b",
    r"\bclass\s+\d+\b",
    r"\b(?:our\s+)?(?:previous|earlier)\s+work\s+on\b",
    r"\bbuilds?\s+on\s+(?:our\s+)?(?:previous|earlier)\s+work\b",
    r"\bas\s+(?:seen|covered)\s+in\s+(?:the\s+)?lecture\b",
    r"\bfrom\s+(?:the\s+)?lecture\s+\d+\b",
    r"\bcovered\s+(?:previously|earlier)\b",
    r"\bwe\s+previously\s+(?:covered|explored|discussed)\b",
)

_MIXED_CONTINUITY_DOMAIN_MARKERS = (
    r"\bwhich\s+(?:are|is|were|was|allow|allows|enable|enables|provide|provides|"
    r"manage|manages|support|supports|make|makes|mean|means)\b",
    r"\bthat\s+(?:are|is|were|was|allow|allows|enable|enables|provide|provides|"
    r"manage|manages|support|supports|make|makes|mean|means)\b",
    r"\band\s+how\b",
    r"\b(?:essential|always|never|faster|slower|widely|necessary|important|"
    r"critical|because|therefore|consequently)\b",
    r"\b(?:define|defines|determines|causes|results in|is used for|are used for)\b",
)

_FUTURE_SEQUENCE_MARKERS = (
    r"\bnext\s+(?:lecture|session|class|lesson|lab|assignment)\b",
    r"\b(?:upcoming|future)\s+(?:lecture|session|class|lesson|lab|assignment)\b",
)

_EXPLICIT_FUTURE_EVIDENCE_MARKERS = (
    r"\b(?:next|upcoming|future|scheduled)\b",
    r"\b(?:roadmap|course plan|sequence)\b",
    r"\bwill cover\b",
)

_CONTINUITY_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "as",
    "at",
    "by",
    "from",
    "into",
    "is",
    "it",
    "its",
    "be",
    "been",
    "being",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "must",
    "not",
    "than",
    "then",
    "there",
    "here",
    "when",
    "where",
    "what",
    "which",
    "who",
    "how",
    "all",
    "some",
    "any",
    "each",
    "both",
    "other",
    "more",
    "most",
    "such",
    "very",
    "also",
    "only",
    "about",
    "after",
    "before",
    "during",
    "through",
    "between",
    "because",
    "while",
    "if",
    "begin",
    "feature",
    "language",
    "let",
    "students",
    "student",
    "allowing",
    "inherited",
    "specialized",
    "solidifying",
    "apply",
    "applies",
    "applied",
    "larger",
    "write",
    "written",
    "writing",
    "implement",
    "implementation",
    "implemented",
    "implementing",
    "learn",
    "reference",
    "base",
    "version",
    "create",
    "created",
    "strategy",
    "strategies",
    "design",
    "application",
    "pattern",
    "robust",
    "interact",
    "combine",
    "we",
    "our",
    "you",
    "your",
    "this",
    "that",
    "understanding",
    "specific",
    "specifically",
    "upon",
    "looking",
    "needing",
    "without",
    "within",
    "them",
    "these",
    "those",
    "their",
    "they",
    "he",
    "she",
    "his",
    "her",
    "me",
    "my",
    "doing",
    "having",
    "using",
    "used",
    "use",
    "make",
    "makes",
    "made",
    "take",
    "takes",
    "taken",
    "give",
    "gives",
    "given",
    "get",
    "gets",
    "got",
    "like",
    "one",
    "two",
    "three",
    "first",
    "second",
    "new",
    "different",
    "same",
    "various",
    "many",
    "much",
    "just",
    "really",
    "well",
    "however",
    "therefore",
    "thus",
    "rather",
    "instead",
    "example",
    "examples",
    "consider",
    "imagine",
    "suppose",
    "scenario",
    "scenarios",
    "decision",
    "architectural",
    "reusable",
    "component",
    "components",
    "via",
    "further",
    "potentially",
    "allow",
    "allows",
    "provide",
    "provides",
    "support",
    "supports",
    "ensure",
    "ensures",
    "result",
    "results",
    "approach",
    "approaches",
    "way",
    "ways",
    "thing",
    "things",
    "part",
    "parts",
    "aspect",
    "aspects",
    "important",
    "generally",
    "in",
    "on",
    "for",
    "with",
    "class",
    "lecture",
    "session",
    "topic",
    "previous",
    "previously",
    "already",
    "today",
}

_STRUCTURAL_CONTINUITY_TERMS = {
    "next",
    "previous",
    "prior",
    "earlier",
    "later",
    "previously",
    "already",
    "upcoming",
    "future",
    "scheduled",
    "session",
    "lecture",
    "class",
    "lesson",
    "lab",
    "assignment",
    "covered",
    "cover",
    "explored",
    "explore",
    "discussed",
    "discuss",
    "build",
    "work",
    "worked",
    "today",
    "will",
    "would",
    "could",
    "may",
}

_GENERIC_CONTINUITY_TERMS = {
    "advanced",
    "concept",
    "topic",
    "programming",
    "code",
}

_LOW_INFORMATION_TOPIC_TERMS = {
    "generic",
    "system",
    "type",
    "strategy",
}

_FUTURE_TOPIC_DESCRIPTOR_TERMS = {"type", "strategy", "expression"}


@dataclass(frozen=True)
class WorkspaceContinuitySelection:
    eligible_claim_ids: list[str]
    reviewable_claim_ids: list[str]
    selected_spans: list[WorkspaceContinuitySpan]
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _WorkspaceContinuityItem:
    item_ref: str
    workspace_item_id: str
    workspace_item_title: str
    role: WorkspaceItemRole
    body: str
    item_order: int


@dataclass(frozen=True)
class _RankedContinuityCandidate:
    span: WorkspaceContinuitySpan
    score: int
    topic_overlap_terms: set[str]
    meaningful_claim_terms: set[str]
    meaningful_candidate_terms: set[str]


def _normalize_for_terms(text: str) -> str:
    cleaned = re.sub(r"\[[^\]]*\]\([^)]+\)", " ", text)
    cleaned = re.sub(r"[*_`#>\-]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().casefold()
    return cleaned


def _normalize_topic_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith(("sses", "shes", "ches", "xes", "zes")):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
        return term[:-1]
    return term


def meaningful_continuity_terms(text: str) -> set[str]:
    """Extract normalized topic words, excluding continuity scaffolding and stop words."""
    terms = set()
    for raw_term in _normalize_for_terms(text).split():
        term = _normalize_topic_term(raw_term)
        if (
            len(term) >= 3
            and term not in _CONTINUITY_STOP_WORDS
            and term not in _STRUCTURAL_CONTINUITY_TERMS
            and term not in _GENERIC_CONTINUITY_TERMS
        ):
            terms.add(term)
    return terms


def _continuity_items(request: GenerationRequest) -> list[_WorkspaceContinuityItem]:
    if request.workspace_context is None:
        return []
    continuity_items = [
        item
        for item in request.workspace_context.items
        if item.role in {WorkspaceItemRole.CONTEXT, WorkspaceItemRole.HISTORY}
    ]
    return [
        _WorkspaceContinuityItem(
            item_ref=f"W{index:02d}",
            workspace_item_id=item.id,
            workspace_item_title=item.title,
            role=item.role,
            body=item.content,
            item_order=index,
        )
        for index, item in enumerate(continuity_items, start=1)
    ]


def _continuity_sentence_spans(text: str) -> list[str]:
    spans: list[str] = []
    in_fence = False
    seen_headers: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith("#"):
            continue
        if re.fullmatch(r"[^\w]*", line):
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", line)
        header_key = _normalize_for_terms(line)
        if header_key and len(header_key.split()) <= 4:
            if header_key in seen_headers:
                continue
            seen_headers.add(header_key)
        line = re.sub(r"\b(?:e\.g|i\.e)\.", lambda match: match.group().replace(".", "<DOT>"), line)
        parts = re.split(r'(?<=[.!?])["”’)]*\s+(?=[A-Z0-9“"($\[])', line)
        for part in parts:
            span = part.replace("<DOT>", ".").strip()
            if not span or not any(character.isalpha() for character in span):
                continue
            terms = meaningful_continuity_terms(span)
            if not terms:
                continue
            if len(span) > 280:
                span = span[:277].rstrip() + "..."
            spans.append(span)
    return spans


def build_workspace_continuity_ledger(request: GenerationRequest) -> list[WorkspaceContinuitySpan]:
    """Create stable continuity spans from selected Workspace context/history only."""

    ledger: list[WorkspaceContinuitySpan] = []
    for item in _continuity_items(request):
        spans = [item.workspace_item_title, *_continuity_sentence_spans(item.body)]
        normalized_spans: list[str] = []
        seen: set[str] = set()
        for span in spans:
            cleaned = " ".join(span.split()).strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized_spans.append(cleaned)
        for span_index, span in enumerate(normalized_spans, start=1):
            ledger.append(
                WorkspaceContinuitySpan(
                    continuity_id=f"{item.item_ref}-C{span_index:03d}",
                    workspace_item_id=item.workspace_item_id,
                    workspace_item_title=item.workspace_item_title,
                    role=item.role,
                    text=span,
                    order=len(ledger) + 1,
                )
            )
    return ledger


def is_workspace_continuity_claim(text: str) -> bool:
    """Conservatively recognize claims about course or project continuity."""

    normalized = " ".join(text.split())
    if not normalized:
        return False
    has_continuity_marker = any(
        re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _CONTINUITY_MARKERS
    )
    has_mixed_domain_claim = any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in _MIXED_CONTINUITY_DOMAIN_MARKERS
    )
    return has_continuity_marker and not has_mixed_domain_claim


def is_mixed_continuity_domain_claim(text: str) -> bool:
    normalized = " ".join(text.split())
    has_continuity_marker = any(
        re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _CONTINUITY_MARKERS
    )
    return has_continuity_marker and any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in _MIXED_CONTINUITY_DOMAIN_MARKERS
    )


def _is_future_sequence_claim(text: str) -> bool:
    return any(
        re.search(pattern, text, flags=re.IGNORECASE) for pattern in _FUTURE_SEQUENCE_MARKERS
    )


def _span_establishes_future_sequence(
    span: WorkspaceContinuitySpan,
    item: _WorkspaceContinuityItem,
) -> bool:
    has_explicit_sequence_marker = any(
        re.search(pattern, span.text, flags=re.IGNORECASE)
        for pattern in _EXPLICIT_FUTURE_EVIDENCE_MARKERS
    )
    numbered_context_item = (
        item.role is WorkspaceItemRole.CONTEXT
        and re.search(r"\b(?:lecture|session|lab|lesson)\s+\d+\b", item.workspace_item_title, re.I)
        is not None
    )
    return has_explicit_sequence_marker or numbered_context_item


def continuity_eligible_claims(claims: list[GroundingClaimInput]) -> list[GroundingClaimInput]:
    return [claim for claim in claims if is_workspace_continuity_claim(claim.text)]


def _continuity_structural_terms(text: str) -> set[str]:
    return {
        term
        for term in _normalize_for_terms(text).split()
        if _normalize_topic_term(term) in _STRUCTURAL_CONTINUITY_TERMS
    }


def _claim_span_score(
    claim_terms: set[str],
    span: WorkspaceContinuitySpan,
) -> tuple[int, set[str]]:
    span_terms = meaningful_continuity_terms(span.text)
    overlap = claim_terms & span_terms
    topical_overlap = overlap - _LOW_INFORMATION_TOPIC_TERMS
    if not topical_overlap:
        return 0, set()
    title_priority = 12 if span.continuity_id.endswith("-C001") else 0
    return title_priority + len(topical_overlap) * 3, topical_overlap


def select_workspace_continuity_support(
    request: GenerationRequest,
    unresolved_claims: list[GroundingClaimInput],
) -> WorkspaceContinuitySelection:
    items = _continuity_items(request)
    eligible_claims = continuity_eligible_claims(unresolved_claims)
    if not eligible_claims or not items:
        return WorkspaceContinuitySelection(
            eligible_claim_ids=[claim.claim_id for claim in eligible_claims],
            reviewable_claim_ids=[],
            selected_spans=[],
            diagnostics={
                "eligible_claim_count": len(eligible_claims),
                "gate_eligible_claim_ids": [claim.claim_id for claim in eligible_claims],
                "reviewable_claim_ids": [],
                "available_item_count": len(items),
                "available_span_count": 0,
                "selected_item_count": 0,
                "selected_span_count": 0,
                "max_spans_per_claim": MAX_CONTINUITY_SPANS_PER_CLAIM,
                "max_total_spans": MAX_CONTINUITY_SPANS_TOTAL,
                "selected_continuity_ids": [],
                "claim_candidates": {},
                "selected_by_claim": {claim.claim_id: [] for claim in eligible_claims},
                "selected_candidate_details": [],
                "rejected_structural_only_candidates": 0,
                "rejected_incomplete_future_claims": [],
                "estimated_prompt_characters": 0,
                "estimated_prompt_tokens": 0,
            },
        )

    all_items_spans = build_workspace_continuity_ledger(request)
    items_by_ref = {item.item_ref: item for item in items}
    selected_by_claim: dict[str, list[WorkspaceContinuitySpan]] = {}
    candidate_details_by_claim: dict[str, list[_RankedContinuityCandidate]] = {}
    rejected_structural_only_candidates = 0
    rejected_incomplete_future_claims: dict[str, list[str]] = {}
    for claim in eligible_claims:
        future_claim = _is_future_sequence_claim(claim.text)
        claim_terms = meaningful_continuity_terms(claim.text)
        if not claim_terms:
            selected_by_claim[claim.claim_id] = []
            candidate_details_by_claim[claim.claim_id] = []
            continue

        allowed_spans = [
            span
            for span in all_items_spans
            if not (
                future_claim
                and not _span_establishes_future_sequence(
                    span, items_by_ref[span.continuity_id.split("-C", 1)[0]]
                )
            )
        ]
        scored_candidates: list[_RankedContinuityCandidate] = []
        for span in allowed_spans:
            candidate_terms = meaningful_continuity_terms(span.text)
            score, overlap = _claim_span_score(claim_terms, span)
            structural_claim_terms = _continuity_structural_terms(claim.text)
            structural_span_terms = _continuity_structural_terms(span.text)
            if not overlap and structural_claim_terms & structural_span_terms:
                rejected_structural_only_candidates += 1
            if score:
                scored_candidates.append(
                    _RankedContinuityCandidate(
                        span=span,
                        score=score,
                        topic_overlap_terms=overlap,
                        meaningful_claim_terms=claim_terms,
                        meaningful_candidate_terms=candidate_terms,
                    )
                )
        ranked = sorted(
            scored_candidates,
            key=lambda candidate: (
                -candidate.score,
                0 if candidate.span.continuity_id.endswith("-C001") else 1,
                int(candidate.span.continuity_id[1:3]),
                int(candidate.span.continuity_id.split("-C", 1)[1]),
                candidate.span.continuity_id,
            ),
        )
        selected_candidates = ranked[:MAX_CONTINUITY_SPANS_PER_CLAIM]
        if future_claim and selected_candidates:
            covered_topic_terms = set().union(
                *(candidate.meaningful_candidate_terms for candidate in selected_candidates)
            )
            missing_topic_terms = sorted(
                claim_terms - _FUTURE_TOPIC_DESCRIPTOR_TERMS - covered_topic_terms
            )
            if missing_topic_terms:
                rejected_incomplete_future_claims[claim.claim_id] = missing_topic_terms
                selected_candidates = []
        selected_by_claim[claim.claim_id] = [candidate.span for candidate in selected_candidates]
        candidate_details_by_claim[claim.claim_id] = selected_candidates

    selected_spans: list[WorkspaceContinuitySpan] = []
    seen_ids: set[str] = set()
    for claim in eligible_claims:
        for span in selected_by_claim.get(claim.claim_id, []):
            if span.continuity_id in seen_ids:
                continue
            seen_ids.add(span.continuity_id)
            selected_spans.append(span)
            if len(selected_spans) >= MAX_CONTINUITY_SPANS_TOTAL:
                break
        if len(selected_spans) >= MAX_CONTINUITY_SPANS_TOTAL:
            break

    selected_ids = {span.continuity_id for span in selected_spans}
    selected_ids_by_claim = {
        claim_id: [span.continuity_id for span in spans if span.continuity_id in selected_ids]
        for claim_id, spans in selected_by_claim.items()
    }
    for claim in eligible_claims:
        if (
            not _is_future_sequence_claim(claim.text)
            or claim.claim_id in rejected_incomplete_future_claims
            or not selected_ids_by_claim.get(claim.claim_id)
        ):
            continue
        covered_topic_terms = set().union(
            *(
                meaningful_continuity_terms(span.text)
                for span in selected_by_claim[claim.claim_id]
                if span.continuity_id in selected_ids
            )
        )
        missing_topic_terms = sorted(
            meaningful_continuity_terms(claim.text)
            - _FUTURE_TOPIC_DESCRIPTOR_TERMS
            - covered_topic_terms
        )
        if missing_topic_terms:
            selected_ids_by_claim[claim.claim_id] = []
            rejected_incomplete_future_claims[claim.claim_id] = missing_topic_terms
    reviewable_claim_ids = [
        claim.claim_id for claim in eligible_claims if selected_ids_by_claim.get(claim.claim_id)
    ]
    selected_candidate_details = [
        {
            "claim_id": claim.claim_id,
            "continuity_id": candidate.span.continuity_id,
            "meaningful_claim_terms": sorted(candidate.meaningful_claim_terms),
            "meaningful_candidate_terms": sorted(candidate.meaningful_candidate_terms),
            "topic_overlap_terms": sorted(candidate.topic_overlap_terms),
            "score": candidate.score,
            "selected": True,
        }
        for claim in eligible_claims
        for candidate in candidate_details_by_claim.get(claim.claim_id, [])
        if candidate.span.continuity_id in selected_ids_by_claim.get(claim.claim_id, [])
    ]
    prompt_chars = sum(len(span.text) + len(span.continuity_id) + 8 for span in selected_spans)
    claim_candidates = selected_ids_by_claim
    selected_item_refs = {span.continuity_id.split("-C", 1)[0] for span in selected_spans}
    return WorkspaceContinuitySelection(
        eligible_claim_ids=[claim.claim_id for claim in eligible_claims],
        reviewable_claim_ids=reviewable_claim_ids,
        selected_spans=selected_spans,
        diagnostics={
            "eligible_claim_count": len(eligible_claims),
            "gate_eligible_claim_ids": [claim.claim_id for claim in eligible_claims],
            "reviewable_claim_ids": reviewable_claim_ids,
            "available_item_count": len(items),
            "available_span_count": len(all_items_spans),
            "selected_item_count": len(selected_item_refs),
            "selected_span_count": len(selected_spans),
            "max_spans_per_claim": MAX_CONTINUITY_SPANS_PER_CLAIM,
            "max_total_spans": MAX_CONTINUITY_SPANS_TOTAL,
            "selected_continuity_ids": [span.continuity_id for span in selected_spans],
            "claim_candidates": claim_candidates,
            "selected_by_claim": claim_candidates,
            "selected_spans": [span.model_dump(mode="json") for span in selected_spans],
            "selected_candidate_details": selected_candidate_details,
            "rejected_structural_only_candidates": rejected_structural_only_candidates,
            "rejected_incomplete_future_claims": [
                {"claim_id": claim_id, "missing_topic_terms": missing_terms}
                for claim_id, missing_terms in rejected_incomplete_future_claims.items()
            ],
            "estimated_prompt_characters": prompt_chars,
            "estimated_prompt_tokens": (prompt_chars + 3) // 4,
        },
    )
