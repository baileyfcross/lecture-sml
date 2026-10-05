"""Offline tests for local knowledge indexing and hybrid retrieval."""

import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from lecture_slm.config.loader import load_model_config
from lecture_slm.generation.models import (
    GenerationProfileName,
    GenerationRequest,
    GenerationStatus,
    SourceMaterial,
)
from lecture_slm.generation.profiles import load_generation_profiles
from lecture_slm.generation.router import GenerationRouter
from lecture_slm.inference.ollama_client import ChatResponse
from lecture_slm.ingestion.discovery import _is_repository_generated_path
from lecture_slm.knowledge.assembler import KnowledgeContextAssembler
from lecture_slm.knowledge.chunking import CHUNKING_VERSION, chunk_document
from lecture_slm.knowledge.config import (
    ChunkingConfig,
    KnowledgeConfig,
    RetrievalConfig,
    SourceFocusConfig,
    load_knowledge_config,
)
from lecture_slm.knowledge.embeddings import EmbeddingProvider
from lecture_slm.knowledge.evaluation.metrics import (
    aggregate_metrics,
    case_metrics,
    classify_resolution_failures,
)
from lecture_slm.knowledge.evaluation.models import RetrievalEvalCase, RetrievalReview
from lecture_slm.knowledge.evaluation.runner import (
    load_cases,
    run_evaluation,
    update_review_summary,
)
from lecture_slm.knowledge.indexer import KnowledgeIndexer
from lecture_slm.knowledge.models import RetrievalMatch, RetrievalRequest, RetrievalResult
from lecture_slm.knowledge.obsidian import parse_obsidian_markdown
from lecture_slm.knowledge.ranking import reciprocal_rank_fusion
from lecture_slm.knowledge.resolution import SourceResolver
from lecture_slm.knowledge.retrieval import (
    KnowledgeRetriever,
    _candidate_term_coverage,
    _query_term_coverage_boost,
    _query_terms,
    _term_coverage,
    normalize_coverage_term,
)
from lecture_slm.knowledge.roles import ChunkRole, classify_chunk_role
from lecture_slm.knowledge.source_focus import SourceFocusKind, classify_source_focus
from lecture_slm.knowledge.stats import knowledge_stats
from lecture_slm.knowledge.storage import KnowledgeStore
from lecture_slm.knowledge.vault import dry_run_report
from lecture_slm.schemas.dataset import TaskType

ROOT = Path(__file__).parents[1]


class FakeEmbeddings(EmbeddingProvider):
    model_name = "fake/test"
    provider_version = "fake-1"
    dimension = 64

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls += 1
        return self._vector(text)

    def _vector(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
            vector[index % self.dimension] += 1.0
        return vector


def knowledge_config(vault: Path, data: Path, *, target: int = 36) -> KnowledgeConfig:
    return KnowledgeConfig(
        vault_path=vault,
        data_dir=data,
        chunking=ChunkingConfig(
            target_tokens=target,
            max_tokens=target + 12,
            overlap_tokens=min(4, target - 1),
        ),
        retrieval=RetrievalConfig(
            lexical_candidates=20,
            semantic_candidates=20,
            fused_candidates=15,
            final_results=8,
            neighbor_expansion=0,
        ),
    )


def write_note(
    path: Path, *, first_section: str = "Rules of inference derive valid conclusions."
) -> None:
    path.write_text(
        f"""---
title: Foundations of Computation
aliases: [Foundations]
tags: [logic, discrete-math]
course: CSC220
unknown_field: retained safely
---

# Foundations of Computation

## 1.6 Deduction

{first_section}

![[Predicate Logic|predicates]]

## 1.7 Proofs

Predicate logic uses quantified variables and relations.
""",
        encoding="utf-8",
    )


def write_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as file:
        writer.write(file)


def test_knowledge_config_environment_override_and_validation(tmp_path: Path) -> None:
    config = load_knowledge_config(
        ROOT / "configs/knowledge/default.yaml",
        environ={"LECTURE_SLM_VAULT_PATH": str(tmp_path / "vault")},
    )
    assert config.vault_path == tmp_path / "vault"
    assert config.retrieval.source_focus.enabled is False
    with pytest.raises(ValidationError, match="overlap_tokens"):
        ChunkingConfig(target_tokens=5, max_tokens=8, overlap_tokens=5)
    with pytest.raises(ValidationError, match="final_results"):
        RetrievalConfig(fused_candidates=4, final_results=5)
    assert _is_repository_generated_path(ROOT / "data/knowledge/test.sqlite")
    assert _is_repository_generated_path(ROOT / "data/processed/train.jsonl")
    assert RetrievalConfig().source_focus.enabled is False
    assert RetrievalConfig().source_focus.focused_note_boost == 0.01
    assert RetrievalConfig().source_focus.overview_tag_adjustment == -0.005


def test_coverage_term_normalization_and_nonlinear_boost() -> None:
    plural_pairs = (
        ("quantifier", "quantifiers"),
        ("variable", "variables"),
        ("predicate", "predicates"),
        ("relation", "relations"),
        ("statement", "statements"),
        ("proposition", "propositions"),
    )
    for singular, plural in plural_pairs:
        assert normalize_coverage_term(singular) == singular
        assert normalize_coverage_term(plural) == singular
        assert _term_coverage([plural], singular)["coverage"] == 1.0
        assert _term_coverage([singular], plural)["coverage"] == 1.0

    assert normalize_coverage_term("  (Quantifiers), ") == "quantifier"
    assert normalize_coverage_term("categories") == "category"
    assert normalize_coverage_term("species") == "species"
    assert normalize_coverage_term("series") == "series"
    assert normalize_coverage_term("movies") == "movie"
    assert normalize_coverage_term("status") == "status"
    assert normalize_coverage_term("analysis") == "analysis"
    assert normalize_coverage_term("class") == "class"

    query_terms = _query_terms("quantifiers and variables")
    assert query_terms == ["quantifiers", "variables"]
    candidate_a = _term_coverage(
        query_terms, "The universal quantifier binds a variable in a predicate."
    )
    candidate_b = _term_coverage(
        query_terms, "Variables represent values in an algebraic expression."
    )
    assert candidate_a["normalized_query_terms"] == ["quantifier", "variable"]
    assert candidate_a["matched_terms"] == ["quantifier", "variable"]
    assert candidate_a["matched_original_terms"] == ["quantifiers", "variables"]
    assert candidate_a["coverage"] == 1.0
    assert candidate_b["coverage"] == 0.5
    assert _query_term_coverage_boost(candidate_a["coverage"], 0.05) == pytest.approx(0.05)
    assert _query_term_coverage_boost(candidate_b["coverage"], 0.05) == pytest.approx(0.0125)
    assert _query_term_coverage_boost(0.75, 0.05) == pytest.approx(0.028125)
    assert _query_term_coverage_boost(0.25, 0.05) == pytest.approx(0.003125)
    assert _query_term_coverage_boost(0.0, 0.05) == 0.0

    repeated = _query_terms("variables variables quantifiers")
    assert repeated == ["variables", "quantifiers"]
    assert len(repeated) == 2
    assert _query_terms("variable variables") == ["variable"]


def test_candidate_coverage_includes_title_and_section_only() -> None:
    coverage = _candidate_term_coverage(
        ["quantifiers", "variables"],
        {
            "title": "Universal Quantifier",
            "section_title": "Bound Variables",
            "text": "A concise factual definition.",
        },
    )
    assert coverage["coverage"] == 1.0
    assert coverage["matched_terms"] == ["quantifier", "variable"]


def test_source_focus_classification_is_component_based() -> None:
    categories = {
        "6 - Full Notes/Universal Quantifier.md": SourceFocusKind.FOCUSED_NOTE,
        "3 - Tags/Formal Logic.md": SourceFocusKind.OVERVIEW_TAG,
        "lecture-notes/Logic.md": SourceFocusKind.NEUTRAL,
    }
    for path, expected in categories.items():
        assert (
            classify_source_focus(
                path,
                focused_note_directories=["6 - Full Notes"],
                overview_tag_directories=["3 - Tags"],
            )
            is expected
        )
    assert (
        classify_source_focus(
            "/private/vault/6 - Full Notes/Universal Quantifier.md",
            focused_note_directories=["6 - Full Notes"],
            overview_tag_directories=["3 - Tags"],
        )
        is SourceFocusKind.FOCUSED_NOTE
    )


def test_obsidian_metadata_and_malformed_frontmatter_are_safe() -> None:
    metadata, warnings = parse_obsidian_markdown(
        "---\naliases: [Proofs]\ntags: [logic]\ncustom: ok\n---\n"
        "# Heading\n#inference [[Lemma|the lemma]] ![[Diagram]]"
    )
    assert not warnings
    assert metadata["aliases"] == ["Proofs"]
    assert metadata["tags"] == ["logic", "inference"]
    assert metadata["custom"] == "ok"
    assert metadata["outgoing_links"] == [
        {"target": "Lemma", "alias": "the lemma", "embedded": False},
        {"target": "Diagram", "alias": "", "embedded": True},
    ]
    _, warnings = parse_obsidian_markdown("---\ntitle: [broken\n")
    assert warnings and "Malformed YAML" in warnings[0]


def test_index_incremental_duplicate_change_and_delete(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    original = vault / "Foundations.md"
    write_note(original)
    duplicate = vault / "copy.md"
    shutil.copy2(original, duplicate)
    original_before = hashlib.sha256(original.read_bytes()).hexdigest()
    data = tmp_path / "local-index"
    embeddings = FakeEmbeddings()
    indexer = KnowledgeIndexer(knowledge_config(vault, data), embeddings)

    first = indexer.index()
    assert first.files_scanned == 2
    assert first.normalized == 1
    assert first.embeddings_generated > 0
    assert first.chunks_created > 0
    assert embeddings.document_calls == 1
    with KnowledgeStore(
        data,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        assert store.stats()["indexed_sources"] == 1
        assert len(store.paths_for_source(store.source_records()[0]["source_id"])) == 2

    second = indexer.index()
    assert second.unchanged == 2
    assert embeddings.document_calls == 1

    write_note(original, first_section="Updated inference rules have a changed derivation.")
    changed = indexer.index()
    assert changed.changed == 1
    assert changed.embeddings_reused > 0
    assert original_before != hashlib.sha256(original.read_bytes()).hexdigest()

    duplicate.unlink()
    deleted = indexer.index()
    assert deleted.deleted == 1
    assert original.exists()


def test_metadata_section_retrieval_uses_title_then_heading(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "foundations.md")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        result = KnowledgeRetriever(store, embeddings, config.retrieval).retrieve(
            RetrievalRequest(query="Foundations of Computation section 1.6 deduction", top_k=4)
        )
        unresolved_section = KnowledgeRetriever(store, embeddings, config.retrieval).retrieve(
            RetrievalRequest(
                query="rules of inference",
                source_title="Foundations of Computation",
                section="9.9",
            )
        )
        assert not unresolved_section.matches
        assert any("No extracted heading matched" in item for item in unresolved_section.warnings)
    assert result.resolved_source is not None
    assert result.resolved_source["title"] == "Foundations of Computation"
    assert result.resolved_section == "1.6 Deduction"
    assert result.matches
    assert all("Deduction" in (match.section or "") for match in result.matches)
    assert result.matches[0].provenance["relative_path"] == "foundations.md"
    assert result.matches[0].lexical_rank is not None
    assert result.matches[0].semantic_rank is not None
    assert result.diagnostics["source_resolution"]["resolved"]["resolution_reason"] == (
        "title named in query"
    )
    assert result.diagnostics["lexical_candidates"]
    assert result.diagnostics["semantic_candidates"]
    assert result.diagnostics["fused_candidates"]
    assert any(
        details["exact_title"] and details["section"]
        for details in result.diagnostics["metadata_boosts"].values()
    )


def test_unmatched_explicit_source_title_fails_closed(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "foundations.md")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        result = KnowledgeRetriever(store, embeddings, config.retrieval).retrieve(
            RetrievalRequest(
                query="rules of inference",
                source_title="Missing course notes",
                top_k=2,
            )
        )

    assert not result.matches
    assert any("No indexed source matched" in warning for warning in result.warnings)
    assert embeddings.query_calls == 0


def test_pdf_page_provenance_survives_index_and_retrieval(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_text_pdf(vault / "networking.pdf", "DNS maps host names to addresses.")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        result = KnowledgeRetriever(store, embeddings, config.retrieval).retrieve(
            RetrievalRequest(query="DNS maps host names", top_k=2)
        )
    assert result.matches[0].page_number == 1
    assert result.matches[0].provenance["source_hash"]
    assert result.matches[0].source_id


def test_status_is_read_only_and_reports_index_features(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "foundations.md")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    database = config.data_dir / "knowledge.sqlite"
    digest_before = hashlib.sha256(database.read_bytes()).hexdigest()
    status = knowledge_stats(config.data_dir)
    digest_after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert digest_before == digest_after
    assert status["indexed_sources"] == 1
    assert status["indexed_chunks"] > 0
    assert status["fts_available"] is True
    assert status["schema_version"] == "2"
    assert status["chunk_size_tokens"]["minimum"] > 0
    assert status["chunk_size_tokens"]["maximum"] >= status["chunk_size_tokens"]["median"]


def test_chunk_roles_exclude_vault_scaffolding_from_factual_retrieval(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Predicate Logic.md").write_text(
        """---
title: Predicate Logic
aliases: [First Order Logic]
tags: [logic]
course: CSC220
---

# Predicate Logic

## Semantics
Predicate logic uses quantifiers and variables to express properties and relations.

## Metadata
2025-06-18 22:54
Status: #seed
Tags: [[logic]]

## References
[[Universal Quantifier.pdf]]
[[Predicate Calculus.pdf]]

## Linked Full Notes
```query
LIST FROM #logic
```
[[Universal Quantifier]]
""",
        encoding="utf-8",
    )
    (vault / "Universal Quantifier.md").write_text(
        """# Universal Quantifier

## Definition
Quantifiers and variables allow a predicate to express a property for every object.
""",
        encoding="utf-8",
    )
    (vault / "Change of Variables.md").write_text(
        """# Change of Variables

## Substitution
Variables can be renamed during substitution in an algebraic expression.
""",
        encoding="utf-8",
    )
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()

    assert classify_chunk_role("References", "An informative reference paragraph.") == (
        ChunkRole.REFERENCE
    )
    assert classify_chunk_role("Semantics", "A concise factual definition.") == (ChunkRole.CONTENT)
    assert classify_chunk_role("Metadata", "Status: #seed\nTags: [[logic]]") == (ChunkRole.METADATA)
    assert classify_chunk_role("Linked Full Notes", "[[One]]") == ChunkRole.NAVIGATION

    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        rows = store.connection.execute(
            "SELECT title, section_title, role, embedding_key, outgoing_links_json FROM chunks"
        ).fetchall()
        roles_by_section = {
            str(row["section_title"]): str(row["role"])
            for row in rows
            if str(row["title"]) == "Predicate Logic"
        }
        assert roles_by_section["Semantics"] == ChunkRole.CONTENT.value
        assert roles_by_section["Metadata"] == ChunkRole.METADATA.value
        assert roles_by_section["References"] == ChunkRole.REFERENCE.value
        assert roles_by_section["Linked Full Notes"] == ChunkRole.NAVIGATION.value
        assert all(
            row["embedding_key"] is None
            for row in rows
            if str(row["role"]) != ChunkRole.CONTENT.value
        )
        assert {
            "Universal Quantifier.pdf",
            "Predicate Calculus.pdf",
            "Universal Quantifier",
        }.issubset(
            json.loads(
                next(
                    row["outgoing_links_json"]
                    for row in rows
                    if row["section_title"] == "References"
                )
            )
        )
        assert store.role_stats()["reference"]["embedded_chunks"] == 0

        retriever = KnowledgeRetriever(store, embeddings, config.retrieval)
        default_result = retriever.retrieve(
            RetrievalRequest(
                query="quantifiers and variables",
                mode="lexical",
                top_k=8,
                neighbor_expansion=2,
            )
        )
        assert default_result.matches
        assert all(match.role is ChunkRole.CONTENT for match in default_result.matches)
        assert any(match.source_title == "Universal Quantifier" for match in default_result.matches)
        assert any(
            candidate["query_term_coverage"]["coverage"] == 1.0
            for candidate in default_result.diagnostics["fused_candidates"]
        )
        scaffolding_match = retriever.retrieve(
            RetrievalRequest(
                query="Predicate Logic",
                source_title="Predicate Logic",
                mode="lexical",
                top_k=8,
            )
        )
        assert scaffolding_match.diagnostics["excluded_non_content_lexical_candidates"]

        reference_result = retriever.retrieve(
            RetrievalRequest(
                query="References",
                source_title="Predicate Logic",
                passage_roles=[ChunkRole.REFERENCE],
                mode="lexical",
                top_k=4,
            )
        )
        assert reference_result.matches
        assert all(match.role is ChunkRole.REFERENCE for match in reference_result.matches)

        topical_title_mention = retriever.retrieve(
            RetrievalRequest(
                query="Predicate Logic quantifiers variables",
                mode="lexical",
                top_k=8,
            )
        )
        assert (
            topical_title_mention.diagnostics["source_resolution"]["hard_constraint_source_id"]
            is None
        )
        assert topical_title_mention.diagnostics["source_resolution"]["strength"] == "inferred"
        assert topical_title_mention.diagnostics["query_terms"] == [
            "predicate",
            "logic",
            "quantifiers",
            "variables",
        ]
        assert topical_title_mention.diagnostics["normalized_query_terms"] == [
            "predicate",
            "logic",
            "quantifier",
            "variable",
        ]
        assert all(
            "normalized_query_terms" in candidate["query_term_coverage"]
            for candidate in topical_title_mention.diagnostics["fused_candidates"]
        )
        assert len({match.source_title for match in topical_title_mention.matches}) > 1
        named_source_id = next(
            str(row["source_id"])
            for row in store.source_records()
            if str(row["title"]) == "Predicate Logic"
        )
        document_lookup = retriever.retrieve(
            RetrievalRequest(
                query="Lecture 5 Predicate Logic",
                mode="lexical",
                top_k=8,
            )
        )
        assert (
            document_lookup.diagnostics["source_resolution"]["hard_constraint_source_id"]
            == named_source_id
        )
        assert document_lookup.diagnostics["source_resolution"]["strength"] == "explicit"
        assert document_lookup.matches
        assert all(match.source_id == named_source_id for match in document_lookup.matches)
        assert all(match.role is ChunkRole.CONTENT for match in document_lookup.matches)


def test_source_focus_is_a_tiebreaker_not_a_filter(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    focused = vault / "6 - Full Notes" / "Focused Concept.md"
    overview = vault / "3 - Tags" / "Tag Overview.md"
    strong_overview = vault / "3 - Tags" / "Broad Evidence.md"
    for path in (focused, overview, strong_overview):
        path.parent.mkdir(parents=True, exist_ok=True)
    focused.write_text(
        "# Focused Concept\n\nQuantifiers bind variables.\n",
        encoding="utf-8",
    )
    overview.write_text(
        "# Tag Overview\n\nQuantifiers bind variables. Overview.\n",
        encoding="utf-8",
    )
    strong_overview.write_text(
        "# Broad Evidence\n\n"
        "Quantifiers variables predicates relations. "
        "Quantifiers variables predicates relations.\n",
        encoding="utf-8",
    )

    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        retriever = KnowledgeRetriever(store, embeddings, config.retrieval)
        tied = retriever.retrieve(
            RetrievalRequest(query="quantifiers variables", mode="lexical", top_k=8)
        )
        assert tied.matches
        by_title = {
            candidate["source_title"]: candidate
            for candidate in tied.diagnostics["fused_candidates"]
        }
        assert by_title["Focused Concept"]["source_focus_kind"] == "focused_note"
        assert by_title["Focused Concept"]["source_focus_enabled"] is False
        assert by_title["Focused Concept"]["source_focus_boost"] == 0.0
        assert by_title["Tag Overview"]["source_focus_kind"] == "overview_tag"
        assert by_title["Tag Overview"]["source_focus_enabled"] is False
        assert by_title["Tag Overview"]["source_focus_boost"] == 0.0
        assert (
            by_title["Focused Concept"]["boost_details"]["source_focus_boost"]
            == by_title["Tag Overview"]["boost_details"]["source_focus_boost"]
            == 0.0
        )

        experimental_config = config.retrieval.model_copy(
            update={"source_focus": SourceFocusConfig(enabled=True)}
        )
        experimental = KnowledgeRetriever(store, embeddings, experimental_config).retrieve(
            RetrievalRequest(query="quantifiers variables", mode="lexical", top_k=8)
        )
        experimental_by_title = {
            candidate["source_title"]: candidate
            for candidate in experimental.diagnostics["fused_candidates"]
        }
        assert experimental_by_title["Focused Concept"]["source_focus_enabled"] is True
        assert experimental_by_title["Focused Concept"]["source_focus_boost"] == pytest.approx(0.01)
        assert experimental_by_title["Tag Overview"]["source_focus_enabled"] is True
        assert experimental_by_title["Tag Overview"]["source_focus_boost"] == pytest.approx(-0.005)
        assert experimental_by_title["Focused Concept"]["fused_score"] - by_title[
            "Focused Concept"
        ]["fused_score"] == pytest.approx(0.01)
        assert experimental_by_title["Tag Overview"]["fused_score"] - by_title["Tag Overview"][
            "fused_score"
        ] == pytest.approx(-0.005)

        stronger_tag = retriever.retrieve(
            RetrievalRequest(
                query="quantifiers variables predicates relations",
                mode="lexical",
                top_k=8,
            )
        )
        assert stronger_tag.matches
        assert stronger_tag.matches[0].source_title == "Broad Evidence"
        assert any(match.source_title == "Focused Concept" for match in stronger_tag.matches)


def test_source_resolution_methods_and_ambiguity_are_explicit(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "foundations.md")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        resolver = SourceResolver(store)
        expectations = {
            "Foundations of Computation": "exact title",
            "foundations of computation": "normalized title",
            "Foundations": "alias",
            "Foundations of Computaton": "fuzzy title match",
        }
        for title, reason in expectations.items():
            source, warnings = resolver.resolve_source("unrelated query", title)
            assert source is not None
            assert source["resolution_reason"] == reason
            assert not warnings
        missing_source, missing_warnings = resolver.resolve_source(
            "explain quantifiers", "Definitely Missing Lecture SLM Source XYZ"
        )
        assert missing_source is None
        assert missing_warnings and "No indexed source matched" in missing_warnings[0]

    write_note(vault / "duplicate-title.md", first_section="A changed source version.")
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        source, warnings = SourceResolver(store).resolve_source(
            "rules", "Foundations of Computation"
        )
    assert source is None
    assert warnings and "ambiguous" in warnings[0].casefold()


def test_lexical_and_semantic_filters_rrf_and_assembler(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "foundations.md")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        retriever = KnowledgeRetriever(store, embeddings, config.retrieval)
        result = retriever.retrieve(
            RetrievalRequest(
                query="rules inference",
                course="CSC220",
                tags=["logic"],
                document_types=["markdown"],
                folder="",
                top_k=3,
            )
        )
        assert result.matches
        assert result.matches[0].lexical_score is not None
        assert result.matches[0].semantic_score is not None
        assert result.filters_applied["course"] == "CSC220"
        assert not retriever.retrieve(
            RetrievalRequest(query="rules inference", course="OTHER", top_k=3)
        ).matches

    scores = reciprocal_rank_fusion(
        {"a": 1, "b": 2},
        {"b": 1, "c": 2},
        constant=60,
    )
    assert scores["b"] > scores["a"]
    assert set(scores) == {"a", "b", "c"}

    match = result.matches[0]
    source_material = KnowledgeContextAssembler().assemble(result, source_context_budget=2000)
    assert source_material
    assert isinstance(source_material[0], SourceMaterial)
    assert source_material[0].metadata["source_origin"] == "retrieved_knowledge"
    assert source_material[0].metadata["chunk_ids"]
    assert KnowledgeContextAssembler().assemble(result, source_context_budget=1) == []
    assert match.text in source_material[0].text or source_material[0].text in match.text


def test_assembler_keeps_page_boundaries() -> None:
    matches = [
        RetrievalMatch(
            chunk_id=f"chunk-{page}",
            source_id="source",
            source_title="Lecture notes",
            source_path="lecture.pdf",
            page_number=page,
            text=f"Content from page {page}.",
            fused_rank=page,
            fused_score=1 / page,
            metadata={"chunk_index": page - 1},
        )
        for page in (1, 2)
    ]
    result = RetrievalResult(query="content", matches=matches)

    assembled = KnowledgeContextAssembler().assemble(result, source_context_budget=100)

    assert len(assembled) == 2
    assert [source.metadata["page_numbers"] for source in assembled] == [[1], [2]]
    assert [source.text for source in assembled] == [
        "Content from page 1.",
        "Content from page 2.",
    ]


def test_failed_source_does_not_abort_and_deleted_content_is_not_retrieved(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "good.md")
    (vault / "broken.pdf").write_bytes(b"not a real pdf")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    metrics = KnowledgeIndexer(config, embeddings).index()
    assert metrics.failed == 1
    assert metrics.normalized == 1
    assert metrics.errors and "broken.pdf" in metrics.errors[0]
    (vault / "broken.pdf").unlink()
    (vault / "good.md").unlink()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        assert store.stats()["indexed_chunks"] == 0
        assert store.stats()["stale_sources"] >= 1


def test_version_mismatch_and_vault_boundary_fail_clearly(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "note.md")
    data = tmp_path / "index"
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(knowledge_config(vault, data), embeddings).index()
    with pytest.raises(ValueError, match="embedding_model"):
        KnowledgeStore(
            data,
            embedding_model="different/model",
            embedding_version=embeddings.provider_version,
            chunking_version=CHUNKING_VERSION,
        )
    with pytest.raises(ValueError, match="outside the read-only vault"):
        KnowledgeIndexer(knowledge_config(vault, vault / "index"), embeddings)
    connection = sqlite3.connect(data / "knowledge.sqlite")
    try:
        connection.execute("UPDATE index_metadata SET value='1' WHERE key='schema_version'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="Unsupported knowledge schema version"):
        KnowledgeStore(
            data,
            embedding_model=embeddings.model_name,
            embedding_version=embeddings.provider_version,
            chunking_version=CHUNKING_VERSION,
        )
    with KnowledgeStore(
        data,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
        rebuild=True,
    ) as store:
        assert store.stats()["indexed_sources"] == 0


def test_failed_changed_source_keeps_last_successful_version(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    source = vault / "book.pdf"
    write_text_pdf(source, "A valid source describing rules of inference.")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    source.write_bytes(b"damaged replacement")

    metrics = KnowledgeIndexer(config, embeddings).index()
    assert metrics.failed == 1
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        assert store.stats()["indexed_chunks"] > 0
        current = store.source_path("book.pdf")
        assert current is not None
        assert current["extraction_status"] == "success"


def test_chunk_ids_and_neighbor_links_are_deterministic(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "note.md"
    write_note(note, first_section=("inference rules " * 70))
    config = knowledge_config(vault, tmp_path / "index", target=10)
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    from lecture_slm.ingestion.extractors.markdown import MarkdownExtractor
    from lecture_slm.ingestion.manifest import sha256_file

    digest = sha256_file(note)
    document = MarkdownExtractor().extract(
        note,
        source_id="src-" + digest,
        source_hash=digest,
        relative_path="note.md",
    )
    first = chunk_document(
        document,
        relative_path="note.md",
        source_version=1,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        config=config.chunking,
        note_metadata=parse_obsidian_markdown(note.read_text(encoding="utf-8"))[0],
    )
    second = chunk_document(
        document,
        relative_path="note.md",
        source_version=1,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        config=config.chunking,
        note_metadata=parse_obsidian_markdown(note.read_text(encoding="utf-8"))[0],
    )
    assert len(first) > 2
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(
        chunk.next_chunk_id == first[index + 1].chunk_id for index, chunk in enumerate(first[:-1])
    )
    assert all(
        chunk.previous_chunk_id == first[index - 1].chunk_id
        for index, chunk in enumerate(first[1:], 1)
    )


def test_retrieval_assembles_request_and_uses_existing_router(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    write_note(vault / "foundations.md")
    config = knowledge_config(vault, tmp_path / "index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        retrieved = KnowledgeRetriever(store, embeddings, config.retrieval).retrieve(
            RetrievalRequest(query="Foundations of Computation section 1.6 deduction", top_k=2)
        )
    retrieved_sources = KnowledgeContextAssembler().assemble(retrieved, source_context_budget=1500)
    explicit_source = SourceMaterial(
        source_id="manual",
        title="Instructor note",
        text="Use a short worked example.",
        metadata={"source_origin": "explicit_file"},
    )
    request = GenerationRequest(
        task=TaskType.EXPLANATION,
        profile=GenerationProfileName.QUICK,
        instruction="Explain deduction.",
        source_material=[explicit_source, *retrieved_sources],
    )
    assert len(request.source_material) >= 2
    assert request.source_material[0].metadata["source_origin"] == "explicit_file"
    assert request.source_material[1].metadata["source_origin"] == "retrieved_knowledge"

    calls: list[str] = []

    class FakeGenerationClient:
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
            format: str | dict[str, object] | None = None,
        ) -> ChatResponse:
            calls.append(user_message)
            return ChatResponse(
                model=model,
                content="Deduction applies valid rules of inference.",
                prompt_tokens=20,
                completion_tokens=10,
                total_duration_ns=1_000_000_000,
                prompt_eval_duration_ns=400_000_000,
                eval_duration_ns=600_000_000,
                completion_reason="stop",
            )

    router = GenerationRouter(
        model_config=load_model_config(ROOT / "configs/models/qwen35-9b.yaml", environ={}),
        profiles=load_generation_profiles(ROOT / "configs/generation/profiles.yaml"),
        client_factory=lambda host, timeout: FakeGenerationClient(),  # type: ignore[arg-type]
    )
    generated = router.route(request)
    assert generated.status is GenerationStatus.COMPLETED
    assert calls and "Supplied source material" in calls[0]
    assert "Foundations of Computation" in calls[0]


def test_retrieval_modes_and_all_mode_evaluation_are_private(tmp_path: Path) -> None:
    vault = tmp_path / "private-vault"
    vault.mkdir()
    write_note(vault / "foundations.md")
    config = knowledge_config(vault, tmp_path / "local-index")
    embeddings = FakeEmbeddings()
    KnowledgeIndexer(config, embeddings).index()
    with KnowledgeStore(
        config.data_dir,
        embedding_model=embeddings.model_name,
        embedding_version=embeddings.provider_version,
        chunking_version=CHUNKING_VERSION,
    ) as store:
        source_id = str(store.source_records()[0]["source_id"])
        case = RetrievalEvalCase(
            id="deduction-section",
            query="rules of inference",
            category="exact_source_section",
            source_title="Foundations of Computation",
            section="1.6",
            expected_source_ids=[source_id],
            expected_source_titles=["Foundations of Computation"],
            expected_sections=["1.6 Deduction"],
        )
        for mode in ("lexical", "semantic", "hybrid"):
            result = KnowledgeRetriever(store, embeddings, config.retrieval).retrieve(
                RetrievalRequest(
                    query=case.query,
                    source_title=case.source_title,
                    section=case.section,
                    mode=mode,
                    neighbor_expansion=0,
                )
            )
            assert result.retrieval_configuration["mode"] == mode
            if mode == "lexical":
                assert result.timings.query_embedding_seconds == 0
            else:
                assert result.timings.query_embedding_seconds >= 0

        case_file = tmp_path / "cases.jsonl"
        case_file.write_text(case.model_dump_json() + "\n", encoding="utf-8")
        loaded = load_cases(case_file)
        assert len(loaded) == 1
        run_dir = run_evaluation(
            cases=loaded,
            store=store,
            embeddings=embeddings,
            config=config.retrieval,
            output_root=config.data_dir / "evaluation" / "results",
            mode="all",
            compare_neighbors=True,
        )

    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert {record["mode"] for record in results} == {
        "lexical",
        "semantic",
        "hybrid",
        "hybrid-neighbors-0",
        "hybrid-neighbors-1",
    }
    assert results[0]["metrics"]["hit_at_k"] is True
    assert results[0]["metrics"]["source_resolution_accuracy"] is True
    assert results[0]["metrics"]["section_resolution_accuracy"] is True
    assert results[0]["context_assembly"]["configured_budget"] > 0
    assert "normalized_query_terms" in results[0]["resolution"]
    assert results[0]["retrieval_configuration"]["query_term_coverage_exponent"] == 2
    assert any(
        "query_term_coverage_boost" in candidate["boost_details"]
        and "source_focus_boost" in candidate["boost_details"]
        for candidate in results[0]["resolution"]["fused_candidates"]
    )
    assert summary["mode_comparison"]["lexical"]["mrr"] is not None
    assert str(vault) not in json.dumps(run_metadata)
    assert (run_dir / "reviews.jsonl").read_text(encoding="utf-8") == ""


def test_retrieval_metrics_handle_unknown_truth_and_fail_closed_categories() -> None:
    missing = RetrievalEvalCase(
        id="missing",
        query="no such source",
        category="missing_source",
    )
    result = RetrievalResult(
        query=missing.query,
        warnings=["No indexed source matched requested title 'No source'"],
    )
    metrics = case_metrics(missing, result, top_k=5)
    assert metrics["hit_at_k"] is None
    assert metrics["mrr"] is None
    assert metrics["fail_closed_accuracy"] is True
    assert classify_resolution_failures(result) == ["source_not_found"]
    assert aggregate_metrics([{"metrics": metrics}])["fail_closed_accuracy"] == 1.0
    explicit_missing = RetrievalEvalCase(
        id="explicit-missing",
        query="explain quantifiers",
        category="missing_explicit_source",
        source_title="Definitely Missing Lecture SLM Source XYZ",
    )
    explicit_missing_metrics = case_metrics(explicit_missing, result, top_k=5)
    assert explicit_missing_metrics["fail_closed_accuracy"] is True

    ambiguous = RetrievalEvalCase(
        id="ambiguous",
        query="same title",
        category="ambiguous_source",
    )
    ambiguous_result = RetrievalResult(
        query=ambiguous.query,
        warnings=["Source resolution is ambiguous: note A, note B"],
    )
    assert (
        case_metrics(ambiguous, ambiguous_result, top_k=5)["ambiguity_detection_accuracy"] is True
    )
    assert classify_resolution_failures(ambiguous_result) == ["source_ambiguous"]


def test_review_persistence_and_private_vault_dry_run(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "ignored").mkdir(parents=True)
    write_note(vault / "one.md")
    shutil.copy2(vault / "one.md", vault / "duplicate.md")
    (vault / "unknown.csv").write_text("not indexed", encoding="utf-8")
    (vault / "ignored" / "hidden.md").write_text("excluded", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    report = dry_run_report(
        vault,
        extensions={".md", ".pdf", ".docx", ".pptx", ".txt"},
        ignored_directories={"ignored"},
        recursive=True,
        data_dir=tmp_path / "index",
        embedding_model="fake/test",
        embedding_cache_dir=cache,
    )
    assert report["supported_files"] == 2
    assert report["unsupported_files"] == 1
    assert report["duplicate_candidate_count"] == 1
    assert report["ignored_directories"] == ["ignored"]
    assert not (tmp_path / "index").exists()

    review = RetrievalReview(
        case_id="case",
        mode="hybrid",
        chunk_id="chunk",
        relevance="relevant",
        context_sufficiency="sufficient",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "reviews.jsonl").write_text(review.model_dump_json() + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    update_review_summary(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["human_review"]["review_count"] == 1
    assert summary["human_review"]["label_counts"] == {
        "relevance": {"relevant": 1},
        "context_sufficiency": {"sufficient": 1},
    }
    assert _is_repository_generated_path(
        ROOT / "data/knowledge/evaluation/results/run/results.jsonl"
    )
