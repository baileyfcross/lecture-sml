import pytest

from lecture_slm.knowledge.query import (
    MAX_RETRIEVAL_EXPANSION_QUERIES,
    build_retrieval_expansion_plan,
    canonicalize_retrieval_query,
)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Can you explain quantum computing?", "quantum computing"),
        ("Could you explain predicate logic?", "predicate logic"),
        ("Please explain DNS.", "DNS"),
        ("Explain predicate logic.", "predicate logic"),
        ("Tell me about reversible computing.", "reversible computing"),
        ("What is quantum annealing?", "quantum annealing"),
        ("What is a finite-state machine?", "a finite-state machine"),
        ("Give me an explanation of Boolean algebra.", "Boolean algebra"),
        ("Create a lecture about DNS.", "DNS"),
        ("Create a lecture on DNS.", "DNS"),
        ("Make slides about predicate logic.", "predicate logic"),
        ("Make slides on section 1.5 deduction.", "section 1.5 deduction"),
        ("Create a lab about Flexbox.", "Flexbox"),
        (
            "Explain why implication is true when the antecedent is false.",
            ("why implication is true when the antecedent is false"),
        ),
        ("Explain P(a) and Q(a).", "P(a) and Q(a)"),
        (
            "Explain post-Von-Neumann computational models.",
            ("post-Von-Neumann computational models"),
        ),
        ("Explainable AI", "Explainable AI"),
        ('Explain "Boolean algebra and logic".', '"Boolean algebra and logic"'),
        ("  Please   explain\n predicate   logic.  ", "predicate logic"),
    ],
)
def test_canonicalize_retrieval_query(instruction: str, expected: str) -> None:
    query = canonicalize_retrieval_query(instruction)

    assert query.original == instruction.strip()
    assert query.canonical == expected


@pytest.mark.parametrize("instruction", ["Explain ."])
def test_unusable_canonical_query_falls_back_to_original(instruction: str) -> None:
    query = canonicalize_retrieval_query(instruction)

    assert query.canonical == instruction.strip()


@pytest.mark.parametrize("instruction", ["", "   "])
def test_empty_instruction_cannot_be_used_as_retrieval_query(instruction: str) -> None:
    with pytest.raises(ValueError, match="empty retrieval instruction"):
        canonicalize_retrieval_query(instruction)


def test_expansion_queries_are_parent_anchored_and_keep_already_anchored_topics() -> None:
    plan = build_retrieval_expansion_plan(
        "quantum computing",
        [
            "general architectures",
            "Quantum algorithms",
            "quantum computing error correction",
        ],
    )

    assert plan.original_query == "quantum computing"
    assert plan.expansion_queries == [
        "quantum computing general architectures",
        "quantum computing algorithms",
        "quantum computing error correction",
    ]


def test_expansion_queries_remove_duplicates_and_ignore_punctuation_only_topics() -> None:
    plan = build_retrieval_expansion_plan(
        "quantum computing",
        ["error correction", "Error Correction", "error correction ", "!!!", "  "],
    )

    assert plan.expansion_queries == ["quantum computing error correction"]
    assert plan.missing_topics == ["error correction"]


def test_expansion_query_that_repeats_the_canonical_query_is_omitted() -> None:
    plan = build_retrieval_expansion_plan(
        "quantum computing",
        ["quantum computing.", "quantum computing error correction"],
    )

    assert plan.expansion_queries == ["quantum computing error correction"]


def test_expansion_query_count_is_bounded_to_three() -> None:
    plan = build_retrieval_expansion_plan(
        "predicate logic",
        ["predicate variables", "universal quantification", "existential quantification", "proofs"],
    )

    assert len(plan.expansion_queries) == MAX_RETRIEVAL_EXPANSION_QUERIES == 3
    assert plan.expansion_queries == [
        "predicate logic variables",
        "predicate logic universal quantification",
        "predicate logic existential quantification",
    ]


def test_empty_canonical_query_cannot_be_expanded() -> None:
    with pytest.raises(ValueError, match="empty canonical retrieval query"):
        build_retrieval_expansion_plan("  ", ["topic"])


def test_punctuation_only_canonical_query_cannot_be_expanded() -> None:
    with pytest.raises(ValueError, match="empty canonical retrieval query"):
        build_retrieval_expansion_plan("!!!", ["topic"])
