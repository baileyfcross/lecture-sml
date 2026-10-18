import pytest

from lecture_slm.knowledge.query import (
    MAX_RETRIEVAL_EXPANSION_QUERIES,
    build_factual_retrieval_query,
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
        ("Write out an introduction to quantum computing", "quantum computing"),
        ("Write an introduction to quantum computing", "quantum computing"),
        ("Create an introduction to quantum computing", "quantum computing"),
        ("Give me an introduction to quantum computing", "quantum computing"),
        ("Create a lecture introducing quantum computing", "quantum computing"),
        ("Teach me about quantum computing", "quantum computing"),
        ("Explain Introduction to Quantum Computing", "Introduction to Quantum Computing"),
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


def test_expansion_queries_prioritize_core_topics_over_optional_and_noncore_gaps() -> None:
    plan = build_retrieval_expansion_plan(
        "quantum computing",
        [
            "quantum error correction",
            "hardware overview",
            "qubits and superposition",
            "measurement and entanglement",
            "gates and circuit models",
        ],
        core_topics=[
            "qubits and superposition",
            "measurement and entanglement",
            "gates and circuit models",
        ],
    )

    assert plan.missing_topics == [
        "qubits superposition",
        "measurement entanglement",
        "gates circuit models",
        "quantum error correction",
        "hardware overview",
    ]
    assert plan.expansion_queries == [
        "quantum computing qubits superposition",
        "quantum computing measurement entanglement",
        "quantum computing gates circuit models",
    ]
    assert len(plan.expansion_queries) == MAX_RETRIEVAL_EXPANSION_QUERIES


def test_empty_canonical_query_cannot_be_expanded() -> None:
    with pytest.raises(ValueError, match="empty canonical retrieval query"):
        build_retrieval_expansion_plan("  ", ["topic"])


def test_punctuation_only_canonical_query_cannot_be_expanded() -> None:
    with pytest.raises(ValueError, match="empty canonical retrieval query"):
        build_retrieval_expansion_plan("!!!", ["topic"])


def test_course_relative_generation_request_uses_concise_workspace_anchor() -> None:
    query, diagnostics = build_factual_retrieval_query(
        "Can you generate lecture 4 for the C# class?",
        workspace_name="C# Class",
        history_titles=["Lecture 3\n# C#.NET Fundamentals: Lecture 3 – Generic Methods"],
    )

    assert query.original == "Can you generate lecture 4 for the C# class?"
    assert query.canonical == "C# Generic Methods"
    assert diagnostics["query_terms"] == ["c#", "generic", "methods"]
    assert diagnostics["workspace_anchor"] == {
        "considered": True,
        "applied": True,
        "value": "C#",
        "reason": "Request depends on the active course sequence.",
    }


def test_self_contained_factual_request_does_not_use_unrelated_workspace_anchor() -> None:
    query, diagnostics = build_factual_retrieval_query(
        "Can you explain quantum computing for me?",
        workspace_name="Java 112",
    )

    assert query.canonical == "quantum computing"
    assert diagnostics["workspace_anchor"] == {
        "considered": True,
        "applied": False,
        "value": None,
        "reason": "Request contains a self-contained factual topic.",
    }


def test_workspace_overlap_preserves_self_contained_quantum_topics() -> None:
    query, diagnostics = build_factual_retrieval_query(
        "Write out an introduction to quantum computing",
        workspace_name="Quantum",
    )

    assert query.canonical == "quantum computing"
    assert diagnostics["raw_instruction"] == "Write out an introduction to quantum computing"
    assert diagnostics["canonical_factual_query"] == "quantum computing"
    assert diagnostics["query_terms"] == ["quantum", "computing"]
    assert diagnostics["workspace_history_topics_used"] == []
    assert diagnostics["workspace_anchor"] == {
        "considered": True,
        "applied": False,
        "value": None,
        "reason": "Request contains a self-contained factual topic.",
    }


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("Explain quantum computing", "quantum computing"),
        ("Write an introduction to quantum gates", "quantum gates"),
        ("Create a lecture introducing quantum computing", "quantum computing"),
    ],
)
def test_workspace_name_does_not_remove_self_contained_topic(
    instruction: str,
    expected: str,
) -> None:
    query, diagnostics = build_factual_retrieval_query(
        instruction,
        workspace_name="Quantum",
    )

    assert query.canonical == expected
    assert diagnostics["workspace_anchor"]["applied"] is False
    assert diagnostics["workspace_anchor"]["reason"] == (
        "Request contains a self-contained factual topic."
    )


def test_ambiguous_single_topic_still_uses_workspace_anchor() -> None:
    query, diagnostics = build_factual_retrieval_query(
        "Explain methods",
        workspace_name="C# Course",
    )

    assert query.canonical == "C# methods"
    assert diagnostics["workspace_anchor"] == {
        "considered": True,
        "applied": True,
        "value": "C#",
        "reason": "Request topic is ambiguous without the active course domain.",
    }


def test_referential_request_uses_course_domain_and_recent_history_title() -> None:
    query, diagnostics = build_factual_retrieval_query(
        "Create the next lecture.",
        workspace_name="C# Class",
        history_titles=["Lecture 3\n# C#.NET Fundamentals: Lecture 3 – Generic Methods"],
    )

    assert query.canonical == "C# Generic Methods"
    assert diagnostics["workspace_anchor"]["applied"] is True


def test_course_relative_request_retains_csc_workspace_history_fallback() -> None:
    query, diagnostics = build_factual_retrieval_query(
        "Create the next lecture",
        workspace_name="CSC 220",
        history_titles=["Lecture 3\n# CSC 220: Lecture 3 – Quantum gates"],
    )

    assert query.canonical == "CSC 220 Quantum gates"
    assert diagnostics["workspace_history_topics_used"] == ["Quantum gates"]
    assert diagnostics["workspace_anchor"]["applied"] is True


def test_expansion_queries_use_normalized_topics_and_domain_anchor() -> None:
    plan = build_retrieval_expansion_plan(
        "C# Generic Methods",
        ["Generic classes syntax and usage"],
        domain_anchor="C#",
    )

    assert plan.missing_topics == ["generic classes syntax usage"]
    assert plan.domain_anchor == "C#"
    assert plan.expansion_queries == [
        "C# generic classes",
        "C# generic classes syntax",
        "C# generic classes usage",
    ]
    assert all("lecture" not in query.casefold() for query in plan.expansion_queries)


def test_simple_gap_does_not_force_three_expansion_queries() -> None:
    plan = build_retrieval_expansion_plan(
        "C# Generic Methods",
        ["generic classes"],
        domain_anchor="C#",
    )

    assert plan.expansion_queries == ["C# generic classes"]


def test_multiple_gaps_share_the_global_expansion_query_cap() -> None:
    plan = build_retrieval_expansion_plan(
        "C# Generic Methods",
        [
            "generic classes syntax and usage",
            "generic constraints and inheritance",
            "type parameters",
        ],
        domain_anchor="C#",
    )

    assert 1 <= len(plan.expansion_queries) <= MAX_RETRIEVAL_EXPANSION_QUERIES == 3
    assert len(set(query.casefold() for query in plan.expansion_queries)) == len(
        plan.expansion_queries
    )
    assert all(query.startswith("C# ") for query in plan.expansion_queries)
