"""Verifier: score recomputation, threshold, retries, and refined queries."""

from __future__ import annotations

import pytest

from research_system.adapters.llm import FakeLLMClient
from research_system.agents.verifier import (
    NO_RESPONSE_ERROR,
    compute_faithfulness,
    reconcile_claims,
    refine_queries,
    route_after_verification,
    verifier_node,
)
from research_system.core.state import RetrievedDocument, create_initial_state
from research_system.errors import LLMError
from research_system.schemas import RefinedQueries, VerifierOutput


def doc(content: str = "evidence") -> RetrievedDocument:
    return RetrievedDocument(content=content, source="a.pdf", score=0.5)


def verified_state(response: str = "The answer.", retry_count: int = 0):
    state = create_initial_state("q")
    state["response"] = response
    state["retrieved_documents"] = [doc()]
    state["retry_count"] = retry_count
    return state


# --- score computation -----------------------------------------------------
def test_faithfulness_is_supported_over_total():
    assert compute_faithfulness(["a", "b", "c", "d"], ["a", "b", "c"]) == pytest.approx(0.75)


def test_no_claims_scores_one_only_for_an_explicit_no_answer():
    assert compute_faithfulness([], [], no_answer=True) == 1.0
    assert compute_faithfulness([], [], no_answer=False) == 0.0


def test_reported_score_is_overridden_by_the_claim_counts(deps):
    deps.llm = FakeLLMClient(
        [
            VerifierOutput(
                claims=["a", "b", "c", "d"],
                supported_claims=["a"],
                unsupported_claims=["b", "c", "d"],
                faithfulness_score=0.99,  # a lie
                reasoning="r",
            ),
            RefinedQueries(queries=["refined b", "refined c"]),
        ]
    )

    update = verifier_node(verified_state(), deps)

    assert update["verification"].faithfulness_score == pytest.approx(0.25)
    assert update["is_verified"] is False
    assert any("using the computed value" in w for w in update["warnings"])


# --- claim reconciliation --------------------------------------------------
def test_claims_classified_but_not_listed_are_absorbed():
    output = VerifierOutput(claims=["a"], supported_claims=["a", "b"], unsupported_claims=[])
    claims, supported, unsupported = reconcile_claims(output)

    assert set(claims) == {"a", "b"}
    assert set(supported) == {"a", "b"}
    assert unsupported == []


def test_unclassified_claims_count_as_unsupported():
    output = VerifierOutput(claims=["a", "b"], supported_claims=["a"], unsupported_claims=[])
    _, supported, unsupported = reconcile_claims(output)

    assert supported == ["a"]
    assert unsupported == ["b"]


def test_a_claim_in_both_lists_counts_as_unsupported():
    output = VerifierOutput(claims=["a"], supported_claims=["a"], unsupported_claims=["a"])
    _, supported, unsupported = reconcile_claims(output)

    assert supported == []
    assert unsupported == ["a"]


# --- threshold -------------------------------------------------------------
def test_score_exactly_at_the_threshold_passes(deps):
    deps.settings.faithfulness_threshold = 0.75
    deps.llm = FakeLLMClient(
        [
            VerifierOutput(
                claims=["a", "b", "c", "d"],
                supported_claims=["a", "b", "c"],
                unsupported_claims=["d"],
            )
        ]
    )

    update = verifier_node(verified_state(), deps)

    assert update["verification"].faithfulness_score == pytest.approx(0.75)
    assert update["is_verified"] is True
    assert update["retry_count"] == 0


# --- retries ---------------------------------------------------------------
def test_failed_verification_increments_retry_count_once(deps):
    deps.llm = FakeLLMClient(
        [
            VerifierOutput(claims=["a", "b"], supported_claims=[], unsupported_claims=["a", "b"]),
            RefinedQueries(queries=["q1"]),
        ]
    )

    update = verifier_node(verified_state(retry_count=0), deps)

    assert update["retry_count"] == 1
    assert update["is_verified"] is False


def test_unsupported_claims_become_refined_sub_queries(deps):
    deps.llm = FakeLLMClient(
        [
            VerifierOutput(
                claims=["x is true"], supported_claims=[], unsupported_claims=["x is true"]
            ),
            RefinedQueries(queries=["evidence that x is true"]),
        ]
    )

    update = verifier_node(verified_state(), deps)

    assert update["sub_queries"] == ["evidence that x is true"]
    assert update["unsupported_claims"] == ["x is true"]


def test_prior_documents_are_carried_forward_for_the_retry(deps):
    deps.llm = FakeLLMClient(
        [
            VerifierOutput(claims=["x"], supported_claims=[], unsupported_claims=["x"]),
            RefinedQueries(queries=["refined"]),
        ]
    )
    state = verified_state()

    update = verifier_node(state, deps)

    assert update["prior_documents"] == state["retrieved_documents"]


def test_refinement_failure_falls_back_to_the_claim_text(deps):
    deps.llm = FakeLLMClient([LLMError("refiner down")])
    queries, warnings = refine_queries(["the sky is green"], "q", deps)

    assert queries == ["the sky is green"]
    assert any("refinement failed" in w for w in warnings)


def test_time_sensitive_claims_flag_web_retrieval(deps):
    deps.llm = FakeLLMClient(
        [
            VerifierOutput(
                claims=["c"], supported_claims=[], unsupported_claims=["the current CEO is X"]
            ),
            RefinedQueries(queries=["who is the CEO"]),
        ]
    )

    update = verifier_node(verified_state(), deps)

    assert any("time-sensitive" in w for w in update["warnings"])


def test_retry_limit_stops_refining_and_warns(deps):
    deps.settings.max_verification_retries = 2
    deps.llm = FakeLLMClient(
        [VerifierOutput(claims=["a"], supported_claims=[], unsupported_claims=["a"])]
    )

    update = verifier_node(verified_state(retry_count=2), deps)

    assert update["retry_count"] == 3
    assert "sub_queries" not in update
    assert any("remains unverified" in w for w in update["warnings"])


# --- safety ----------------------------------------------------------------
def test_empty_response_ends_safely(deps):
    deps.llm = FakeLLMClient([])  # must not be called

    update = verifier_node(verified_state(response=""), deps)

    assert update["error"] == NO_RESPONSE_ERROR
    assert update["is_verified"] is False
    assert deps.llm.calls == []


def test_verifier_outage_is_not_reported_as_verified(deps):
    deps.llm = FakeLLMClient([LLMError("verifier down")])

    update = verifier_node(verified_state(), deps)

    assert update["is_verified"] is False
    assert update["verification"] is None
    assert any("could not run" in w for w in update["warnings"])


# --- routing ---------------------------------------------------------------
def test_route_respond_when_verified(deps):
    state = create_initial_state("q")
    state["is_verified"] = True
    assert route_after_verification(state, deps.settings) == "respond"


def test_route_retrieve_when_unverified_with_budget(deps):
    state = create_initial_state("q")
    state["retry_count"] = 1
    state["unsupported_claims"] = ["a"]
    assert route_after_verification(state, deps.settings) == "retrieve"


def test_route_respond_at_the_retry_limit(deps):
    state = create_initial_state("q")
    state["retry_count"] = deps.settings.max_verification_retries
    state["unsupported_claims"] = ["a"]
    assert route_after_verification(state, deps.settings) == "respond"


def test_route_respond_when_there_is_nothing_to_search_for(deps):
    state = create_initial_state("q")
    state["unsupported_claims"] = []
    assert route_after_verification(state, deps.settings) == "respond"


def test_route_respond_when_there_was_no_response(deps):
    state = create_initial_state("q")
    state["error"] = NO_RESPONSE_ERROR
    assert route_after_verification(state, deps.settings) == "respond"
