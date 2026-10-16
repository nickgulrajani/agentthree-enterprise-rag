"""Proof tests: hybrid retrieval, ACL, grounding, refusal, evaluation."""

from __future__ import annotations

from pathlib import Path

from agentthree_enterprise_rag.agent import KnowledgeAgent
from agentthree_enterprise_rag.evaluation import run_evaluation
from agentthree_enterprise_rag.hybrid_search import HybridRetriever
from agentthree_enterprise_rag.ingest import load_documents

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "policies"
ACL = ROOT / "configs" / "acl_policy.yaml"
FIXTURES = ROOT / "evaluation" / "fixtures" / "rag_cases.yaml"


def _agent() -> KnowledgeAgent:
    return KnowledgeAgent(POLICIES, ACL)


def test_ingest_loads_policy_chunks():
    chunks = load_documents(POLICIES)
    assert len(chunks) >= 4
    ids = {c.doc_id for c in chunks}
    assert "POL-HR-001" in ids
    assert "POL-ENG-002" in ids


def test_hybrid_finds_stipend():
    chunks = load_documents(POLICIES)
    retriever = HybridRetriever()
    retriever.index(chunks)
    hits = retriever.search("monthly home-office stipend remote", top_k=3)
    assert hits
    assert any(h.chunk.doc_id == "POL-HR-001" for h in hits)


def test_alice_gets_grounded_stipend_answer():
    result = _agent().ask(
        "alice_hr",
        "What is the monthly home-office stipend for fully remote employees?",
    )
    assert result.answered
    assert any(c.doc_id == "POL-HR-001" for c in result.citations)
    assert "$75" in result.answer or any("$75" in c.excerpt for c in result.citations)


def test_bob_cannot_read_hr_confidential():
    result = _agent().ask(
        "bob_eng",
        "What is the monthly home-office stipend for fully remote employees?",
    )
    assert not result.answered
    assert not any(c.doc_id == "POL-HR-001" for c in result.citations)


def test_refuses_when_no_evidence():
    result = _agent().ask(
        "alice_hr",
        "What is our pet-friendly office aquarium cleaning schedule?",
    )
    assert not result.answered
    assert "will not invent" in result.answer.lower() or "insufficient" in result.answer.lower()


def test_contractor_can_read_public_code_of_conduct():
    result = _agent().ask(
        "eve_contractor",
        "How do I report harassment or ethics concerns?",
    )
    assert result.answered
    assert any(c.doc_id == "POL-PUB-003" for c in result.citations)


def test_offline_eval_suite_passes():
    report = run_evaluation(_agent(), FIXTURES)
    assert report.ok, [(r.case_id, r.checks) for r in report.results if not r.passed]
