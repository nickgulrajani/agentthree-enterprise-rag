"""Offline RAG evaluation — faithfulness, citation presence, ACL correctness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .agent import KnowledgeAgent
from .grounding import GroundedAnswer


@dataclass
class EvalCaseResult:
    case_id: str
    passed: bool
    checks: dict[str, bool]
    detail: str = ""


@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    score: float
    results: list[EvalCaseResult]

    @property
    def ok(self) -> bool:
        return self.failed == 0


def _citation_doc_ids(answer: GroundedAnswer) -> set[str]:
    return {c.doc_id for c in answer.citations}


def evaluate_case(agent: KnowledgeAgent, case: dict[str, Any]) -> EvalCaseResult:
    principal_id = case["principal"]
    question = case["question"]
    expect = case.get("expect", {})
    result = agent.ask(principal_id, question)

    checks: dict[str, bool] = {}

    if "answered" in expect:
        checks["answered"] = result.answered is bool(expect["answered"])

    if expect.get("must_refuse"):
        checks["must_refuse"] = not result.answered

    if "must_cite_docs" in expect:
        required = set(expect["must_cite_docs"])
        checks["must_cite_docs"] = required.issubset(_citation_doc_ids(result))

    if "must_not_cite_docs" in expect:
        forbidden = set(expect["must_not_cite_docs"])
        checks["must_not_cite_docs"] = not (
            forbidden & _citation_doc_ids(result)
        )

    if expect.get("citations_required") and result.answered:
        checks["citations_required"] = len(result.citations) > 0

    if expect.get("grounded_only") and result.answered:
        # Faithfulness proxy: every citation excerpt must appear in corpus chunks
        # (already true by construction); assert answer claims citation markers.
        checks["grounded_only"] = all(
            c.doc_id in result.answer for c in result.citations
        )

    passed = all(checks.values()) if checks else False
    detail = f"answered={result.answered} citations={len(result.citations)}"
    return EvalCaseResult(
        case_id=str(case.get("id", question[:40])),
        passed=passed,
        checks=checks,
        detail=detail,
    )


def run_evaluation(agent: KnowledgeAgent, fixtures_path: Path) -> EvalReport:
    data = yaml.safe_load(fixtures_path.read_text(encoding="utf-8"))
    cases = list(data.get("cases") or [])
    results = [evaluate_case(agent, c) for c in cases]
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    failed = total - passed
    score = passed / total if total else 0.0
    return EvalReport(
        total=total, passed=passed, failed=failed, score=score, results=results
    )
