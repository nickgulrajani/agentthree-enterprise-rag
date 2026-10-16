"""Grounded answers with citations — extractive, no LLM API, refuse when thin."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embeddings import tokenize
from .vector_store import ScoredChunk

# Minimum evidence to answer (hybrid RRF scores are typically small)
MIN_TOP_SCORE = 0.015
MIN_EVIDENCE_CHUNKS = 1
# Distinctive query tokens that must appear in evidence (coverage gate)
MIN_COVERAGE = 0.45
MIN_ABSOLUTE_OVERLAP = 2


REFUSAL_TEMPLATE = (
    "I do not have sufficient authorized evidence to answer that question. "
    "I will not invent policy details. Please rephrase, escalate to a "
    "principal with higher clearance, or consult the owning policy owner."
)


@dataclass
class Citation:
    doc_id: str
    title: str
    section: str
    chunk_id: str
    classification: str
    source: str
    excerpt: str


@dataclass
class GroundedAnswer:
    question: str
    answered: bool
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refusal_reason: str = ""
    evidence_count: int = 0
    top_score: float = 0.0


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are",
        "be", "by", "with", "what", "how", "when", "who", "does", "do", "can",
        "may", "our", "we", "i", "my", "me", "about", "please", "must", "should",
        "quickly", "from", "into", "that", "this", "have", "has",
    }
    return {t for t in tokenize(text) if t not in stop and len(t) > 2}


def _best_sentence(text: str, query_tokens: set[str]) -> str:
    # Drop markdown headings from excerpt candidates
    cleaned = re.sub(r"^#+\s*.+$", "", text, flags=re.MULTILINE).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        sentences = [cleaned or text.strip()]
    best, best_hit = sentences[0], -1
    for sent in sentences:
        hits = len(query_tokens & set(tokenize(sent)))
        if hits > best_hit:
            best, best_hit = sent, hits
    return best.strip()


def _evidence_overlap(question_tokens: set[str], evidence: list[ScoredChunk]) -> set[str]:
    corpus_tokens: set[str] = set()
    for item in evidence:
        corpus_tokens |= _content_tokens(item.chunk.text)
    return question_tokens & corpus_tokens


def ground_answer(
    question: str,
    evidence: list[ScoredChunk],
    max_citations: int = 3,
) -> GroundedAnswer:
    """Build an extractive grounded answer or refuse if evidence is insufficient."""
    top_score = evidence[0].score if evidence else 0.0
    q_tokens = _content_tokens(question)

    if len(evidence) < MIN_EVIDENCE_CHUNKS or top_score < MIN_TOP_SCORE:
        return GroundedAnswer(
            question=question,
            answered=False,
            answer=REFUSAL_TEMPLATE,
            refusal_reason="insufficient_retrieval_score",
            evidence_count=len(evidence),
            top_score=top_score,
        )

    # Coverage gate: enough distinctive query tokens must appear in evidence.
    # Prevents answering unrelated questions from weakly scored / colliding chunks.
    overlap = _evidence_overlap(q_tokens, evidence)
    coverage = len(overlap) / max(len(q_tokens), 1)
    if len(overlap) < MIN_ABSOLUTE_OVERLAP or coverage < MIN_COVERAGE:
        return GroundedAnswer(
            question=question,
            answered=False,
            answer=REFUSAL_TEMPLATE,
            refusal_reason="no_lexical_grounding",
            evidence_count=len(evidence),
            top_score=top_score,
        )

    # Prefer chunks that actually overlap the query for citations
    ranked = sorted(
        evidence,
        key=lambda s: len(q_tokens & _content_tokens(s.chunk.text)),
        reverse=True,
    )
    usable = [s for s in ranked if q_tokens & _content_tokens(s.chunk.text)]
    if not usable:
        return GroundedAnswer(
            question=question,
            answered=False,
            answer=REFUSAL_TEMPLATE,
            refusal_reason="no_lexical_grounding",
            evidence_count=len(evidence),
            top_score=top_score,
        )

    citations: list[Citation] = []
    bullets: list[str] = []
    for item in usable[:max_citations]:
        c = item.chunk
        excerpt = _best_sentence(c.text, q_tokens)
        citations.append(
            Citation(
                doc_id=c.doc_id,
                title=c.title,
                section=c.section,
                chunk_id=c.chunk_id,
                classification=c.classification,
                source=c.source,
                excerpt=excerpt,
            )
        )
        label = f"[{c.doc_id}" + (f" · {c.section}" if c.section else "") + "]"
        bullets.append(f"- {label} {excerpt}")

    answer = (
        "Based on authorized documents only:\n"
        + "\n".join(bullets)
        + "\n\nAll claims above are grounded in the cited excerpts."
    )
    return GroundedAnswer(
        question=question,
        answered=True,
        answer=answer,
        citations=citations,
        evidence_count=len(evidence),
        top_score=top_score,
    )
