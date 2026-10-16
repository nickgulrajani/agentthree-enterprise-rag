"""Enterprise knowledge agent: ingest → hybrid retrieve → ACL → ground."""

from __future__ import annotations

from pathlib import Path

from .acl import AccessController, Principal
from .grounding import GroundedAnswer, ground_answer
from .hybrid_search import HybridRetriever
from .ingest import load_documents


class KnowledgeAgent:
    def __init__(
        self,
        policy_dir: Path,
        acl_path: Path,
        top_k: int = 5,
    ) -> None:
        self.acl = AccessController(acl_path)
        self.retriever = HybridRetriever()
        self.top_k = top_k
        chunks = load_documents(policy_dir)
        self.retriever.index(chunks)
        self.chunk_count = len(chunks)

    def ask(self, principal_id: str, question: str) -> GroundedAnswer:
        principal = self.acl.get_principal(principal_id)
        raw = self.retriever.search(question, top_k=self.top_k * 2)
        allowed = self.acl.filter(principal, raw)[: self.top_k]
        return ground_answer(question, allowed)

    def describe_principal(self, principal_id: str) -> Principal:
        return self.acl.get_principal(principal_id)
