"""Metadata-based access controls for retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .ingest import Chunk
from .vector_store import ScoredChunk


@dataclass
class Principal:
    principal_id: str
    display_name: str
    department: str
    clearance: str
    roles: list[str]


class AccessController:
    def __init__(self, policy_path: Path) -> None:
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        self.clearance_rank: dict[str, int] = dict(raw.get("clearance_rank") or {})
        self._principals: dict[str, Principal] = {}
        for pid, meta in (raw.get("principals") or {}).items():
            self._principals[pid] = Principal(
                principal_id=pid,
                display_name=str(meta.get("display_name", pid)),
                department=str(meta.get("department", "")),
                clearance=str(meta.get("clearance", "public")),
                roles=list(meta.get("roles") or []),
            )
        self.defaults: dict[str, Any] = dict(raw.get("defaults") or {})

    def get_principal(self, principal_id: str) -> Principal:
        if principal_id not in self._principals:
            raise KeyError(f"Unknown principal: {principal_id}")
        return self._principals[principal_id]

    def list_principals(self) -> list[Principal]:
        return list(self._principals.values())

    def _rank(self, level: str) -> int:
        return self.clearance_rank.get(level, -1)

    def may_read(self, principal: Principal, chunk: Chunk) -> bool:
        if self.defaults.get("require_clearance_at_least_doc_classification", True):
            if self._rank(principal.clearance) < self._rank(chunk.classification):
                return False
        if chunk.allowed_departments and principal.department not in chunk.allowed_departments:
            return False
        if chunk.allowed_roles:
            if not set(principal.roles) & set(chunk.allowed_roles):
                return False
        return True

    def filter(self, principal: Principal, scored: list[ScoredChunk]) -> list[ScoredChunk]:
        return [s for s in scored if self.may_read(principal, s.chunk)]
