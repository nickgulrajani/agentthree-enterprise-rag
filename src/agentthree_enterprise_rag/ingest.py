"""Document ingestion: load markdown policies, chunk, attach metadata."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    classification: str
    allowed_departments: list[str]
    allowed_roles: list[str]
    source: str
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(raw.strip())
    if not match:
        return {}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    return meta, body


def _chunk_by_heading(body: str, max_chars: int = 600) -> list[tuple[str, str]]:
    """Split on ## headings; fall back to paragraph packs under max_chars."""
    parts = re.split(r"(?=^## )", body, flags=re.MULTILINE)
    chunks: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        section = ""
        heading = re.match(r"^##\s+(.+)$", part, re.MULTILINE)
        if heading:
            section = heading.group(1).strip()
        if len(part) <= max_chars:
            chunks.append((section, part))
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", part) if p.strip()]
        buf: list[str] = []
        size = 0
        for para in paras:
            if buf and size + len(para) > max_chars:
                chunks.append((section, "\n\n".join(buf)))
                buf, size = [], 0
            buf.append(para)
            size += len(para)
        if buf:
            chunks.append((section, "\n\n".join(buf)))
    return chunks or [("", body)]


def load_documents(policy_dir: Path) -> list[Chunk]:
    """Ingest all .md files under policy_dir into Chunk objects."""
    chunks: list[Chunk] = []
    paths = sorted(policy_dir.glob("*.md"))
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        doc_id = str(meta.get("doc_id", path.stem))
        title = str(meta.get("title", path.stem))
        classification = str(meta.get("classification", "internal"))
        allowed_departments = list(meta.get("allowed_departments") or [])
        allowed_roles = list(meta.get("allowed_roles") or [])
        source = str(meta.get("source", path.name))
        for i, (section, text) in enumerate(_chunk_by_heading(body)):
            chunk_id = f"{doc_id}#c{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    classification=classification,
                    allowed_departments=allowed_departments,
                    allowed_roles=allowed_roles,
                    source=source,
                    section=section,
                    metadata={
                        "path": str(path.name),
                        "effective_date": meta.get("effective_date"),
                    },
                )
            )
    return chunks
