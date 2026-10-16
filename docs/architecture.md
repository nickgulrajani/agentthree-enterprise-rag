# Architecture — Enterprise RAG & Knowledge Architecture

Non-billable portfolio demo. Everything runs offline: hashing embeddings, in-memory BM25, extractive grounding. No OpenAI / Anthropic / Pinecone / cloud vector DB.

## Control flow

```text
 Policy Markdown          Ingest + Chunk           Hybrid Index
 ┌──────────────┐        ┌──────────────┐        ┌─────────────────┐
 │ frontmatter  │───────►│ section split│───────►│ Hashing vectors │
 │ classification│        │ metadata ACL │        │ BM25 keywords   │
 │ departments  │        └──────────────┘        └────────┬────────┘
 └──────────────┘                                         │
                                                          ▼
 Principal ──► Hybrid retrieve ──► ACL filter ──► Ground or refuse
                  (RRF fuse)       (clearance +      │
                                   dept + roles)     ▼
                                              Citations / refusal
```

## Components

| Capability | Local implementation |
|------------|----------------------|
| Document ingestion | Markdown + YAML frontmatter → section chunks |
| Embeddings | Deterministic hashing embedder (256-d), no model download |
| Vector search | Cosine similarity over in-memory matrix |
| Keyword retrieval | BM25 |
| Hybrid | Reciprocal rank fusion (RRF) |
| Access control | Clearance rank + department + role intersection |
| Grounded answers | Extractive sentences from authorized chunks only |
| Refuse to fabricate | Score + lexical overlap gates |
| RAG evaluation | YAML fixtures: answered, cite, refuse, ACL |

## Production mapping (documented, not deployed)

| Demo piece | Enterprise analogue |
|------------|---------------------|
| Hashing embedder | Managed embedding endpoint / on-prem encoder |
| In-memory index | OpenSearch / Azure AI Search / Vertex Vector Search |
| ACL YAML | IdP groups + document ACLs + row-level security |
| Extractive grounder | Grounded LLM with citation force + abstain policy |
| Offline eval | Continuous RAG eval (faithfulness, context recall, ACL) |

## Design principles

1. **Retrieve then constrain** — ACL filters *after* retrieval so denials are auditable.
2. **Fail closed** — thin evidence ⇒ refusal, never hallucinated policy.
3. **Citations are mandatory** — every answered claim points at `doc_id` + excerpt.
4. **Zero billable surface** — no API keys, no cloud apply, no model downloads required.
