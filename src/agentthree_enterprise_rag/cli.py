"""CLI for the non-billable Enterprise RAG demo."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import KnowledgeAgent
from .evaluation import run_evaluation

console = Console()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICIES = ROOT / "data" / "policies"
DEFAULT_ACL = ROOT / "configs" / "acl_policy.yaml"
DEFAULT_EVAL = ROOT / "evaluation" / "fixtures" / "rag_cases.yaml"


def _agent() -> KnowledgeAgent:
    return KnowledgeAgent(DEFAULT_POLICIES, DEFAULT_ACL)


def _print_answer(principal_id: str, question: str) -> None:
    agent = _agent()
    principal = agent.describe_principal(principal_id)
    result = agent.ask(principal_id, question)

    header = (
        f"{principal.display_name} · clearance={principal.clearance} · "
        f"dept={principal.department}"
    )
    console.print(Panel(question, title="Question", border_style="cyan"))
    console.print(f"[dim]{header}[/dim]")

    style = "green" if result.answered else "yellow"
    title = "Grounded answer" if result.answered else "Refusal (insufficient evidence)"
    console.print(Panel(result.answer, title=title, border_style=style))

    if result.citations:
        table = Table(title="Citations")
        table.add_column("Doc")
        table.add_column("Section")
        table.add_column("Class")
        table.add_column("Excerpt")
        for c in result.citations:
            table.add_row(c.doc_id, c.section or "—", c.classification, c.excerpt[:80])
        console.print(table)
    elif result.refusal_reason:
        console.print(f"[dim]refusal_reason={result.refusal_reason}[/dim]")


@click.group()
def main() -> None:
    """AgentThree Enterprise RAG — local hybrid retrieval with ACL + citations."""


@main.command("demo")
def demo() -> None:
    """Interview walkthrough: authorized answer, ACL deny path, and refusal."""
    agent = _agent()
    console.print(
        Panel.fit(
            "[bold]AgentThree Enterprise RAG[/bold]\n"
            "Local · zero API cost · hybrid retrieval · metadata ACL · grounded citations",
            border_style="magenta",
        )
    )
    console.print(f"[dim]Indexed {agent.chunk_count} chunks from {DEFAULT_POLICIES}[/dim]\n")

    scenarios = [
        (
            "1 · Authorized HR policy question",
            "alice_hr",
            "What is the monthly home-office stipend for fully remote employees?",
        ),
        (
            "2 · Same question, insufficient clearance (engineering)",
            "bob_eng",
            "What is the monthly home-office stipend for fully remote employees?",
        ),
        (
            "3 · Contractor limited to public docs",
            "eve_contractor",
            "How do I report harassment or ethics concerns?",
        ),
        (
            "4 · Fabrication guard — no evidence in corpus",
            "alice_hr",
            "What is our pet-friendly office aquarium cleaning schedule?",
        ),
    ]

    for title, principal_id, question in scenarios:
        console.rule(title)
        _print_answer(principal_id, question)
        console.print()


@main.command("ask")
@click.option("--principal", "-p", required=True, help="Principal id from acl_policy.yaml")
@click.argument("question")
def ask(principal: str, question: str) -> None:
    """Ask a policy question as a named principal."""
    _print_answer(principal, question)


@main.command("eval")
@click.option(
    "--fixtures",
    type=click.Path(path_type=Path),
    default=DEFAULT_EVAL,
    show_default=True,
)
def eval_cmd(fixtures: Path) -> None:
    """Run offline RAG evaluation fixtures (no LLM APIs)."""
    agent = _agent()
    report = run_evaluation(agent, fixtures)
    table = Table(title=f"RAG eval  score={report.score:.0%}  ({report.passed}/{report.total})")
    table.add_column("Case")
    table.add_column("Pass")
    table.add_column("Checks")
    table.add_column("Detail")
    for r in report.results:
        checks = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in r.checks.items())
        table.add_row(r.case_id, "PASS" if r.passed else "FAIL", checks, r.detail)
    console.print(table)
    if not report.ok:
        raise SystemExit(1)


@main.command("principals")
def principals() -> None:
    """List demo principals and clearances."""
    agent = _agent()
    table = Table(title="Principals")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Dept")
    table.add_column("Clearance")
    table.add_column("Roles")
    for p in agent.acl.list_principals():
        table.add_row(p.principal_id, p.display_name, p.department, p.clearance, ",".join(p.roles))
    console.print(table)


if __name__ == "__main__":
    main()
