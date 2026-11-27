"""`ScenarioLLM`: a deterministic, *stateless* stand-in for a language model.

It is **not** a real LLM and is labelled as such in the UI. What makes it
useful for demonstrations is that it is *content-driven*: every output is
computed from what is actually in the conversation it receives (tool results,
earlier agents' messages, recalled memories, human revision feedback), and
it makes genuine tool calls (a search, then a calculator call with numbers it
parsed). So retries, memory recall and human feedback visibly change the
result. Because it is stateless, node retries are safe and idempotent.

Presentation lens: put ``Output lens: <name>`` on its own line in the query.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from agentic_research.agents.base import LLMResponse, ToolCallRequest

LENSES = ("Structured Synthesis", "Executive Brief", "Deep Analytical Breakdown", "Risk & Caveat Audit")

_LENS_RE = re.compile(r"^\s*Output lens:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_PCT_RE = re.compile(r"([+-]\d+(?:\.\d+)?)%")
_ITEM_RE = re.compile(r"^- \[(\w+)\] (.+?) \((https?://\S+)\)\s*(?:— (.*))?$", re.MULTILINE)
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "does", "do",
         "what", "how", "why", "reconcile", "summarize", "summarise", "evidence", "competing"}


def split_query(query: str) -> tuple[str, str | None]:
    """Return ``(question, lens)`` from a raw query that may contain a lens line."""
    m = _LENS_RE.search(query)
    lens = m.group(1) if m else None
    question = _LENS_RE.sub("", query).strip()
    return question, lens


def _tool_results(messages: list[dict[str, str]], tool: str) -> list[str]:
    prefix = f"[Tool result for {tool}]: "
    return [m["content"][len(prefix):] for m in messages[1:] if m["content"].startswith(prefix)]


def _first(messages: list[dict[str, str]], marker: str) -> str | None:
    for m in messages[1:]:
        if marker in m["content"]:
            return m["content"]
    return None


def _fmt(x: float) -> str:
    return f"{x:+.1f}" if abs(x - round(x)) > 1e-9 else f"{x:+.0f}"


class ScenarioLLM:
    """Implements the project's `LLMClient` protocol."""

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if "Researcher agent" in system:
            return self._researcher(messages)
        if "Analyst agent" in system:
            return self._analyst(messages)
        if "Writer agent" in system:
            return self._writer(messages)
        return LLMResponse(content="(unrecognised role)")

    # ---------------------------------------------------------- researcher
    def _researcher(self, messages: list[dict[str, str]]) -> LLMResponse:
        question, _ = split_query(messages[0]["content"])
        results = _tool_results(messages, "web_search")
        failures = [r for r in results if r.startswith("Error")]
        ok = [r for r in results if not r.startswith("Error")]
        if not ok:
            if len(failures) >= 2:
                return LLMResponse(content="Researcher: search is unavailable; no sources retrieved.")
            query = question
            if failures:  # graceful degradation: retry with a broader, shorter query
                words = [w for w in re.findall(r"[A-Za-z0-9]+", question) if w.lower() not in _STOP]
                query = " ".join(words[:4])
            return LLMResponse(
                content=f"I'll search for sources on: {query}",
                tool_calls=[ToolCallRequest("web_search", {"query": query, "max_results": 5},
                                            call_id=f"search-{len(failures) + 1}")],
                stop_reason="tool_use",
            )
        try:
            items = ast.literal_eval(ok[-1])
        except (ValueError, SyntaxError):
            items = []
        lines = []
        for it in items:
            m = re.match(r"\[(\w+)\]\s*(.*)", it["snippet"])
            stance, text = (m.group(1), m.group(2)) if m else ("context", it["snippet"])
            lines.append(f"- [{stance}] {it['title']} ({it['url']}) — {text}")
        note = ""
        if failures:
            note = f" (recovered after {len(failures)} failed search attempt(s) using a broader query)"
        if not lines:
            return LLMResponse(content="Researcher: the search returned no sources.")
        return LLMResponse(content=f"Researcher findings ({len(lines)} sources){note}:\n" + "\n".join(lines))

    # ------------------------------------------------------------- analyst
    def _evidence(self, messages: list[dict[str, str]]) -> list[tuple[str, str, str, str]]:
        block = _first(messages, "Researcher findings") or ""
        return [(m.group(1), m.group(2), m.group(3), m.group(4) or "") for m in _ITEM_RE.finditer(block)]

    def _analyst(self, messages: list[dict[str, str]]) -> LLMResponse:
        evidence = self._evidence(messages)
        nums = [float(n) for _, _, _, text in evidence for n in _PCT_RE.findall(text)]
        calc = _tool_results(messages, "calculator")
        if len(nums) >= 2 and not calc:
            mean_expr = "(" + " + ".join(f"({n:g})" for n in nums) + f") / {len(nums)}"
            spread_expr = f"({max(nums):g}) - ({min(nums):g})"
            return LLMResponse(
                content="Quantifying the effects reported across sources.",
                tool_calls=[
                    ToolCallRequest("calculator", {"expression": mean_expr}, "calc-mean"),
                    ToolCallRequest("calculator", {"expression": spread_expr}, "calc-spread"),
                ],
                stop_reason="tool_use",
            )
        stances = [s for s, *_ in evidence]
        sup, dis, ctx = (stances.count(x) for x in ("supports", "disputes", "context"))
        recalled = _first(messages, "Recalled from long-term memory")
        parts = [f"Analyst assessment: {sup} supporting, {dis} disputing, {ctx} contextual source(s)."]
        good = [c for c in calc if not c.startswith("Error")]
        if len(good) >= 2:
            parts.append(f"Mean reported effect {_fmt(float(good[0]))}%; spread between extremes {float(good[1]):.0f} points.")
        elif calc:
            parts.append("Quantitative synthesis unavailable (calculator error); relying on qualitative balance.")
        if recalled:
            parts.append("Earlier-run findings recalled from memory were weighed alongside the new evidence.")
        if sup and dis:
            parts.append("Verdict: evidence is genuinely split; the conflict is driven by context, not noise.")
        elif sup or dis:
            parts.append("Verdict: evidence points one way, with limited counter-evidence.")
        else:
            parts.append("Verdict: insufficient evidence.")
        return LLMResponse(content=" ".join(parts))

    # -------------------------------------------------------------- writer
    def _writer(self, messages: list[dict[str, str]]) -> LLMResponse:
        question, lens = split_query(messages[0]["content"])
        lens = lens or "Structured Synthesis"
        evidence = self._evidence(messages)
        analysis = next((m["content"] for m in messages[1:] if m["content"].startswith("Analyst assessment")), "")
        revisions = [m["content"].split(":", 1)[1].strip() for m in messages[1:]
                     if m["content"].startswith("Revision requested by human reviewer:")]
        recalled = _first(messages, "Recalled from long-term memory")
        n_recalled = recalled.count("\n- ") if recalled else 0
        prior = _first(messages, "Previous run's final report")
        prior_conclusion = ""
        if prior:
            m = re.search(r"## Reconciliation\n(.+)", prior)
            prior_conclusion = (m.group(1).strip().split(". ")[0] if m else "").rstrip(".")
        by = {k: [e for e in evidence if e[0] == k] for k in ("supports", "disputes", "context")}
        mean = re.search(r"Mean reported effect ([+-][\d.]+)%", analysis)
        spread = re.search(r"spread between extremes (\d+) points", analysis)

        def bullets(rows: list[tuple[str, str, str, str]]) -> str:
            return "\n".join(f"- {t.replace('Fixture: ', '')}: {x} ({u})" for _, t, u, x in rows) or "- None found."

        head = f"# {lens}: {question}\n"
        if revisions:
            head += f"\n_Revision {len(revisions)} — addressing reviewer feedback: “{revisions[-1]}”_\n"
        cont = ""
        if n_recalled:
            cont = (f"\n## Continuity\nBuilt on {n_recalled} finding(s) recalled from an earlier run, "
                    f"combined with {len(evidence)} newly retrieved source(s).\n")
            if prior_conclusion:
                cont += f"\nPrior conclusion carried forward (from checkpointed report): {prior_conclusion}.\n"
        quant = ""
        if mean and spread:
            quant = f"Mean reported effect {mean.group(1)}%, spread of {spread.group(1)} points between extremes."

        if lens == "Executive Brief":
            lead = by["supports"][:1] + by["disputes"][:1]
            body = (f"\n**Bottom line:** {self._verdict(by)}\n\n## Key points\n"
                    + "\n".join(f"- {t.replace('Fixture: ', '')}" for _, t, _, _ in lead)
                    + (f"\n- {quant}" if quant else "")
                    + "\n\n## Recommended action\nProceed with a scoped pilot and track the disputed metrics explicitly.\n")
        elif lens == "Deep Analytical Breakdown":
            rows = "\n".join(f"| {s} | {t.replace('Fixture: ', '')} | {', '.join(_PCT_RE.findall(x)) or '—'} |"
                             for s, t, _, x in evidence)
            body = ("\n## Method\nEvidence retrieved by the Researcher, quantified by the Analyst with the calculator tool.\n"
                    f"\n## Evidence table\n| Stance | Source | Reported effect (%) |\n|---|---|---|\n{rows}\n"
                    f"\n## Quantitative synthesis\n{quant or 'Not enough numeric evidence to quantify.'}\n"
                    f"\n## Interpretation\n{analysis}\n"
                    "\n## Limitations\nFixture data; small samples; heterogeneous measures.\n")
        elif lens == "Risk & Caveat Audit":
            risks = [(t.replace("Fixture: ", ""), x) for _, t, _, x in by["disputes"]] or [("No disputing evidence found", "")]
            table = "\n".join(f"| {i + 1} | {t} | {'High' if i == 0 else 'Medium'} | {x} |" for i, (t, x) in enumerate(risks))
            caveats = "\n".join(f"- {t.replace('Fixture: ', '')}: {x}" for _, t, _, x in by["context"]) or "- None recorded."
            body = (f"\n## Risk register\n| # | Risk | Severity | Evidence |\n|---|---|---|---|\n{table}\n"
                    f"\n## Caveats\n{caveats}\n"
                    "\n## What would change this conclusion\nLarger, independently measured samples in the disputed segments.\n")
        else:  # Structured Synthesis
            body = (f"\n## Position A — supporting evidence\n{bullets(by['supports'])}\n"
                    f"\n## Position B — disputing evidence\n{bullets(by['disputes'])}\n"
                    f"\n## Context and caveats\n{bullets(by['context'])}\n"
                    f"\n## Reconciliation\n{self._verdict(by)} {quant}\n"
                    f"\n## Confidence\n{'Moderate' if by['supports'] and by['disputes'] else 'Low-to-moderate'} "
                    f"— {len(evidence)} source(s), fixture-grade.\n")
        sources = "\n".join(f"- {u}" for _, _, u, _ in evidence)
        return LLMResponse(content=f"{head}{cont}{body}\n## Sources\n{sources}\n".strip())

    @staticmethod
    def _verdict(by: dict[str, list]) -> str:
        if by["supports"] and by["disputes"]:
            return ("The sources disagree because they measure different things in different settings; "
                    "both positive and negative effects are real and context-dependent.")
        if by["supports"]:
            return "The available sources lean supportive, with little counter-evidence."
        if by["disputes"]:
            return "The available sources lean sceptical."
        return "There is not enough evidence to conclude."
