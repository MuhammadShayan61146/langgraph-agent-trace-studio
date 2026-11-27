"""Run dossiers: a clean, exportable summary of one run, built from its events.

A dossier is derived *only* from the event stream (so it can be rebuilt from
a saved ``.jsonl`` file) and contains: the final deliverable, the routing
path with reasons, a step timeline, a tool audit trail, a memory audit
trail, fault/recovery history and any human decisions. Render it as JSON,
Markdown or a self-contained HTML page.
"""

from __future__ import annotations

import html
import json
from typing import Any

from agentic_research.telemetry.events import EventType as T
from agentic_research.telemetry.events import preview


def summarize_event(e: dict[str, Any]) -> str:
    """One-line human-readable description of an event."""
    t, d, n = e["type"], e.get("data", {}), e.get("node")
    if t == T.RUN_STARTED:
        return f"Run started: {preview(d.get('query', ''), 120)}"
    if t == T.NODE_STARTED:
        return f"{n} started (attempt {d.get('attempt', 1)})"
    if t == T.NODE_FINISHED:
        return f"{n} finished in {d.get('duration_ms', 0)} ms \u2192 {d.get('task_status')}"
    if t == T.NODE_ERROR:
        retry = "will retry" if d.get("will_retry") else "giving up"
        return f"{n} failed (attempt {d.get('attempt')}, {d.get('error_type')}): {preview(d.get('message', ''), 100)} \u2014 {retry}"
    if t == T.ROUTE:
        return f"{d.get('source')} \u2192 {d.get('target')}: {d.get('reason')}"
    if t == T.LLM_REQUEST:
        return f"{n} called the LLM (round {d.get('round')}, {d.get('message_count')} messages)"
    if t == T.LLM_RESPONSE:
        calls = d.get("tool_calls") or []
        return (f"{n} LLM requested {len(calls)} tool call(s)" if calls
                else f"{n} LLM answered ({len(d.get('content', ''))} chars)")
    if t == T.TOOL_CALL:
        return f"{n} called {d.get('tool')}({preview(d.get('arguments'), 100)})"
    if t == T.TOOL_RESULT:
        return (f"{d.get('tool')} returned ok" if d.get("success")
                else f"{d.get('tool')} failed: {preview(d.get('error'), 100)}")
    if t == T.STATE_SNAPSHOT:
        c = d.get("counts", {})
        return f"State after {n}: {d.get('task_status')} ({c.get('messages')} msgs, {c.get('findings')} findings)"
    if t == T.MEMORY_CHECKPOINT:
        return f"Checkpoint saved after {n} (step {d.get('step')})"
    if t in (T.MEMORY_WRITE, T.MEMORY_READ):
        return d.get("summary") or f"{t} ({d.get('source')})"
    if t == T.FAULT_INJECTED:
        return f"Fault injected [{d.get('kind')}:{d.get('target')}] {d.get('message')} ({d.get('firing')}/{d.get('of')})"
    if t == T.RECOVERY:
        return d.get("message", "Recovered")
    if t == T.HUMAN_INTERRUPT:
        return "Paused at human gate awaiting a decision"
    if t == T.HUMAN_DECISION:
        fb = f" \u2014 \u201c{d.get('feedback')}\u201d" if d.get("feedback") else ""
        return f"Human decision: {d.get('decision')}{fb}"
    if t == T.RUN_PAUSED:
        return "Run paused"
    if t == T.RUN_RESUMED:
        return "Run resumed from checkpoint"
    if t == T.RUN_COMPLETED:
        return f"Run completed in {d.get('steps')} step(s)"
    if t == T.RUN_FAILED:
        return f"Run failed: {preview(d.get('error'), 160)}"
    return t


def build_dossier(events: list[dict[str, Any]], *, include_events: bool = True) -> dict[str, Any]:
    """Build a dossier dict from one run's event dicts (in sequence order)."""
    events = sorted(events, key=lambda e: e["seq"])
    if not events:
        raise ValueError("Cannot build a dossier from zero events.")
    first = events[0]
    d: dict[str, Any] = {
        "run_id": first["run_id"], "group_id": first.get("group_id"),
        "query": "", "status": "running", "duration_ms": events[-1].get("t_ms", 0),
        "steps": 0, "thread_id": None, "final_report": None, "human_approval_status": None,
        "error": None, "meta": {}, "routes": [], "node_runs": [], "tool_audit": [], "llm_calls": 0,
        "memory_audit": [], "faults": [], "human": [], "timeline": [], "final_state": None,
    }
    pending_tools: dict[tuple[str | None, str], dict[str, Any]] = {}
    for e in events:
        t, data, node = e["type"], e.get("data", {}), e.get("node")
        if t != T.STATE_SNAPSHOT:
            d["timeline"].append({"seq": e["seq"], "t_ms": e.get("t_ms", 0), "type": t,
                                  "node": node, "summary": summarize_event(e)})
        if t == T.RUN_STARTED:
            d["query"], d["thread_id"] = data.get("query", ""), data.get("thread_id")
            d["meta"] = data.get("meta") or {}
        elif t == T.RUN_PAUSED:
            d["status"] = "paused"
        elif t == T.RUN_RESUMED:
            d["status"] = "running"
        elif t == T.RUN_COMPLETED:
            d.update(status="completed", final_report=data.get("final_report"),
                     human_approval_status=data.get("human_approval_status"), steps=data.get("steps", 0))
        elif t == T.RUN_FAILED:
            d.update(status="failed", error=data.get("error"), steps=data.get("steps", 0),
                     final_report=data.get("final_report"))
        elif t == T.ROUTE:
            d["routes"].append({"source": data.get("source"), "target": data.get("target"),
                                "kind": data.get("kind"), "reason": data.get("reason")})
        elif t in (T.NODE_FINISHED, T.NODE_ERROR):
            d["node_runs"].append({"node": node, "attempt": data.get("attempt"),
                                   "duration_ms": data.get("duration_ms"),
                                   "outcome": "ok" if t == T.NODE_FINISHED else f"error: {data.get('error_type')}"})
        elif t == T.LLM_REQUEST:
            d["llm_calls"] += 1
        elif t == T.TOOL_CALL:
            pending_tools[(node, data.get("tool"))] = {"node": node, "tool": data.get("tool"),
                                                       "arguments": data.get("arguments")}
        elif t == T.TOOL_RESULT:
            row = pending_tools.pop((node, data.get("tool")),
                                    {"node": node, "tool": data.get("tool"), "arguments": None})
            row.update(success=data.get("success"), output=preview(data.get("output"), 300),
                       error=data.get("error"), duration_ms=data.get("duration_ms"))
            d["tool_audit"].append(row)
        elif t in (T.MEMORY_CHECKPOINT, T.MEMORY_WRITE, T.MEMORY_READ):
            d["memory_audit"].append({"seq": e["seq"], "op": t.split(".")[1],
                                      "source": data.get("source", "checkpoint"), "node": node,
                                      "detail": summarize_event(e)})
        elif t in (T.HUMAN_INTERRUPT, T.HUMAN_DECISION):
            d["human"].append({"type": t, "detail": summarize_event(e),
                               "payload": data.get("payload"), "decision": data.get("decision"),
                               "feedback": data.get("feedback")})
        elif t == T.STATE_SNAPSHOT:
            d["final_state"] = data.get("state")
        if t in (T.FAULT_INJECTED, T.RECOVERY, T.NODE_ERROR):
            d["faults"].append({"type": t, "node": node, "detail": summarize_event(e)})
    if d["steps"] == 0:
        d["steps"] = sum(1 for e in events if e["type"] == T.NODE_FINISHED)
    if include_events:
        d["events"] = events
    return d


def render_json(dossier: dict[str, Any]) -> str:
    return json.dumps(dossier, indent=2, ensure_ascii=False)


def _cell(x: Any) -> str:
    return str(x if x is not None else "\u2014").replace("|", "\\|").replace("\n", " ")


def render_markdown(d: dict[str, Any]) -> str:
    out = [f"# Run dossier \u2014 {d['run_id']}", "",
           f"- **Status:** {d['status']}",
           f"- **Query:** {d['query']}",
           f"- **Steps:** {d['steps']}  \u00b7  **LLM calls:** {d['llm_calls']}  \u00b7  **Duration:** {d['duration_ms']} ms",
           f"- **Human approval:** {d['human_approval_status'] or 'n/a'}"]
    if d.get("error"):
        out.append(f"- **Error:** {d['error']}")
    out += ["", "## Final deliverable", "", d["final_report"] or "_No report was produced._", "",
            "## Routing path (and why)", "", "| # | From | To | Kind | Reason |", "|---|---|---|---|---|"]
    out += [f"| {i} | {_cell(r['source'])} | {_cell(r['target'])} | {_cell(r['kind'])} | {_cell(r['reason'])} |"
            for i, r in enumerate(d["routes"], 1)]
    out += ["", "## Step timeline", "", "| t (ms) | Event | Detail |", "|---|---|---|"]
    out += [f"| {x['t_ms']} | {x['type']} | {_cell(x['summary'])} |" for x in d["timeline"]]
    out += ["", "## Tool audit trail", ""]
    if d["tool_audit"]:
        out += ["| Node | Tool | Arguments | OK | Output / error |", "|---|---|---|---|---|"]
        out += [f"| {_cell(r['node'])} | {_cell(r['tool'])} | {_cell(preview(r['arguments'], 120))} | "
                f"{'yes' if r['success'] else 'no'} | {_cell(r['output'] if r['success'] else r['error'])} |"
                for r in d["tool_audit"]]
    else:
        out.append("_No tools were called._")
    out += ["", "## Memory audit trail", ""]
    out += [f"- {m['detail']}" for m in d["memory_audit"]] or ["_No memory operations._"]
    out += ["", "## Faults & recovery", ""]
    out += [f"- {f['detail']}" for f in d["faults"]] or ["_None._"]
    out += ["", "## Human-in-the-loop", ""]
    out += [f"- {h['detail']}" for h in d["human"]] or ["_Not used._"]
    return "\n".join(out) + "\n"


_CSS = """
:root{--bg:#f6f4ef;--fg:#1d1b17;--mut:#6b665c;--card:#fff;--line:#e3ded2;--acc:#3b5bdb}
@media(prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8e6e1;--mut:#9a968c;--card:#1c1f25;--line:#2a2e36;--acc:#8fa6ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px;letter-spacing:.02em}
.meta{color:var(--mut);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600}pre{white-space:pre-wrap;margin:0;font:inherit}
.pill{display:inline-block;padding:1px 10px;border-radius:99px;border:1px solid var(--line);font-size:12px;color:var(--acc)}
"""


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c if c is not None else chr(8212)))}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def render_html(d: dict[str, Any]) -> str:
    esc = html.escape
    parts = [f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Run dossier {esc(d['run_id'])}</title><style>{_CSS}</style><main>",
             f"<h1>Run dossier</h1><p class='meta'><span class='pill'>{esc(d['status'])}</span> &nbsp;{esc(d['run_id'])} &middot; "
             f"{d['steps']} steps &middot; {d['llm_calls']} LLM calls &middot; {d['duration_ms']} ms</p>",
             f"<div class='card'><strong>Query</strong><br>{esc(d['query'])}</div>"]
    if d.get("error"):
        parts.append(f"<h2>Error</h2><div class='card'>{esc(d['error'])}</div>")
    parts.append(f"<h2>Final deliverable</h2><div class='card'><pre>{esc(d['final_report'] or 'No report was produced.')}</pre></div>")
    parts.append("<h2>Routing path</h2>" + _table(["From", "To", "Kind", "Reason"],
                 [[r["source"], r["target"], r["kind"], r["reason"]] for r in d["routes"]]))
    parts.append("<h2>Step timeline</h2>" + _table(["t (ms)", "Event", "Detail"],
                 [[x["t_ms"], x["type"], x["summary"]] for x in d["timeline"]]))
    parts.append("<h2>Tool audit trail</h2>" + (_table(["Node", "Tool", "Arguments", "OK", "Output / error"],
                 [[r["node"], r["tool"], preview(r["arguments"], 140), "yes" if r["success"] else "no",
                   r["output"] if r["success"] else r["error"]] for r in d["tool_audit"]]) if d["tool_audit"] else "<p class='meta'>No tools were called.</p>"))
    for title, rows in (("Memory audit trail", [m["detail"] for m in d["memory_audit"]]),
                        ("Faults &amp; recovery", [f["detail"] for f in d["faults"]]),
                        ("Human-in-the-loop", [h["detail"] for h in d["human"]])):
        parts.append(f"<h2>{title}</h2>" + ("<ul>" + "".join(f"<li>{esc(r)}</li>" for r in rows) + "</ul>" if rows else "<p class='meta'>None.</p>"))
    parts.append("</main></html>")
    return "".join(parts)
