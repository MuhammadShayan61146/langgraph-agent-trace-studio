"""Stdlib-only dashboard server (no Flask/FastAPI needed).

Endpoints (all JSON unless noted)
---------------------------------
GET  /                               the single-page dashboard
GET  /api/health
GET  /api/scenarios                  scenario catalogue (presets, variants)
GET  /api/topology                   graph nodes/edges
GET  /api/runs                       every known run (live + archived on disk)
POST /api/runs                       {scenario, variant?, query?} -> {group_id}
GET  /api/groups/<gid>               group status (+ interrupt payload when paused)
GET  /api/groups/<gid>/stream        text/event-stream of that group's trace events
POST /api/groups/<gid>/decision      {decision: approved|rejected|revision_requested, feedback?}
GET  /api/runs/<id>/events           events as JSON      (.jsonl variant: events.jsonl)
GET  /api/runs/<id>/dossier.json|md|html   exportable run dossier

Security: binds to 127.0.0.1 by default; POST requires ``Content-Type:
application/json`` (a cross-site form cannot send that without a CORS
preflight, which this server never answers).
"""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agentic_research.experiments import ExperimentRunner, scenario_catalog
from agentic_research.experiments.runner import GroupSession
from agentic_research.human.decision import ApprovalDecision
from agentic_research.telemetry.dossier import build_dossier, render_html, render_json, render_markdown
from agentic_research.telemetry.events import EventBus, EventType, JsonlSink, TraceEvent, load_jsonl
from agentic_research.telemetry.topology import topology

STATIC_DIR = Path(__file__).parent / "static"


class DashboardApp:
    """Application state shared by all request handlers."""

    def __init__(self, data_dir: str | Path, pace: float = 0.4) -> None:
        self.data_dir = Path(data_dir)
        self.bus = EventBus()
        self.runner = ExperimentRunner(self.bus, data_dir=self.data_dir / "state", pace=pace)
        self.runs_dir = self.data_dir / "runs"
        self.bus.subscribe(JsonlSink(self.runs_dir))
        self._archive: dict[str, list[dict[str, Any]]] = {}
        self._load_archive()

    # --------------------------------------------------------------- queries

    def _load_archive(self) -> None:
        if not self.runs_dir.exists():
            return
        for path in sorted(self.runs_dir.glob("*.jsonl")):
            try:
                events = load_jsonl(path)
            except (OSError, ValueError):
                continue
            if events:
                self._archive[events[0]["run_id"]] = events

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        live = [e.to_dict() for e in self.bus.history(run_id=run_id)]
        return live or self._archive.get(run_id, [])

    def list_runs(self) -> list[dict[str, Any]]:
        ids: dict[str, list[dict[str, Any]]] = dict(self._archive)
        for e in self.bus.history():
            ids.setdefault(e.run_id, [])
        rows = []
        for run_id in ids:
            events = self.run_events(run_id)
            if not events:
                continue
            d = build_dossier(events, include_events=False)
            rows.append({
                "run_id": run_id, "group_id": d["group_id"], "status": d["status"],
                "query": d["query"], "meta": d["meta"], "started": events[0]["ts"],
                "steps": d["steps"], "events": len(events),
            })
        rows.sort(key=lambda r: r["started"], reverse=True)
        return rows

    # -------------------------------------------------------------- commands

    def start_group(self, scenario: str, variant: str | None, query: str | None) -> GroupSession:
        group = self.runner.create(scenario, variant=variant, query=query,
                                   group_id=f"grp-{uuid.uuid4().hex[:8]}")
        self._spawn(group, lambda: self.runner.execute(group))
        return group

    def decide(self, group_id: str, decision: str, feedback: str | None) -> GroupSession:
        group = self.runner.groups.get(group_id)
        if group is None:
            raise KeyError(group_id)
        ApprovalDecision(decision)  # ValueError on bad input
        if group.status != "paused":
            raise ValueError(f"Group is not awaiting a decision (status={group.status}).")
        self._spawn(group, lambda: self.runner.resume(group_id, decision, feedback))
        return group

    def _spawn(self, group: GroupSession, work: Any) -> None:
        def target() -> None:
            try:
                work()
            except Exception as exc:  # noqa: BLE001 - never leave a group stuck "running"
                group.status = "failed"
                run_id = group.runs[-1].run_id if group.runs else group.group_id
                self.bus.publish(run_id=run_id, group_id=group.group_id, type=EventType.RUN_FAILED,
                                 node=None, t_ms=0,
                                 data={"error": f"{type(exc).__name__}: {exc}", "steps": 0})

        threading.Thread(target=target, daemon=True, name=f"group-{group.group_id}").start()


def make_handler(app: DashboardApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "AgenticDashboard/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        # ----------------------------------------------------------- helpers
        def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _body(self) -> dict[str, Any]:
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                raise ValueError("Content-Type must be application/json.")
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object.")
            return data

        # --------------------------------------------------------------- GET
        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                if not parts:
                    return self._static("index.html")
                if parts[0] == "static" and len(parts) >= 2:
                    return self._static("/".join(parts[1:]))
                if parts[:2] == ["api", "health"]:
                    return self._json({"ok": True})
                if parts[:2] == ["api", "scenarios"]:
                    return self._json(scenario_catalog())
                if parts[:2] == ["api", "topology"]:
                    return self._json(topology())
                if parts == ["api", "runs"]:
                    return self._json(app.list_runs())
                if len(parts) == 4 and parts[:2] == ["api", "runs"]:
                    return self._run_resource(parts[2], parts[3])
                if len(parts) == 3 and parts[:2] == ["api", "groups"]:
                    return self._group(parts[2])
                if len(parts) == 4 and parts[:2] == ["api", "groups"] and parts[3] == "stream":
                    return self._stream(parts[2], parse_qs(url.query))
                return self._error(404, "Not found")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:  # noqa: BLE001
                return self._error(500, f"{type(exc).__name__}: {exc}")

        def _static(self, rel: str) -> None:
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                return self._error(404, "Not found")
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        def _group(self, gid: str) -> None:
            group = app.runner.groups.get(gid)
            if group is None:
                return self._error(404, "Unknown group")
            self._json(group.to_dict())

        def _run_resource(self, run_id: str, name: str) -> None:
            events = app.run_events(run_id)
            if not events:
                return self._error(404, "Unknown run")
            if name == "events":
                return self._json(events)
            if name == "events.jsonl":
                body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events).encode()
                return self._send(200, body, "application/x-ndjson",
                                  {"Content-Disposition": f'attachment; filename="{run_id}.events.jsonl"'})
            if name.startswith("dossier."):
                dossier = build_dossier(events)
                fmt = name.split(".", 1)[1]
                if fmt == "json":
                    return self._send(200, render_json(dossier).encode(), "application/json; charset=utf-8",
                                      {"Content-Disposition": f'attachment; filename="{run_id}.dossier.json"'})
                if fmt == "md":
                    return self._send(200, render_markdown(dossier).encode(), "text/markdown; charset=utf-8",
                                      {"Content-Disposition": f'attachment; filename="{run_id}.dossier.md"'})
                if fmt == "html":
                    return self._send(200, render_html(dossier).encode(), "text/html; charset=utf-8",
                                      {"Content-Disposition": f'attachment; filename="{run_id}.dossier.html"'})
            return self._error(404, "Not found")

        def _stream(self, gid: str, query: dict[str, list[str]]) -> None:
            since = int((query.get("since") or ["0"])[0])
            group = app.runner.groups.get(gid)
            q: queue.Queue[TraceEvent] = queue.Queue()
            unsubscribe = app.bus.subscribe(lambda e: q.put(e) if e.group_id == gid else None)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                last = since
                for e in app.bus.history(group_id=gid, since_seq=since):
                    self._sse(e)
                    last = e.seq
                idle = 0
                while True:
                    try:
                        e = q.get(timeout=1.0)
                        idle = 0
                        if e.seq > last:
                            self._sse(e)
                            last = e.seq
                        continue
                    except queue.Empty:
                        idle += 1
                    done = group is None or group.status in ("completed", "failed")
                    if done and q.empty():
                        self.wfile.write(b"event: end\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    if idle % 10 == 0:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                unsubscribe()
                self.close_connection = True

        def _sse(self, e: TraceEvent) -> None:
            self.wfile.write(b"data: " + json.dumps(e.to_dict(), ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        # -------------------------------------------------------------- POST
        def do_POST(self) -> None:  # noqa: N802
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            try:
                body = self._body()
                if parts == ["api", "runs"]:
                    scenario = str(body.get("scenario") or "")
                    variant = body.get("variant") or None
                    group = app.start_group(scenario, variant, body.get("query") or None)
                    return self._json({"group_id": group.group_id}, HTTPStatus.ACCEPTED)
                if len(parts) == 4 and parts[:2] == ["api", "groups"] and parts[3] == "decision":
                    app.decide(parts[2], str(body.get("decision") or ""), body.get("feedback") or None)
                    return self._json({"ok": True}, HTTPStatus.ACCEPTED)
                return self._error(404, "Not found")
            except KeyError as exc:
                return self._error(404, f"Unknown: {exc}")
            except (ValueError, json.JSONDecodeError) as exc:
                return self._error(400, str(exc))
            except Exception as exc:  # noqa: BLE001
                return self._error(500, f"{type(exc).__name__}: {exc}")

    return Handler


def create_server(host: str, port: int, data_dir: str | Path, pace: float = 0.4) -> tuple[ThreadingHTTPServer, DashboardApp]:
    app = DashboardApp(data_dir, pace=pace)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server, app
