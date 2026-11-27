"""Integration tests for the stdlib dashboard server (real HTTP on an ephemeral port)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from agentic_research.viz.server import create_server


@pytest.fixture()
def server(tmp_path):
    srv, app = create_server("127.0.0.1", 0, tmp_path, pace=0.0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, app, tmp_path
    srv.shutdown()
    srv.server_close()


def _get(url, raw=False):
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read()
        return (r.status, r.headers, body) if raw else json.loads(body)


def _post(url, payload, ctype="application/json"):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait(base, gid, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        g = _get(f"{base}/api/groups/{gid}")
        if g["status"] in statuses:
            return g
        time.sleep(0.02)
    raise AssertionError(f"group never reached {statuses}: {g}")


def test_static_assets_and_catalog_endpoints(server) -> None:
    base, _, _ = server
    status, headers, body = _get(base + "/", raw=True)
    assert status == 200 and b"Agent trace viewer" in body and "text/html" in headers["Content-Type"]
    assert b"derive(" in _get(base + "/static/app.js", raw=True)[2]
    assert _get(base + "/api/health") == {"ok": True}
    assert len(_get(base + "/api/scenarios")) == 5
    topo = _get(base + "/api/topology")
    assert {n["id"] for n in topo["nodes"]} >= {"supervisor", "human_approval"} and topo["edges"]


def test_static_path_traversal_and_unknown_routes_are_404(server) -> None:
    base, _, _ = server
    for path in ("/static/../server.py", "/static/%2e%2e/server.py", "/nope", "/api/runs/missing/events"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base + path)
        assert exc.value.code == 404


def test_post_requires_json_content_type_and_valid_input(server) -> None:
    base, _, _ = server
    status, body = _post(base + "/api/runs", {"scenario": "contradiction"}, ctype="text/plain")
    assert status == 400 and "Content-Type" in body["error"]
    assert _post(base + "/api/runs", {"scenario": "nope"})[0] == 404
    assert _post(base + "/api/groups/none/decision", {"decision": "approved"})[0] == 404


def test_run_a_scenario_stream_events_and_export_dossier(server) -> None:
    base, app, tmp = server
    status, body = _post(base + "/api/runs", {"scenario": "contradiction"})
    assert status == 202
    gid = body["group_id"]
    _wait(base, gid, {"completed"})
    # SSE: replays history and terminates with an `end` event
    with urllib.request.urlopen(f"{base}/api/groups/{gid}/stream", timeout=10) as r:
        assert "text/event-stream" in r.headers["Content-Type"]
        text = r.read().decode()
    payloads = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ") and l != "data: {}"]
    assert payloads[0]["type"] == "run.started" and payloads[-1]["type"] == "run.completed"
    assert "event: end" in text
    run_id = payloads[0]["run_id"]
    assert _get(f"{base}/api/runs/{run_id}/events")[0]["run_id"] == run_id
    md = _get(f"{base}/api/runs/{run_id}/dossier.md", raw=True)
    assert b"## Final deliverable" in md[2] and "attachment" in md[1]["Content-Disposition"]
    assert b"<!doctype html>" in _get(f"{base}/api/runs/{run_id}/dossier.html", raw=True)[2]
    assert json.loads(_get(f"{base}/api/runs/{run_id}/dossier.json", raw=True)[2])["status"] == "completed"
    assert len(_get(f"{base}/api/runs/{run_id}/events.jsonl", raw=True)[2].splitlines()) == len(payloads)
    assert any(r["run_id"] == run_id and r["status"] == "completed" for r in _get(base + "/api/runs"))
    assert (tmp / "runs" / f"{run_id}.jsonl").exists()  # persisted for replay


def test_human_in_the_loop_over_http(server) -> None:
    base, _, _ = server
    gid = _post(base + "/api/runs", {"scenario": "human-checkpoint"})[1]["group_id"]
    g = _wait(base, gid, {"paused"})
    assert g["interrupt_payload"]["draft_report"]
    assert _post(f"{base}/api/groups/{gid}/decision", {"decision": "maybe"})[0] == 400
    assert _post(f"{base}/api/groups/{gid}/decision", {"decision": "approved"})[0] == 202
    g = _wait(base, gid, {"completed"})
    assert g["runs"][0]["status"] == "completed"
    # deciding again after completion is rejected
    assert _post(f"{base}/api/groups/{gid}/decision", {"decision": "approved"})[0] == 400


def test_archived_runs_are_listed_after_restart(tmp_path) -> None:
    srv, app = create_server("127.0.0.1", 0, tmp_path, pace=0.0)
    group = app.runner.run_to_completion("contradiction")
    srv.server_close()
    srv2, app2 = create_server("127.0.0.1", 0, tmp_path, pace=0.0)
    try:
        assert [r["run_id"] for r in app2.list_runs()] == [group.runs[0].run_id]
        assert app2.run_events(group.runs[0].run_id)[-1]["type"] == "run.completed"
    finally:
        srv2.server_close()
