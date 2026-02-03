import base64
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from runner.opencode_client import OpenCodeClient


@dataclass
class _ServerState:
    expected_auth: str
    requests: list[dict[str, Any]] = field(default_factory=list)


def _make_handler(state: _ServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:  # pragma: no cover
            return

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return None
            return json.loads(raw.decode("utf-8", errors="replace"))

        def _write_json(self, code: int, data: Any) -> None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _require_auth(self) -> bool:
            got = self.headers.get("Authorization") or ""
            if got != state.expected_auth:
                self._write_json(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            state.requests.append({"method": "GET", "path": self.path, "body": None})
            if self.path == "/global/health":
                self._write_json(200, {"ok": True})
                return
            self._write_json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            body = self._read_json()
            state.requests.append({"method": "POST", "path": self.path, "body": body})
            if self.path == "/session":
                self._write_json(200, {"id": "s1"})
                return
            if self.path == "/session/s1/message":
                # Minimal contract check.
                assert isinstance(body, dict)
                assert isinstance(body.get("parts"), list)
                assert body["parts"] and body["parts"][0]["type"] == "text"
                self._write_json(200, {"parts": [{"type": "text", "text": "hello"}]})
                return
            if self.path == "/instance/dispose":
                self._write_json(200, True)
                return
            self._write_json(404, {"error": "not_found"})

        def do_PATCH(self) -> None:  # noqa: N802
            if not self._require_auth():
                return
            body = self._read_json()
            state.requests.append({"method": "PATCH", "path": self.path, "body": body})
            if self.path == "/config":
                assert isinstance(body, dict)
                assert "permission" in body
                self._write_json(200, {"ok": True})
                return
            self._write_json(404, {"error": "not_found"})

    return Handler


def test_opencode_client_http_happy_path(tmp_path: Path):
    user = "u"
    pwd = "p"
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    expected_auth = f"Basic {token}"
    state = _ServerState(expected_auth=expected_auth)

    server = HTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = OpenCodeClient(
            repo=tmp_path,
            plan_rel="PLAN.md",
            pipeline_rel="pipeline.yml",
            model="openai/gpt-4o-mini",
            base_url=f"http://{host}:{port}",
            timeout_seconds=5,
            bash_mode="restricted",
            unattended="strict",
            username=user,
            password=pwd,
            session_title="t",
        )
        try:
            res = client.run("hi", fsm_state="S2_PLAN_UPDATE", iter_idx=1, purpose="plan_update_attempt_1")
            assert res.assistant_text.strip() == "hello"
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    paths = [(r["method"], r["path"]) for r in state.requests]
    assert ("GET", "/global/health") in paths
    assert ("POST", "/session") in paths
    assert ("POST", "/session/s1/message") in paths
