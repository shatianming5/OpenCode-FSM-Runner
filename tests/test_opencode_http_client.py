import base64
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from runner.opencode_client import OpenCodeClient, OpenCodeRequestError


@dataclass
class _ServerState:
    """中文说明：
    - 含义：测试用的轻量 HTTP 服务器状态容器。
    - 内容：保存期望的 Authorization 值，以及记录所有收到的请求（method/path/body）。
    - 可简略：是（仅用于测试；也可用 dict 代替）。
    """

    expected_auth: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    fail_message_once_with_503: bool = False
    reject_context_once: bool = False
    message_returns_empty_once: bool = False
    message_calls: int = 0


def _make_handler(state: _ServerState):
    """中文说明：
    - 含义：为 `HTTPServer` 构造一个绑定了 `state` 的请求处理器类（BaseHTTPRequestHandler 子类）。
    - 内容：实现 health/session/message/config/instance/dispose 的最小 OpenCode HTTP 合同，并记录请求轨迹。
    - 可简略：可能（可以用更小的 handler 覆盖当前测试；但这里集中实现便于复用与调试）。
    """

    class Handler(BaseHTTPRequestHandler):
        """中文说明：
        - 含义：测试用 OpenCode “伪服务端”的请求处理器。
        - 内容：校验 Basic Auth、解析/返回 JSON，并按路径返回固定响应以驱动 OpenCodeClient 走通流程。
        - 可简略：可能（很多分支只为覆盖客户端路径；可按测试需要裁剪）。
        """

        def log_message(self, _format: str, *_args: Any) -> None:  # pragma: no cover
            """中文说明：
            - 含义：覆盖 BaseHTTPRequestHandler 的日志输出，避免测试打印到 stderr。
            - 内容：直接 return（不输出任何日志）。
            - 可简略：是（纯噪音控制；也可用 logging 配置代替）。
            """
            return

        def _read_json(self) -> Any:
            """中文说明：
            - 含义：读取请求体并尝试解析为 JSON（用于 POST/PATCH）。
            - 内容：按 Content-Length 读取字节并 utf-8 解码；空 body 返回 None。
            - 可简略：是（测试 helper；也可在各 handler 方法内联）。
            """
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return None
            return json.loads(raw.decode("utf-8", errors="replace"))

        def _write_json(self, code: int, data: Any) -> None:
            """中文说明：
            - 含义：以 JSON 格式写入响应（用于 GET/POST/PATCH）。
            - 内容：`json.dumps` → bytes，设置 Content-Type/Length，并写入到 wfile。
            - 可简略：是（测试 helper；也可用更高层框架替代）。
            """
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _require_auth(self) -> bool:
            """中文说明：
            - 含义：校验 Authorization 头是否符合预期（Basic 认证）。
            - 内容：不匹配则返回 401 JSON，并阻断后续处理；匹配则返回 True。
            - 可简略：是（测试 helper；可内联到各方法）。
            """
            got = self.headers.get("Authorization") or ""
            if got != state.expected_auth:
                self._write_json(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            """中文说明：
            - 含义：实现最小 GET 路由（目前用于 /global/health）。
            - 内容：先校验 auth；记录请求；health 返回 ok，其它路径 404。
            - 可简略：可能（取决于客户端覆盖面；当前为了可读性保留分支）。
            """
            if not self._require_auth():
                return
            state.requests.append({"method": "GET", "path": self.path, "body": None})
            if self.path == "/global/health":
                self._write_json(200, {"ok": True})
                return
            self._write_json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            """中文说明：
            - 含义：实现最小 POST 路由（session 创建、发送消息、dispose）。
            - 内容：先校验 auth；读取 JSON；按路径返回固定响应，并做少量 contract assert。
            - 可简略：可能（很多 assert/路径可随测试精简）。
            """
            if not self._require_auth():
                return
            body = self._read_json()
            state.requests.append({"method": "POST", "path": self.path, "body": body})
            if self.path == "/session":
                self._write_json(200, {"id": "s1"})
                return
            if self.path == "/session/s1/message":
                state.message_calls += 1
                if state.message_returns_empty_once and state.message_calls == 1:
                    # Simulate a buggy/empty transport: 200 OK but no JSON body.
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if state.fail_message_once_with_503 and state.message_calls == 1:
                    self._write_json(503, {"error": "try_again"})
                    return
                if (
                    state.reject_context_once
                    and state.message_calls == 1
                    and isinstance(body, dict)
                    and body.get("contextLength") is not None
                ):
                    self._write_json(422, {"error": "invalid_context_length"})
                    return
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
            """中文说明：
            - 含义：实现最小 PATCH 路由（用于 /config 权限配置）。
            - 内容：先校验 auth；读取 JSON；断言包含 permission 字段并返回 ok；其它路径 404。
            - 可简略：可能（随客户端实现变化；当前仅覆盖必要路径）。
            """
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
    """中文说明：
    - 含义：验证 OpenCodeClient 在 HTTP 模式下能走通 health → session → message → dispose 的最小闭环。
    - 内容：启动本地 HTTPServer（Basic auth），构造 OpenCodeClient 发消息，断言回复与请求轨迹均符合预期。
    - 可简略：否（是 HTTP 客户端契约的关键端到端覆盖；建议保留）。
    """
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


def test_opencode_client_retries_transient_503(tmp_path: Path):
    user = "u"
    pwd = "p"
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    expected_auth = f"Basic {token}"
    state = _ServerState(expected_auth=expected_auth, fail_message_once_with_503=True)

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
            request_retry_attempts=2,
            request_retry_backoff_seconds=0.01,
            bash_mode="restricted",
            unattended="strict",
            username=user,
            password=pwd,
            session_title="t-retry",
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

    message_posts = [r for r in state.requests if r.get("method") == "POST" and r.get("path") == "/session/s1/message"]
    assert len(message_posts) == 2


def test_opencode_client_retries_empty_message_body(tmp_path: Path):
    """If OpenCode returns 200 with an empty body, the client should treat it as a transient failure and retry."""
    user = "u"
    pwd = "p"
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    expected_auth = f"Basic {token}"
    state = _ServerState(expected_auth=expected_auth, message_returns_empty_once=True)

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
            request_retry_attempts=2,
            request_retry_backoff_seconds=0.01,
            bash_mode="restricted",
            unattended="strict",
            username=user,
            password=pwd,
            session_title="t-empty",
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


def test_opencode_client_context_length_fallback_and_prompt_clip(tmp_path: Path):
    user = "u"
    pwd = "p"
    token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    expected_auth = f"Basic {token}"
    state = _ServerState(expected_auth=expected_auth, reject_context_once=True)

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
            request_retry_attempts=1,
            request_retry_backoff_seconds=0.01,
            context_length=8192,
            max_prompt_chars=120,
            bash_mode="restricted",
            unattended="strict",
            username=user,
            password=pwd,
            session_title="t-context",
        )
        try:
            long_prompt = "x" * 500
            res = client.run(long_prompt, fsm_state="S2_PLAN_UPDATE", iter_idx=1, purpose="plan_update_attempt_1")
            assert res.assistant_text.strip() == "hello"
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    message_posts = [r for r in state.requests if r.get("method") == "POST" and r.get("path") == "/session/s1/message"]
    assert len(message_posts) == 2
    first_body = message_posts[0]["body"]
    second_body = message_posts[1]["body"]
    assert isinstance(first_body, dict)
    assert isinstance(second_body, dict)
    assert first_body.get("contextLength") == 8192
    assert "contextLength" not in second_body

    first_text = (((first_body.get("parts") or [{}])[0]).get("text") if isinstance(first_body.get("parts"), list) else "") or ""
    assert isinstance(first_text, str)
    assert len(first_text) <= 120


def test_post_message_retry_recovers_local_transport_error() -> None:
    client = OpenCodeClient.__new__(OpenCodeClient)
    client._request_retry_attempts = 2
    client._request_retry_backoff_seconds = 0.0
    client._session_recover_attempts = 1
    client._session_recover_backoff_seconds = 0.0
    client._context_length = None
    client._owns_local_server = True

    calls = {"post": 0, "recover": 0}

    def fake_post(*, model: Any, text: str, include_context: bool = True) -> Any:
        calls["post"] += 1
        if calls["post"] == 1:
            raise OpenCodeRequestError(
                method="POST",
                url="http://127.0.0.1:1/session/s1/message",
                status=None,
                detail="urlopen error [Errno 111] Connection refused",
            )
        return {"parts": [{"type": "text", "text": "ok"}]}

    def fake_recover(*, reason: str) -> None:
        calls["recover"] += 1

    client._post_message = fake_post
    client._recover_local_server_session = fake_recover
    client._sleep_retry_backoff = lambda **_kwargs: None
    client._sleep_session_recover_backoff = lambda **_kwargs: None
    client._should_retry_request_error = lambda _err: True

    out = OpenCodeClient._post_message_with_retry(client, model={"providerID": "openai", "modelID": "gpt-4o-mini"}, text="hi")
    assert isinstance(out, dict)
    assert calls["recover"] == 1
    assert calls["post"] == 2


def test_post_message_retry_does_not_recover_for_external_server() -> None:
    client = OpenCodeClient.__new__(OpenCodeClient)
    client._request_retry_attempts = 1
    client._request_retry_backoff_seconds = 0.0
    client._session_recover_attempts = 3
    client._session_recover_backoff_seconds = 0.0
    client._context_length = None
    client._owns_local_server = False

    calls = {"recover": 0}

    def fake_post(*, model: Any, text: str, include_context: bool = True) -> Any:
        raise OpenCodeRequestError(
            method="POST",
            url="http://example.com/session/s1/message",
            status=None,
            detail="urlopen error [Errno 111] Connection refused",
        )

    def fake_recover(*, reason: str) -> None:
        calls["recover"] += 1

    client._post_message = fake_post
    client._recover_local_server_session = fake_recover
    client._sleep_retry_backoff = lambda **_kwargs: None
    client._sleep_session_recover_backoff = lambda **_kwargs: None
    client._should_retry_request_error = lambda _err: False

    with pytest.raises(OpenCodeRequestError):
        OpenCodeClient._post_message_with_retry(
            client,
            model={"providerID": "openai", "modelID": "gpt-4o-mini"},
            text="hi",
        )
    assert calls["recover"] == 0
