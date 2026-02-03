from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_client import AgentClient, AgentResult
from .opencode_tooling import ToolPolicy, execute_tool_calls, format_tool_results, parse_tool_calls
from .subprocess_utils import tail


def _normalize_base_url(url: str) -> str:
    s = str(url or "").strip()
    if not s:
        raise ValueError("empty_url")
    return s.rstrip("/")


def _basic_auth_value(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _split_model(model: str) -> tuple[str, str]:
    s = str(model or "").strip()
    if not s:
        return "openai", "gpt-4o-mini"
    if "/" in s:
        provider_id, model_id = s.split("/", 1)
        provider_id = provider_id.strip() or "openai"
        model_id = model_id.strip() or "gpt-4o-mini"
        return provider_id, model_id
    return "openai", s


def _extract_assistant_text(message: Any) -> str:
    if not isinstance(message, dict):
        return str(message)
    parts = message.get("parts")
    if not isinstance(parts, list):
        return str(message)
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join([t for t in texts if t.strip()]) or str(message)


def _extract_opencode_error(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    info = message.get("info")
    if not isinstance(info, dict):
        return None
    err = info.get("error")
    if not isinstance(err, dict):
        return None
    name = str(err.get("name") or "").strip() or "Error"
    data = err.get("data")
    detail = ""
    if isinstance(data, dict):
        detail = str(data.get("message") or "").strip()
    if not detail:
        detail = str(data).strip() if data is not None else ""
    if detail:
        return f"{name}: {detail}"
    return name


@dataclass(frozen=True)
class OpenCodeServerConfig:
    base_url: str
    username: str
    password: str


class OpenCodeRequestError(RuntimeError):
    def __init__(self, *, method: str, url: str, status: int | None, detail: str):
        super().__init__(f"OpenCode request failed: {method} {url} ({status}) {detail}")
        self.method = method
        self.url = url
        self.status = status
        self.detail = detail


class OpenCodeClient(AgentClient):
    def __init__(
        self,
        *,
        repo: Path,
        plan_rel: str,
        pipeline_rel: str | None,
        model: str,
        base_url: str | None,
        timeout_seconds: int,
        bash_mode: str,
        unattended: str,
        server_log_path: Path | None = None,
        username: str | None = None,
        password: str | None = None,
        session_title: str | None = None,
    ) -> None:
        self._repo = repo
        self._plan_rel = str(plan_rel or "PLAN.md").strip() or "PLAN.md"
        self._pipeline_rel = str(pipeline_rel).strip() if pipeline_rel else None
        self._timeout_seconds = int(timeout_seconds) if timeout_seconds else 300
        self._bash_mode = (bash_mode or "restricted").strip().lower()
        if self._bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_bash_mode")
        self._unattended = str(unattended or "strict").strip().lower() or "strict"

        provider_id, model_id = _split_model(model)
        self._model_obj: dict[str, str] = {"providerID": provider_id, "modelID": model_id}
        self._model_str: str = f"{provider_id}/{model_id}"

        self._proc: subprocess.Popen[str] | None = None
        self._server_log_file = None

        if base_url:
            self._server = OpenCodeServerConfig(
                base_url=_normalize_base_url(base_url),
                username=(username or "opencode").strip() or "opencode",
                password=(password or "").strip(),
            )
        else:
            self._server = self._start_local_server(
                repo=repo, server_log_path=server_log_path, username=username
            )

        self._wait_for_health()
        self._session_id = self._create_session(title=session_title or f"runner:{repo.name}")

    def close(self) -> None:
        try:
            # Best-effort dispose (does not necessarily terminate the server process).
            if self._proc is not None:
                try:
                    self._request_json("POST", "/instance/dispose", body=None, require_auth=True)
                except Exception:
                    pass
        finally:
            if self._proc is not None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass

    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        policy = ToolPolicy(
            repo=self._repo.resolve(),
            plan_path=(self._repo / self._plan_rel).resolve(),
            pipeline_path=((self._repo / self._pipeline_rel).resolve() if self._pipeline_rel else None),
            purpose=purpose,
            bash_mode=self._bash_mode,
            unattended=self._unattended,
        )

        prompt = text
        last_msg: Any | None = None
        for _turn in range(20):
            try:
                msg = self._post_message(model=self._model_obj, text=prompt)
            except OpenCodeRequestError as e:
                # Compatibility fallback: some builds may accept model as a string.
                if e.status in (400, 422):
                    msg = self._post_message(model=self._model_str, text=prompt)
                else:
                    raise
            last_msg = msg

            err = _extract_opencode_error(msg)
            if err:
                raise RuntimeError(f"OpenCode agent error: {err}")

            assistant_text = _extract_assistant_text(msg)
            calls = parse_tool_calls(assistant_text)
            if not calls:
                return AgentResult(assistant_text=assistant_text, raw=msg)

            results = execute_tool_calls(calls, repo=self._repo, policy=policy)
            prompt = format_tool_results(results)

        raise RuntimeError("OpenCode tool loop exceeded 20 turns without a final response.")

    def _start_local_server(self, *, repo: Path, server_log_path: Path | None, username: str | None) -> OpenCodeServerConfig:
        if not shutil.which("opencode"):
            raise RuntimeError("`opencode` not found in PATH. Install it from https://opencode.ai/")

        host = "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            _host, port = s.getsockname()

        user = (username or "opencode").strip() or "opencode"
        pwd = secrets.token_urlsafe(24)
        env = dict(os.environ)
        # OpenCode's OpenAI-compatible provider reads `OPENAI_BASE_URL`.
        # Keep compatibility with `.env` files using `OPENAI_API_BASE`.
        if not str(env.get("OPENAI_BASE_URL") or "").strip():
            api_base = str(env.get("OPENAI_API_BASE") or "").strip().rstrip("/")
            if api_base:
                env["OPENAI_BASE_URL"] = api_base if api_base.endswith("/v1") else (api_base + "/v1")
        env["OPENCODE_SERVER_USERNAME"] = user
        env["OPENCODE_SERVER_PASSWORD"] = pwd

        cmd = ["opencode", "serve", "--hostname", host, "--port", str(port)]

        stdout = subprocess.DEVNULL
        if server_log_path is not None:
            server_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._server_log_file = server_log_path.open("w", encoding="utf-8")
            stdout = self._server_log_file

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stdout,
        )
        return OpenCodeServerConfig(base_url=f"http://{host}:{port}", username=user, password=pwd)

    def _wait_for_health(self) -> None:
        deadline = time.time() + 20
        last_err = ""
        while time.time() < deadline:
            try:
                self._request_json("GET", "/global/health", body=None, require_auth=bool(self._server.password))
                return
            except Exception as e:
                last_err = str(e)
                time.sleep(0.2)
        raise RuntimeError(f"OpenCode server failed health check: {tail(last_err, 2000)}")

    def _create_session(self, *, title: str) -> str:
        body = {"title": title}
        data = self._request_json("POST", "/session", body=body, require_auth=bool(self._server.password))
        if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
            return data["id"]
        if isinstance(data, dict) and isinstance(data.get("sessionID"), str) and data["sessionID"].strip():
            return data["sessionID"]
        raise RuntimeError(f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}")

    def _post_message(self, *, model: Any, text: str) -> Any:
        body = {
            "agent": "build",
            "model": model,
            "parts": [{"type": "text", "text": text}],
        }
        return self._request_json(
            "POST",
            f"/session/{self._session_id}/message",
            body=body,
            require_auth=bool(self._server.password),
        )

    def _request_json(self, method: str, path: str, *, body: Any, require_auth: bool) -> Any:
        url = f"{self._server.base_url}{path}"
        headers = {"Accept": "application/json"}
        if require_auth and self._server.password:
            headers["Authorization"] = _basic_auth_value(self._server.username, self._server.password)

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = Request(url, method=method, data=data, headers=headers)
        try:
            with urlopen(req, timeout=self._timeout_seconds) as resp:
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8", errors="replace"))
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise OpenCodeRequestError(method=method, url=url, status=int(getattr(e, "code", 0) or 0), detail=tail(detail, 2000))
        except URLError as e:
            raise OpenCodeRequestError(method=method, url=url, status=None, detail=str(e))
