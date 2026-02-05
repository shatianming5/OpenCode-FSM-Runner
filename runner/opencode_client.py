from __future__ import annotations

import base64
import json
import os
import signal
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
    """中文说明：
    - 含义：规范化 OpenCode server base URL（去掉末尾 `/`，并拒绝空字符串）。
    - 内容：用于把用户传入的 `--opencode-url` 或内部启动的 base_url 统一成可拼接 path 的形式。
    - 可简略：可能（小工具；但集中做输入清洗更可靠）。
    """
    s = str(url or "").strip()
    if not s:
        raise ValueError("empty_url")
    return s.rstrip("/")


def _basic_auth_value(username: str, password: str) -> str:
    """中文说明：
    - 含义：生成 HTTP Basic Auth 的 Authorization header 值。
    - 内容：`base64(username:password)`，用于访问带密码的 OpenCode server。
    - 可简略：可能（工具函数；但集中实现便于测试与避免拼写错误）。
    """
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _split_model(model: str) -> tuple[str, str]:
    """中文说明：
    - 含义：把 `provider/model` 形式的模型字符串拆分为 (provider_id, model_id)。
    - 内容：空字符串回退到 `openai/gpt-4o-mini`；只有模型名时默认 provider=openai。
    - 可简略：可能（与 CLI/env_local 的模型解析存在重复；可抽公共模块）。
    """
    s = str(model or "").strip()
    if not s:
        return "openai", "gpt-4o-mini"
    if "/" in s:
        provider_id, model_id = s.split("/", 1)
        provider_id = provider_id.strip() or "openai"
        model_id = model_id.strip() or "gpt-4o-mini"
        return provider_id, model_id
    return "openai", s


def select_bash_mode(*, purpose: str, default_bash_mode: str, scaffold_bash_mode: str) -> str:
    """中文说明：
    - 含义：根据用途选择 tool-call 的 bash 权限模式（restricted/full）。
    - 内容：当 purpose==scaffold_contract 时允许用单独的 scaffold_bash_mode（通常更宽松以便生成合同文件）；其它情况用 default_bash_mode。
    - 可简略：可能（小策略函数；但把规则集中有利于审计）。
    """
    p = str(purpose or "").strip().lower()
    default = str(default_bash_mode or "restricted").strip().lower() or "restricted"
    scaffold = str(scaffold_bash_mode or default).strip().lower() or default
    if p in ("scaffold_contract", "repair_contract"):
        return scaffold
    return default


def _extract_assistant_text(message: Any) -> str:
    """中文说明：
    - 含义：从 OpenCode message JSON 中提取所有 text parts 并拼接为字符串。
    - 内容：兼容 message 非 dict 或 parts 格式不符的情况；用于后续解析 tool-calls 或作为最终回答。
    - 可简略：可能（依赖 OpenCode 返回结构；集中封装更利于兼容不同版本）。
    """
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
    """中文说明：
    - 含义：从 OpenCode message JSON 中提取错误信息（如果存在）。
    - 内容：读取 `message.info.error`，把 name/message 合成为人类可读字符串；用于 fail-fast。
    - 可简略：可能（兼容性/可读性辅助；但保留能提升错误诊断体验）。
    """
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
    """中文说明：
    - 含义：OpenCode server 连接信息（base_url + 基本认证）。
    - 内容：当 runner 自己启动本地 server 时会生成随机 password；也支持连接外部 server（用户提供 base_url/username/password）。
    - 可简略：可能（字段很少；但作为显式结构便于传递与测试）。
    """

    base_url: str
    username: str
    password: str


class OpenCodeRequestError(RuntimeError):
    """中文说明：
    - 含义：OpenCode HTTP 请求失败时抛出的异常（带 method/url/status/detail）。
    - 内容：用于把 HTTPError/URLError 统一包装成稳定异常类型，方便上层做兼容性回退或报错。
    - 可简略：否（稳定错误类型对诊断与测试很重要）。
    """

    def __init__(self, *, method: str, url: str, status: int | None, detail: str):
        """中文说明：
        - 含义：构造带上下文信息的请求异常。
        - 内容：把 method/url/status/detail 写入属性，便于调用方判断（如 400/422 时做 model 字段兼容回退）。
        - 可简略：可能（实现简单；但建议保留属性字段以便调试）。
        """
        super().__init__(f"OpenCode request failed: {method} {url} ({status}) {detail}")
        self.method = method
        self.url = url
        self.status = status
        self.detail = detail


class OpenCodeClient(AgentClient):
    """中文说明：
    - 含义：基于 OpenCode server API 的 AgentClient 实现（带 tool-call 执行闭环）。
    - 内容：可连接外部 OpenCode server，也可在 repo 内自动启动 `opencode serve`；每次 `run()` 发送消息、解析 tool-calls、由 runner 执行并回灌结果，直到拿到最终文本。
    - 可简略：否（这是当前项目的核心 agent 适配层）。
    """

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
        scaffold_bash_mode: str = "full",
        unattended: str,
        server_log_path: Path | None = None,
        username: str | None = None,
        password: str | None = None,
        session_title: str | None = None,
    ) -> None:
        """中文说明：
        - 含义：初始化 OpenCodeClient（模型选择、server 连接/启动、健康检查、创建 session）。
        - 内容：当 `base_url` 为空时会启动本地 server 进程并写日志；随后创建 session 并保留 session_id；bash 权限由 bash_mode/scaffold_bash_mode 与 purpose 共同决定。
        - 可简略：否（涉及进程管理/认证/兼容性与关键默认值）。
        """
        self._repo = repo
        self._plan_rel = str(plan_rel or "PLAN.md").strip() or "PLAN.md"
        self._pipeline_rel = str(pipeline_rel).strip() if pipeline_rel else None
        self._timeout_seconds = int(timeout_seconds) if timeout_seconds else 300
        self._bash_mode = (bash_mode or "restricted").strip().lower()
        if self._bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_bash_mode")
        self._scaffold_bash_mode = (scaffold_bash_mode or "full").strip().lower()
        if self._scaffold_bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_scaffold_bash_mode")
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

        try:
            self._wait_for_health()
            self._session_id = self._create_session(title=session_title or f"runner:{repo.name}")
        except Exception:
            # If init fails after starting a local server, ensure we don't leak the process.
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        """中文说明：
        - 含义：释放 OpenCodeClient 资源（best-effort）。
        - 内容：尝试调用 `/instance/dispose`；若是本地启动的 server，则 terminate/kill 进程并关闭日志文件句柄。
        - 可简略：否（避免后台进程泄漏）。
        """
        try:
            # Best-effort dispose (does not necessarily terminate the server process).
            if self._proc is not None:
                try:
                    # Do not block shutdown on a potentially wedged server.
                    self._request_json("POST", "/instance/dispose", body=None, require_auth=True, timeout_seconds=5)
                except Exception:
                    pass
        finally:
            if self._proc is not None:
                try:
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                        except Exception:
                            self._proc.terminate()
                    else:  # pragma: no cover
                        self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5)
                    except Exception:
                        if os.name == "posix":
                            try:
                                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                            except Exception:
                                self._proc.kill()
                        else:  # pragma: no cover
                            self._proc.kill()
                except Exception:
                    pass
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass

    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：执行一次 agent 调用（含 tool loop）。
        - 内容：发送 prompt → 获取 assistant 输出 → 解析 tool-calls → 执行并回灌 `tool_result` → 重复最多 20 轮；若无 tool-call 则直接返回最终文本。
        - 可简略：否（tool loop 是闭环能力的核心实现）。
        """
        policy = ToolPolicy(
            repo=self._repo.resolve(),
            plan_path=(self._repo / self._plan_rel).resolve(),
            pipeline_path=((self._repo / self._pipeline_rel).resolve() if self._pipeline_rel else None),
            purpose=purpose,
            bash_mode=select_bash_mode(
                purpose=purpose,
                default_bash_mode=self._bash_mode,
                scaffold_bash_mode=self._scaffold_bash_mode,
            ),
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
        """中文说明：
        - 含义：在本地启动 OpenCode server（`opencode serve`）。
        - 内容：选择一个空闲端口；生成随机 password；设置 OPENCODE_SERVER_* 环境变量；可选写 server log 到 artifacts。
        - 可简略：否（进程/端口/认证管理是关键；实现需要谨慎）。
        """
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
            start_new_session=True,
        )
        return OpenCodeServerConfig(base_url=f"http://{host}:{port}", username=user, password=pwd)

    def _wait_for_health(self) -> None:
        """中文说明：
        - 含义：等待 OpenCode server health endpoint 可用。
        - 内容：轮询 `/global/health` 最多约 20 秒；失败则抛错并包含最近一次错误尾部。
        - 可简略：可能（轮询参数可配置；但保留健康检查可避免后续难以理解的失败）。
        """
        deadline = time.time() + 60
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
        """中文说明：
        - 含义：在 OpenCode server 上创建一个会话（session），并返回 session id。
        - 内容：兼容不同字段命名（`id` 或 `sessionID`）；若响应不符合预期则抛错并附带 JSON 尾部。
        - 可简略：可能（主要是兼容性处理；但对不同 OpenCode 版本很有用）。
        """
        body = {"title": title}
        data = self._request_json("POST", "/session", body=body, require_auth=bool(self._server.password))
        if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
            return data["id"]
        if isinstance(data, dict) and isinstance(data.get("sessionID"), str) and data["sessionID"].strip():
            return data["sessionID"]
        raise RuntimeError(f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}")

    def _post_message(self, *, model: Any, text: str) -> Any:
        """中文说明：
        - 含义：向当前 session 发送一条消息（prompt），返回 server 的 message JSON。
        - 内容：body 里包含 agent 类型、model、parts；由 `_request_json` 完成 HTTP 细节与错误包装。
        - 可简略：可能（薄封装；但把请求体结构集中在一处更易维护）。
        """
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

    def _request_json(self, method: str, path: str, *, body: Any, require_auth: bool, timeout_seconds: float | None = None) -> Any:
        """中文说明：
        - 含义：执行一次 OpenCode HTTP 请求并解析 JSON 响应。
        - 内容：支持可选 Basic Auth；body 会被编码为 JSON；HTTPError/URLError 会被包装为 OpenCodeRequestError（截断 detail 以控制体积）。
        - 可简略：否（HTTP I/O 与错误处理的核心封装；影响稳定性与可诊断性）。
        """
        url = f"{self._server.base_url}{path}"
        headers = {"Accept": "application/json"}
        if require_auth and self._server.password:
            headers["Authorization"] = _basic_auth_value(self._server.username, self._server.password)

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = Request(url, method=method, data=data, headers=headers)
        timeout = self._timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        try:
            with urlopen(req, timeout=timeout) as resp:
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
