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


def _looks_like_transport_unavailable(detail: str) -> bool:
    """Best-effort classification for transient transport failures."""
    d = str(detail or "").strip().lower()
    if not d:
        return False
    needles = (
        "connection refused",
        "failed to establish a new connection",
        "connection reset",
        "connection aborted",
        "connection closed",
        "remote end closed",
        "network is unreachable",
        "name or service not known",
        "temporary failure in name resolution",
        "timed out",
        "timeout",
    )
    return any(n in d for n in needles)


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
        request_retry_attempts: int = 2,
        request_retry_backoff_seconds: float = 2.0,
        session_recover_attempts: int | None = None,
        session_recover_backoff_seconds: float | None = None,
        context_length: int | None = None,
        max_prompt_chars: int | None = None,
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
        self._request_retry_attempts = max(0, int(request_retry_attempts or 0))
        try:
            _backoff = float(request_retry_backoff_seconds)
        except Exception:
            _backoff = 2.0
        self._request_retry_backoff_seconds = max(0.0, _backoff)
        _recover_attempts_raw = (
            session_recover_attempts
            if session_recover_attempts is not None
            else os.environ.get("AIDER_OPENCODE_SESSION_RECOVER_ATTEMPTS", "2")
        )
        try:
            self._session_recover_attempts = max(0, int(_recover_attempts_raw or 0))
        except Exception:
            self._session_recover_attempts = 2
        _recover_backoff_raw = (
            session_recover_backoff_seconds
            if session_recover_backoff_seconds is not None
            else os.environ.get("AIDER_OPENCODE_SESSION_RECOVER_BACKOFF_SECONDS", "2.0")
        )
        try:
            self._session_recover_backoff_seconds = max(0.0, float(_recover_backoff_raw or 0.0))
        except Exception:
            self._session_recover_backoff_seconds = 2.0
        try:
            _context_length = int(context_length or 0)
        except Exception:
            _context_length = 0
        self._context_length: int | None = _context_length if _context_length > 0 else None
        try:
            _max_prompt_chars = int(max_prompt_chars or 0)
        except Exception:
            _max_prompt_chars = 0
        self._max_prompt_chars: int | None = _max_prompt_chars if _max_prompt_chars > 0 else None
        self._bash_mode = (bash_mode or "restricted").strip().lower()
        if self._bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_bash_mode")
        self._scaffold_bash_mode = (scaffold_bash_mode or "full").strip().lower()
        if self._scaffold_bash_mode not in ("restricted", "full"):
            raise ValueError("invalid_scaffold_bash_mode")
        self._unattended = str(unattended or "strict").strip().lower() or "strict"
        self._session_title = str(session_title or f"runner:{repo.name}")
        self._server_log_path = server_log_path.resolve() if server_log_path is not None else None

        provider_id, model_id = _split_model(model)
        self._model_obj: dict[str, str] = {"providerID": provider_id, "modelID": model_id}
        self._model_str: str = f"{provider_id}/{model_id}"

        self._proc: subprocess.Popen[str] | None = None
        self._server_log_file = None
        self._owns_local_server = not bool(base_url)

        if base_url:
            self._server = OpenCodeServerConfig(
                base_url=_normalize_base_url(base_url),
                username=(username or "opencode").strip() or "opencode",
                password=(password or "").strip(),
            )
        else:
            self._server = self._start_local_server(
                repo=repo, server_log_path=self._server_log_path, username=username
            )

        try:
            self._wait_for_health()
            self._session_id = self._create_session(title=self._session_title)
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
        self._stop_local_server()
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None

    def _stop_local_server(self) -> None:
        """Best-effort stop for locally-owned OpenCode server process."""
        if self._proc is not None:
            try:
                # Do not block shutdown on a potentially wedged server.
                self._request_json("POST", "/instance/dispose", body=None, require_auth=True, timeout_seconds=5)
            except Exception:
                pass
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
            finally:
                self._proc = None
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None

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
        trace: list[dict[str, Any]] = []
        last_msg: Any | None = None
        for turn_idx in range(20):
            try:
                msg = self._post_message_with_retry(model=self._model_obj, text=prompt)
            except OpenCodeRequestError as e:
                # Compatibility fallback: some builds may accept model as a string.
                if e.status in (400, 422):
                    msg = self._post_message_with_retry(model=self._model_str, text=prompt)
                else:
                    raise
            last_msg = msg

            err = _extract_opencode_error(msg)
            if err:
                raise RuntimeError(f"OpenCode agent error: {err}")

            assistant_text = _extract_assistant_text(msg)
            calls = parse_tool_calls(assistant_text)
            if not calls:
                trace.append(
                    {
                        "turn": int(turn_idx + 1),
                        "assistant_text_tail": tail(assistant_text or "", 4000),
                        "calls": [],
                        "results": [],
                    }
                )
                return AgentResult(assistant_text=assistant_text, raw=msg, tool_trace=trace)

            results = execute_tool_calls(calls, repo=self._repo, policy=policy)
            compact_results: list[dict[str, Any]] = []
            for r in results:
                detail = dict(r.detail or {})
                if isinstance(detail.get("content"), str):
                    detail["content"] = tail(detail["content"], 4000)
                if isinstance(detail.get("stdout"), str):
                    detail["stdout"] = tail(detail["stdout"], 4000)
                if isinstance(detail.get("stderr"), str):
                    detail["stderr"] = tail(detail["stderr"], 4000)
                compact_results.append(detail | {"tool": r.kind, "ok": bool(r.ok)})
            trace.append(
                {
                    "turn": int(turn_idx + 1),
                    "assistant_text_tail": tail(assistant_text or "", 4000),
                    "calls": [
                        {
                            "kind": str(c.kind),
                            "payload": c.payload if isinstance(c.payload, (dict, list, str, int, float, bool)) else str(c.payload),
                        }
                        for c in calls
                    ],
                    "results": compact_results,
                }
            )

            # For scaffold runs, we don't need the agent to "finish talking" if the contract
            # is already valid. Some models keep emitting extra tool calls indefinitely.
            if str(purpose or "").strip().lower() == "scaffold_contract" and self._pipeline_rel:
                try:
                    from .pipeline_spec import load_pipeline_spec  # local import to avoid overhead for non-scaffold runs
                    from .scaffold_validation import validate_scaffold_contract

                    pipeline_path = (self._repo / self._pipeline_rel).resolve()
                    if pipeline_path.exists():
                        parsed = load_pipeline_spec(pipeline_path)
                        report = validate_scaffold_contract(self._repo, pipeline=parsed, require_metrics=True)
                        if report.ok:
                            return AgentResult(assistant_text=assistant_text, raw=msg, tool_trace=trace)
                except Exception:
                    pass

            prompt = format_tool_results(results)

        raise RuntimeError("OpenCode tool loop exceeded 20 turns without a final response.")

    def _start_local_server(
        self,
        *,
        repo: Path,
        server_log_path: Path | None,
        username: str | None,
        append_log: bool = False,
    ) -> OpenCodeServerConfig:
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
            if self._server_log_file is not None:
                try:
                    self._server_log_file.close()
                except Exception:
                    pass
            self._server_log_file = server_log_path.open(
                "a" if append_log else "w",
                encoding="utf-8",
            )
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

    def _sleep_retry_backoff(self, *, attempt_idx: int) -> None:
        """中文说明：
        - 含义：按指数退避等待下一次 OpenCode 请求重试。
        - 内容：delay = base * 2^(attempt_idx-1)，并限制到 30 秒，避免无上限等待。
        - 可简略：可能（简单策略函数；但集中处理更便于调参与测试）。
        """
        base = float(self._request_retry_backoff_seconds or 0.0)
        if base <= 0:
            return
        delay = min(30.0, base * (2 ** max(0, int(attempt_idx) - 1)))
        if delay > 0:
            time.sleep(delay)

    def _sleep_session_recover_backoff(self, *, recover_idx: int) -> None:
        base = float(self._session_recover_backoff_seconds or 0.0)
        if base <= 0:
            return
        delay = min(30.0, base * (2 ** max(0, int(recover_idx) - 1)))
        if delay > 0:
            time.sleep(delay)

    def _is_transport_unavailable_error(self, err: OpenCodeRequestError) -> bool:
        if err.status is not None:
            return False
        return _looks_like_transport_unavailable(err.detail)

    def _recover_local_server_session(self, *, reason: str) -> None:
        """Restart local OpenCode server and create a fresh session."""
        if not self._owns_local_server:
            raise RuntimeError("session_recover_not_local_server")
        username = (
            str(getattr(self, "_server", None).username).strip()
            if getattr(self, "_server", None) is not None
            else "opencode"
        ) or "opencode"
        self._stop_local_server()
        self._server = self._start_local_server(
            repo=self._repo,
            server_log_path=self._server_log_path,
            username=username,
            append_log=True,
        )
        self._wait_for_health()
        self._session_id = self._create_session(title=self._session_title)

    def _should_retry_request_error(self, err: OpenCodeRequestError) -> bool:
        """中文说明：
        - 含义：判断 OpenCode 请求错误是否属于可重试类别。
        - 内容：网络类（status=None）以及常见瞬时 HTTP 错误（408/409/425/429/5xx）会重试。
        - 可简略：可能（也可在调用点内联；集中定义更可维护）。
        """
        if err.status is None:
            return True
        try:
            code = int(err.status)
        except Exception:
            return True
        if code in (408, 409, 425, 429):
            return True
        return code >= 500

    def _clip_prompt_text(self, text: str) -> str:
        """中文说明：
        - 含义：按配置裁剪超长 prompt（保留头尾），避免超过服务端可接受上下文。
        - 内容：未设置 `max_prompt_chars` 时不裁剪；裁剪时插入标记便于定位。
        - 可简略：可能（可直接尾裁剪；但头尾保留更稳妥）。
        """
        s = str(text or "")
        cap = self._max_prompt_chars
        if cap is None or cap <= 0 or len(s) <= cap:
            return s
        if cap < 128:
            return s[-cap:]
        marker = "\n...[TRUNCATED_FOR_OPENCODE_CONTEXT]...\n"
        head = max(32, cap // 2)
        tail_keep = max(32, cap - head - len(marker))
        return s[:head] + marker + s[-tail_keep:]

    def _post_message_with_retry(self, *, model: Any, text: str) -> Any:
        """中文说明：
        - 含义：发送 message 请求并在超时/瞬时错误时自动重试。
        - 内容：支持 `request_retry_attempts` + 指数退避；若 `contextLength` 字段不被服务端接受，会自动降级重发一次。
        - 可简略：否（是提升 scaffold/repair 稳定性的关键逻辑）。
        """
        attempts = 1 + int(self._request_retry_attempts or 0)
        include_context = True
        last_err: OpenCodeRequestError | None = None
        recover_budget = int(self._session_recover_attempts or 0) if self._owns_local_server else 0
        recover_tries = 0

        for attempt in range(1, attempts + 1):
            try:
                return self._post_message(model=model, text=text, include_context=include_context)
            except OpenCodeRequestError as e:
                last_err = e
                # Some OpenCode builds may reject unknown fields; degrade gracefully.
                if include_context and self._context_length is not None and e.status in (400, 422):
                    include_context = False
                    try:
                        return self._post_message(model=model, text=text, include_context=False)
                    except OpenCodeRequestError as e2:
                        last_err = e2
                        e = e2
                if self._is_transport_unavailable_error(e) and recover_tries < recover_budget:
                    recover_tries += 1
                    try:
                        self._recover_local_server_session(reason=e.detail)
                    except Exception as recover_exc:
                        last_err = OpenCodeRequestError(
                            method=e.method,
                            url=e.url,
                            status=e.status,
                            detail=f"{e.detail}; recover_failed: {tail(str(recover_exc), 1200)}",
                        )
                    else:
                        self._sleep_session_recover_backoff(recover_idx=recover_tries)
                        continue
                if attempt >= attempts or not self._should_retry_request_error(e):
                    raise
                self._sleep_retry_backoff(attempt_idx=attempt)

        if last_err is not None:
            raise last_err
        raise RuntimeError("opencode_retry_failed_without_error")

    def _post_message(self, *, model: Any, text: str, include_context: bool = True) -> Any:
        """中文说明：
        - 含义：向当前 session 发送一条消息（prompt），返回 server 的 message JSON。
        - 内容：body 里包含 agent 类型、model、parts；由 `_request_json` 完成 HTTP 细节与错误包装。
        - 可简略：可能（薄封装；但把请求体结构集中在一处更易维护）。
        """
        clipped_text = self._clip_prompt_text(text)
        body = {
            "agent": "build",
            "model": model,
            "parts": [{"type": "text", "text": clipped_text}],
        }
        if include_context and self._context_length is not None:
            # Best-effort: different OpenCode versions may ignore this field.
            body["contextLength"] = int(self._context_length)
        data = self._request_json(
            "POST",
            f"/session/{self._session_id}/message",
            body=body,
            require_auth=bool(self._server.password),
        )
        # Some OpenCode builds/transports may respond with 200 + empty body; treat it as a transient transport failure
        # so the caller can retry or recover the local session instead of silently returning "None".
        if data is None:
            url = f"{self._server.base_url}/session/{self._session_id}/message"
            raise OpenCodeRequestError(method="POST", url=url, status=None, detail="connection closed: empty_response_body")
        return data

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
        except (TimeoutError, socket.timeout) as e:
            raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"timeout: {e}")
        except OSError as e:
            raise OpenCodeRequestError(method=method, url=url, status=None, detail=f"os_error: {e}")
