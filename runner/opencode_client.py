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


def select_bash_mode(*, purpose: str, default_bash_mode: str, scaffold_bash_mode: str) -> str:
    """中文说明：
    - 含义：根据用途选择 tool-call 的 bash 权限模式（restricted/full）。
    - 内容：当 purpose==scaffold_contract 时允许用单独的 scaffold_bash_mode（通常更宽松以便生成合同文件）；其它情况用 default_bash_mode。
    - 可简略：可能（小策略函数；但把规则集中有利于审计）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈12 行；引用次数≈8（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/opencode_client.py:89；类型=function；引用≈8；规模≈12行
    p = str(purpose or "").strip().lower()
    default = str(default_bash_mode or "restricted").strip().lower() or "restricted"
    scaffold = str(scaffold_bash_mode or default).strip().lower() or default
    if p in ("scaffold_contract", "repair_contract"):
        return scaffold
    return default


@dataclass(frozen=True)
class OpenCodeServerConfig:
    """中文说明：
    - 含义：OpenCode server 连接信息（base_url + 基本认证）。
    - 内容：当 runner 自己启动本地 server 时会生成随机 password；也支持连接外部 server（用户提供 base_url/username/password）。
    - 可简略：可能（字段很少；但作为显式结构便于传递与测试）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈10 行；引用次数≈4（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/opencode_client.py:150；类型=class；引用≈4；规模≈10行

    base_url: str
    username: str
    password: str


class OpenCodeRequestError(RuntimeError):
    """中文说明：
    - 含义：OpenCode HTTP 请求失败时抛出的异常（带 method/url/status/detail）。
    - 内容：用于把 HTTPError/URLError 统一包装成稳定异常类型，方便上层做兼容性回退或报错。
    - 可简略：否（稳定错误类型对诊断与测试很重要）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈18 行；引用次数≈18（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/opencode_client.py:162；类型=class；引用≈18；规模≈18行

    def __init__(self, *, method: str, url: str, status: int | None, detail: str):
        """中文说明：
        - 含义：构造带上下文信息的请求异常。
        - 内容：把 method/url/status/detail 写入属性，便于调用方判断（如 400/422 时做 model 字段兼容回退）。
        - 可简略：部分
        - 原因：实现简单但保留结构化字段能显著提升诊断与兼容性回退（例如按 status 做分支）。
        """
        # 作用：中文说明：
        # 能否简略：是
        # 原因：规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
        # 证据：位置=runner/opencode_client.py:169；类型=method；引用≈1；规模≈11行
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
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈542 行；引用次数≈21（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/opencode_client.py:182；类型=class；引用≈21；规模≈542行

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
        - 可简略：否
        - 原因：涉及进程管理/端口选择/认证、兼容性回退与关键默认值；简化容易引入资源泄漏或连接不稳定。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈105 行；引用次数≈1（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/opencode_client.py:211；类型=method；引用≈1；规模≈105行
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

        model_str = str(model or "").strip()
        if not model_str:
            provider_id, model_id = "openai", "gpt-4o-mini"
        elif "/" in model_str:
            provider_id, model_id = model_str.split("/", 1)
            provider_id = provider_id.strip() or "openai"
            model_id = model_id.strip() or "gpt-4o-mini"
        else:
            provider_id, model_id = "openai", model_str
        self._model_obj: dict[str, str] = {"providerID": provider_id, "modelID": model_id}
        self._model_str: str = f"{provider_id}/{model_id}"

        self._proc: subprocess.Popen[str] | None = None
        self._server_log_file = None
        self._owns_local_server = not bool(base_url)

        if base_url:
            base_url_s = str(base_url).strip()
            if not base_url_s:
                raise ValueError("empty_url")
            self._server = OpenCodeServerConfig(
                base_url=base_url_s.rstrip("/"),
                username=(username or "opencode").strip() or "opencode",
                password=(password or "").strip(),
            )
        else:
            self._server = self._start_local_server(
                repo=repo, server_log_path=self._server_log_path, username=username
            )

        try:
            deadline = time.time() + 60
            last_err = ""
            while time.time() < deadline:
                try:
                    self._request_json("GET", "/global/health", body=None, require_auth=bool(self._server.password))
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(0.2)
            else:
                raise RuntimeError(f"OpenCode server failed health check: {tail(last_err, 2000)}")

            data = self._request_json(
                "POST",
                "/session",
                body={"title": self._session_title},
                require_auth=bool(self._server.password),
            )
            if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
                self._session_id = data["id"]
            elif isinstance(data, dict) and isinstance(data.get("sessionID"), str) and data["sessionID"].strip():
                self._session_id = data["sessionID"]
            else:
                raise RuntimeError(
                    f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}"
                )
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
        - 可简略：否
        - 原因：资源回收语义关键，必须避免本地 server 进程或文件句柄泄漏。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈13 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/opencode_client.py:295；类型=method；引用≈10；规模≈13行
        self._stop_local_server()
        if self._server_log_file is not None:
            try:
                self._server_log_file.close()
            except Exception:
                pass
            self._server_log_file = None

    def _stop_local_server(self) -> None:
        """中文说明：
        - 含义：best-effort 停止由当前 OpenCodeClient 启动的本地 server 进程。
        - 内容：先尝试调用 `/instance/dispose` 做优雅释放；随后 terminate/kill 进程组；最后关闭 server log 文件句柄。
        - 可简略：部分
        - 原因：可以抽出更小的进程清理 helper，但必须保留“优雅释放 + 强制 kill 兜底 + 不泄漏后台进程”的语义。
        """
        # 作用：Best-effort stop for locally-owned OpenCode server process.
        # 能否简略：部分
        # 原因：规模≈36 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/opencode_client.py:305；类型=method；引用≈2；规模≈36行
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
        - 可简略：否
        - 原因：tool loop 是“能实际改文件/跑命令并回灌结果”的闭环核心；简化会直接削弱 contract scaffold/repair 能力。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈95 行；引用次数≈29（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/opencode_client.py:346；类型=method；引用≈29；规模≈95行
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
        for turn_idx in range(20):
            try:
                msg = self._post_message_with_retry(model=self._model_obj, text=prompt)
            except OpenCodeRequestError as e:
                # Compatibility fallback: some builds may accept model as a string.
                if e.status in (400, 422):
                    msg = self._post_message_with_retry(model=self._model_str, text=prompt)
                else:
                    raise
            opencode_err = None
            if isinstance(msg, dict):
                info = msg.get("info")
                if isinstance(info, dict):
                    err_obj = info.get("error")
                    if isinstance(err_obj, dict):
                        name = str(err_obj.get("name") or "").strip() or "Error"
                        data = err_obj.get("data")
                        detail = ""
                        if isinstance(data, dict):
                            detail = str(data.get("message") or "").strip()
                        if not detail:
                            detail = str(data).strip() if data is not None else ""
                        opencode_err = f"{name}: {detail}" if detail else name
            if opencode_err:
                raise RuntimeError(f"OpenCode agent error: {opencode_err}")

            if not isinstance(msg, dict):
                assistant_text = str(msg)
            else:
                parts = msg.get("parts")
                if not isinstance(parts, list):
                    assistant_text = str(msg)
                else:
                    texts: list[str] = []
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            t = part["text"]
                            if t.strip():
                                texts.append(t)
                    assistant_text = "\n".join(texts) or str(msg)
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
                        if not report.errors:
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
        - 可简略：否
        - 原因：进程/端口/认证管理属于稳定性与安全关键路径；实现需要谨慎，简化容易导致端口冲突或认证失效。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈60 行；引用次数≈2（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/opencode_client.py:449；类型=method；引用≈2；规模≈60行
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

    def _post_message_with_retry(self, *, model: Any, text: str) -> Any:
        """中文说明：
        - 含义：发送 message 请求并在超时/瞬时错误时自动重试。
        - 内容：支持 `request_retry_attempts` + 指数退避；若 `contextLength` 字段不被服务端接受，会自动降级重发一次。
        - 可简略：否
        - 原因：这是提升 scaffold/repair 稳定性的关键逻辑（重试、退避、字段兼容回退、会话恢复）。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈46 行；引用次数≈4（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/opencode_client.py:612；类型=method；引用≈4；规模≈46行
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

                transport_unavailable = False
                if e.status is None:
                    d = str(e.detail or "").strip().lower()
                    if d:
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
                        transport_unavailable = any(n in d for n in needles)

                if transport_unavailable and recover_tries < recover_budget:
                    recover_tries += 1
                    try:
                        recover_fn = getattr(self, "_recover_local_server_session", None)
                        if callable(recover_fn):
                            recover_fn(reason=e.detail)
                        else:
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

                            deadline = time.time() + 60
                            last_health_err = ""
                            while time.time() < deadline:
                                try:
                                    self._request_json(
                                        "GET",
                                        "/global/health",
                                        body=None,
                                        require_auth=bool(self._server.password),
                                    )
                                    break
                                except Exception as health_exc:
                                    last_health_err = str(health_exc)
                                    time.sleep(0.2)
                            else:
                                raise RuntimeError(
                                    f"OpenCode server failed health check: {tail(last_health_err, 2000)}"
                                )

                            data = self._request_json(
                                "POST",
                                "/session",
                                body={"title": self._session_title},
                                require_auth=bool(self._server.password),
                            )
                            if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"].strip():
                                self._session_id = data["id"]
                            elif (
                                isinstance(data, dict)
                                and isinstance(data.get("sessionID"), str)
                                and data["sessionID"].strip()
                            ):
                                self._session_id = data["sessionID"]
                            else:
                                raise RuntimeError(
                                    f"unexpected /session response: {tail(json.dumps(data, ensure_ascii=False), 2000)}"
                                )
                    except Exception as recover_exc:
                        last_err = OpenCodeRequestError(
                            method=e.method,
                            url=e.url,
                            status=e.status,
                            detail=f"{e.detail}; recover_failed: {tail(str(recover_exc), 1200)}",
                        )
                    else:
                        sleep_fn = getattr(self, "_sleep_session_recover_backoff", None)
                        if callable(sleep_fn):
                            sleep_fn(recover_idx=recover_tries)
                        else:
                            base = float(self._session_recover_backoff_seconds or 0.0)
                            if base > 0:
                                delay = min(30.0, base * (2 ** max(0, int(recover_tries) - 1)))
                                if delay > 0:
                                    time.sleep(delay)
                        continue

                should_retry_fn = getattr(self, "_should_retry_request_error", None)
                if callable(should_retry_fn):
                    should_retry = bool(should_retry_fn(e))
                elif e.status is None:
                    should_retry = True
                else:
                    try:
                        code = int(e.status)
                    except Exception:
                        should_retry = True
                    else:
                        should_retry = code in (408, 409, 425, 429) or code >= 500

                if attempt >= attempts or not should_retry:
                    raise

                sleep_fn = getattr(self, "_sleep_retry_backoff", None)
                if callable(sleep_fn):
                    sleep_fn(attempt_idx=attempt)
                else:
                    base = float(self._request_retry_backoff_seconds or 0.0)
                    if base > 0:
                        delay = min(30.0, base * (2 ** max(0, int(attempt) - 1)))
                        if delay > 0:
                            time.sleep(delay)

        if last_err is not None:
            raise last_err
        raise RuntimeError("opencode_retry_failed_without_error")

    def _post_message(self, *, model: Any, text: str, include_context: bool = True) -> Any:
        """中文说明：
        - 含义：向当前 session 发送一条消息（prompt），返回 server 的 message JSON。
        - 内容：body 里包含 agent 类型、model、parts；由 `_request_json` 完成 HTTP 细节与错误包装。
        - 可简略：部分
        - 原因：属于薄封装，但把请求体结构（字段名/裁剪策略/contextLength）集中在一处更易维护与做兼容回退。
        """
        # 作用：中文说明：
        # 能否简略：部分
        # 原因：规模≈27 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/opencode_client.py:659；类型=method；引用≈2；规模≈27行
        s = str(text or "")
        cap = self._max_prompt_chars
        if cap is None or cap <= 0 or len(s) <= cap:
            clipped_text = s
        elif cap < 128:
            clipped_text = s[-cap:]
        else:
            marker = "\n...[TRUNCATED_FOR_OPENCODE_CONTEXT]...\n"
            head = max(32, cap // 2)
            tail_keep = max(32, cap - head - len(marker))
            clipped_text = s[:head] + marker + s[-tail_keep:]
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
        - 可简略：否
        - 原因：这是 HTTP I/O 与错误处理的核心封装，直接影响稳定性、可诊断性与上层的兼容性回退行为。
        """
        # 作用：中文说明：
        # 能否简略：部分
        # 原因：规模≈37 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
        # 证据：位置=runner/opencode_client.py:687；类型=method；引用≈4；规模≈37行
        url = f"{self._server.base_url}{path}"
        headers = {"Accept": "application/json"}
        if require_auth and self._server.password:
            token = base64.b64encode(
                f"{self._server.username}:{self._server.password}".encode("utf-8")
            ).decode("ascii")
            headers["Authorization"] = f"Basic {token}"

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
