from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentResult:
    """中文说明：
    - 含义：一次 agent 调用的返回结果（最小封装）。
    - 内容：包含 assistant 输出文本（用于解析 tool-calls 或作为最终回复），以及可选 raw（底层响应对象，便于调试）。
    - 可简略：可能（字段很少；但作为稳定边界类型有助于测试与替换不同 agent 后端）。
    """

    assistant_text: str
    raw: Any | None = None
    tool_trace: list[dict[str, Any]] | None = None


class AgentClient(Protocol):
    """中文说明：
    - 含义：Runner 与“模型/agent 后端”的最小协议接口。
    - 内容：Runner 只依赖 `run()`（一次对话/一次任务）与 `close()`（资源回收）；具体实现可对接 OpenCode server 或其它实现。
    - 可简略：否（这是解耦边界；去掉会导致 runner 强绑定某个实现）。
    """

    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：向 agent 发送 prompt 并拿到 assistant 回复。
        - 内容：`fsm_state/iter_idx/purpose` 用于日志与策略（例如 scaffold 合同阶段允许更高 bash 权限）。
        - 可简略：否（核心协议方法；参数也用于审计/安全策略）。
        """

        ...

    def close(self) -> None:
        """中文说明：
        - 含义：关闭/清理 agent 相关资源（如本地 server 进程/文件句柄）。
        - 内容：通常在 runner 退出或完成一次独立任务后调用。
        - 可简略：可能（某些实现可以是 no-op，但接口应保留）。
        """

        ...
