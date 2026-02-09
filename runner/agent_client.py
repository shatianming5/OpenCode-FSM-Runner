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
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈10 行；引用次数≈9（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/agent_client.py:14；类型=class；引用≈9；规模≈10行

    assistant_text: str
    raw: Any | None = None
    tool_trace: list[dict[str, Any]] | None = None


class AgentClient(Protocol):
    """中文说明：
    - 含义：Runner 与“模型/agent 后端”的最小协议接口。
    - 内容：Runner 只依赖 `run()`（一次对话/一次任务）与 `close()`（资源回收）；具体实现可对接 OpenCode server 或其它实现。
    - 可简略：否（这是解耦边界；去掉会导致 runner 强绑定某个实现）。
    """
    # 作用：中文说明：
    # 能否简略：否
    # 原因：规模≈24 行；引用次数≈6（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
    # 证据：位置=runner/agent_client.py:26；类型=class；引用≈6；规模≈24行

    def run(self, text: str, *, fsm_state: str, iter_idx: int, purpose: str) -> AgentResult:
        """中文说明：
        - 含义：向 agent 发送 prompt 并拿到 assistant 回复。
        - 内容：`fsm_state/iter_idx/purpose` 用于日志与策略（例如 scaffold 合同阶段允许更高 bash 权限）。
        - 可简略：否
        - 原因：这是 runner 与 agent 后端解耦的核心协议；这些参数会影响审计/权限边界与可追溯性，简化或改签名容易导致行为漂移。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈8 行；引用次数≈29（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/agent_client.py:33；类型=method；引用≈29；规模≈8行

        ...

    def close(self) -> None:
        """中文说明：
        - 含义：关闭/清理 agent 相关资源（如本地 server 进程/文件句柄）。
        - 内容：通常在 runner 退出或完成一次独立任务后调用。
        - 可简略：部分
        - 原因：部分后端实现可以是 no-op，但协议层需要显式的资源回收入口以避免进程/连接泄漏。
        """
        # 作用：中文说明：
        # 能否简略：否
        # 原因：规模≈8 行；引用次数≈10（静态近似，可能包含注释/字符串）；多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性
        # 证据：位置=runner/agent_client.py:42；类型=method；引用≈10；规模≈8行

        ...
