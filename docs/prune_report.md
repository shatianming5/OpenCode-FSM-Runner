# Prune Report（可省略/无关多余项清单）

生成时间：2026-02-04  
基于提交：`1cc4a19`（本地工作区）  

> 本报告的“可省略/无关多余”是**相对于本项目的核心目标**（一个 OpenCode 驱动的闭环 runner：plan→execute→verify→metrics）而言：  
> - **Core**：实现闭环所必需（删了就跑不起来或核心语义缺失）  
> - **Entry**：入口/包装（可用更少入口替代）  
> - **Optional**：可选能力（按需保留；删了不会影响核心，但会减少功能面）  
> - **Dev/Test**：仅开发/测试需要（发布/部署时可不带）  
> - **Docs/Examples**：说明/模板（发布时可不带；但利于集成）  
> - **Legacy/Deprecated**：历史兼容/旧设计文档（可删或迁移到外部存档）  

本轮约束：**只列清单，不做删除/重构**（你如果要我真正瘦身，我会按本报告的 “建议动作” 分批提交）。

---

## 1) 项目入口与“核心闭环”定义

### 1.1 运行入口（Entry Points）

- `runner/__main__.py`：`python -m runner` 入口，调用 `runner/cli.py:main()`  
- `fsm_runner.py`：脚本包装入口，调用 `runner/cli.py:main()`（与 `python -m runner` 功能重叠）  
- `runner/env_local.py`：同机调用 API（供其它 Python 项目 import 调用 rollout/evaluation）  

### 1.2 核心闭环链路（Core Path）

从 CLI 跑通闭环的最小链路：

1) `runner/cli.py`：解析参数、准备 repo、加载/自动 scaffold pipeline、组装 `RunnerConfig`  
2) `runner/runner.py`：主 FSM（scaffold 合同→preflight→plan update→execute step→verify→fix/replan→mark done）  
3) `runner/opencode_client.py`：与 OpenCode server 通讯、驱动 agent + tool-call loop  
4) `runner/opencode_tooling.py`：解析/执行工具调用（文件读写/命令执行）+ 安全策略  
5) `runner/pipeline_spec.py`：pipeline 合同 schema（v1）  
6) `runner/pipeline_verify.py`：按 pipeline 执行 tests/deploy/rollout/evaluation/benchmark + metrics 校验 + artifacts 落盘  
7) `runner/security.py`：命令 allow/deny 与 unattended（strict/guided）策略  
8) `runner/plan_format.py`：PLAN.md 的机器可解析格式与解析器  
9) `runner/snapshot.py`：采样 repo 状态并喂给 agent  
10) `runner/state.py`：`.aider_fsm/state.json` 与 `.aider_fsm/logs/*.jsonl`  

---

## 2) `runner/` 模块地图（职责 + 对外 API）

> 这里只列对外（非 `_` 前缀）符号；完整细节见对应文件。

- `runner/cli.py`：**CLI** 参数解析与入口（对外：`main`）  
- `runner/runner.py`：**闭环 FSM**（对外：`RunnerConfig`, `run`）  
- `runner/opencode_client.py`：OpenCode server 客户端（对外：`OpenCodeClient`, `OpenCodeRequestError`, `select_bash_mode`）  
- `runner/opencode_tooling.py`：tool-call 解析/执行（对外：`parse_tool_calls`, `execute_tool_calls`, `ToolPolicy` 等）  
- `runner/pipeline_spec.py`：pipeline v1 schema（对外：`PipelineSpec`, `load_pipeline_spec`）  
- `runner/pipeline_verify.py`：pipeline 执行器（对外：`run_pipeline_verification`, `stage_rc`, `fmt_stage_tail`）  
- `runner/security.py`：命令策略（对外：`cmd_allowed`, `looks_interactive`, `safe_env`）  
- `runner/plan_format.py`：PLAN.md 格式（对外：`ensure_plan_file`, `parse_plan`, `parse_next_step` 等）  
- `runner/snapshot.py`：snapshot（对外：`build_snapshot`, `get_git_changed_files`, `git_checkout`, `non_plan_changes`）  
- `runner/state.py`：state/log IO（对外：`ensure_dirs`, `load_state`, `save_state`, `append_jsonl` 等）  
- `runner/bootstrap.py`：`.aider_fsm/bootstrap.yml`（对外：`load_bootstrap_spec`, `run_bootstrap`）  
- `runner/actions.py`：`.aider_fsm/actions.yml`（对外：`load_actions_spec`, `run_pending_actions`）  
- `runner/repo_resolver.py`：repo 解析（本地路径 / git clone / GitHub zip fallback / HF dataset 下载）（对外：`prepare_repo`, `is_probably_repo_url`）  
- `runner/env_local.py`：同机调用 API（对外：`open_env`, `rollout_and_evaluate` 等）  
- `runner/dotenv.py`：简化版 dotenv loader（对外：`load_dotenv`）  
- `runner/paths.py` / `runner/subprocess_utils.py` / `runner/types.py`：基础工具与类型  
- `runner/agent_client.py`：Agent 协议与返回类型（对外：`AgentClient`, `AgentResult`）  

---

## 3) Import 证据（哪些模块“只为可选入口服务”）

用静态 import 图（基于本 repo tracked files）得到：

### 3.1 CLI 与 `env_local` 的可达模块

- **CLI & env_local 共享（12 个）**：  
  `runner.agent_client`, `runner.bootstrap`, `runner.opencode_client`, `runner.opencode_tooling`, `runner.paths`, `runner.pipeline_spec`, `runner.pipeline_verify`, `runner.prompts`, `runner.repo_resolver`, `runner.security`, `runner.subprocess_utils`, `runner.types`
- **仅 CLI 侧需要（7 个）**：  
  `runner.actions`, `runner.cli`, `runner.dotenv`, `runner.plan_format`, `runner.runner`, `runner.snapshot`, `runner.state`
- **仅 env_local 侧需要（1 个）**：  
  `runner.env_local`

> 结论：如果你的最终形态只要 “CLI runner” 或只要 “库 API”，可以按 reachability 拆包；本轮只做清单不拆。

---

## 4) “可省略/无关多余”候选清单（逐文件覆盖，不漏项）

说明：
- “可省略”不等于“必须删”；只是指出：**对核心闭环不是必须**，可在瘦身/拆包/发布时移除。  
- 风险等级：Low（安全移除，顶多少文档/少测试）/ Medium（会减少某项功能）/ High（会破核心或大量行为）。

### 4.1 Root 级文件

| Path | 角色 | 可省略? | 风险 | 理由 / 影响 | 建议动作 |
|---|---|---:|---|---|---|
| `.gitignore` | Dev | 否 | High | 影响 repo hygiene | 保留 |
| `requirements.txt` | Core | 否 | High | 运行依赖 | 保留 |
| `README.md` | Docs | 是 | Low | 运行不依赖 | 若需精简发布，可放到外部文档 |
| `fsm_runner.py` | Entry | 是 | Low | 只是 `runner.cli:main` 的包装；`python -m runner` 已足够 | 若只保留一种入口，可删除该脚本 |

### 4.2 `runner/`（核心实现）

| Path | 角色 | 可省略? | 风险 | 理由 / 影响 | 建议动作 |
|---|---|---:|---|---|---|
| `runner/__init__.py` | Core | 否 | High | 包定义 | 保留 |
| `runner/__main__.py` | Entry | 视情况 | Medium | 仅用于 `python -m runner`；若你只用 `fsm_runner.py`（或反之）可二选一 | 统一入口后删掉另一个 |
| `runner/cli.py` | Core | 否 | High | CLI 主入口 | 保留 |
| `runner/runner.py` | Core | 否 | High | 闭环 FSM | 保留 |
| `runner/opencode_client.py` | Core | 否 | High | OpenCode agent/serve 管理与 HTTP 调用 | 保留 |
| `runner/opencode_tooling.py` | Core | 否 | High | tool-call 解析与安全执行 | 保留 |
| `runner/pipeline_spec.py` | Core | 否 | High | pipeline v1 schema | 保留 |
| `runner/pipeline_verify.py` | Core | 否 | High | pipeline 执行与 metrics 校验 | 保留 |
| `runner/security.py` | Core | 否 | High | 安全策略（deny/allow/unattended） | 保留 |
| `runner/plan_format.py` | Core | 否 | High | PLAN.md 机器可解析格式 | 保留 |
| `runner/snapshot.py` | Core | 否 | High | snapshot（给 agent 的事实输入） | 保留 |
| `runner/state.py` | Core | 否 | High | state/log 持久化 | 保留 |
| `runner/bootstrap.py` | Optional | 是 | Medium | 可选 `.aider_fsm/bootstrap.yml`；删掉会失去 repo-owned env setup 能力 | 需要最小化时可拆到 optional 包 |
| `runner/actions.py` | Optional | 是 | Medium | 可选 `.aider_fsm/actions.yml`；删掉会失去“失败后由模型请求 actions 修复环境”的能力 | 需要最小化时可拆到 optional 包 |
| `runner/repo_resolver.py` | Core | 否（但可裁剪子能力） | Medium | repo URL → 本地路径；HF/GitHub fallback 属于增强功能 | 可把 HF 支持/zip fallback 标成可选插件 |
| `runner/env_local.py` | Optional | 是 | Medium | “同机调用 rollout/evaluation API”；删掉不影响 CLI 闭环 | 若只要 CLI，可移除或拆包 |
| `runner/dotenv.py` | Optional | 是 | Low | 只为 CLI `--env-file` 服务；可改成依赖 python-dotenv 或完全不加载 | 若要瘦身，可移除并在 CLI 侧降级 |
| `runner/paths.py` | Core | 否 | High | workdir/path 约束 | 保留 |
| `runner/subprocess_utils.py` | Core | 否 | High | 命令执行与 artifacts 写入 | 保留 |
| `runner/types.py` | Core | 否 | High | 类型定义 | 保留 |
| `runner/prompts.py` | Core | 否（但可裁剪部分 prompt） | Medium | scaffolding/plan/execute/fix 等 prompt 集合 | 如果你去掉 scaffold 或 plan loop，可按功能裁剪 |
| `runner/agent_client.py` | Core | 否 | High | Agent 协议 | 保留 |

### 4.3 `docs/`（文档）

| Path | 角色 | 可省略? | 风险 | 理由 / 影响 | 建议动作 |
|---|---|---:|---|---|---|
| `docs/overview.md` | Docs | 是 | Low | 运行不依赖 | 发布瘦身可移出 repo |
| `docs/integration.md` | Docs | 是 | Low | 运行不依赖 | 同上 |
| `docs/pipeline_spec.md` | Docs | 是 | Low | 运行不依赖 | 同上 |
| `docs/bootstrap_spec.md` | Docs | 是 | Low | 运行不依赖 | 同上 |
| `docs/metrics_schema.md` | Docs | 是 | Low | 运行不依赖 | 同上 |
| `docs/security_model.md` | Docs | 是 | Low | 运行不依赖 | 同上 |

已删除的 legacy 文档（不再存在于 repo 中）：

- `docs/legacy_proposal.md`：Aider 时代 proposal（与当前 OpenCode runner 不一致）

### 4.4 `examples/`（模板）

| Path | 角色 | 可省略? | 风险 | 理由 / 影响 | 建议动作 |
|---|---|---:|---|---|---|
| `examples/pipeline.example.yml` | Examples | 是 | Low | 运行不依赖 | 发布瘦身可移出 |
| `examples/pipeline.benchmark_skeleton.yml` | Examples | 是 | Low | 运行不依赖 | 同上 |
| `examples/actions.example.yml` | Examples | 是 | Low | 运行不依赖 | 同上 |
| `examples/bootstrap.example.yml` | Examples | 是 | Low | 运行不依赖 | 同上 |

### 4.5 `tests/`（测试）

> 所有 `tests/*` 都属于 Dev/Test，可在“发布/部署”时整体不带。  
> 但它们对保障安全性与契约稳定性很关键（例如防止写 pipeline、保证 scaffold 合同可解析）。

| Path | 角色 | 可省略? | 风险 | 理由 / 影响 | 建议动作 |
|---|---|---:|---|---|---|
| `tests/conftest.py` | Dev/Test | 是 | Low | 测试路径注入 | 发布瘦身可移出 |
| `tests/test_actions.py` | Dev/Test | 是 | Low | 覆盖 `.aider_fsm/actions.yml` 执行与安全 | 同上 |
| `tests/test_bootstrap.py` | Dev/Test | 是 | Low | 覆盖 `.aider_fsm/bootstrap.yml` 解析/执行 | 同上 |
| `tests/test_cli.py` | Dev/Test | 是 | Low | 覆盖 CLI 对 pipeline 自动发现 | 同上 |
| `tests/test_dotenv.py` | Dev/Test | 是 | Low | 覆盖 dotenv loader | 同上 |
| `tests/test_env_local.py` | Dev/Test | 是 | Low | 覆盖同机调用 API | 同上 |
| `tests/test_no_repo_specific_refs.py` | Dev/Test | 是 | Low | 防止 repo 侧硬编码残留 | 同上 |
| `tests/test_opencode_bash_override.py` | Dev/Test | 是 | Low | 覆盖 scaffold bash 权限选择 | 同上 |
| `tests/test_opencode_http_client.py` | Dev/Test | 是 | Low | 覆盖 OpenCode HTTP 客户端错误处理 | 同上 |
| `tests/test_opencode_tooling.py` | Dev/Test | 是 | Low | 覆盖 tool-call 解析与执行策略 | 同上 |
| `tests/test_pipeline.py` | Dev/Test | 是 | Low | 覆盖 pipeline schema/verify/metrics 校验 | 同上 |
| `tests/test_plan_parse.py` | Dev/Test | 是 | Low | 覆盖 PLAN.md 解析器 | 同上 |
| `tests/test_repo_resolver.py` | Dev/Test | 是 | Low | 覆盖 repo URL 解析（含 GitHub fallback/HF dataset） | 同上 |
| `tests/test_runner_agent_integration.py` | Dev/Test | 是 | Low | 覆盖 runner 对“非法文件改动”的回滚保护 | 同上 |
| `tests/test_scaffold_flow_agent_first_fallback.py` | Dev/Test | 是 | Low | 覆盖 scaffold 合同的成功/失败快速退出 | 同上 |
| `tests/test_snapshot.py` | Dev/Test | 是 | Low | 覆盖 snapshot 输出 | 同上 |
| `tests/test_state_io.py` | Dev/Test | 是 | Low | 覆盖 state/log IO | 同上 |

---

## 5) 代码级“多余/可合并/可裁剪”点（不删，只列出来）

### 5.1 明确重复（Duplicate）

1) **模型选择逻辑重复**  
   - `runner/cli.py`：`_list_opencode_models()` + `_resolve_model()`  
   - `runner/env_local.py`：`_list_opencode_models()` + `_resolve_model()`  
   影响：改一处容易忘另一处；可抽到 `runner/model_resolver.py` 之类的共享模块（或并入 `opencode_client`）。

2) **scaffold 合同的 pipeline 校验逻辑重复**  
   - `runner/runner.py`：scaffold 后对 `pipeline.yml` 的“必需字段/metrics 约束”校验  
   - `runner/env_local.py`：`_validate_scaffolded_pipeline()`  
   影响：两套校验标准可能漂移；建议抽成共享函数（例如 `runner/scaffold_validation.py`）。

3) **OPENAI base URL 兼容逻辑分散**  
   - `runner/cli.py`：从 `OPENAI_API_BASE` 推导 `OPENAI_BASE_URL` 且做 reachable 探测  
   - `runner/opencode_client.py:_start_local_server`：也做 `OPENAI_API_BASE` → `OPENAI_BASE_URL` 的兼容  
   影响：行为不一致；建议统一到一个地方，CLI 与 server 启动都复用。

### 5.2 Legacy/Deprecated（可移除的“兼容负担”）

1) **CLI 中已废弃的 flags**（仍保留解析以避免 silent break）  
   - `runner/cli.py`：`--ensure-tools/--ensure-kind/--kind-*/--full-quickstart`  
   建议：如果你不需要兼容旧用法，可以删除这些参数与分支。

2) **pipeline schema 中的 tooling 字段仍存在但不再被 runner 使用**  
   - `runner/pipeline_spec.py`：`tooling_ensure_tools`, `tooling_ensure_kind_cluster`, `tooling_kind_*`  
   建议：如果你确认不再支持旧 schema，可移除这些字段与解析逻辑，减少 schema 表面积。

### 5.3 可选子功能（Optional Sub-features）

1) `runner/repo_resolver.py` 的 HuggingFace dataset URL 支持  
   - 若你只跑 Git repo，可以删掉 `_parse_hf_dataset/_download_hf_dataset_snapshot` 相关逻辑。  
2) GitHub ZIP fallback（在 `git clone` 失败时）  
   - 若你运行环境保证 `git clone` 可用，可移除 `_archive_clone_github/_extract_github_zip`。  
3) `.aider_fsm/bootstrap.yml` 与 `.aider_fsm/actions.yml`  
   - 如果你只想要“纯验收，不做环境修复/自动 bootstrap”，可整体移除 `runner/bootstrap.py`、`runner/actions.py` 与 runner 内相关调用点。

---

## 6) 本地非 tracked 的“多余物”（可直接清理，不属于代码）

这些不在 `git ls-files` 中，但在工作区可能存在：

- `.venv/`：本地虚拟环境目录（应在 `.gitignore`；可随时删除重建）  
- `.pytest_cache/`：pytest 缓存目录（可删除）  
- `runner/__pycache__/`, `tests/__pycache__/`：pyc 缓存（可删除）  
- `.env`：**强建议不要提交到 git**（放到 `.gitignore` / 用示例文件替代）  

---

## 7) 如果你下一步要真正“删/拆/瘦身”，推荐的验证方式（仅建议）

按你要瘦身的方向，逐步删除并验证：

1) 只删 docs/examples/tests：跑 `pytest -q`（或至少 `python -m runner --help`）  
2) 去掉 `fsm_runner.py`（只保留 `python -m runner`）：跑一个最小 repo preflight（已有 tests 覆盖）  
3) 去掉 `env_local`：删 `runner/env_local.py` 与相关 tests，跑 `pytest -q`  
4) 去掉 HF / GitHub ZIP fallback：用本地 repo + 正常 git URL 跑一次 `--preflight-only`  
5) 合并重复逻辑：保证 `pytest -q` + 基于一个真实 remote repo 的 smoke（`--preflight-only`）  
