> NOTE (2026-02-03): 本文是 **Aider 集成**的历史 proposal。当前仓库已迁移为 **OpenCode agent + OpenCode server API** 驱动闭环。
> 以 `README.md` / `docs/overview.md` 为准；本文仅保留做设计背景参考。

下面给你一份**按“Python `Coder.run()` 驱动 FSM”**来做的、**最小不可分（MVP 级、但闭环完整）** proposal。核心目标：在**同一进程里保持上下文**，用一个**有限状态机**反复执行：

> **读计划+读repo状态 → 更新计划/选下一步 → 执行修改 → 验收（测试/检查）→ 失败则修复或改计划 → 直到 Done 或 Stop**

并且完全基于 aider 官方给出的 Python 脚本化入口：`Coder.create(...)` + `coder.run(...)`，以及可在 `coder.run()` 中执行斜杠命令（含 `/test`）。([Aider][1])

---

## 0) 一句话定义（你要构建的东西是什么）

**Aider-FSM Runner**：一个 Python 程序（单文件也行），把 aider 当成“可编辑+可自检”的执行器，用状态机把它组织成**可停机、可验收、可迭代更新计划**的闭环。

---

## 1) 最小不可分：系统边界与成败定义

### 1.1 成功（Done）的唯一判据

* **验收命令全绿**（最小先只支持一个 `TEST_CMD`）：例如 `pytest -q` / `npm test` / `cargo test` 等。
* 或者你明确写入 `PLAN.md` 的 Acceptance Checklist 全部满足（runner 会把 checklist 对应到命令/文件存在性）。

> 这里“正确性”= **相对于验收检查**的正确，不做形而上“绝对正确”。

### 1.2 非目标（MVP 刻意不做）

* 不追求“无限自治跑任意 shell 命令装依赖/改系统”。MVP 只允许验证命令（/test）+ 只在需要时把失败输出喂回修复。`/test` 本身就是为“跑命令→失败才把输出加回对话”设计的。([Aider][2])
* 不做复杂 UI、不做并发、不做多 repo。

### 1.3 关键现实约束（必须写进 proposal）

* **Python scripting API 不稳定**：官方明确说 python API 不被正式支持，未来可能不兼容。([Aider][1])
  → 所以 proposal 必须包含“版本锁定 + 适配层 + CLI 回退方案”。

---

## 2) 最小组件（必须有且仅有这些）

> 你要的是“最小不可分”，所以我只保留能跑通闭环的组件。

1. `fsm_runner.py`（唯一代码文件）
2. `PLAN.md`（人类可读 + 机器可解析的计划与进度）
3. `.aider_fsm/state.json`（机器状态：迭代次数、当前任务 id、失败次数、最近一次验收结果）
4. `.aider_fsm/logs/`（每轮日志，可选但强烈建议）

**不引入数据库、不引入复杂配置系统。** model/key 从环境变量或你现有方式拿即可。

---

## 3) 输入/输出契约（I/O spec）

### 输入

* `repo_path`：仓库根目录
* `goal`：一句话目标（写入 `PLAN.md` 顶部）
* `seed_files`：最少 1–3 个入口文件（或直接只给 `PLAN.md`，靠 aider 自己加文件也行）
* `test_cmd`：验收命令（字符串）
* `max_iters`：最大轮数
* `max_fix_attempts_per_step`：单步修复上限（防死循环）
* `model_name`：例如 `gpt-4-turbo`（示例来自官方文档）([Aider][1])

### 输出

* 更新后的 `PLAN.md`（含 Done/Next/Blocked/Notes）
* 仓库代码改动（git commit 可选；MVP 先不强制）
* `.aider_fsm/state.json`（最终状态 + 退出原因）
* `.aider_fsm/logs/run_*.jsonl`（每轮：prompt、aider 返回、测试结果摘要）

---

## 4) `PLAN.md` 作为“可执行计划”的最小格式（机器可解析）

你要让 FSM 能**不依赖模型输出的随意文本**来决定“下一步做啥”，所以 `PLAN.md` 必须有一个**稳定的机器段**。

推荐格式（最小可解析）：

```md
# PLAN

## Goal
- <一句话目标>

## Acceptance
- [ ] TEST_CMD passes: `<pytest -q>`

## Next (exactly ONE item)
- [ ] (STEP_ID=001) <一句话下一步>  // 只能有一条

## Backlog
- [ ] (STEP_ID=002) ...
- [ ] (STEP_ID=003) ...

## Done
- [x] (STEP_ID=000) 初始化计划文件

## Notes
- 最近失败原因/重要约束/不确定性
```

**FSM 规则（硬约束）**：

* `Next` 里**只能有一个** step；其余都在 Backlog。
* 每个 step 必须是**最小原子**：能在一次编辑+一次验收内闭合（不然就拆）。
* `Acceptance` 至少包含 `TEST_CMD`。

---

## 5) 状态机定义（FSM spec）

### 状态集合（MVP 6 状态）

1. **S0_BOOTSTRAP**：若无 `PLAN.md`，创建最小模板
2. **S1_SNAPSHOT**：采集 repo 状态（git diff、失败测试摘要、关键文件列表）
3. **S2_PLAN_UPDATE**：让 aider 更新 `PLAN.md`（只改计划，不改业务代码）
4. **S3_EXECUTE_STEP**：执行 `PLAN.md -> Next` 这一个 step（允许改业务代码）
5. **S4_VERIFY**：跑验收（`/test {TEST_CMD}` + runner 侧 subprocess 再确认一次）
6. **S5_DECIDE**：通过→标记 Done+挑下一步；失败→让 aider 修复或改计划；满足停止条件则退出

### 停止条件（必须有）

* 验收通过且 `Backlog` 为空 → Done
* `iter >= max_iters` → Stop（给出“最后状态 + 下一建议”）
* 单步修复失败次数超过 `max_fix_attempts_per_step` → 把该 step 标记 Blocked，并让模型在 Notes 写“阻塞原因+需要人类决策点”

---

## 6) 你要的“一轮”最小不可分闭环（逐步执行定义）

> 下面这套是你要求的“检查 plan+repo → 做事 → 检查是否完成 → 可改 plan/加 plan”的最小实现。

### Step A：S1_SNAPSHOT（外部收集事实）

runner 做这些（不让模型瞎猜）：

* `git status --porcelain`
* `git diff --stat`（可选）
* 读取 `PLAN.md`
* 读取 `.aider_fsm/state.json`
* （可选）快速探测：存在 `pytest.ini`/`package.json`/`pyproject.toml`，用于提示模型怎么跑测试

把这些合成一个 `snapshot_text`，作为每轮 prompt 的“事实输入”。

### Step B：S2_PLAN_UPDATE（只允许更新计划文件）

调用：

* `coder.run(<PLAN_UPDATE_PROMPT with snapshot_text>)`

硬约束写进 prompt：

* **只允许修改 `PLAN.md`**
* `Next` 必须只留一个最小 step
* 任何不确定点写入 Notes，并把 step 标记 Blocked/拆分

### Step C：S3_EXECUTE_STEP（执行 Next 的唯一 step）

runner 从 `PLAN.md` 解析出 `Next` 的 step 文本，然后调用：

* `coder.run(<EXECUTE_PROMPT that contains the step text + constraints>)`

执行后 runner 再做一次 `git diff --name-only`，把“改了哪些文件”记日志。

### Step D：S4_VERIFY（验收）

两段式（避免依赖 python API 的返回语义）：

1. `coder.run(f"/test {TEST_CMD}")` 让失败输出进入 aider 上下文，便于修复。([Aider][2])
2. runner 自己 `subprocess.run(TEST_CMD, capture_output=True)` 得到明确 pass/fail（写入 state/log）

### Step E：S5_DECIDE（通过 or 修复 or 改计划）

* 若 pass：runner 把当前 step 从 Next 移到 Done（可以让 aider 做，也可以 runner 直接改 PLAN；MVP 推荐让 aider 做，保持一致）
* 若 fail：调用 `coder.run(<FIX_OR_REPLAN_PROMPT with failing output>)`

  * 要求模型二选一：

    1. 修复代码直到测试过
    2. 承认阻塞点，回到计划层拆 step / 标记 Blocked，并写出“需要人类提供的信息”

---

## 7) Aider Python 集成方式（最小、直接可用）

官方文档给的最小脚本化范式就是：([Aider][1])

* `from aider.coders import Coder`
* `from aider.models import Model`
* `coder = Coder.create(main_model=model, fnames=fnames)`
* `coder.run("...")`
* 也可以 `coder.run("/tokens")`（证明 slash 命令可跑）([Aider][1])
* 如需自动确认（等价 `--yes`）：`InputOutput(yes=True)` ([Aider][1])

### MVP 伪代码骨架（你可以直接按这个实现）

```python
from pathlib import Path
import json, subprocess, time
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

def run_cmd(cmd: str, cwd: Path):
    p = subprocess.run(cmd, cwd=str(cwd), shell=True, text=True,
                       capture_output=True)
    return p.returncode, p.stdout[-8000:], p.stderr[-8000:]

def main(repo: str, model_name: str, test_cmd: str, seed_files: list[str]):
    repo = Path(repo)
    state_dir = repo / ".aider_fsm"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "logs").mkdir(exist_ok=True)

    io = InputOutput(yes=True)
    model = Model(model_name)

    # 至少把 PLAN.md 加进 context；seed_files 再加 1~3 个入口文件
    fnames = ["PLAN.md"] + seed_files
    coder = Coder.create(main_model=model, fnames=fnames, io=io)

    for iter_idx in range(1, 999999):
        # S1 snapshot
        rc, gs, ge = run_cmd("git status --porcelain", repo)
        rc2, diff, de = run_cmd("git diff --stat", repo)
        plan_text = (repo / "PLAN.md").read_text() if (repo / "PLAN.md").exists() else ""

        snapshot = f"""[SNAPSHOT]
git_status:
{gs}
git_diff_stat:
{diff}
plan_md:
{plan_text}
"""

        # S2 plan update
        coder.run(make_plan_update_prompt(snapshot, test_cmd))

        # parse Next step from PLAN.md
        step = parse_next_step((repo / "PLAN.md").read_text())
        if not step:
            # 没 next：尝试结束
            break

        # S3 execute
        coder.run(make_execute_prompt(snapshot, step))

        # S4 verify
        coder.run(f"/test {test_cmd}")
        trc, tout, terr = run_cmd(test_cmd, repo)

        if trc == 0:
            coder.run(make_mark_done_prompt(step))
            continue
        else:
            coder.run(make_fix_or_replan_prompt(step, tout, terr))
            # loop continues

```

---

## 8) 三个关键 Prompt 模板（保证“最小不可分”）

> 你要的是“反复检查 plan+repo → 做事 → 验收 → 可改 plan”，这 3 个 prompt 就是最小原子闭环。

### 8.1 PLAN_UPDATE_PROMPT（只改 PLAN.md）

要求：更新计划、确保 Next 只有一个原子任务、写清验收、阻塞点显式化。

* 输入：`snapshot_text + test_cmd`
* 输出：只改 `PLAN.md`

要点：

* **禁止改业务代码**
* 把 Next 控制成 **exactly one**
* 如果发现缺信息→把当前 step 标记 Blocked，并写“需要人类提供什么”

### 8.2 EXECUTE_PROMPT（只做 Next 这一步）

* 输入：`snapshot_text + next_step_text`
* 输出：修改必要文件实现该 step
* 约束：只做这一件事，不要顺便重构；改动越小越好；若发现 step 不可闭合，停止执行并回到计划（让它修改 PLAN.md）

### 8.3 FIX_OR_REPLAN_PROMPT（失败后：修复或改计划）

* 输入：`next_step_text + failing stdout/stderr`
* 输出：二选一：

  1. 修复直到 `TEST_CMD` 通过
  2. 修改 `PLAN.md`：拆步 / Blocked / 追加需要的信息

---

## 9) 风险与对策（proposal 必须写清）

### 风险 R1：Python API 可能变更/不兼容

官方原话：python scripting API “not officially supported… could change… without backwards compatibility”。([Aider][1])

**对策（最小不可分也必须做）**：

1. **版本锁定**：在你的 runner repo 里固定 `aider-chat==X.Y.Z`（requirements/uv/poetry 随你）
2. **适配层**：把 `Coder.create` / `coder.run` 包一层 `AiderBackend`，未来变更只改这一层
3. **回退路径**：一旦 import/API 失败，自动切到 CLI `aider --message-file ...`（这能保证系统不报废）

### 风险 R2：长循环导致上下文膨胀

**对策（MVP 可选，但建议）**：

* 每 N 轮让模型把关键状态压缩写回 `PLAN.md -> Notes`
* 必要时 `/save` + `/reset` + 重新 `/add`（斜杠命令支持）([Aider][2])

### 风险 R3：模型在 EXECUTE 阶段偷偷“改计划”或“做多件事”

**对策**：

* 在 prompt 里写硬约束：只能做 Next step；超出则视为失败并回滚（可用 git）
* runner 侧加入 guard：若 `PLAN.md` 在 EXECUTE 被改动且不是预期，立刻进入 REPLAN 状态

---

## 10) 实施计划（最小不可分的里程碑）

### Milestone 0（1 次提交就能跑通）

* 加入 `PLAN.md` 模板
* 写 `fsm_runner.py`：能 `Coder.create` + `coder.run("...")` 连续两条指令（验证“同进程上下文”）([Aider][1])

### Milestone 1（闭环跑通）

* 实现 6 状态 FSM
* 实现 3 类 prompt（plan / execute / fix-or-replan）
* 接入 `TEST_CMD`（`/test` + subprocess 双保险）([Aider][2])

### Milestone 2（可靠性最小增强）

* state.json + jsonl logs
* 停止条件（max_iters / max_fix_attempts）
* 版本锁定 + CLI 回退

> 到 Milestone 2，你的“自动反复检查计划+执行+验收+改计划”就已经是一个可长期用的工具了。

---

如果你希望我再把它**具体化到“可直接复制粘贴”的 `fsm_runner.py`**（带 `parse_next_step()`、三个 prompt 文本、以及最小 argparse），我可以在下一条消息直接给出完整单文件实现草案。

[1]: https://aider.chat/docs/scripting.html "Scripting aider | aider"
[2]: https://aider.chat/docs/usage/commands.html?utm_source=chatgpt.com "In-chat commands"
