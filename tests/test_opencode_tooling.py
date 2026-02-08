from __future__ import annotations

from pathlib import Path

from runner.opencode_tooling import ToolPolicy, execute_tool_calls, parse_tool_calls


def test_parse_tool_calls_detects_file_write_json_fence():
    """中文说明：
    - 含义：验证 `parse_tool_calls` 能识别 fenced `json` 里的 file-write payload。
    - 内容：提供一个包含 ```json ...``` 的文本，断言解析出 1 个 kind=file 且 payload 字段正确。
    - 可简略：可能（可并入更大的表驱动测试；但单测粒度清晰）。
    """
    text = "hi\n```json\n{\"filePath\":\"PLAN.md\",\"content\":\"# PLAN\\n\"}\n```\n"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "PLAN.md"
    assert payload["content"].startswith("# PLAN")


def test_parse_tool_calls_detects_bash_call():
    """中文说明：
    - 含义：验证 `parse_tool_calls` 能识别 fenced `bash` 的工具调用。
    - 内容：提供一个包含 bash tool 的 JSON 参数，断言解析出 kind=bash 且 command 正确。
    - 可简略：可能（可参数化更多命令；当前覆盖主路径）。
    """
    text = "```bash\nbash\n{\"command\":\"git status --porcelain\"}\n```\n"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["command"] == "git status --porcelain"


def test_parse_tool_calls_detects_self_closing_write_tag():
    """中文说明：
    - 含义：验证 `parse_tool_calls` 能识别自闭合 `<write ... />` 写文件标签。
    - 内容：输入 `<write filePath=... content=... />`，断言解析出 kind=file 且内容正确解码。
    - 可简略：可能（主要覆盖标签语法的兼容性）。
    """
    text = '<write filePath=\"hello.txt\" content=\"hello\\n\" />'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "hello.txt"
    assert payload["content"] == "hello\n"


def test_parse_tool_calls_detects_paired_write_tag_with_body_content():
    text = '<write filePath="pipeline.yml">version: 1\nsecurity:\n  mode: safe\n</write>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "pipeline.yml"
    assert payload["content"].startswith("version: 1")


def test_parse_tool_calls_unescapes_xml_entities_in_write_body_content():
    # Models sometimes emit XML-escaped file bodies, e.g. `<<` becomes `&lt;&lt;`.
    text = '<write filePath="deploy_setup.sh">cat &lt;&lt; EOF\nhello\nEOF\n</write>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "deploy_setup.sh"
    assert "cat << EOF" in payload["content"]


def test_parse_tool_calls_unescapes_double_escaped_xml_entities():
    # Some model outputs are double-escaped (e.g. `&amp;amp;`), so we unescape a few times.
    text = '<write filePath="x.sh">echo hi 2&gt;&amp;amp;1\n</write>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "x.sh"
    assert "2>&1" in payload["content"]


def test_parse_tool_calls_detects_self_closing_edit_tag():
    text = '<edit filePath="hello.txt" oldString="hello" newString="world" />'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "hello.txt"
    assert payload["oldString"] == "hello"
    assert payload["newString"] == "world"


def test_execute_tool_calls_applies_edit_old_new(tmp_path: Path):
    repo = tmp_path.resolve()
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")

    policy = ToolPolicy(
        repo=repo,
        plan_path=(repo / "PLAN.md"),
        pipeline_path=(repo / "pipeline.yml"),
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    calls = parse_tool_calls('<edit filePath="hello.txt" oldString="hello\\n" newString="world\\n" />')
    results = execute_tool_calls(calls, repo=repo, policy=policy)
    assert results and results[0].ok
    assert (repo / "hello.txt").read_text(encoding="utf-8") == "world\n"


def test_execute_tool_calls_edit_without_old_string_overwrites(tmp_path: Path):
    repo = tmp_path.resolve()
    (repo / "hello.txt").write_text("old", encoding="utf-8")

    policy = ToolPolicy(
        repo=repo,
        plan_path=(repo / "PLAN.md"),
        pipeline_path=(repo / "pipeline.yml"),
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    calls = parse_tool_calls('<edit filePath="hello.txt" newString="new" />')
    results = execute_tool_calls(calls, repo=repo, policy=policy)
    assert results and results[0].ok
    assert (repo / "hello.txt").read_text(encoding="utf-8") == "new"


def test_execute_tool_calls_edit_requires_unique_old_string(tmp_path: Path):
    repo = tmp_path.resolve()
    (repo / "hello.txt").write_text("aa", encoding="utf-8")

    policy = ToolPolicy(
        repo=repo,
        plan_path=(repo / "PLAN.md"),
        pipeline_path=(repo / "pipeline.yml"),
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    calls = parse_tool_calls('<edit filePath="hello.txt" oldString="a" newString="b" />')
    results = execute_tool_calls(calls, repo=repo, policy=policy)
    assert results and not results[0].ok
    assert (repo / "hello.txt").read_text(encoding="utf-8") == "aa"


def test_execute_tool_calls_read_directory_returns_error_instead_of_throwing(tmp_path: Path):
    repo = tmp_path.resolve()
    (repo / ".aider_fsm").mkdir(parents=True, exist_ok=True)

    policy = ToolPolicy(
        repo=repo,
        plan_path=(repo / "PLAN.md"),
        pipeline_path=(repo / "pipeline.yml"),
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    calls = parse_tool_calls('<read filePath=".aider_fsm" />')
    results = execute_tool_calls(calls, repo=repo, policy=policy)
    assert results and results[0].kind == "read"
    assert not results[0].ok
    assert results[0].detail.get("error") == "read_failed"


def test_parse_tool_calls_detects_paired_bash_tag_with_attrs():
    text = '<bash command="mkdir -p .aider_fsm/stages" description="create stages"></bash>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["command"] == "mkdir -p .aider_fsm/stages"


def test_parse_tool_calls_parses_self_closing_bash_with_slash_in_command():
    text = '<bash command="ls -la .aider_fsm/" description="list" />'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["command"] == "ls -la .aider_fsm/"


def test_parse_tool_calls_normalizes_malformed_bash_tag_missing_leading_angle_bracket():
    text = 'bash<command="ls -la .aider_fsm/" description="list" />'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["command"] == "ls -la .aider_fsm/"


def test_parse_tool_calls_unescapes_quoted_attr_values():
    text = (
        '<bash command="find . -name \\"*.json\\" -o -name \\"Dockerfile*\\" | head -20" '
        'description="find files" />'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "bash"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["command"] == 'find . -name "*.json" -o -name "Dockerfile*" | head -20'


def test_parse_tool_calls_tolerates_malformed_single_quoted_write_content():
    text = """
<write
filePath=".aider_fsm/stages/evaluation.sh"
content='#!/bin/bash
set -euo pipefail
python -c "print(f'X')"
'
/>
"""
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == ".aider_fsm/stages/evaluation.sh"
    assert "print(f'X')" in payload["content"]


def test_parse_tool_calls_handles_large_malformed_write_without_backtracking_hang():
    blob = ("echo line with quote f'X' and slash /tmp/path\n" * 2500).strip()
    text = (
        "<write filePath=\".aider_fsm/stages/evaluation.sh\" "
        f"content='{blob}'/>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == ".aider_fsm/stages/evaluation.sh"
    assert "f'X'" in payload["content"]


def test_parse_tool_calls_detects_raw_json_inside_tool_call_block():
    """中文说明：
    - 含义：兼容 `<tool_call>{...json...}</tool_call>` 这种不带 `<read>/<bash>` 内层标签的输出格式。
    - 内容：提供 raw JSON filePath tool_call，断言可解析为 kind=file。
    - 可简略：否（该格式在部分模型/代理输出中较常见，缺失会导致 scaffold 无法落盘）。
    """
    text = "<tool_call>\n{\"filePath\": \"README.md\"}\n</tool_call>\n"
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "README.md"


def test_parse_tool_calls_detects_wrapped_name_arguments_tool_call_with_missing_key_quote():
    """中文说明：
    - 含义：兼容 OpenCode 输出的 `{name, arguments}` 包装格式，并容忍缺少 key 开头引号的常见 JSON 生成错误。
    - 内容：提供 `<tool_call>{"name":"write","arguments":{...}}</tool_call>`，断言解析为 file 工具调用。
    - 可简略：否（真实 strict scaffold 日志中出现；缺失会导致 stage scripts 无法落盘，进而触发 incomplete_contract）。
    """
    text = '<tool_call>{"name":"write","arguments":{"content":"hello\\n",filePath":"hello.txt"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "hello.txt"
    assert payload["content"] == "hello\n"


def test_parse_tool_calls_repairs_invalid_escape_in_wrapped_name_arguments_tool_call():
    """中文说明：
    - 含义：兼容模型在 JSON 字符串里输出 `\\${VAR}` 这类无效 escape（例如为了阻止 shell 展开）。
    - 内容：提供包含 `\\${...}` 的 `<tool_call>{"name":"write","arguments":{...}}</tool_call>`，断言仍可解析为 file 工具调用并保留字面反斜杠。
    - 可简略：否（真实 strict scaffold 日志中出现，导致 json.loads 失败进而漏执行 tool-call）。
    """
    text = '<tool_call>{"name":"write","arguments":{"filePath":"hello.txt","content":"echo \\${VAR}\\n"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].kind == "file"
    payload = calls[0].payload
    assert isinstance(payload, dict)
    assert payload["filePath"] == "hello.txt"
    assert payload["content"] == "echo \\${VAR}\n"


def test_parse_tool_calls_detects_inline_read_and_write_json():
    """中文说明：
    - 含义：兼容模型直接输出 `read{...}` 与 `write\\n{...}` 的 inline 工具调用格式。
    - 内容：输入一段包含 read+write 的原始文本，断言都能解析为 kind=file 且 payload 字段正确。
    - 可简略：否（该格式在真实 strict scaffold 日志中出现，缺失会导致工具调用被漏执行）。
    """
    text = (
        'I will inspect docs first. read{"filePath":"README.md"}\n'
        "Then create the contract file:</think>write\n"
        '{"filePath":"pipeline.yml","content":"version: 1\\n"}\n'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2

    first = calls[0].payload
    assert isinstance(first, dict)
    assert calls[0].kind == "file"
    assert first["filePath"] == "README.md"

    second = calls[1].payload
    assert isinstance(second, dict)
    assert calls[1].kind == "file"
    assert second["filePath"] == "pipeline.yml"
    assert second["content"] == "version: 1\n"


def test_tool_policy_plan_update_only_allows_plan_md(tmp_path: Path):
    """中文说明：
    - 含义：验证 plan_update_attempt_1 场景下 ToolPolicy 仅允许写 PLAN.md。
    - 内容：构造 policy 并分别尝试写 PLAN.md 与其它文件，断言允许/拒绝与 reason 符合预期。
    - 可简略：否（属于安全边界测试；建议保留以防回归）。
    """
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="plan_update_attempt_1",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(plan)
    assert ok and reason is None

    ok, reason = policy.allow_file_write(repo / "foo.txt")
    assert not ok
    assert reason == "plan_update_allows_only_plan_md"


def test_tool_policy_execute_step_denies_plan_and_pipeline(tmp_path: Path):
    """中文说明：
    - 含义：验证 execute_step 场景禁止写 PLAN.md 与 pipeline.yml，但允许写普通代码文件。
    - 内容：分别对 plan/pipeline/src 文件调用 `allow_file_write`，断言拒绝原因与允许结果正确。
    - 可简略：否（是执行阶段的关键保护，避免 agent 越权改契约/计划）。
    """
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(plan)
    assert not ok and reason == "execute_step_disallows_plan_md"

    ok, reason = policy.allow_file_write(pipeline)
    assert not ok and reason == "execute_step_disallows_pipeline_yml"

    ok, reason = policy.allow_file_write(repo / "src" / "x.py")
    assert ok and reason is None


def test_tool_policy_scaffold_contract_allows_only_pipeline_and_aider_fsm(tmp_path: Path):
    """中文说明：
    - 含义：验证 scaffold_contract 场景只允许写 pipeline.yml 与 `.aider_fsm/*`。
    - 内容：policy.allow_file_write 对 pipeline/bootstrap.yml 放行，对 src/app.py 拒绝并返回原因。
    - 可简略：否（是 scaffold 合同的关键边界；建议保留）。
    """
    repo = tmp_path.resolve()
    plan = repo / "PLAN.md"
    pipeline = repo / "pipeline.yml"

    policy = ToolPolicy(
        repo=repo,
        plan_path=plan,
        pipeline_path=pipeline,
        purpose="scaffold_contract",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_file_write(pipeline)
    assert ok and reason is None

    ok, reason = policy.allow_file_write(repo / ".aider_fsm" / "bootstrap.yml")
    assert ok and reason is None

    ok, reason = policy.allow_file_write(repo / "src" / "app.py")
    assert not ok
    assert reason == "scaffold_contract_allows_only_pipeline_yml_and_aider_fsm"


def test_tool_policy_restricted_bash_blocks_shell_metacharacters(tmp_path: Path):
    """中文说明：
    - 含义：验证 restricted bash 模式会拦截包含重定向等 shell 元字符的命令。
    - 内容：调用 `allow_bash('echo \"hi\" > hello.txt')`，期望返回 not ok 且 reason 属于预期集合。
    - 可简略：否（属于安全防线测试；建议保留以防策略回退）。
    """
    repo = tmp_path.resolve()
    policy = ToolPolicy(
        repo=repo,
        plan_path=repo / "PLAN.md",
        pipeline_path=None,
        purpose="execute_step",
        bash_mode="restricted",
        unattended="strict",
    )

    ok, reason = policy.allow_bash('echo \"hi\" > hello.txt')
    assert not ok
    assert reason in ("blocked_shell_metacharacters", "blocked_by_restricted_bash_mode")
