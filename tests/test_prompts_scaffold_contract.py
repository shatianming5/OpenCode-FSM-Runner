from __future__ import annotations

from pathlib import Path

from runner.prompts import make_scaffold_contract_prompt, make_scaffold_contract_retry_prompt


def test_scaffold_contract_prompt_mentions_trained_model_dir_and_runtime_env():
    p = make_scaffold_contract_prompt(Path("/tmp/repo"), pipeline_rel="pipeline.yml", require_metrics=True)
    assert "AIDER_RUNTIME_ENV_PATH" in p
    assert "AIDER_TRAINED_MODEL_DIR" in p
    assert "AIDER_LLM_KIND" in p
    assert "AIDER_LLM_MODEL" in p
    assert "model_dir" in p


def test_scaffold_contract_prompt_includes_doc_command_hints_when_provided():
    p = make_scaffold_contract_prompt(
        Path("/tmp/repo"),
        pipeline_rel="pipeline.yml",
        require_metrics=True,
        command_hints=["python -m benchtool.evaluate --dataset demo --model openai"],
    )
    assert "[CANDIDATE_COMMAND_HINTS]" in p
    assert "benchtool.evaluate" in p


def test_scaffold_contract_prompt_forbids_fake_tool_transcripts():
    p = make_scaffold_contract_prompt(Path("/tmp/repo"), pipeline_rel="pipeline.yml", require_metrics=True)
    assert "Do NOT fabricate `<tool_result>` blocks" in p
    assert "Do NOT print pseudo tool snippets as plain text" in p


def test_scaffold_contract_prompt_mentions_opencode_xml_tool_formats():
    p = make_scaffold_contract_prompt(Path("/tmp/repo"), pipeline_rel="pipeline.yml", require_metrics=True)
    assert '<read filePath="PATH" />' in p
    assert '<write filePath="PATH">' in p
    assert '<bash command="..." description="..." />' in p


def test_scaffold_contract_retry_prompt_forbids_pseudo_tool_syntax():
    p = make_scaffold_contract_retry_prompt(
        Path("/tmp/repo"),
        pipeline_rel="pipeline.yml",
        require_metrics=True,
        attempt=2,
        max_attempts=3,
        previous_failure="missing_pipeline_yml",
    )
    assert "Do NOT output pseudo tool syntax" in p
    assert "emit fake `<tool_result>` blocks" in p
