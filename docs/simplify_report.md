# Simplification Report (static)
Generated via AST scan + approximate reference counting.

Notes:
- `refs≈` is a regex count across `runner/` + `tests/` (may include comments/strings; may miss reflection).
- `action` is a suggestion only; acceptance is `pytest -q`.

## Summary
- total symbols: 289
- by action:
  - INLINE: 40
  - SIMPLIFY: 83
  - KEEP: 166
- by simplifiable:
  - 是: 45
  - 部分: 83
  - 否: 161

## Per-file Candidates (non-KEEP)
Only symbols with suggested action != `KEEP` are listed below. Full surface is in CSV.

### `runner/bootstrap.py`

- **INLINE** `_is_sensitive_key` [function] simp=是 refs≈2 lines≈12
  - reason: 规模≈12 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/bootstrap.py:209; kind=function; refs≈2; lines≈12
- **INLINE** `_redact_env` [function] simp=是 refs≈2 lines≈17
  - reason: 规模≈17 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/bootstrap.py:223; kind=function; refs≈2; lines≈17
- **INLINE** `_apply_env_mapping` [function] simp=是 refs≈2 lines≈20
  - reason: 规模≈20 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/bootstrap.py:128; kind=function; refs≈2; lines≈20
- **INLINE** `_expand_env_value._bare` [function] simp=是 refs≈3 lines≈11
  - reason: 规模≈11 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/bootstrap.py:111; kind=function; refs≈3; lines≈11
- **INLINE** `_expand_env_value._brace` [function] simp=是 refs≈3 lines≈11
  - reason: 规模≈11 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/bootstrap.py:99; kind=function; refs≈3; lines≈11
- **SIMPLIFY** `_normalize_bootstrap_mapping` [function] simp=部分 refs≈3 lines≈21
  - reason: 规模≈21 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/bootstrap.py:311; kind=function; refs≈3; lines≈21
- **SIMPLIFY** `_as_optional_int` [function] simp=部分 refs≈4 lines≈17
  - reason: 规模≈17 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/bootstrap.py:237; kind=function; refs≈4; lines≈17

### `runner/contract_hints.py`

- **INLINE** `_split_script_lines` [function] simp=是 refs≈2 lines≈13
  - reason: 规模≈13 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/contract_hints.py:180; kind=function; refs≈2; lines≈13
- **INLINE** `_iter_md_paths` [function] simp=是 refs≈3 lines≈11
  - reason: 规模≈11 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/contract_hints.py:103; kind=function; refs≈3; lines≈11
- **INLINE** `_tokenize_hint` [function] simp=是 refs≈3 lines≈11
  - reason: 规模≈11 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/contract_hints.py:281; kind=function; refs≈3; lines≈11
- **INLINE** `_yaml_safe_load` [function] simp=是 refs≈3 lines≈13
  - reason: 规模≈13 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/contract_hints.py:132; kind=function; refs≈3; lines≈13
- **INLINE** `_iter_workflow_paths` [function] simp=是 refs≈3 lines≈14
  - reason: 规模≈14 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/contract_hints.py:116; kind=function; refs≈3; lines≈14
- **SIMPLIFY** `_join_with_continuations` [function] simp=部分 refs≈2 lines≈22
  - reason: 规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_hints.py:195; kind=function; refs≈2; lines≈22
- **SIMPLIFY** `_extract_workflow_run_scripts` [function] simp=部分 refs≈2 lines≈30
  - reason: 规模≈30 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_hints.py:148; kind=function; refs≈2; lines≈30
- **SIMPLIFY** `_looks_like_command` [function] simp=部分 refs≈2 lines≈32
  - reason: 规模≈32 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_hints.py:70; kind=function; refs≈2; lines≈32
- **SIMPLIFY** `_strip_prompt_prefix` [function] simp=部分 refs≈4 lines≈16
  - reason: 规模≈16 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_hints.py:29; kind=function; refs≈4; lines≈16

### `runner/contract_provenance.py`

- **INLINE** `_normalize_rel_to_repo` [function] simp=是 refs≈3 lines≈17
  - reason: 规模≈17 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/contract_provenance.py:67; kind=function; refs≈3; lines≈17
- **SIMPLIFY** `_status` [function] simp=部分 refs≈4 lines≈16
  - reason: 规模≈16 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_provenance.py:112; kind=function; refs≈4; lines≈16
- **SIMPLIFY** `_file_meta` [function] simp=部分 refs≈5 lines≈14
  - reason: 规模≈14 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_provenance.py:11; kind=function; refs≈5; lines≈14

### `runner/contract_repair.py`

- **SIMPLIFY** `_build_contract_validation_snapshot` [function] simp=部分 refs≈2 lines≈28
  - reason: 规模≈28 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_repair.py:27; kind=function; refs≈2; lines≈28
- **SIMPLIFY** `repair_contract` [function] simp=部分 refs≈10 lines≈304
  - reason: 规模≈304 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/contract_repair.py:134; kind=function; refs≈10; lines≈304

### `runner/env.py`

- **INLINE** `EnvSession._set_llm` [method] simp=是 refs≈2 lines≈9
  - reason: 规模≈9 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/env.py:470; kind=method; refs≈2; lines≈9
- **INLINE** `_now_run_id` [function] simp=是 refs≈3 lines≈6
  - reason: 规模≈6 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/env.py:38; kind=function; refs≈3; lines≈6
- **INLINE** `EnvSession._maybe_teardown` [method] simp=是 refs≈3 lines≈14
  - reason: 规模≈14 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/env.py:559; kind=method; refs≈3; lines≈14
- **INLINE** `_verification_errors_summary` [function] simp=是 refs≈3 lines≈18
  - reason: 规模≈18 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/env.py:160; kind=function; refs≈3; lines≈18
- **SIMPLIFY** `EnvSession._evaluation` [method] simp=部分 refs≈1 lines≈168
  - reason: 规模≈168 行；引用次数≈1（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:696; kind=method; refs≈1; lines≈168
- **SIMPLIFY** `_inject_openai_base_compat` [function] simp=部分 refs≈2 lines≈22
  - reason: 规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:113; kind=function; refs≈2; lines≈22
- **SIMPLIFY** `EnvSession._apply_llm_overrides` [method] simp=部分 refs≈2 lines≈24
  - reason: 规模≈24 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:481; kind=method; refs≈2; lines≈24
- **SIMPLIFY** `_runtime_openai_config` [function] simp=部分 refs≈2 lines≈26
  - reason: 规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:132; kind=function; refs≈2; lines≈26
- **SIMPLIFY** `_hf_parquet_qa_rows` [function] simp=部分 refs≈2 lines≈34
  - reason: 规模≈34 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:180; kind=function; refs≈2; lines≈34
- **SIMPLIFY** `_resolve_llm` [function] simp=部分 refs≈2 lines≈38
  - reason: 规模≈38 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:59; kind=function; refs≈2; lines≈38
- **SIMPLIFY** `EnvSession._apply_runtime_env_inference_overrides` [method] simp=部分 refs≈3 lines≈21
  - reason: 规模≈21 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:537; kind=method; refs≈3; lines≈21
- **SIMPLIFY** `EnvSession._base_overrides` [method] simp=部分 refs≈3 lines≈31
  - reason: 规模≈31 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:505; kind=method; refs≈3; lines≈31
- **SIMPLIFY** `_validate_rollout_samples._norm_ws` [function] simp=部分 refs≈4 lines≈6
  - reason: 规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:346; kind=function; refs≈4; lines≈6
- **SIMPLIFY** `EnvSession._audit_mode` [method] simp=部分 refs≈4 lines≈9
  - reason: 规模≈9 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:460; kind=method; refs≈4; lines≈9
- **SIMPLIFY** `_resolve_path` [function] simp=部分 refs≈5 lines≈6
  - reason: 规模≈6 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:46; kind=function; refs≈5; lines≈6
- **SIMPLIFY** `_resolve_run_root` [function] simp=部分 refs≈5 lines≈11
  - reason: 规模≈11 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:94; kind=function; refs≈5; lines≈11
- **SIMPLIFY** `_validate_rollout_samples` [function] simp=部分 refs≈10 lines≈165
  - reason: 规模≈165 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env.py:280; kind=function; refs≈10; lines≈165

### `runner/env_local.py`

- **SIMPLIFY** `_classify_opencode_transport_error` [function] simp=部分 refs≈4 lines≈22
  - reason: 规模≈22 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env_local.py:31; kind=function; refs≈4; lines≈22
- **SIMPLIFY** `_list_opencode_models` [function] simp=部分 refs≈4 lines≈32
  - reason: 规模≈32 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env_local.py:147; kind=function; refs≈4; lines≈32
- **SIMPLIFY** `open_env` [function] simp=部分 refs≈9 lines≈287
  - reason: 规模≈287 行；引用次数≈9（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/env_local.py:273; kind=function; refs≈9; lines≈287

### `runner/eval_audit.py`

- **SIMPLIFY** `_looks_like_python_exec` [function] simp=部分 refs≈3 lines≈27
  - reason: 规模≈27 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/eval_audit.py:47; kind=function; refs≈3; lines≈27

### `runner/generic_rollout.py`

- **INLINE** `_read_hf_manifest` [function] simp=是 refs≈3 lines≈9
  - reason: 规模≈9 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/generic_rollout.py:57; kind=function; refs≈3; lines≈9
- **INLINE** `_read_text` [function] simp=是 refs≈3 lines≈12
  - reason: 规模≈12 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/generic_rollout.py:43; kind=function; refs≈3; lines≈12
- **INLINE** `_answers_match` [function] simp=是 refs≈3 lines≈18
  - reason: 规模≈18 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/generic_rollout.py:226; kind=function; refs≈3; lines≈18
- **SIMPLIFY** `_build_prompts` [function] simp=部分 refs≈3 lines≈31
  - reason: 规模≈31 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:354; kind=function; refs≈3; lines≈31
- **SIMPLIFY** `_resolve_openai_base` [function] simp=部分 refs≈3 lines≈31
  - reason: 规模≈31 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:68; kind=function; refs≈3; lines≈31
- **SIMPLIFY** `_now_iso` [function] simp=部分 refs≈4 lines≈6
  - reason: 规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:35; kind=function; refs≈4; lines≈6
- **SIMPLIFY** `_norm_answer_str` [function] simp=部分 refs≈4 lines≈8
  - reason: 规模≈8 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:205; kind=function; refs≈4; lines≈8
- **SIMPLIFY** `_extract_last_number` [function] simp=部分 refs≈4 lines≈9
  - reason: 规模≈9 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:215; kind=function; refs≈4; lines≈9
- **SIMPLIFY** `_to_fraction` [function] simp=部分 refs≈4 lines≈10
  - reason: 规模≈10 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:173; kind=function; refs≈4; lines≈10
- **SIMPLIFY** `_extract_final_line` [function] simp=部分 refs≈4 lines≈18
  - reason: 规模≈18 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:185; kind=function; refs≈4; lines≈18
- **SIMPLIFY** `_norm_number_str` [function] simp=部分 refs≈5 lines≈6
  - reason: 规模≈6 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/generic_rollout.py:165; kind=function; refs≈5; lines≈6

### `runner/hints_exec.py`

- **INLINE** `normalize_hint_command._maybe_rewrite_python_tools._is_pip` [function] simp=是 refs≈3 lines≈9
  - reason: 规模≈9 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:780; kind=function; refs≈3; lines≈9
- **INLINE** `normalize_hint_command._maybe_rewrite_python_tools._is_py` [function] simp=是 refs≈3 lines≈9
  - reason: 规模≈9 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:770; kind=function; refs≈3; lines≈9
- **INLINE** `_extract_score_from_json_file` [function] simp=是 refs≈3 lines≈10
  - reason: 规模≈10 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:544; kind=function; refs≈3; lines≈10
- **INLINE** `run_hints._probe_rank` [function] simp=是 refs≈3 lines≈10
  - reason: 规模≈10 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:1725; kind=function; refs≈3; lines≈10
- **INLINE** `_find_latest_scaffold_hints_file` [function] simp=是 refs≈3 lines≈14
  - reason: 规模≈14 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:40; kind=function; refs≈3; lines≈14
- **INLINE** `_contains_openai_auth_error` [function] simp=是 refs≈3 lines≈15
  - reason: 规模≈15 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:443; kind=function; refs≈3; lines≈15
- **INLINE** `run_hints._hint_kind` [function] simp=是 refs≈3 lines≈15
  - reason: 规模≈15 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:1738; kind=function; refs≈3; lines≈15
- **INLINE** `_read_hints_file` [function] simp=是 refs≈3 lines≈16
  - reason: 规模≈16 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:22; kind=function; refs≈3; lines≈16
- **INLINE** `run_hints._looks_like_py_incompat_build_failure` [function] simp=是 refs≈3 lines≈17
  - reason: 规模≈17 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:1256; kind=function; refs≈3; lines≈17
- **INLINE** `normalize_hint_command._rewrite_line` [function] simp=是 refs≈3 lines≈19
  - reason: 规模≈19 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:727; kind=function; refs≈3; lines≈19
- **INLINE** `run_hints._hint_workdir` [function] simp=是 refs≈3 lines≈19
  - reason: 规模≈19 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/hints_exec.py:1420; kind=function; refs≈3; lines≈19
- **SIMPLIFY** `_extract_score_from_text` [function] simp=部分 refs≈2 lines≈26
  - reason: 规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:482; kind=function; refs≈2; lines≈26
- **SIMPLIFY** `_docker_available` [function] simp=部分 refs≈2 lines≈29
  - reason: 规模≈29 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1020; kind=function; refs≈2; lines≈29
- **SIMPLIFY** `normalize_hint_command._looks_like_fire_cli` [function] simp=部分 refs≈2 lines≈30
  - reason: 规模≈30 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:821; kind=function; refs≈2; lines≈30
- **SIMPLIFY** `_infer_repo_python_pin` [function] simp=部分 refs≈2 lines≈40
  - reason: 规模≈40 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:166; kind=function; refs≈2; lines≈40
- **SIMPLIFY** `run_hints._maybe_prepare_dataset_override` [function] simp=部分 refs≈2 lines≈142
  - reason: 规模≈142 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1447; kind=function; refs≈2; lines≈142
- **SIMPLIFY** `_extract_invoked_command` [function] simp=部分 refs≈3 lines≈22
  - reason: 规模≈22 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1046; kind=function; refs≈3; lines≈22
- **SIMPLIFY** `normalize_hint_command._maybe_normalize_fire_flag_aliases` [function] simp=部分 refs≈3 lines≈22
  - reason: 规模≈22 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:851; kind=function; refs≈3; lines≈22
- **SIMPLIFY** `_hint_runtime_compatible` [function] simp=部分 refs≈3 lines≈23
  - reason: 规模≈23 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:599; kind=function; refs≈3; lines≈23
- **SIMPLIFY** `run_hints._parse_pytest_counts` [function] simp=部分 refs≈3 lines≈28
  - reason: 规模≈28 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1584; kind=function; refs≈3; lines≈28
- **SIMPLIFY** `_extract_score_from_json_obj` [function] simp=部分 refs≈3 lines≈33
  - reason: 规模≈33 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:510; kind=function; refs≈3; lines≈33
- **SIMPLIFY** `normalize_hint_command._maybe_bound_openai_codegen_eval` [function] simp=部分 refs≈3 lines≈34
  - reason: 规模≈34 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:874; kind=function; refs≈3; lines≈34
- **SIMPLIFY** `normalize_hint_command._strip_pytest_xdist_flags` [function] simp=部分 refs≈3 lines≈39
  - reason: 规模≈39 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:915; kind=function; refs≈3; lines≈39
- **SIMPLIFY** `_is_remote_openai_hint` [function] simp=部分 refs≈4 lines≈6
  - reason: 规模≈6 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:435; kind=function; refs≈4; lines≈6
- **SIMPLIFY** `_canonical_base_url` [function] simp=部分 refs≈4 lines≈12
  - reason: 规模≈12 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:359; kind=function; refs≈4; lines≈12
- **SIMPLIFY** `_hint_backend` [function] simp=部分 refs≈4 lines≈13
  - reason: 规模≈13 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:420; kind=function; refs≈4; lines≈13
- **SIMPLIFY** `_normalize_score` [function] simp=部分 refs≈4 lines≈14
  - reason: 规模≈14 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:465; kind=function; refs≈4; lines≈14
- **SIMPLIFY** `run_hints._looks_like_openai_codegen_eval` [function] simp=部分 refs≈4 lines≈19
  - reason: 规模≈19 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1400; kind=function; refs≈4; lines≈19
- **SIMPLIFY** `run_hints._exec` [function] simp=部分 refs≈4 lines≈27
  - reason: 规模≈27 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1828; kind=function; refs≈4; lines≈27
- **SIMPLIFY** `run_hints._priority` [function] simp=部分 refs≈4 lines≈38
  - reason: 规模≈38 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1349; kind=function; refs≈4; lines≈38
- **SIMPLIFY** `_extract_cli_flag_value_any` [function] simp=部分 refs≈5 lines≈10
  - reason: 规模≈10 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:408; kind=function; refs≈5; lines≈10
- **SIMPLIFY** `_as_major_minor` [function] simp=部分 refs≈5 lines≈19
  - reason: 规模≈19 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:144; kind=function; refs≈5; lines≈19
- **SIMPLIFY** `_matched_anchors` [function] simp=部分 refs≈5 lines≈20
  - reason: 规模≈20 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:1186; kind=function; refs≈5; lines≈20
- **SIMPLIFY** `_extract_cli_flag_value` [function] simp=部分 refs≈5 lines≈21
  - reason: 规模≈21 行；引用次数≈5（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/hints_exec.py:385; kind=function; refs≈5; lines≈21

### `runner/opencode_client.py`

- **INLINE** `OpenCodeClient._is_transport_unavailable_error` [method] simp=是 refs≈1 lines≈8
  - reason: 规模≈8 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:630; kind=method; refs≈1; lines≈8
- **INLINE** `OpenCodeClient._sleep_session_recover_backoff` [method] simp=是 refs≈1 lines≈11
  - reason: 规模≈11 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:618; kind=method; refs≈1; lines≈11
- **INLINE** `OpenCodeClient._sleep_retry_backoff` [method] simp=是 refs≈1 lines≈16
  - reason: 规模≈16 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:606; kind=method; refs≈1; lines≈16
- **INLINE** `OpenCodeClient._should_retry_request_error` [method] simp=是 refs≈1 lines≈19
  - reason: 规模≈19 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:667; kind=method; refs≈1; lines≈19
- **INLINE** `OpenCodeClient._clip_prompt_text` [method] simp=是 refs≈1 lines≈20
  - reason: 规模≈20 行；引用次数≈1（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:687; kind=method; refs≈1; lines≈20
- **INLINE** `_basic_auth_value` [function] simp=是 refs≈2 lines≈12
  - reason: 规模≈12 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:70; kind=function; refs≈2; lines≈12
- **INLINE** `_normalize_base_url` [function] simp=是 refs≈2 lines≈14
  - reason: 规模≈14 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:54; kind=function; refs≈2; lines≈14
- **INLINE** `OpenCodeClient._create_session` [method] simp=是 refs≈2 lines≈17
  - reason: 规模≈17 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:588; kind=method; refs≈2; lines≈17
- **INLINE** `_split_model` [function] simp=是 refs≈2 lines≈19
  - reason: 规模≈19 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:84; kind=function; refs≈2; lines≈19
- **INLINE** `OpenCodeClient._wait_for_health` [method] simp=是 refs≈2 lines≈20
  - reason: 规模≈20 行；引用次数≈2（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/opencode_client.py:567; kind=method; refs≈2; lines≈20
- **SIMPLIFY** `OpenCodeClient._recover_local_server_session` [method] simp=部分 refs≈1 lines≈22
  - reason: 规模≈22 行；引用次数≈1（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_client.py:640; kind=method; refs≈1; lines≈22
- **SIMPLIFY** `_extract_assistant_text` [function] simp=部分 refs≈2 lines≈22
  - reason: 规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_client.py:123; kind=function; refs≈2; lines≈22
- **SIMPLIFY** `_looks_like_transport_unavailable` [function] simp=部分 refs≈2 lines≈23
  - reason: 规模≈23 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_client.py:25; kind=function; refs≈2; lines≈23
- **SIMPLIFY** `_extract_opencode_error` [function] simp=部分 refs≈2 lines≈28
  - reason: 规模≈28 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_client.py:147; kind=function; refs≈2; lines≈28
- **SIMPLIFY** `OpenCodeClient._post_message` [method] simp=部分 refs≈2 lines≈31
  - reason: 规模≈31 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_client.py:759; kind=method; refs≈2; lines≈31
- **SIMPLIFY** `OpenCodeClient._stop_local_server` [method] simp=部分 refs≈2 lines≈40
  - reason: 规模≈40 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_client.py:357; kind=method; refs≈2; lines≈40

### `runner/opencode_tooling.py`

- **SIMPLIFY** `_find_tag_gt` [function] simp=部分 refs≈2 lines≈32
  - reason: 规模≈32 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:350; kind=function; refs≈2; lines≈32
- **SIMPLIFY** `_extract_json_object` [function] simp=部分 refs≈2 lines≈35
  - reason: 规模≈35 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:313; kind=function; refs≈2; lines≈35
- **SIMPLIFY** `_sanitized_env` [function] simp=部分 refs≈3 lines≈24
  - reason: 规模≈24 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:663; kind=function; refs≈3; lines≈24
- **SIMPLIFY** `_is_env_like` [function] simp=部分 refs≈4 lines≈20
  - reason: 规模≈20 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:624; kind=function; refs≈4; lines≈20
- **SIMPLIFY** `_decode_attr_value` [function] simp=部分 refs≈4 lines≈26
  - reason: 规模≈26 行；引用次数≈4（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:88; kind=function; refs≈4; lines≈26
- **SIMPLIFY** `execute_tool_calls` [function] simp=部分 refs≈8 lines≈286
  - reason: 规模≈286 行；引用次数≈8（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/opencode_tooling.py:867; kind=function; refs≈8; lines≈286

### `runner/pipeline_verify.py`

- **INLINE** `_validate_metrics` [function] simp=是 refs≈3 lines≈15
  - reason: 规模≈15 行；引用次数≈3（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/pipeline_verify.py:93; kind=function; refs≈3; lines≈15
- **SIMPLIFY** `run_pipeline_verification._teardown_allowed` [function] simp=部分 refs≈2 lines≈21
  - reason: 规模≈21 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/pipeline_verify.py:382; kind=function; refs≈2; lines≈21
- **SIMPLIFY** `_validate_hints_used` [function] simp=部分 refs≈2 lines≈35
  - reason: 规模≈35 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/pipeline_verify.py:106; kind=function; refs≈2; lines≈35
- **SIMPLIFY** `_validate_hints_run` [function] simp=部分 refs≈2 lines≈39
  - reason: 规模≈39 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/pipeline_verify.py:143; kind=function; refs≈2; lines≈39
- **SIMPLIFY** `run_pipeline_verification._env_get` [function] simp=部分 refs≈3 lines≈22
  - reason: 规模≈22 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/pipeline_verify.py:422; kind=function; refs≈3; lines≈22

### `runner/prompts.py`

- **SIMPLIFY** `make_scaffold_contract_prompt` [function] simp=部分 refs≈10 lines≈239
  - reason: 规模≈239 行；引用次数≈10（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/prompts.py:105; kind=function; refs≈10; lines≈239

### `runner/repo_resolver.py`

- **SIMPLIFY** `_repo_slug` [function] simp=部分 refs≈2 lines≈22
  - reason: 规模≈22 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:96; kind=function; refs≈2; lines≈22
- **SIMPLIFY** `_parse_hf_dataset` [function] simp=部分 refs≈2 lines≈26
  - reason: 规模≈26 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:237; kind=function; refs≈2; lines≈26
- **SIMPLIFY** `_extract_github_zip` [function] simp=部分 refs≈2 lines≈27
  - reason: 规模≈27 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:208; kind=function; refs≈2; lines≈27
- **SIMPLIFY** `_hf_dataset_api_info` [function] simp=部分 refs≈2 lines≈27
  - reason: 规模≈27 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:271; kind=function; refs≈2; lines≈27
- **SIMPLIFY** `_parse_github_owner_repo` [function] simp=部分 refs≈2 lines≈36
  - reason: 规模≈36 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:120; kind=function; refs≈2; lines≈36
- **SIMPLIFY** `_download_hf_dataset_snapshot` [function] simp=部分 refs≈2 lines≈128
  - reason: 规模≈128 行；引用次数≈2（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:305; kind=function; refs≈2; lines≈128
- **SIMPLIFY** `_default_clones_base` [function] simp=部分 refs≈3 lines≈25
  - reason: 规模≈25 行；引用次数≈3（静态近似，可能包含注释/字符串）；可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联
  - evidence: runner/repo_resolver.py:26; kind=function; refs≈3; lines≈25

### `runner/scaffold_validation.py`

- **SIMPLIFY** `ScaffoldValidationReport.ok` [method] simp=是 refs≈0 lines≈6
  - reason: 规模≈6 行；引用次数≈0（静态近似，可能包含注释/字符串）；逻辑短且低复用，适合 inline/合并以减少符号面
  - evidence: runner/scaffold_validation.py:59; kind=method; refs≈0; lines≈6

