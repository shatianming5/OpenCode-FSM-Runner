from runner.plan_format import parse_backlog_open_count, parse_next_step, parse_plan


def _plan(text: str) -> str:
    return (
        "# PLAN\n\n"
        "## Goal\n"
        "- g\n\n"
        "## Acceptance\n"
        "- [ ] TEST_CMD passes: `pytest -q`\n\n"
        "## Next (exactly ONE item)\n"
        f"{text}\n"
        "## Backlog\n"
        "- [ ] (STEP_ID=002) b\n\n"
        "## Done\n"
        "- [x] (STEP_ID=000) d\n\n"
        "## Notes\n"
        "- \n"
    )


def test_parse_next_step_ok():
    step, err = parse_next_step(_plan("- [ ] (STEP_ID=001) do x\n"))
    assert err is None
    assert step == {"id": "001", "text": "do x"}


def test_parse_next_step_missing_section():
    step, err = parse_next_step("# PLAN\n")
    assert step is None
    assert err == "missing_next_section"


def test_parse_next_step_multiple_items():
    step, err = parse_next_step(_plan("- [ ] (STEP_ID=001) a\n- [ ] (STEP_ID=003) b\n"))
    assert step is None
    assert err == "next_count_not_one"


def test_parse_next_step_checked_is_error():
    step, err = parse_next_step(_plan("- [x] (STEP_ID=001) done\n"))
    assert step is None
    assert err == "next_is_checked"


def test_parse_next_step_bad_line_is_error():
    step, err = parse_next_step(_plan("- [ ] STEP_ID=001 bad\n"))
    assert step is None
    assert err == "bad_next_line"


def test_backlog_open_count_ok():
    plan = _plan("- [ ] (STEP_ID=001) x\n") + "\n"
    open_count, err = parse_backlog_open_count(plan)
    assert err is None
    assert open_count == 1


def test_duplicate_step_id_is_error():
    plan = (
        "# PLAN\n\n"
        "## Goal\n- g\n\n"
        "## Acceptance\n- [ ] TEST_CMD passes: `pytest -q`\n\n"
        "## Next (exactly ONE item)\n"
        "- [ ] (STEP_ID=001) a\n\n"
        "## Backlog\n"
        "- [ ] (STEP_ID=001) dup\n\n"
        "## Done\n"
        "- [x] (STEP_ID=000) d\n\n"
        "## Notes\n- \n"
    )
    parsed = parse_plan(plan)
    assert "duplicate_step_id" in parsed["errors"]
