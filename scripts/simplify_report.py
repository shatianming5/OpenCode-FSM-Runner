from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATE = ROOT / "scripts" / "annotate_symbols.py"


def _load_annotate_module():
    name = "_aider_runner_annotate_symbols"
    spec = importlib.util.spec_from_file_location(name, ANNOTATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed_to_load: {ANNOTATE}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses expects the module to be in sys.modules during exec.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[call-arg]
    return mod


def _is_dunder(name: str) -> bool:
    s = str(name or "")
    return s.startswith("__") and s.endswith("__") and len(s) >= 5


def _action_priority(action: str) -> int:
    return {"MERGE": 0, "INLINE": 1, "SIMPLIFY": 2, "DELETE": 3, "KEEP": 9}.get(str(action or ""), 50)


def _suggest_action(
    *,
    name: str,
    kind: str,
    qualname: str,
    simplifiable: str,
    core_api: bool,
    dup_count: int,
    in_tests: bool,
) -> str:
    if core_api:
        return "KEEP"
    if in_tests:
        return "KEEP"
    if _is_dunder(name):
        return "KEEP"
    is_top_level_fn = kind == "function" and "." not in str(qualname or "")
    if name.startswith("_") and is_top_level_fn and dup_count >= 2:
        return "MERGE"
    if simplifiable == "是":
        return "INLINE" if name.startswith("_") else "SIMPLIFY"
    if simplifiable == "部分":
        return "SIMPLIFY"
    return "KEEP"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate simplification candidates report (static).")
    ap.add_argument("--include-tests", action="store_true", help="Include targets from tests/ in the report.")
    ap.add_argument("--out-md", default=str(ROOT / "docs" / "simplify_report.md"))
    ap.add_argument("--out-csv", default=str(ROOT / "docs" / "simplify_report.csv"))
    ns = ap.parse_args(argv)

    mod = _load_annotate_module()

    runner_files = list(mod._iter_py_files(ROOT / "runner"))
    test_files = list(mod._iter_py_files(ROOT / "tests"))
    report_files = runner_files + (test_files if ns.include_tests else [])

    # Count refs across runner+tests so test-only usage still influences keep/inline decisions.
    texts: dict[Path, str] = {p: mod._read_text(p) for p in (runner_files + test_files)}

    # Detect duplicate top-level private functions across runner/.
    name_occurrences: Counter[str] = Counter()
    for p in runner_files:
        text = texts[p]
        for t in mod._collect_targets(p, text):
            if t.kind != "function":
                continue
            if "." in str(t.qualname or ""):
                continue
            if not str(t.name).startswith("_") or _is_dunder(t.name):
                continue
            name_occurrences[str(t.name)] += 1

    rows: list[dict[str, object]] = []
    for p in report_files:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        in_tests = rel.startswith("tests/")
        text = texts.get(p) or mod._read_text(p)
        for t in mod._collect_targets(p, text):
            explicit_simplify = mod._extract_simplify(t.doc)
            ref_count = mod._count_refs(t.name, texts=texts, kind=("method" if t.kind == "method" else "name"))
            core_api = bool(mod._is_core_api(p, t.qualname))
            simplifiable = mod._classify_simplifiable(
                explicit=explicit_simplify,
                name=t.name,
                path=p,
                qualname=t.qualname,
                size_lines=t.size_lines,
                ref_count=ref_count,
                in_tests=in_tests,
            )
            dup_count = int(name_occurrences.get(str(t.name), 0))
            action = _suggest_action(
                name=str(t.name),
                kind=str(t.kind),
                qualname=str(t.qualname),
                simplifiable=str(simplifiable),
                core_api=core_api,
                dup_count=dup_count,
                in_tests=in_tests,
            )
            reason = mod._build_reason(
                simplifiable=simplifiable,
                explicit_from_doc=bool(explicit_simplify),
                size_lines=t.size_lines,
                ref_count=ref_count,
                in_tests=in_tests,
                core_api=core_api,
            )
            evidence = f"{rel}:{getattr(t, 'insert_idx', 0) + 1}; kind={t.kind}; refs≈{ref_count}; lines≈{t.size_lines}"
            rows.append(
                {
                    "path": rel,
                    "kind": str(t.kind),
                    "qualname": str(t.qualname),
                    "name": str(t.name),
                    "action": action,
                    "simplifiable": str(simplifiable),
                    "lines": int(t.size_lines),
                    "refs": int(ref_count),
                    "dup_count": int(dup_count),
                    "core_api": bool(core_api),
                    "in_tests": bool(in_tests),
                    "reason": str(reason),
                    "evidence": str(evidence),
                }
            )

    # CSV: full surface.
    out_csv = Path(ns.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "kind",
        "qualname",
        "name",
        "action",
        "simplifiable",
        "lines",
        "refs",
        "dup_count",
        "core_api",
        "in_tests",
        "reason",
        "evidence",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # MD: only candidates (non-KEEP).
    out_md = Path(ns.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    by_action = Counter(str(r["action"]) for r in rows)
    by_simp = Counter(str(r["simplifiable"]) for r in rows)
    by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_file[str(r["path"])].append(r)

    dup_names = sorted([(n, c) for n, c in name_occurrences.items() if c >= 2], key=lambda x: (-x[1], x[0]))

    parts: list[str] = []
    parts.append("# Simplification Report (static)\n")
    parts.append("Generated via AST scan + approximate reference counting.\n\n")
    parts.append("Notes:\n")
    parts.append("- `refs≈` is a regex count across `runner/` + `tests/` (may include comments/strings; may miss reflection).\n")
    parts.append("- `action` is a suggestion only; acceptance is `pytest -q`.\n\n")

    parts.append("## Summary\n")
    parts.append(f"- total symbols: {len(rows)}\n")
    parts.append("- by action:\n")
    for k, v in sorted(by_action.items(), key=lambda kv: (_action_priority(kv[0]), kv[0])):
        parts.append(f"  - {k}: {v}\n")
    parts.append("- by simplifiable:\n")
    for k, v in sorted(by_simp.items(), key=lambda kv: ({"是": 0, "部分": 1, "否": 2}.get(kv[0], 9), kv[0])):
        parts.append(f"  - {k}: {v}\n")
    parts.append("\n")

    if dup_names:
        parts.append("## Duplicate Top-level Private Functions (runner/)\n")
        parts.append("Good MERGE candidates if semantics match.\n\n")
        for n, c in dup_names[:80]:
            parts.append(f"- `{n}`: {c} occurrences\n")
        parts.append("\n")

    parts.append("## Per-file Candidates (non-KEEP)\n")
    parts.append("Only symbols with suggested action != `KEEP` are listed below. Full surface is in CSV.\n\n")

    for path in sorted(by_file.keys()):
        file_rows = by_file[path]
        cands = [r for r in file_rows if str(r["action"]) != "KEEP"]
        if not cands:
            continue
        parts.append(f"### `{path}`\n\n")
        cands.sort(key=lambda r: (_action_priority(str(r["action"])), int(r["refs"]), int(r["lines"]), str(r["qualname"])))
        for r in cands:
            action = str(r["action"])
            simp = str(r["simplifiable"])
            kind = str(r["kind"])
            qual = str(r["qualname"])
            refs = int(r["refs"])
            lines = int(r["lines"])
            dup = int(r["dup_count"])
            extra: list[str] = []
            if dup >= 2 and str(r["name"]).startswith("_") and kind == "function" and "." not in qual:
                extra.append(f"dup={dup}")
            extra_s = f" ({', '.join(extra)})" if extra else ""
            parts.append(f"- **{action}** `{qual}` [{kind}] simp={simp} refs≈{refs} lines≈{lines}{extra_s}\n")
            parts.append(f"  - reason: {r['reason']}\n")
            parts.append(f"  - evidence: {r['evidence']}\n")
        parts.append("\n")

    out_md.write_text("".join(parts), encoding="utf-8")
    print(f"wrote: {out_md.relative_to(ROOT)}")
    print(f"wrote: {out_csv.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

