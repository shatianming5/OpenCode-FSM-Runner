from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [ROOT / "runner", ROOT / "tests"]


_MEANING_RE = re.compile(r"(?:^-\\s*)?含义[:：]\\s*(?P<text>.+?)\\s*$")
_SIMPLIFY_RE = re.compile(r"(?:^-\\s*)?可简略[:：]\\s*(?P<text>.+?)\\s*$")


def _iter_py_files(base: Path) -> list[Path]:
    out: list[Path] = []
    if not base.exists():
        return out
    for p in base.rglob("*.py"):
        try:
            if not p.is_file():
                continue
        except Exception:
            continue
        # Skip caches.
        if "__pycache__" in p.parts:
            continue
        out.append(p.resolve())
    out.sort()
    return out


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _leading_ws(line: str) -> str:
    i = 0
    while i < len(line) and line[i] in (" ", "\t"):
        i += 1
    return line[:i]


def _docstring_of(node: ast.AST) -> str:
    try:
        return ast.get_docstring(node) or ""
    except Exception:
        return ""


def _extract_meaning(doc: str) -> str:
    lines = [ln.rstrip() for ln in (doc or "").splitlines()]
    for ln in lines:
        m = _MEANING_RE.match(ln.strip())
        if m:
            return str(m.group("text") or "").strip()
    for ln in lines:
        s = ln.strip()
        if s:
            return s
    return ""


def _extract_simplify(doc: str) -> str:
    lines = [ln.rstrip() for ln in (doc or "").splitlines()]
    for ln in lines:
        m = _SIMPLIFY_RE.match(ln.strip())
        if not m:
            continue
        raw = str(m.group("text") or "").strip()
        low = raw.lower()
        if "否" in raw or low.startswith("no"):
            return "否"
        if "是" in raw or low.startswith("yes"):
            return "是"
        # "可能" / "可" / "maybe" -> partial
        return "部分"
    return ""


def _is_core_api(path: Path, qualname: str) -> bool:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    if rel == "runner/env.py" and qualname in ("setup", "EnvSession.rollout", "EnvSession.evaluate"):
        return True
    return False


def _count_refs(symbol: str, *, texts: dict[Path, str], kind: str) -> int:
    if not symbol:
        return 0
    if kind == "method":
        pat = re.compile(rf"\.{re.escape(symbol)}\s*\(")
    else:
        pat = re.compile(rf"\b{re.escape(symbol)}\b")
    n = 0
    for txt in texts.values():
        n += len(pat.findall(txt))
    return int(n)


def _classify_simplifiable(
    *,
    explicit: str,
    name: str,
    path: Path,
    qualname: str,
    size_lines: int,
    ref_count: int,
    in_tests: bool,
) -> str:
    if explicit:
        return explicit
    if _is_core_api(path, qualname):
        return "否"
    if name.startswith("test_"):
        if size_lines >= 120:
            return "部分"
        return "否"
    if in_tests and size_lines >= 80:
        return "部分"
    if name.startswith("_") and size_lines <= 20 and ref_count <= 3:
        return "是"
    if size_lines <= 8 and ref_count <= 2:
        return "是"
    if size_lines <= 40 and ref_count <= 5 and (name.startswith("_") or in_tests):
        return "部分"
    if ref_count >= 12:
        return "否"
    if size_lines >= 120:
        return "部分"
    return "否"


def _build_reason(
    *,
    simplifiable: str,
    explicit_from_doc: bool,
    size_lines: int,
    ref_count: int,
    in_tests: bool,
    core_api: bool,
) -> str:
    # Keep each reason concrete and auditable (even if approximate).
    parts: list[str] = []
    if core_api:
        parts.append("公共 API/关键编排点")
    if in_tests:
        parts.append("测试代码（优先可读性）")
    if explicit_from_doc:
        parts.append("判定来自原 docstring 的“可简略”说明")
    parts.append(f"规模≈{size_lines} 行")
    parts.append(f"引用次数≈{ref_count}（静态近似，可能包含注释/字符串）")

    if simplifiable == "是":
        parts.append("逻辑短且低复用，适合 inline/合并以减少符号面")
    elif simplifiable == "部分":
        parts.append("可通过拆分/去重复/抽 helper 减少复杂度，但不建议完全内联")
    else:
        parts.append("多点复用或涉及副作用/协议验收，过度简化会增加回归风险或降低可审计性")
    return "；".join(parts)


def _annotation_lines(*, indent: str, meaning: str, simplifiable: str, reason: str, evidence: str) -> list[str]:
    return [
        f"{indent}# 作用：{meaning}\n",
        f"{indent}# 能否简略：{simplifiable}\n",
        f"{indent}# 原因：{reason}\n",
        f"{indent}# 证据：{evidence}\n",
    ]


@dataclass(frozen=True)
class _Target:
    path: Path
    qualname: str
    name: str
    kind: str  # class|function|method
    insert_idx: int  # 0-based insertion index in file lines
    start_idx: int  # 0-based start index (inclusive) of the search window for existing blocks
    end_idx: int  # 0-based end index (inclusive) of the search window for existing blocks
    indent: str
    size_lines: int
    doc: str


def _collect_targets(path: Path, text: str) -> list[_Target]:
    mod = ast.parse(text, filename=str(path))
    lines = text.splitlines(keepends=True)
    out: list[_Target] = []

    class _V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []
            self.class_stack: list[bool] = []

        def _push(self, name: str, *, is_class: bool) -> None:
            self.stack.append(name)
            self.class_stack.append(bool(is_class))

        def _pop(self) -> None:
            self.stack.pop()
            self.class_stack.pop()

        def _qual(self, name: str) -> str:
            return ".".join(self.stack + [name]) if self.stack else name

        def _is_method(self) -> bool:
            return bool(self.class_stack and self.class_stack[-1] is True)

        def _add(self, node: ast.AST, *, name: str, kind: str) -> None:
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                return
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                return

            doc_expr = None
            first_stmt = body[0]
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(getattr(first_stmt, "value", None), ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                doc_expr = first_stmt

            if doc_expr is not None:
                end = int(getattr(doc_expr, "end_lineno", doc_expr.lineno))
                # Insert on the line *after* the docstring expression.
                insert_idx = end
                indent = _leading_ws(lines[int(doc_expr.lineno) - 1]) if int(doc_expr.lineno) - 1 < len(lines) else " " * (
                    int(getattr(node, "col_offset", 0)) + 4
                )
                start_idx = max(0, int(end))
                if len(body) >= 2:
                    next_stmt = body[1]
                    end_idx = max(start_idx, int(getattr(next_stmt, "lineno")) - 1)
                else:
                    end_idx = min(len(lines), start_idx + 50)
            else:
                # Insert right after the header (between `def ...:` and the first statement),
                # not immediately before the first statement. This matches the test's
                # annotation window and avoids false "missing" reports for blocks that are
                # already placed at the top of the body.
                start_idx = max(0, int(getattr(node, "lineno")))
                insert_idx = start_idx
                end_idx = max(start_idx, int(getattr(first_stmt, "lineno")) - 1)
                indent = (
                    _leading_ws(lines[int(getattr(first_stmt, "lineno")) - 1])
                    if int(getattr(first_stmt, "lineno")) - 1 < len(lines)
                    else " " * (int(getattr(node, "col_offset", 0)) + 4)
                )

            size_lines = int(getattr(node, "end_lineno")) - int(getattr(node, "lineno")) + 1
            doc = _docstring_of(node)
            qualname = self._qual(name)
            out.append(
                _Target(
                    path=path,
                    qualname=qualname,
                    name=name,
                    kind=kind,
                    insert_idx=max(0, int(insert_idx)),
                    start_idx=max(0, int(start_idx)),
                    end_idx=max(0, int(end_idx)),
                    indent=indent,
                    size_lines=max(1, int(size_lines)),
                    doc=doc,
                )
            )

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._add(node, name=str(node.name), kind="class")
            self._push(str(node.name), is_class=True)
            self.generic_visit(node)
            self._pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            kind = "method" if self._is_method() else "function"
            self._add(node, name=str(node.name), kind=kind)
            self._push(str(node.name), is_class=False)
            self.generic_visit(node)
            self._pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            kind = "method" if self._is_method() else "function"
            self._add(node, name=str(node.name), kind=kind)
            self._push(str(node.name), is_class=False)
            self.generic_visit(node)
            self._pop()

    _V().visit(mod)
    return out


def _has_block(lines: list[str], *, start_idx: int, end_idx: int, indent: str) -> bool:
    start_idx = max(0, int(start_idx))
    end_idx = min(len(lines), max(start_idx, int(end_idx)))
    for start in range(start_idx, min(len(lines), end_idx + 1)):
        if lines[start].startswith(indent + "# 作用："):
            needed = (
                indent + "# 作用：",
                indent + "# 能否简略：",
                indent + "# 原因：",
                indent + "# 证据：",
            )
            for off, prefix in enumerate(needed):
                if start + off >= len(lines):
                    return False
                if not lines[start + off].startswith(prefix):
                    return False
            return True
    return False


def annotate_file(path: Path, *, texts: dict[Path, str]) -> bool:
    text = _read_text(path)
    lines = text.splitlines(keepends=True)
    targets = _collect_targets(path, text)
    if not targets:
        return False

    in_tests = str(path.relative_to(ROOT)).replace("\\", "/").startswith("tests/")

    edits: list[tuple[int, list[str]]] = []
    for t in targets:
        if _has_block(lines, start_idx=t.start_idx, end_idx=t.end_idx, indent=t.indent):
            continue

        meaning = _extract_meaning(t.doc) or (
            "pytest 测试用例：验证行为契约" if t.name.startswith("test_") else f"内部符号：{t.qualname}"
        )
        explicit_simplify = _extract_simplify(t.doc)
        ref_count = _count_refs(t.name, texts=texts, kind=("method" if t.kind == "method" else "name"))
        core_api = _is_core_api(path, t.qualname)
        simplifiable = _classify_simplifiable(
            explicit=explicit_simplify,
            name=t.name,
            path=path,
            qualname=t.qualname,
            size_lines=t.size_lines,
            ref_count=ref_count,
            in_tests=in_tests,
        )
        reason = _build_reason(
            simplifiable=simplifiable,
            explicit_from_doc=bool(explicit_simplify),
            size_lines=t.size_lines,
            ref_count=ref_count,
            in_tests=in_tests,
            core_api=core_api,
        )
        evidence = f"位置={path.relative_to(ROOT)}:{getattr(t, 'insert_idx', 0) + 1}；类型={t.kind}；引用≈{ref_count}；规模≈{t.size_lines}行"
        ann = _annotation_lines(indent=t.indent, meaning=meaning, simplifiable=simplifiable, reason=reason, evidence=evidence)
        edits.append((t.insert_idx, ann))

    if not edits:
        return False

    # Apply bottom-up to keep indices stable.
    edits.sort(key=lambda x: x[0], reverse=True)
    for idx, ann in edits:
        insert_at = min(max(0, int(idx)), len(lines))
        lines[insert_at:insert_at] = ann

    path.write_text("".join(lines), encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    write = "--write" in argv
    files: list[Path] = []
    for d in TARGET_DIRS:
        files.extend(_iter_py_files(d))

    # Preload texts for approximate reference counting.
    texts = {p: _read_text(p) for p in files}

    changed = 0
    for p in files:
        before = texts[p]
        if not write:
            # Dry-run: just report how many symbols would be annotated.
            targets = _collect_targets(p, before)
            if not targets:
                continue
            lines = before.splitlines(keepends=True)
            missing = 0
            for t in targets:
                if not _has_block(lines, start_idx=t.start_idx, end_idx=t.end_idx, indent=t.indent):
                    missing += 1
            if missing:
                print(f"{p.relative_to(ROOT)}: missing={missing}")
            continue

        did = annotate_file(p, texts=texts)
        if did:
            changed += 1

    if not write:
        print("dry-run complete (use --write to apply)")
    else:
        print(f"annotated {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
