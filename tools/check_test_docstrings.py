#!/usr/bin/env python3
"""Check that pytest tests carry GIVEN / WHEN / THEN docstrings.

Application code in sphinx-asr uses Google-style docstrings, enforced by
ruff. Tests instead describe a scenario: what is set up and what the
baseline is (Given), the single action under test (When), every assertion
(Then) and, when the test expects an exception, which ones (Raises). This
module enforces that format with the standard-library ``ast`` module only,
so it runs under any Python 3.12+ without third-party dependencies.

A *test* is a ``test_*`` function (sync or async) at module level, or a
``test_*`` method of a module-level class whose name starts with ``Test``.

Rules:
    TDS001: The test has no docstring.
    TDS002: The first line is not a non-empty sentence ending in a period.
    TDS003: A required section (Given, When, Then) is missing.
    TDS004: Sections are not in the order Given, When, Then, Raises.
    TDS005: A section header has no body text.
    TDS006: The body asserts an exception (``pytest.raises``,
        ``self.assertRaises``, ``self.assertRaisesRegex``) but there is no
        Raises section, or the section does not name every exception.
    TDS007: There is a Raises section but no exception assertion in the body.
    TDS008: The Given section does not state a baseline: it must contain the
        word "baseline" or "expected" (case-insensitive).

Section headers are the exact title-case words ``Given:``, ``When:``,
``Then:`` and ``Raises:`` alone on a line at the docstring's base
indentation. Section bodies are the indented lines that follow.

Usage:
    python tools/check_test_docstrings.py PATH [PATH ...]

    A PATH may be a file or a directory (searched recursively for ``*.py``).
    Findings are printed as ``path:line: TDSnnn message``. The exit status is
    0 when clean, 1 when there are findings, and 2 when a file cannot be read
    or parsed.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

SECTION_ORDER = ("Given", "When", "Then", "Raises")
REQUIRED_SECTIONS = ("Given", "When", "Then")
HEADER_RE = re.compile(r"^(Given|When|Then|Raises):\s*$")
BASELINE_RE = re.compile(r"\b(baseline|expected)\b", re.IGNORECASE)
RAISES_METHODS = frozenset({"assertRaises", "assertRaisesRegex"})


@dataclass(frozen=True)
class Finding:
    """A single rule violation.

    Attributes:
        path: File the violation was found in, as given on the command line.
        line: 1-based line number of the offending ``def``.
        code: Rule code such as ``TDS001``.
        message: Human-readable description of the violation.
    """

    path: str
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        """Render the finding as ``path:line: CODE message``.

        Returns:
            The formatted finding.
        """
        return f"{self.path}:{self.line}: {self.code} {self.message}"


@dataclass
class _Section:
    """A parsed docstring section.

    Attributes:
        name: Section name, one of ``SECTION_ORDER``.
        body: Non-blank body lines, stripped.
    """

    name: str
    body: list[str]


def _iter_tests(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield every function pytest would collect as a test.

    Args:
        tree: Parsed module.

    Yields:
        Module-level ``test_*`` functions and ``test_*`` methods of
        module-level ``Test*`` classes, in source order.
    """
    func_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, func_types) and node.name.startswith("test"):
            yield node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, func_types) and item.name.startswith("test"):
                    yield item


def _parse_sections(lines: list[str]) -> list[_Section]:
    """Split docstring lines into sections.

    Args:
        lines: Docstring lines after ``inspect.cleandoc``-style dedenting.

    Returns:
        Sections in the order their headers appear. Text before the first
        header (the summary and description) is not part of any section.
    """
    sections: list[_Section] = []
    for line in lines:
        match = HEADER_RE.match(line)
        if match:
            sections.append(_Section(match.group(1), []))
        elif sections and line.strip():
            sections[-1].body.append(line.strip())
    return sections


def _exception_names(node: ast.expr) -> list[str]:
    """Extract exception class names from a ``raises`` argument.

    Args:
        node: First argument to ``pytest.raises`` or ``assertRaises``: a
            name, a dotted attribute, or a tuple of those.

    Returns:
        The final component of each name (``yaml.YAMLError`` becomes
        ``YAMLError``). Unrecognised expressions yield no names.
    """
    if isinstance(node, ast.Tuple):
        return [name for elt in node.elts for name in _exception_names(elt)]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []


def _is_raises_call(call: ast.Call) -> bool:
    """Decide whether a call is an exception assertion.

    Args:
        call: Any call expression in a test body.

    Returns:
        True for ``pytest.raises(...)``, bare ``raises(...)`` and
        ``self.assertRaises(...)`` / ``self.assertRaisesRegex(...)``.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "raises"
    if isinstance(func, ast.Attribute):
        if func.attr in RAISES_METHODS:
            return True
        return (
            func.attr == "raises"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytest"
        )
    return False


def _asserted_exceptions(func: ast.AST) -> tuple[bool, list[str]]:
    """Find exception assertions in a test body.

    Args:
        func: The test function node.

    Returns:
        A pair ``(found, names)``: whether any exception assertion occurs,
        and the exception names asserted, de-duplicated in source order.
    """
    found = False
    names: list[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and _is_raises_call(node):
            found = True
            if node.args:
                for name in _exception_names(node.args[0]):
                    if name not in names:
                        names.append(name)
    return found, names


def _check_test(
    func: ast.FunctionDef | ast.AsyncFunctionDef, path: str
) -> list[Finding]:
    """Apply every TDS rule to one test function.

    Args:
        func: The test function node.
        path: File path used in findings.

    Returns:
        Findings for this test, ordered by rule code.
    """
    findings: list[Finding] = []

    def report(code: str, message: str) -> None:
        findings.append(Finding(path, func.lineno, code, message))

    doc = ast.get_docstring(func, clean=True)
    if not doc or not doc.strip():
        report("TDS001", f"{func.name}: missing docstring")
        return findings

    lines = doc.splitlines()
    summary = lines[0].strip()
    if not summary or HEADER_RE.match(summary) or not summary.endswith("."):
        report(
            "TDS002",
            f"{func.name}: first line must be a one-sentence summary "
            "ending in a period",
        )

    sections = _parse_sections(lines)
    present = {s.name: s for s in reversed(sections)}  # first occurrence wins

    for name in REQUIRED_SECTIONS:
        if name not in present:
            report("TDS003", f"{func.name}: missing '{name}:' section")

    ranks = [SECTION_ORDER.index(s.name) for s in sections]
    if any(b <= a for a, b in zip(ranks, ranks[1:], strict=False)):
        order = ", ".join(s.name for s in sections)
        report(
            "TDS004",
            f"{func.name}: sections out of order ({order}); expected "
            "Given, When, Then, then optional Raises",
        )

    for section in sections:
        if not section.body:
            report("TDS005", f"{func.name}: '{section.name}:' section is empty")

    asserts_raises, exc_names = _asserted_exceptions(func)
    raises = present.get("Raises")
    if asserts_raises and raises is None:
        report(
            "TDS006",
            f"{func.name}: asserts an exception but has no 'Raises:' section",
        )
    elif raises is not None:
        raises_text = " ".join(raises.body)
        for name in exc_names:
            if not re.search(rf"\b{re.escape(name)}\b", raises_text):
                report(
                    "TDS006",
                    f"{func.name}: 'Raises:' section does not mention {name}",
                )
        if not asserts_raises:
            report(
                "TDS007",
                f"{func.name}: 'Raises:' section present but the body has "
                "no pytest.raises / assertRaises",
            )

    given = present.get("Given")
    if given is not None and given.body:
        if not BASELINE_RE.search(" ".join(given.body)):
            report(
                "TDS008",
                f"{func.name}: 'Given:' must state a baseline (use the word "
                "'baseline' or 'expected')",
            )

    findings.sort(key=lambda f: f.code)
    return findings


def check_source(source: str, path: str = "<string>") -> list[Finding]:
    """Check every test in a module's source code.

    A ``SyntaxError`` from ``ast.parse`` propagates unchanged when
    ``source`` does not parse; ``main`` turns it into exit status 2.

    Args:
        source: Python source text.
        path: Name used in findings.

    Returns:
        All findings, ordered by test position and then rule code.
    """
    tree = ast.parse(source, filename=path)
    findings: list[Finding] = []
    for func in _iter_tests(tree):
        findings.extend(_check_test(func, path))
    return findings


def _expand(paths: Sequence[str]) -> Iterator[Path]:
    """Expand command-line arguments into Python files.

    Args:
        paths: File or directory paths.

    Yields:
        Each file as given, and every ``*.py`` under each directory in
        sorted order.
    """
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        else:
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker as a command-line tool.

    Args:
        argv: Paths to check. Defaults to ``sys.argv[1:]``.

    Returns:
        0 if clean, 1 if any findings were printed, 2 if any file could not
        be read or parsed (this takes precedence over 1).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    status = 0
    for path in _expand(args):
        try:
            source = path.read_text(encoding="utf-8")
            findings = check_source(source, str(path))
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as exc:
            print(f"{path}: cannot check: {exc}", file=sys.stderr)
            status = 2
            continue
        for finding in findings:
            print(finding)
        if findings and status == 0:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
