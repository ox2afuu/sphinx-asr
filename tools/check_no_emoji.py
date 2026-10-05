#!/usr/bin/env python3
"""Reject emoji and pictographic symbols in text files.

The repository convention is plain text: no emoji in code, comments, docs,
commit-tracked config or test output. This checker scans the given files
for code points in the emoji and pictograph blocks and reports each hit.
Binary files (anything that is not valid UTF-8 or contains a NUL byte) are
skipped silently, so it is safe to run over every staged file.

Ordinary typographic symbols such as arrows (U+2190 to U+21FF), the
multiplication sign and box-drawing characters are allowed.

Usage:
    python tools/check_no_emoji.py PATH [PATH ...]

    Findings are printed as ``path:line:col: EMJ001 U+XXXX``. The exit
    status is 0 when clean and 1 when any emoji is found.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # mahjong .. symbols and pictographs ext-A
    "\u2600-\u27bf"  # misc symbols, dingbats
    "\u2b00-\u2bff"  # misc symbols and arrows (stars, squares, circles)
    "\u231a\u231b\u2328\u23cf\u23e9-\u23fa"  # watch, hourglass, media keys
    "\u24c2\u3030\u303d\u3297\u3299"
    "\ufe0f"  # emoji presentation selector
    "\U000e0020-\U000e007f"  # tag characters (flag sequences)
    "]"
)


def find_emoji(text: str) -> list[tuple[int, int, str]]:
    """Locate every emoji code point in a string.

    Args:
        text: The text to scan.

    Returns:
        ``(line, column, char)`` triples, 1-based, in reading order.
    """
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in EMOJI_RE.finditer(line):
            hits.append((lineno, match.start() + 1, match.group()))
    return hits


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker as a command-line tool.

    Args:
        argv: Paths to check. Defaults to ``sys.argv[1:]``.

    Returns:
        0 if no emoji were found, 1 otherwise.
    """
    status = 0
    for raw in sys.argv[1:] if argv is None else argv:
        path = Path(raw)
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line, col, char in find_emoji(text):
            print(f"{path}:{line}:{col}: EMJ001 emoji U+{ord(char):04X} not allowed")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
