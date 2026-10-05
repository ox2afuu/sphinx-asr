"""Shared helpers used by every sphinx-asr script.

This module is imported by the scripts under `scripts/` (`setup.py`,
`train.py`, `decode.py`, `feats.py`, `lm.py`, `new_experiment.py`) and by
`lib/config.py`. No `sphinx.sh` subcommand runs it directly.

It provides fatal-error reporting (`err`), project-root resolution
(`get_sphinx_root`), and two carriage-return progress renderers
(`waitbar`, `spinner`) used by `decode.py`.

Environment variables read:

- `SPHINX_ROOT`: absolute path of the sphinx-asr checkout, read by
  `get_sphinx_root`. `sphinx.sh` exports it before running any script.

Files read or written: none.

Attributes:
    SPINNER_CHARS: Frames cycled through by `spinner`, in display order.
"""
import os
import sys
from pathlib import Path

SPINNER_CHARS = ["|", "/", "-", "\\"]


def err(msg: str):
    """Print an error message to stderr and terminate the process.

    The message is prefixed with `[ERROR] ` and written to `sys.stderr`, then
    the process exits with status 1. The function never returns, so callers
    use it as the last statement on an unrecoverable path.

    Args:
        msg: Human-readable description of the failure. May contain
            newlines.

    Raises:
        SystemExit: Always, with exit code 1.
    """  # noqa: DOC502
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def get_sphinx_root() -> Path:
    """Return the sphinx-asr project root from the `SPHINX_ROOT` variable.

    All scripts resolve corpus, vendor and experiment paths relative to this
    directory instead of the current working directory. The value is used
    as-is: it is not resolved, made absolute, or checked for existence.

    Returns:
        The value of `SPHINX_ROOT` as a `Path`.

    Raises:
        SystemExit: If `SPHINX_ROOT` is unset or empty (exit code 1, via
            `err`).
    """  # noqa: DOC502
    root = os.environ.get("SPHINX_ROOT")
    if not root:
        err("SPHINX_ROOT environment variable not set. Exiting.")
    return Path(root)


def waitbar(current: int, total: int, label: str = "") -> str:
    """Render a one-line text progress bar sized to the terminal.

    The bar width is the terminal width minus the counter suffix, with a
    minimum of 10 characters. If the terminal size cannot be queried (for
    example when stdout is not a TTY) a terminal width of 40 columns is
    assumed. The string begins with a carriage return and has no trailing
    newline, so printing it with `end=""` redraws the bar in place.

    Args:
        current: Number of completed items. Values above `total` are not
            clamped and overfill the bar.
        total: Total number of items. When 0, the bar is drawn empty and the
            percentage is reported as 0.0.
        label: Optional text appended after the counter.

    Returns:
        A carriage return followed by a bar such as
            `[=====>     ] 45/100 (45.0%) label`.
    """
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 40

    suffix = f" {current}/{total} ({(current/total*100) if total else 0:.1f}%)"
    suffix += f" {label}" if label else ""
    bar_width = term_width - len(suffix) - 2
    bar_width = max(bar_width, 10)

    if total == 0:
        pct = 0.0
        filled = 0
    else:
        pct = current / total
        filled = int(bar_width * pct)

    bar = "=" * filled
    if filled < bar_width:
        bar += ">"
    bar = bar.ljust(bar_width)

    return f"\r[{bar}]{suffix}"


def spinner(idx: int, label: str = "") -> str:
    r"""Render one frame of a text spinner animation.

    Args:
        idx: Frame counter. It is taken modulo the number of frames in
            `SPINNER_CHARS`, so callers can increment it on every redraw.
        label: Optional text shown after the spinner character.

    Returns:
        A carriage return, the spinner character and, if given, a space and
            the label. There is no trailing newline.

    Examples:
        >>> spinner(1, "aligning...")
        '\r/ aligning...'
        >>> spinner(4)
        '\r|'
    """
    char = SPINNER_CHARS[idx % len(SPINNER_CHARS)]
    suffix = f" {label}" if label else ""
    return f"\r{char}{suffix}"
