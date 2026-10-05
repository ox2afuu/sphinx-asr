---
title: sphinx-asr docstring and test standards
description: Google-style docstrings for application code, GIVEN/WHEN/THEN docstrings for tests, and the hooks that enforce them.
---

# sphinx-asr docstring and test standards

sphinx-asr's Python code (`scripts/`, `tools/`) and its test suite (`tests/`)
follow two docstring formats, each checked by a tool:

| Code             | Format                | Checked by                                          |
| ---------------- | --------------------- | --------------------------------------------------- |
| `scripts/**/*.py`, `tools/**/*.py` | Google style | `ruff` pydocstyle (`D`, convention `google`) and pydoclint (`DOC`, preview) |
| `tests/**/*.py`  | GIVEN / WHEN / THEN   | `tools/check_test_docstrings.py` (rules `TDS001`-`TDS008`) |

All tool configuration lives in `pyproject.toml`. Git hooks live in
`lefthook.yml`.

!!! note "Runtime environment is untouched"
    The dev tools run through `uv` / `uvx` with `--no-project`. They never
    install into `.venv/`, which `sphinx.sh` builds from `requirements.txt`
    and which is prebuilt from airgapped wheels on the ivaliceCluster.
    `requirements.txt` stays the runtime source of truth; `pyproject.toml`
    only mirrors it.

## Application code: Google style

Every public module, class, method and function has a docstring:

- A one-line summary in the imperative mood ending in a period, on the same
  line as the opening quotes, followed by a blank line if there is more.
- `Args:` for every parameter (once the section exists, ruff `D417` requires
  all of them).
- `Returns:` (or `Yields:`) whenever the function returns (yields) a value
  (`DOC201`, `DOC402`).
- `Raises:` for every exception the function raises **explicitly** with a
  `raise` statement (`DOC501`). Do not list exceptions that only propagate
  from a callee: `DOC502` rejects them. Describe propagated exceptions in
  the prose body instead.
- Optional `Warning:`, `Note:` and `Examples:` sections.

### Example

Based on `scripts/lib/config.py::load_yaml`:

```python
def load_yaml(path: Path) -> dict:
    """Load a YAML file whose top-level value must be a mapping.

    Used for both ``experiment.yml`` and ``corpus/<name>/corpus.yml``.
    ``OSError`` from opening the file propagates unchanged.

    Args:
        path: YAML file to read.

    Returns:
        The parsed mapping. An empty file yields ``{}``.

    Raises:
        ValueError: If the file is not valid YAML. The original
            ``yaml.YAMLError`` is chained as ``__cause__``.
        TypeError: If the top-level YAML value is not a mapping.

    Warning:
        An empty file is not an error. Callers that need specific keys must
        validate them, as ``load_experiment`` does via
        ``validate_experiment``.

    Examples:
        >>> corpus = load_yaml(Path("corpus/librispeech/corpus.yml"))
        >>> corpus["sample_rate"]
        16000
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: invalid YAML syntax\n{e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected yaml mapping in {path}")
    return data
```

## Tests: GIVEN / WHEN / THEN

Tests follow Martin Fowler's
[GivenWhenThen](https://martinfowler.com/bliki/GivenWhenThen.html) and
Robert C. Martin's clean-tests guidance: one behaviour per test, one action,
assertions that read as the specification. The docstring is that
specification and must cover:

- **Given**: the inputs and fixtures, and the **baseline**: the expected or
  reference state the result is judged against.
- **When**: the single action under test.
- **Then**: every assertion the test makes.
- **Raises** (only when the test asserts an exception): each exception class
  it expects.

The body mirrors the docstring: arrange, a blank line, act, a blank line,
assert.

### Example

From `tests/scripts/test_config.py`:

```python
def test_validate_split_rejects_unknown_split():
    """An unknown split raises ``ValueError`` listing the available ones.

    Given:
        A corpus with splits ``train`` and ``test``; expected: ``dev`` is
        rejected.
    When:
        ``_validate_split`` is called for ``dev``.
    Then:
        The error message names the corpus and lists ``train, test``.
    Raises:
        ValueError: because ``dev`` is not a split of the corpus.
    """
    corpus = {"splits": {"train": {}, "test": {}}}

    with pytest.raises(ValueError, match=r"'dev'.*'demo'.*train, test"):
        _validate_split("demo", "dev", corpus)
```

Section headers are the exact words `Given:`, `When:`, `Then:` and `Raises:`
alone on a line at the docstring's base indentation; their bodies are
indented beneath them.

### Which functions are checked

Functions whose name starts with `test` (sync or async) at module level, and
methods whose name starts with `test` on module-level classes whose name
starts with `Test`, matching pytest's default collection. Helpers and
fixtures are not checked by `TDS` rules but still need a docstring under
ruff's `D103`.

### Rules

| Code   | Rule                                                                                         |
| ------ | -------------------------------------------------------------------------------------------- |
| TDS001 | The test has no docstring.                                                                   |
| TDS002 | The first line is not a non-empty, one-sentence summary ending in a period.                  |
| TDS003 | A required section is missing: `Given`, `When` and `Then` are all required (one finding each). |
| TDS004 | Sections are out of order. The order is `Given`, `When`, `Then`, then optional `Raises`; repeating a section also counts. |
| TDS005 | A section header has no body text.                                                           |
| TDS006 | The body asserts an exception (`pytest.raises(...)`, `with pytest.raises(...)`, bare `raises(...)`, `self.assertRaises`, `self.assertRaisesRegex`) but there is no `Raises:` section, or the section does not name every exception asserted. Dotted names match on their last part, so `yaml.YAMLError` needs `YAMLError`. |
| TDS007 | There is a `Raises:` section but the body has no exception assertion.                        |
| TDS008 | The `Given:` section does not state a baseline: it must contain the word `baseline` or `expected` (any case). This keeps every test explicit about what "correct" is measured against. |

Output is `path:line: TDSnnn message`, where `line` is the test's `def` line.
Exit status: `0` clean, `1` findings, `2` a file could not be read or parsed.

### Ruff and tests

`tests/**` keeps ruff's `D100`-`D107` (a docstring must exist) and `D2xx`
formatting rules, but ignores the Google section rules (`D405`-`D417`) and
all `DOC` rules, since `Given`/`When`/`Then` are not Google sections and a
test's `Raises:` documents what it *expects*, not what it raises.

### Known defects as strict xfails

Defects found in review are encoded as tests that assert the **correct**
behaviour, marked:

```python
@pytest.mark.xfail(strict=True, reason="Known defect (review 2026-10-04 #6): ...")
```

While the bug exists the test reports `XFAIL`. When someone fixes it, the
test `XPASS`es, which strict mode turns into a failure: remove the marker so
the test becomes a regression guard. See `tests/scripts/test_known_defects.py`.

## Running the checks

```bash
# Tests (pytest + runtime deps in a throwaway env; .venv/ is not touched)
uv run --no-project --with pytest --with-requirements requirements.txt python -m pytest -q

# Google docstrings
uvx ruff@0.16.10 check --select D,DOC,F --preview scripts tools

# GIVEN/WHEN/THEN docstrings (files or directories)
uv run --no-project python tools/check_test_docstrings.py tests

# No emoji in text files
uv run --no-project python tools/check_no_emoji.py <files...>
```

## Git hooks

Install once per clone:

```bash
lefthook install
```

| Hook       | Job                 | Scope                                       | Command                                      |
| ---------- | ------------------- | ------------------------------------------- | -------------------------------------------- |
| pre-commit | `docstrings-google` | staged `scripts/**/*.py`, `tools/**/*.py`   | `ruff check --select D,DOC,F --preview`      |
| pre-commit | `docstrings-tests`  | staged `tests/**/*.py`                      | `tools/check_test_docstrings.py`             |
| pre-commit | `no-emoji`          | all staged files except `vendor/**`         | `tools/check_no_emoji.py`                    |
| pre-commit | `pytest-fast`       | runs if anything under `tests/`, `scripts/` or `tools/` is staged | `pytest -m "not slow"` |
| pre-push   | `docstrings-google` | whole `scripts/` and `tools/`               | `ruff check --select D,DOC,F --preview`      |
| pre-push   | `docstrings-tests`  | whole `tests/`                              | `tools/check_test_docstrings.py`             |
| pre-push   | `pytest`            | whole suite                                 | `pytest`                                     |

!!! tip "The pre-commit hook is a ratchet"
    Pre-commit only checks staged files. Any file you touch must meet the
    standard, but untouched legacy files do not block a commit. Pre-push
    checks the whole tree, so legacy files must be brought up to standard
    before a branch can be pushed.

Mark slow tests with `@pytest.mark.slow` to keep them out of `pytest-fast`;
they still run on pre-push.
