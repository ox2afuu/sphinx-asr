"""Known defects from the 2026-10-04 code review, encoded as strict xfails.

Each xfail test asserts the CORRECT behaviour. While the defect exists the
test fails and is reported as XFAIL. ``xfail_strict = true`` means that once
the defect is fixed the test XPASSes, which pytest reports as a failure: at
that point delete the ``xfail`` marker so the test becomes a regression
guard. Passing tests in this module pin behaviour that a fix must preserve.
"""

import ast
import builtins
from pathlib import Path

import pytest
import train
from lib.config import _apply_overrides

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


# --------------------------------------------------------------- review #6


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect (review 2026-10-04 #6): _apply_overrides "
        "(scripts/lib/config.py:279-303) silently drops overrides for keys "
        "absent from the sphinx_train.cfg template."
    ),
)
def test_override_for_key_absent_from_template_is_not_dropped():
    """An override for a variable missing from the template is still emitted.

    Given:
        A template that only assigns ``$CFG_X`` and an override for
        ``CFG_NEW``; expected: the user's ``sphinxtrain:`` key reaches
        the generated config (or setup fails loudly).
    When:
        ``_apply_overrides`` is applied.
    Then:
        The output contains ``$CFG_NEW = 5;``.
    """
    cfg = "$CFG_X = 1;\n"

    result = _apply_overrides(cfg, {"CFG_NEW": 5})

    assert "$CFG_NEW = 5;" in result


# -------------------------------------------------------------- review #11


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect (review 2026-10-04 #11): _apply_overrides passes the "
        "value into re.sub's replacement template unescaped, so a backslash "
        "group reference such as \\1 is expanded and corrupts the line."
    ),
)
def test_backslash_group_reference_in_value_is_written_literally():
    r"""A value containing ``\1`` is written verbatim.

    Given:
        A template line ``$CFG_X = 1;`` and the value ``C:\1``; expected:
        ``$CFG_X = 'C:\1';``. Today ``\1`` expands to the first regex group.
    When:
        ``_apply_overrides`` sets ``CFG_X``.
    Then:
        The line is exactly ``$CFG_X = 'C:\1';``.
    """
    cfg = "$CFG_X = 1;\n"

    result = _apply_overrides(cfg, {"CFG_X": "C:\\1"})

    assert result == "$CFG_X = 'C:\\1';\n"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect (review 2026-10-04 #11): an unknown backslash escape "
        "such as \\q in an override value makes re.sub raise re.error."
    ),
)
def test_unknown_backslash_escape_in_value_is_written_literally():
    r"""A value containing ``\q`` is written verbatim instead of crashing.

    Given:
        A template line ``$CFG_X = 1;`` and the value ``a\q``; expected:
        ``$CFG_X = 'a\q';``. Today ``re.sub`` raises ``re.error``.
    When:
        ``_apply_overrides`` sets ``CFG_X``.
    Then:
        The line is exactly ``$CFG_X = 'a\q';``.
    """
    cfg = "$CFG_X = 1;\n"

    result = _apply_overrides(cfg, {"CFG_X": "a\\q"})

    assert result == "$CFG_X = 'a\\q';\n"


# --------------------------------------------------------------- review #1


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` that replays canned output.

    Attributes:
        stdout: Iterator over the canned output lines.
        returncode: Exit status reported after the ``with`` block.
    """

    def __init__(self, lines: list[str], returncode: int):
        """Store the canned output and exit status.

        Args:
            lines: Lines the fake process prints.
            returncode: Exit status to report.
        """
        self.stdout = iter(lines)
        self.returncode = returncode

    def __enter__(self):
        """Enter the context manager.

        Returns:
            This fake process.
        """
        return self

    def __exit__(self, *exc_info):
        """Exit the context manager without suppressing exceptions.

        Args:
            *exc_info: Exception triple, ignored.

        Returns:
            False, so exceptions propagate.
        """
        return False


def _run_fake_step(monkeypatch, tmp_path, step, returncode):
    """Run ``train.run_step`` with Perl replaced by a fake process.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Experiment directory to use.
        step: The ``TrainingStep`` to run.
        returncode: Exit status of the fake Perl process.

    Returns:
        Whatever ``run_step`` returns.
    """
    monkeypatch.setattr(
        train.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakePopen(["working\n"], returncode),
    )
    return train.run_step(
        step, tmp_path / "scripts", tmp_path, "db", tmp_path / "logdir"
    )


def test_check_artifacts_is_vacuous_for_step_without_artifacts(tmp_path):
    """A step with no declared artifacts always passes the artifact check.

    Given:
        A ``TrainingStep`` with ``artifacts=[]`` and an empty model dir;
        baseline: this ``[]`` is what ``run_step`` reads as "artifacts
        present", which is the root of review #1.
    When:
        ``check_artifacts`` is called.
    Then:
        It returns an empty error list.
    """
    step = train.TrainingStep(script="00.verify/verify_all.pl")

    errors = train.check_artifacts(step, tmp_path, "db")

    assert errors == []


def test_nonzero_exit_with_present_artifacts_is_tolerated(monkeypatch, tmp_path):
    """MLLT's exit-1-on-convergence is tolerated when artifacts exist.

    Given:
        A step declaring ``{db}.mllt`` which exists and is non-empty, and a
        fake Perl run exiting 1; expected: intended MLLT behaviour that a
        fix for #1 must preserve.
    When:
        ``train.run_step`` runs the step.
    Then:
        It returns the elapsed time instead of exiting.
    """
    (tmp_path / "model_parameters").mkdir()
    (tmp_path / "model_parameters" / "db.mllt").write_text("transform")
    step = train.TrainingStep(
        script="02.mllt_train/slave_mllt.pl", artifacts=["{db}.mllt"]
    )

    elapsed = _run_fake_step(monkeypatch, tmp_path, step, returncode=1)

    assert isinstance(elapsed, float)


def test_nonzero_exit_with_missing_artifacts_exits(monkeypatch, tmp_path):
    """A nonzero exit with missing artifacts aborts training.

    Given:
        A step declaring ``{db}.lda`` which does not exist, and a fake Perl
        run exiting 3; expected: the step's exit status is propagated.
    When:
        ``train.run_step`` runs the step.
    Then:
        ``SystemExit`` is raised with code 3.
    Raises:
        SystemExit: from ``sys.exit(ret)`` in ``run_step``.
    """
    step = train.TrainingStep(
        script="01.lda_train/slave_lda.pl", artifacts=["{db}.lda"]
    )

    with pytest.raises(SystemExit) as exc_info:
        _run_fake_step(monkeypatch, tmp_path, step, returncode=3)

    assert exc_info.value.code == 3


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect (review 2026-10-04 #1): scripts/train.py:234-251 "
        "treats a nonzero exit as success when the step declares no "
        "artifacts, because check_artifacts() returns [] vacuously."
    ),
)
def test_nonzero_exit_without_declared_artifacts_exits(monkeypatch, tmp_path):
    """A failing step with no declared artifacts must abort training.

    Given:
        The artifact-less ``00.verify`` step and a fake Perl run exiting 1;
        expected: training stops with that status.
    When:
        ``train.run_step`` runs the step.
    Then:
        ``SystemExit`` is raised with code 1.
    Raises:
        SystemExit: the expected abort; today none is raised.
    """
    step = train.TrainingStep(script="00.verify/verify_all.pl")

    with pytest.raises(SystemExit) as exc_info:
        _run_fake_step(monkeypatch, tmp_path, step, returncode=1)

    assert exc_info.value.code == 1


# --------------------------------------------------------------- review #2


def _undefined_called_names(path: Path) -> list[str]:
    """List bare names that are called but never bound in a module.

    A name counts as bound if it is a builtin, or is defined anywhere in the
    module by ``def``, ``class``, ``import``, a parameter, an assignment
    target or an ``except ... as`` clause. This is deliberately scope-blind:
    it finds typos and stale names, not shadowing bugs.

    Args:
        path: Python source file to analyse.

    Returns:
        Sorted names called as ``name(...)`` with no binding.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = set(dir(builtins))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    return sorted(called - bound)


def _script_params():
    """Build one pytest param per script.

    ``lm.py`` was the known-bad case (review 2026-10-04 #2, fixed); it now
    runs as a plain regression test like every other script.

    Returns:
        A list of ``pytest.param`` objects keyed by relative path.
    """
    return [
        pytest.param(path, id=path.relative_to(SCRIPTS_DIR).as_posix())
        for path in sorted(SCRIPTS_DIR.rglob("*.py"))
    ]


@pytest.mark.parametrize("path", _script_params())
def test_every_called_name_is_defined(path):
    """Every bare function call in a script resolves to a known name.

    Given:
        One ``scripts/**/*.py`` file parsed with ``ast``; expected: each
        ``name(...)`` call refers to a builtin, import or definition.
    When:
        Called names are compared with bound names.
    Then:
        No called name is undefined.
    """
    undefined = _undefined_called_names(path)

    assert undefined == []
