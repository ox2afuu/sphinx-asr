#!/usr/bin/env python3
"""Run the 13-step SphinxTrain acoustic model training pipeline.

Invoked as `sphinx.sh train <exp_dir> [--from-step STEP]`. `sphinx.sh`
submits the run with `qsub` when that command is on `PATH` (an untested
Torque path) and otherwise runs `train.py` locally. A relative
`<exp_dir>` is resolved against `SPHINX_ROOT`.

The script changes the working directory to `<exp_dir>` and runs each
step's Perl script from `vendor/sphinxtrain/scripts/` in sequence (see
`STEPS`), streaming its output. After each step it checks the declared
artifacts and scans the step's log directory for failure patterns, and
it exits with a nonzero status at the first step judged to have failed.

Files read:

- `<exp_dir>/experiment.yml` and the referenced `corpus.yml` files
- `<exp_dir>/logdir/<step>/*.log` (log scanning)
- `<exp_dir>/model_parameters/<db>.lda` and `<db>.mllt` (artifact
  checks)

Files written: none directly. The Perl steps write under `<exp_dir>` as
configured by `etc/sphinx_train.cfg`.

Environment variables:

- `SPHINX_ROOT` (read): project root.
- `SPHINXASR_MLLT_SEED` (set or removed): passed to the Perl steps,
  derived from `pipeline.mllt_init`; removed for the deterministic
  `eye` initialisation.

Usage: `train.py <experiment_dir> [--from-step STEP]`

Attributes:
    STEPS: The pipeline in execution order, from `00.verify` to
        `90.deleted_interpolation`. Only `01.lda_train` and
        `02.mllt_train` declare artifacts.
"""

import argparse
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from lib.asr_util import get_sphinx_root
from lib.config import load_experiment

@dataclass
class TrainingStep:
    """One SphinxTrain pipeline step and how to validate it.

    Attributes:
        script: Path of the step's Perl script relative to
            `vendor/sphinxtrain/scripts/`, for example
            `"30.cd_hmm_untied/slave_convg.pl"`.
        artifacts: File name templates, relative to
            `<exp_dir>/model_parameters/`, that must exist and be non-empty
            after the step. `{db}` is replaced by the database name. Empty
            for most steps.
        log_dir: Subdirectory of `<exp_dir>/logdir/` holding the step's
            logs. Empty to skip log scanning.
        fatal_patterns: Case-sensitive substrings that mark a log line as
            fatal.
        warning_patterns: Case-sensitive substrings that mark a log line as
            a warning. Only checked for lines that match no fatal pattern.
    """
    script: str
    # artifact files that must exist and not be empty after this step
    artifacts: list[str] = field(default_factory=list)
    # log directory to scan for errors for this step
    log_dir: str = ""
    # fatal log patterns
    fatal_patterns: list[str] = field(
        default_factory=lambda: ["0 frames", "FATAL"]
    )
    # warning log patterns
    warning_patterns: list[str] = field(
        default_factory=lambda: ["failed", "Aborting"]
    )

    @property
    def name(self) -> str:
        """Short display name: the script's directory, e.g. `30.cd_hmm_untied`."""
        return self.script.rsplit("/", 1)[0]


STEPS = [
    TrainingStep(
        script="00.verify/verify_all.pl",
    ),
    TrainingStep(
        script="01.lda_train/slave_lda.pl",
        artifacts=["{db}.lda"],
        log_dir="01.lda_train",
    ),
    TrainingStep(
        script="02.mllt_train/slave_mllt.pl",
        artifacts=["{db}.mllt"],
        log_dir="02.mllt_train",
    ),
    TrainingStep(
        script="05.vector_quantize/slave.VQ.pl",
    ),
    TrainingStep(
        script="10.falign_ci_hmm/slave_convg.pl",
        log_dir="10.falign_ci_hmm",
    ),
    TrainingStep(
        script="11.force_align/slave_align.pl",
        log_dir="11.force_align",
    ),
    TrainingStep(
        script="12.vtln_align/slave_align.pl",
        log_dir="12.vtln_align",
    ),
    TrainingStep(
        script="20.ci_hmm/slave_convg.pl",
        log_dir="20.ci_hmm",
    ),
    TrainingStep(
        script="30.cd_hmm_untied/slave_convg.pl",
        log_dir="30.cd_hmm_untied",
    ),
    TrainingStep(
        script="40.buildtrees/slave.treebuilder.pl",
        log_dir="40.buildtrees",
    ),
    TrainingStep(
        script="45.prunetree/slave.state-tying.pl",
        log_dir="45.prunetree",
    ),
    TrainingStep(
        script="50.cd_hmm_tied/slave_convg.pl",
        log_dir="50.cd_hmm_tied",
    ),
    TrainingStep(
        script="90.deleted_interpolation/deleted_interpolation.pl",
    ),
]


def check_artifacts(
        step: TrainingStep,
        model_dir: Path,
        db_name: str
) -> list[str]:
    """Check that a step's declared artifacts exist and are non-empty.

    Args:
        step: The step whose `artifacts` templates are checked.
        model_dir: Directory the templates are relative to, normally
            `<exp_dir>/model_parameters`.
        db_name: Database name substituted for `{db}`.

    Returns:
        One message per problem, of the form `  missing: <path>` or
            `  empty: <path>`. The list is empty when every artifact is
            present, which includes the case where the step declares none.
    """
    errors = []
    for tmpl in step.artifacts:
        artifact = model_dir / tmpl.format(db=db_name)
        if not artifact.is_file():
            errors.append(f"  missing: {artifact}")
        elif artifact.stat().st_size == 0:
            errors.append(f"  empty: {artifact}")
    return errors


def check_logs(
        step: TrainingStep,
        log_dir: Path
) -> tuple[list[str], list[str]]:
    """Scan a step's log files for fatal and warning patterns.

    Every `*.log` file in `<log_dir>/<step.log_dir>/` is read in sorted name
    order, with undecodable bytes replaced. A line containing any of
    `step.fatal_patterns` is reported as fatal; otherwise a line containing
    any of `step.warning_patterns` is reported as a warning. Files that
    cannot be read are skipped.

    Args:
        step: The step whose `log_dir` and patterns are used.
        log_dir: The experiment's `logdir` directory.

    Returns:
        A tuple `(fatals, warnings)` of lists of strings formatted as
            `  <file>:<lineno>: <line>`. Both lists are empty if the step has
            no `log_dir` or the directory does not exist.

    Warning:
        Every `*.log` file in the directory is scanned, including stale logs
        left by an earlier run of the same experiment, so an old failure can
        make a successful rerun fail, and old warnings are reported again.
        Matching is plain substring search, so `"failed"` also matches
        benign lines that contain the word.
    """
    if not step.log_dir:
        return ([], [])
    subdir = log_dir / step.log_dir
    if not subdir.is_dir():
        return ([], [])

    fatals = []
    warnings = []
    for log_file in sorted(subdir.glob("*.log")):
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line, in enumerate(lines, 1):
            hit = f"  {log_file.name}:{lineno}: {line.strip()}"
            if any(pat in line for pat in step.fatal_patterns):
                fatals.append(hit)
            elif any(pat in line for pat in step.warning_patterns):
                warnings.append(hit)
    return fatals, warnings


def print_failure_logs(step: TrainingStep, log_dir: Path, tail_lines: int = 20):
    """Print the tail of a step's summary logs after it fails.

    Prints the last `tail_lines` lines of every
    `<log_dir>/<step.log_dir>/*.<suffix>.log`, where `<suffix>` is the part
    of `step.log_dir` after its last `.` (for example
    `*.cd_hmm_untied.log`). Does nothing if the step has no `log_dir` or the
    directory does not exist.

    Args:
        step: The step that failed.
        log_dir: The experiment's `logdir` directory.
        tail_lines: Number of trailing lines to print per log file.
    """
    if not step.log_dir:
        return
    subdir = log_dir / step.log_dir
    if not subdir.is_dir():
        return

    step_short = step.log_dir.split(".")[-1]
    for summary_log in sorted(subdir.glob(f"*.{step_short}.log")):
        lines = summary_log.read_text(errors="replace").splitlines()
        print(f"\n--- {summary_log.name} (last {tail_lines} lines) ---")
        print("\n".join(lines[-tail_lines:]))


def validate_step(
        step: TrainingStep,
        exp_dir: Path,
        db_name: str,
        log_dir: Path
):
    """Validate a finished step and exit the process if it failed.

    First checks the declared artifacts with `check_artifacts`, then scans
    the step's logs with `check_logs` and prints any warning matches.
    Warning matches never fail the step.

    Args:
        step: The step to validate.
        exp_dir: Experiment directory. Artifacts are looked up under its
            `model_parameters/`.
        db_name: Database name substituted into artifact templates.
        log_dir: The experiment's `logdir` directory.

    Raises:
        SystemExit: With exit code 1 if an artifact is missing or empty, or
            if any fatal pattern is found in the step's logs.

    Note:
        The error messages say the step "returned 0", but `run_step` also
        calls this function after a tolerated nonzero exit.
    """  # noqa: DOC502
    artifact_errors = check_artifacts(
        step,
        exp_dir / "model_parameters",
        db_name
    )
    if artifact_errors:
        print(
            f"[ERROR] {step.name} returned 0 "
            "but expected output is missing or empty:"
        )
        for msg in artifact_errors:
            print(msg)
        sys.exit(1)

    fatals, warnings = check_logs(step, log_dir)

    if warnings:
        print(f"[WARNING] {step.name} produced warning strings in logs:")
        for msg in warnings:
            print(msg)

    if fatals:
        print(
            f"[FATAL] {step.name} returned 0 but "
            "critical errors found in logs:"
        )
        for msg in fatals:
            print(msg)
        sys.exit(1)


def run_step(
        step: TrainingStep,
        scripts_dir: Path,
        exp_dir: Path,
        db_name: str,
        log_dir: Path
):
    """Run one SphinxTrain step and decide whether it succeeded.

    Runs `perl <scripts_dir>/<step.script>` in the current working directory
    (which `main` sets to the experiment directory), streams the merged
    stdout and stderr to stdout while capturing it, and prints the step's
    wall-clock time. The outcome is then decided as follows:

    1. Nonzero exit: if `check_artifacts` reports no problems, the failure
       is downgraded to a printed warning and the run continues. This is
       meant to tolerate MLLT exiting with status 1 on a convergence warning
       after writing a valid transform. Otherwise the step's summary logs
       are printed and the process exits with the step's exit code.
    2. If the captured output contains `Skipped` anywhere, the step is
       accepted without further validation.
    3. Otherwise `validate_step` checks the artifacts and logs.

    Args:
        step: The step to run.
        scripts_dir: `vendor/sphinxtrain/scripts` under the project root.
        exp_dir: Experiment directory.
        db_name: Database name substituted into artifact templates.
        log_dir: The experiment's `logdir` directory.

    Returns:
        elapsed (float): Wall-clock duration of the step in seconds, measured
            with `time.monotonic()` from process start to the end of its
            output.

    Raises:
        SystemExit: With the step's exit code if it exits nonzero and an
            artifact is missing or empty, or with code 1 if `validate_step`
            rejects the step.

    Warning:
        - A nonzero exit is tolerated whenever `check_artifacts` reports no
          errors, and a step that declares no artifacts (every step except
          `01.lda_train` and `02.mllt_train`) always reports none. Such steps
          therefore never fail on their exit code alone; only the log scan
          can stop them, and the printed warning still claims that the
          artifacts are present.
        - Any occurrence of the substring `Skipped` in the step's output
          bypasses artifact and log validation, even when it comes from an
          unrelated message.
    """  # noqa: DOC502
    print(f"\n=== {step.name} ===")
    step_start = time.monotonic()

    output_lines = []
    with subprocess.Popen(
        ["perl", str(scripts_dir / step.script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
    ret = proc.returncode
    output = "".join(output_lines)

    elapsed = time.monotonic() - step_start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"=== Step time: {minutes}m {seconds}s ===")
    if ret != 0:
        # MLLT exits on 1 on convergence warning even with a valid transform
        # so check if the artifacts exist, and if they do ouput a warning
        # instead of step failure
        artifact_errors = check_artifacts(
            step,
            exp_dir / "model_parameters",
            db_name
        )
        if not artifact_errors:
            print(
                f"[WARNING] {step.name} exited {ret} "
                "but artifacts are present, continuing."
            )
        else:
            print(f"Error: {step.name} failed (exit {ret})")
            print_failure_logs(step, log_dir)
            sys.exit(ret)

    if "Skipped" in output:
        return elapsed

    validate_step(step, exp_dir, db_name, log_dir)
    return elapsed


def resolve_mllt_seed(experiment: dict) -> int | None:
    """Translate `pipeline.mllt_init` into an MLLT random seed.

    Args:
        experiment: Experiment mapping. `experiment["pipeline"]["mllt_init"]`
            is read and defaults to `"eye"`.

    Returns:
        `None` for `"eye"` (deterministic identity initialisation) and for
            values that cannot be converted to `int` (after printing a
            warning). For `"random"`, a new seed drawn from the unseeded
            `random` module in `[0, 2**31 - 1]`. Otherwise the value
            converted with `int()`.

    Note:
        A `"random"` seed is not saved to any file. It is only recorded in
        the console output printed by `main`.
    """
    pipeline_options = experiment.get("pipeline", {})
    mllt_init = pipeline_options.get("mllt_init", "eye")
    if mllt_init == "eye":
        return None
    if mllt_init == "random":
        return random.randint(0, 2**31 -1)
    try:
        return int(mllt_init)
    except (ValueError, TypeError):
        print(
            f"[WARNING] invalid mllt_init value '{mllt_init}', "
            "falling back to eye"
        )
        return None


def main():
    """Run `sphinx.sh train`: execute the training pipeline for an experiment.

    Parses `exp_dir` and `--from-step`, resolves `exp_dir` against
    `SPHINX_ROOT` if it is relative, changes the working directory to it
    with `os.chdir`, loads the experiment, sets or removes
    `SPHINXASR_MLLT_SEED` in the environment inherited by the Perl steps,
    and runs every step from the selected starting point with `run_step`.
    Per-step times and the total time are printed.

    Raises:
        SystemExit: On invalid arguments, if `SPHINX_ROOT` is unset, if
            `experiment.yml` fails validation, if `--from-step` matches no
            step (code 1), or if a step fails (see `run_step`).

    Warning:
        `--from-step` selects the first step whose name contains the given
        string as a substring, so short values can resolve to an unintended
        earlier step: `--from-step 2` resumes at `02.mllt_train`, not
        `20.ci_hmm`, and `--from-step 0` restarts at `00.verify`.
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(description="Run sphinxtrain training.")
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument(
        "--from-step",
        metavar="STEP",
        help="resume from step (e.g. '30.cd_hmm_untied' or '30')",
    )
    args = parser.parse_args()

    root = get_sphinx_root()
    exp_dir = args.exp_dir
    if not exp_dir.is_absolute():
        exp_dir = root / exp_dir

    db_name = exp_dir.name
    scripts_dir = root / "vendor" / "sphinxtrain" / "scripts"
    log_dir = exp_dir / "logdir"
    os.chdir(exp_dir)

    experiment = load_experiment(exp_dir, root)

    # set MLLT seed
    mllt_seed = resolve_mllt_seed(experiment)
    if mllt_seed is not None:
        os.environ["SPHINXASR_MLLT_SEED"] = str(mllt_seed)
        print(f"MLLT seed: {mllt_seed}")
    else:
        os.environ.pop("SPHINXASR_MLLT_SEED", None)
        print("MLLT init: eye (deterministic)")

    steps = STEPS
    if args.from_step:
        matches = [s for s in STEPS if args.from_step in s.name]
        if not matches:
            print(f"[ERROR] no step matching '{args.from_step}'")
            print("Available steps:")
            for s in STEPS:
                print(f"  {s.name}")
            sys.exit(1)
        start_idx = STEPS.index(matches[0])
        steps = STEPS[start_idx:]
        print(f"Resuming from: {matches[0].name}")

    total_start = time.monotonic()
    step_times: dict[str, float] = {}
    for step in steps:
        elapsed = run_step(step, scripts_dir, exp_dir, db_name, log_dir)
        step_times[step.name] = elapsed
    total_elapsed = time.monotonic() - total_start
    hours, remainder = divmod(int(total_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining complete. Total time: {hours}h {minutes}m {seconds}s")

    #TODO write results.yml with step times


if __name__ == "__main__":
    main()
