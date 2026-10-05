#!/usr/bin/env python3
"""Decode an experiment's test set with the trained model and report WER.

Invoked as `sphinx.sh decode <exp_dir>`. As with `train`, `sphinx.sh`
submits the run with `qsub` when that command is on `PATH` and otherwise
runs `decode.py` locally. A relative `<exp_dir>` is resolved against
`SPHINX_ROOT`.

The script runs SphinxTrain's `vendor/sphinxtrain/scripts/decode/slave.pl`
from inside `<exp_dir>`; it decodes `<db>_decode.fileids` in
`DEC_CFG_NPART` parts and scores the hypotheses. While it runs, a
background thread shows progress by counting lines in the per-part
`.match` files. Lines of the Perl output containing `WORD ERROR` or
`SENTENCE ERROR` are printed at the end.

Files read:

- `<exp_dir>/etc/sphinx_train.cfg` (`DEC_CFG_NPART`)
- `<exp_dir>/etc/<db>_decode.fileids` (utterance count)
- `<exp_dir>/result/*.match` (progress)
- `<exp_dir>/logdir/decode/*.log` (printed on failure)

Files deleted before decoding: `<exp_dir>/result/*.match` and
`<exp_dir>/result/*.matchseg`. Files written: none directly; `slave.pl`
writes the decoding results under `<exp_dir>/result/`.

Environment variables read: `SPHINX_ROOT`.

Usage: `decode.py <experiment_dir>`
"""

import argparse
import os
import subprocess
import sys
import time
import threading
import re
from pathlib import Path

from lib.asr_util import get_sphinx_root, err, waitbar, spinner

def get_npart(exp_dir: Path) -> int:
    """Read `DEC_CFG_NPART` from the experiment's `sphinx_train.cfg`.

    The first active (uncommented) line assigning an integer literal to
    `$DEC_CFG_NPART` is used.

    Args:
        exp_dir: Experiment directory.

    Returns:
        The number of decode parts, or 1 if the file is missing or contains
            no such assignment.
    """
    cfg_path = exp_dir / "etc" / "sphinx_train.cfg"
    if not cfg_path.is_file():
        return 1
    for line in cfg_path.read_text().splitlines():
        m = re.match(r'^\s*\$DEC_CFG_NPART\s*=\s*(\d+)', line)
        if m:
            return int(m.group(1))
    return 1


def get_total_utterances(exp_dir: Path, db_name: str) -> int:
    """Count the utterances in `<exp_dir>/etc/<db_name>_decode.fileids`.

    Args:
        exp_dir: Experiment directory.
        db_name: Database name (the experiment directory name).

    Returns:
        The number of lines in the fileids file (after any OOV filtering by
            `setup`), or 0 if the file does not exist.
    """
    fileids = exp_dir / "etc" / f"{db_name}_decode.fileids"
    if not fileids.is_file():
        return 0
    return sum(1 for _ in fileids.open())


def poll_progress(
        result_dir: Path,
        db_name: str,
        npart: int,
        total: int,
        stop_event: threading.Event
):
    """Display decode progress until `stop_event` is set.

    Intended to run on a daemon thread. Every 5 seconds it sums the line
    counts of `<db_name>-<i>-<npart>.match` for every part `i` and redraws a
    progress bar. Once the combined `<db_name>.match` file exists, which the
    code treats as the start of the alignment (scoring) phase, it draws a
    full bar once and then switches to a spinner redrawn every 0.25 seconds.
    A final newline is printed when polling stops.

    Args:
        result_dir: The experiment's `result/` directory.
        db_name: Database name (the experiment directory name).
        npart: Number of decode parts (`DEC_CFG_NPART`).
        total: Expected number of utterances, used as the bar's total.
        stop_event: Event set by the main thread to stop polling. Also used
            as an interruptible sleep between redraws.
    """
    spin_idx = 0
    in_align = False
    while not stop_event.is_set():
        combined = result_dir / f"{db_name}.match"

        if combined.is_file():
            # alignment
            if not in_align:
                print(
                    waitbar(total, total, label="decoding"),
                    end="",
                    flush=True
                )
                print("Decode phase complete. Moving to aligning phase.")
                in_align = True
            print(spinner(spin_idx, "aligning..."), end="", flush=True)
            spin_idx += 1
            stop_event.wait(timeout=0.25)
        else:
            # decode
            completed = 0
            for i in range(1, npart + 1):
                match = result_dir / f"{db_name}-{i}-{npart}.match"
                if match.is_file():
                    try:
                        completed += sum(1 for _ in match.open())
                    except OSError:
                        pass
            print(waitbar(completed, total, label="decoding"), end="", flush=True)
            stop_event.wait(timeout=5)
    print()


# TODO clean this up by splitting into multiple functions
def main():
    """Run `sphinx.sh decode`: decode the test set and print WER and SER.

    Checks that `etc/sphinx_train.cfg` exists, that `model_parameters/` is
    non-empty and that the decode script exists; deletes stale `.match` and
    `.matchseg` files from `result/`; changes the working directory to the
    experiment with `os.chdir`; starts `poll_progress` on a daemon thread;
    and runs `perl slave.pl`, capturing its stdout. On success the captured
    lines containing `WORD ERROR` or `SENTENCE ERROR` are printed.

    Raises:
        SystemExit: On invalid arguments, if `SPHINX_ROOT` is unset, if a
            prerequisite is missing (code 1, via `err`), or with the Perl
            exit code if decoding fails. In the last case the final 20 lines
            of the first `logdir/decode/*.log` (by name) are printed first.

    Warning:
        - The decoder's stderr is sent to `/dev/null`, so its diagnostics are
          lost. Only stdout is captured, and it is not echoed while decoding
          runs.
        - The WER and SER lines are printed as reported, without comparing
          the number of decoded utterances with the number of lines in
          `<db>_decode.fileids`, so an incomplete decode is not flagged.

    Note:
        The "decode script not found" message lacks an f-string prefix and
        prints `{decode_script}` literally.
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(description="Run sphinxtrain decoding.")
    parser.add_argument("exp_dir", type=Path)
    args = parser.parse_args()

    root = get_sphinx_root()
    exp_dir = args.exp_dir
    if not exp_dir.is_absolute():
        exp_dir = root / exp_dir

    db_name = exp_dir.name

    cfg_file = exp_dir / "etc" / "sphinx_train.cfg"
    if not cfg_file.is_file():
        err(f"{cfg_file} not found. Run 'sphinx setup {args.exp_dir}' first")

    # check for trained model
    model_dir = exp_dir / "model_parameters"
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        err(
            f"no trained models found in {model_dir}. "
            f"Run 'sphinx train {args.exp_dir}' first"
        )

    decode_script = (
        root / "vendor" / "sphinxtrain" / "scripts" / "decode" / "slave.pl"
    )
    if not decode_script.is_file():
        err("decode script not found at {decode_script}")

    npart = get_npart(exp_dir)
    total_utts = get_total_utterances(exp_dir, db_name)
    result_dir = exp_dir / "result"

    # cleanup old files
    for f in result_dir.glob("*.match"):
        f.unlink()
    for f in result_dir.glob("*.matchseg"):
        f.unlink()

    os.chdir(exp_dir)
    print(f"Decoding experiment {exp_dir.name}...")

    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=poll_progress,
        args=(result_dir, db_name, npart, total_utts, stop_event),
        daemon=True,
    )
    progress_thread.start()

    start = time.monotonic()

    output_lines = []
    with subprocess.Popen(
        ["perl", str(decode_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) as proc:
        for line in proc.stdout:
            output_lines.append(line)
    ret = proc.returncode

    elapsed = time.monotonic() - start
    minutes, seconds = divmod(int(elapsed), 60)

    stop_event.set()
    progress_thread.join()

    if ret != 0:
        print(f"\nError: decoding failed (exit {ret})")
        log_dir = exp_dir / "logdir" / "decode"
        if log_dir.is_dir():
            for log in sorted(log_dir.glob("*.log"))[:1]:
                lines = log.read_text(errors="replace").splitlines()
                print(f"\n--- {log.name} (last 20 lines) ---")
                print("\n".join(lines[-20:]))
        sys.exit(ret)

    print(f"\nDecoding complete. Time: {minutes}m {seconds}s")

    # print WER
    for line in output_lines:
        if "WORD ERROR" in line or "SENTENCE ERROR" in line:
            print(line.strip())

    if result_dir.is_dir() and any(result_dir.iterdir()):
        print(f"Results written to {result_dir}")


if __name__ == "__main__":
    main()
