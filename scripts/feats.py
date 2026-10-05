#!/usr/bin/env python3
"""Pre-extract MFCC feature files (`.mfc`) for corpus splits.

Invoked as `sphinx.sh feats <corpus> <split> [-j N]`,
`sphinx.sh feats <corpus> all`, or `sphinx.sh feats --list`. Extraction
only needs to run once per split: audio files that already have a
non-trivial `.mfc` are skipped. `sphinx_fe` runs once per audio file on a
thread pool, and each `.mfc` is written next to its audio file (same
path, `.mfc` suffix). SphinxTrain finds it there because fileids are
relative to `SPHINX_ROOT` and `CFG_FEATFILES_DIR` is set to the project
root.

Front-end parameters (`sample_rate`, `num_filt`, `lo_filt`, `hi_filt`,
`audio_format`, `audio_type`) come from `corpus.yml`; `-transform dct`
and `-lifter 22` are fixed.

Files read: `corpus/<corpus>/corpus.yml` and every `*.<audio_format>`
file below the split's audio directory. Files written: one `.mfc` per
audio file.

Environment variables read: `SPHINX_ROOT`. `sphinx_fe` is looked up in
`bin/<arch>/` first, then on `PATH`.

Usage: `feats.py <corpus> <split>` or `feats.py --list`

Attributes:
    NATIVE_FORMATS: Audio formats that `sphinx_fe` reads without `sox`.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lib.asr_util import err, get_sphinx_root
from lib.config import load_corpus

# formats sphinx_fe can directly read (without sox)
NATIVE_FORMATS = {"wav", "mswav", "nist", "raw", "sph"}


def find_sphinx_fe(root: Path) -> Path:
    """Locate the `sphinx_fe` binary.

    Args:
        root: Project root. `<root>/bin/<platform.machine()>/sphinx_fe` is
            checked first.

    Returns:
        The binary in the project's `bin/<arch>/` directory if it exists,
            otherwise the first match on `PATH`.

    Raises:
        SystemExit: If `sphinx_fe` is found in neither place (exit code 1,
            via `err`).
    """  # noqa: DOC502
    fe = root / "bin" / platform.machine() / "sphinx_fe"
    if fe.is_file():
        return fe
    found = shutil.which("sphinx_fe")
    if found:
        return Path(found)
    err("sphinx_fe not found.")


# TODO just pass corpus_config
def extract_one(
        sphinx_fe: str,
        audio_file: Path,
        mfc_file: Path,
        audio_ext: str,
        audio_type: str,
        sample_rate: int,
        num_filt: int = 25,
        lo_filt: int = 130,
        hi_filt: int = 6800,
) -> bool:
    """Run `sphinx_fe` on one audio file.

    The input format flags are chosen from `audio_type` and `audio_ext`; the
    first matching rule wins:

    - `nist` type or `sph` extension: `-nist yes`
    - `raw` type or extension: `-raw yes -input_endian little`
    - `mswav` type, or `wav`/`mswav` extension: `-mswav yes`
    - any other extension not in `NATIVE_FORMATS`: `-sox yes`
    - otherwise: `-input_endian little`

    On failure, `FAIL: <audio_file>` and `sphinx_fe`'s stderr are printed.

    Args:
        sphinx_fe: Path of the `sphinx_fe` binary, as a string.
        audio_file: Input audio file.
        mfc_file: Output feature file, written directly by `sphinx_fe`. Its
            parent directory must exist.
        audio_ext: Audio file extension without the dot (`audio_format` in
            `corpus.yml`).
        audio_type: `audio_type` from `corpus.yml` (for example `mswav`,
            `nist` or `raw`), or an empty string.
        sample_rate: Sampling rate in Hz.
        num_filt: Number of mel filters (`-nfilt`).
        lo_filt: Lower edge of the filterbank in Hz (`-lowerf`).
        hi_filt: Upper edge of the filterbank in Hz (`-upperf`).

    Returns:
        `True` if `sphinx_fe` exited with status 0, otherwise `False`.

    Note:
        A failed run does not delete whatever `sphinx_fe` already wrote to
        `mfc_file`.
    """
    args = [
        sphinx_fe,
        "-i", str(audio_file),
        "-o", str(mfc_file),
        "-samprate", str(sample_rate),
        "-nfilt", str(num_filt),
        "-lowerf", str(lo_filt),
        "-upperf", str(hi_filt),
        "-transform", "dct",
        "-lifter", "22",
    ]

    if audio_type == "nist" or audio_ext == "sph":
        args.extend(["-nist", "yes"])
    elif audio_type == "raw" or audio_ext == "raw":
        args.extend(["-raw", "yes", "-input_endian", "little"])
    elif audio_type == "mswav" or audio_ext in ("wav", "mswav"):
        args.extend(["-mswav", "yes"])
    elif audio_ext not in NATIVE_FORMATS:
        args.extend(["-sox", "yes"])
    else:
        args.extend(["-input_endian", "little"])

    ret = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )

    if ret.returncode != 0:
        print(f"FAIL: {audio_file}")
        if ret.stderr:
            print(ret.stderr)

    return ret.returncode == 0


def extract_features(
        sphinx_fe: Path,
        audio_dir: Path,
        audio_ext: str,
        audio_type: str,
        sample_rate: int,
        num_filt: int = 25,
        lo_filt: int = 130,
        hi_filt: int = 6800,
        jobs: int = 1
) -> tuple[int, int, int]:
    """Extract `.mfc` features for every audio file below a directory.

    Audio files are found recursively by extension (`*.<audio_ext>`) and
    sorted. A file is skipped as cached when its `.mfc` sibling exists and is
    larger than 4 bytes; a file of 4 bytes or less is treated as corrupt and
    regenerated. The remaining files are processed by `extract_one` on a
    `ThreadPoolExecutor` with `jobs` workers, and a progress line is printed
    every 500 files and at the end.

    Args:
        sphinx_fe: Path of the `sphinx_fe` binary.
        audio_dir: Directory searched recursively for audio files.
        audio_ext: Audio file extension without the dot.
        audio_type: `audio_type` from `corpus.yml`, or an empty string.
        sample_rate: Sampling rate in Hz.
        num_filt: Number of mel filters.
        lo_filt: Lower edge of the filterbank in Hz.
        hi_filt: Upper edge of the filterbank in Hz.
        jobs: Number of worker threads, each running one `sphinx_fe` process
            at a time. Must be at least 1.

    Returns:
        A tuple `(extracted, skipped, failed)` of file counts. All three are
            0 if no audio files are found.

    Raises:
        OSError: If an output directory cannot be created. The error is
            raised in a worker thread and re-raised by `future.result()`.

    Warning:
        - The cache test only checks that the `.mfc` is larger than 4 bytes.
          A partially written file, from an interrupted run or a failed
          `sphinx_fe` call, is accepted as valid on the next run and never
          regenerated.
        - There is no locking. Two runs over the same split, for example on
          different nodes sharing the tree over NFS, can both decide that a
          file needs extracting and write the same `.mfc` concurrently.
    """  # noqa: DOC502
    audio_files = sorted(audio_dir.rglob(f"*.{audio_ext}"))
    total = len(audio_files)

    if total == 0:
        print(f"  no .{audio_ext} files found in {audio_dir}")
        return (0, 0, 0)

    to_extract = []
    skipped = 0
    for audio_file in audio_files:
        mfc_file = audio_file.with_suffix(".mfc")

        # if size is only 4 bytes that means its a corrputed/invalid
        # mfc file, so we'd need to regenerate
        if mfc_file.is_file() and mfc_file.stat().st_size > 4:
            skipped += 1
        else:
            to_extract.append((audio_file, mfc_file))

    if not to_extract:
        print(f"  All {skipped} files already cached.")
        return (0, skipped, 0)

    print(f"  {len(to_extract)} to extract, {skipped} already cached")

    extracted = 0
    failed = 0
    fe_str = str(sphinx_fe)

    def _do_one(pair):
        """Create the output directory and extract one audio file.

        Closes over the enclosing call's `sphinx_fe` path and front-end
        parameters. Submitted to the thread pool once per file.

        Args:
            pair (tuple[Path, Path]): `(audio_file, mfc_file)` to process.

        Returns:
            success (bool): The result of `extract_one`.
        """
        audio_file, mfc_file = pair
        mfc_file.parent.mkdir(parents=True, exist_ok=True)
        return extract_one(
            fe_str,
            audio_file,
            mfc_file,
            audio_ext,
            audio_type,
            sample_rate,
            num_filt,
            lo_filt,
            hi_filt
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_do_one, pair): pair for pair in to_extract}
        for future in as_completed(futures):
            if future.result():
                extracted += 1
            else:
                failed += 1
            done = extracted + failed
            if done % 500 == 0 or done == len(to_extract):
                print(
                    f"  {done}/{len(to_extract)} ({extracted} ok, "
                    f"{failed} failed)"
                )
        return (extracted, skipped, failed)


# TODO maybe move this to lib/
# also theres a similar function in new_experiment.py
# could be combined into one with a flag denoting whether or not to list splits
def list_corpora(root: Path):
    """Print every corpus and whether each split's audio is present.

    A corpus is listed if `corpus/<name>/corpus.yml` exists. A split is shown
    as `[available]` if `corpus/<name>/<audio>` is a directory, where
    `<audio>` is the split's `audio` key or, if that is absent, the split
    name; otherwise it is shown as `[not downloaded]`.

    Args:
        root: Project root.

    Note:
        The corpus-level `audio_dir` fallback used by `process_split` is not
        considered, so splits that rely on it are reported as
        `[not downloaded]`.
    """
    corpus_dir = root / "corpus"
    if not corpus_dir.is_dir():
        print("No corpora found.")
        return

    for corpus_path in sorted(corpus_dir.iterdir()):
        yml = corpus_path / "corpus.yml"
        if not yml.is_file():
            continue
        corpus = load_corpus(corpus_path.name, root)
        print(f"\n{corpus_path.name}:")
        for split_name, split_cfg in corpus.get("splits", {}).items():
            audio_dir = corpus_path / split_cfg.get("audio", split_name)
            status = "[available]" if audio_dir.is_dir() else "[not downloaded]"
            print(f"  {split_name:<25} {status}")


def process_split(
        split: str,
        corpus_config: dict,
        root: Path,
        jobs: int
) -> tuple[int, int, int]:
    """Extract features for one split of a corpus.

    Resolves the split's audio directory (`split_cfg["audio"]`, else the
    corpus's `audio_dir`, relative to the corpus directory) and the
    front-end parameters from `corpus.yml`, prints a summary, and calls
    `extract_features`.

    Args:
        split: Split name, a key of `corpus_config["splits"]`.
        corpus_config: Corpus mapping from `load_corpus`. It must contain
            `name`.
        root: Project root, used to locate `sphinx_fe`.
        jobs: Number of parallel extraction threads.

    Returns:
        A tuple `(extracted, skipped, failed)` of file counts.

    Raises:
        SystemExit: If the split is not defined, no audio directory is
            configured, the audio directory does not exist, or `sphinx_fe`
            cannot be found (exit code 1, via `err`).
    """  # noqa: DOC502
    splits = corpus_config.get("splits", {})
    if split not in splits:
        available = ", ".join(splits.keys())
        err(f"split '{split}' not found. Available: {available}")

    split_cfg = splits[split]
    corpus_dir = corpus_config["_dir"]
    audio_rel = split_cfg.get("audio") or corpus_config.get("audio_dir")
    if not audio_rel:
        err(f"no audio directory defined for split '{split}' or corpus")
    audio_dir = corpus_dir / audio_rel
    audio_ext = corpus_config.get("audio_format", "wav")
    audio_type = corpus_config.get("audio_type", "")
    sample_rate = int(corpus_config.get("sample_rate", 16000))
    num_filt = int(corpus_config.get("num_filt", 25))
    lo_filt = int(corpus_config.get("lo_filt", 130))
    hi_filt = int(corpus_config.get("hi_filt", 6800))

    if not audio_dir.is_dir():
        err(f"audio directory not found: {audio_dir}")

    sphinx_fe = find_sphinx_fe(root)

    print(f"Extracting features for {corpus_config["name"]}/{split}")
    print(f"  Audio: {audio_dir}")
    print(f"  Format: {audio_ext} @ {sample_rate}Hz")
    print(f"  Jobs: {jobs}")

    return extract_features(
        sphinx_fe,
        audio_dir,
        audio_ext,
        audio_type,
        sample_rate,
        num_filt,
        lo_filt,
        hi_filt,
        jobs
    )


def main():
    """Run `sphinx.sh feats`: extract features for one split or all splits.

    With `--list`, prints the available corpora and exits. Otherwise both
    `corpus` and `split` are required; `split` may be `all` to process every
    split in `corpus.yml` in order, printing per-split counts. `-j/--jobs`
    defaults to the CPU count, or 4 if it is unknown.

    Raises:
        SystemExit: With code 0 after `--list`; with code 1 if `corpus` or
            `split` is missing (after printing help), if any file failed to
            extract, or on errors reported through `err`; with code 2 on
            argparse errors.
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(
        description="Pre-extract MFC features for a corpus split"
    )
    parser.add_argument(
        "corpus",
        nargs="?",
        help="corpus name"
    )
    parser.add_argument(
        "split",
        nargs="?",
        help="split name or 'all' for all splits"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available corpora and splits"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Parallel jobs (default: {os.cpu_count() or 4})",
    )
    args = parser.parse_args()
    root = get_sphinx_root()

    if args.list:
        list_corpora(root)
        sys.exit(0)

    if not args.corpus or not args.split:
        parser.print_help()
        sys.exit(1)

    # load corpus config
    corpus = load_corpus(args.corpus, root)
    extracted = 0
    skipped = 0
    failed = 0
    if args.split == "all":
        split_list = list(corpus["splits"])
        print("Generating features for the following splits:")
        for split in split_list:
            print(f"  {args.corpus}/{split}")
        for split in split_list:
            results = process_split(split, corpus, root, args.jobs)
            extracted += results[0]
            skipped += results[1]
            failed += results[2]
            print(f"Processed split: {split}\n")
            print(
                f"  {results[0]} extracted, {results[1]} cached, "
                f"{results[2]} failed"
            )
    else:
        extracted, skipped, failed = process_split(args.split, corpus, root, args.jobs)

    print(f"\nDone. {extracted} extracted, {skipped} cached, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
