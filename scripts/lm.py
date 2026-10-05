#!/usr/bin/env python3
"""Build an ARPA n-gram language model from corpus transcripts.

Invoked as `sphinx.sh lm <corpus> <split>` (one split) or
`sphinx.sh lm <corpus> --all` (the corpus's `full_transcripts` file),
with optional `-n/--order` (default 3) and `-o/--output`. The text comes
from the corpus adapter, each sentence is wrapped in `<s> ... </s>`, and
the CMU-Cambridge SLM toolkit chain `text2wfreq`, `wfreq2vocab`,
`text2idngram`, `idngram2lm` is run in a temporary directory.

Files read: `corpus/<corpus>/corpus.yml` and the transcripts read by the
adapter. Files written: the ARPA model, by default
`corpus/<corpus>/lm/<split>.arpa`, or `corpus/<corpus>/lm/<corpus>.arpa`
with `--all`. Intermediate files live in a temporary directory that is
removed afterwards.

Environment variables read: `SPHINX_ROOT`. Toolkit binaries are looked
up in `bin/<arch>/` first, then on `PATH`.

Usage: `lm.py <corpus> <split>` or `lm.py <corpus> --all`

Warning:
    The per-split mode is broken: `main` calls `extract_text`, which is
    not defined, so `sphinx.sh lm <corpus> <split>` raises `NameError`.
    Only `--all` reaches the toolkit.
"""

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from corpus import get_adapter
from lib.asr_util import err, get_sphinx_root
from lib.config import load_corpus


def find_tool(name: str, root: Path) -> Path:
    """Locate a CMU SLM toolkit binary.

    Args:
        name: Executable name, for example `"text2wfreq"`.
        root: Project root. `<root>/bin/<platform.machine()>/<name>` is
            checked first.

    Returns:
        The binary in the project's `bin/<arch>/` directory if it exists,
            otherwise the first match on `PATH`.

    Raises:
        SystemExit: If the binary is found in neither place (exit code 1, via
            `err`).
    """  # noqa: DOC502
    local = root / "bin" / platform.machine() / name
    if local.is_file():
        return local
    found = shutil.which(name)
    if found:
        return Path(found)
    err(f"{name} not found. Make sure to run 'make' to build the CMU toolkit.")


def extract_text_from_split(corpus: dict, split_name: str, root: Path) -> str:
    """Return LM training text for one split of a corpus.

    Args:
        corpus: Corpus mapping from `load_corpus`. It must contain a `name`
            key (declared in `corpus.yml`) and the split under `splits`.
        split_name: Split to read.
        root: Project root. Unused.

    Returns:
        One `<s> TEXT </s>` line per utterance, newline-terminated.

    Note:
        Nothing calls this function. `main` calls the undefined name
        `extract_text` instead.
    """
    split_cfg = corpus["splits"][split_name]
    adapter = get_adapter(corpus["name"])
    lines = []
    utterances = adapter.get_utterances(
        corpus["_dir"],
        split_name,
        split_cfg,
        corpus
    )
    for _, text in utterances:
        lines.append(f"<s> {text} </s>")
    return "\n".join(lines) + "\n"


def extract_text_from_full(corpus: dict, root: Path) -> str:
    """Return LM training text from a corpus's `full_transcripts` file.

    The adapter is called with the split name `"full"` and a synthetic split
    config `{"transcripts": <full_transcripts>}`. This suits adapters that
    read the transcript file named by `split_cfg["transcripts"]`
    (Switchboard). The LibriSpeech adapter ignores that key and looks for a
    `full/` audio directory instead.

    Args:
        corpus: Corpus mapping from `load_corpus`. It must contain `name`
            and should contain `full_transcripts` (a path relative to the
            corpus directory).
        root: Project root. Unused.

    Returns:
        One `<s> TEXT </s>` line per utterance, newline-terminated.

    Raises:
        SystemExit: If `full_transcripts` is not defined or the file does not
            exist (exit code 1, via `err`).
    """  # noqa: DOC502
    full_trans_rel = corpus.get("full_transcripts")
    if not full_trans_rel:
        err(
            f"corpus '{corpus['name']}' has no 'full_transcripts' defined in "
            "corpus.yml. Add a 'full_transcripts' key pointing to the full "
            "transcript file, or use a specific split instead:"
            f"\n  lm.py {corpus['name']} <split>"
        )
    trans_path = corpus["_dir"] / full_trans_rel
    if not trans_path.is_file():
        err(f"full transcript not found: {trans_path}")

    # create throwaway split config for get_utterances that points to the full
    # transcript
    full_split_cfg = {"transcripts": full_trans_rel}
    adapter = get_adapter(corpus["name"])
    lines = []
    utterances = adapter.get_utterances(
        corpus["_dir"],
        "full",
        full_split_cfg,
        corpus
    )
    for _, text in utterances:
        lines.append(f"<s> {text} </s>")
    return "\n".join(lines) + "\n"


def build_lm(
        text: str,
        output_path: Path,
        root: Path,
        ngram_order: int = 3
) -> Path:
    """Build an ARPA language model with the CMU-Cambridge SLM toolkit.

    Writes `text` to a temporary directory and runs four shell commands with
    `_run`: `text2wfreq` (word frequencies), `wfreq2vocab` (vocabulary),
    `text2idngram -n <ngram_order>` (id n-gram counts) and
    `idngram2lm -arpa` (the model). Other toolkit options are left at their
    defaults. The temporary directory is removed on return.

    Args:
        text: Training text, one `<s> ... </s>` sentence per line.
        output_path: Destination `.arpa` file. Its parent directories are
            created.
        root: Project root, used to locate the toolkit binaries.
        ngram_order: N-gram order passed to `text2idngram`.

    Returns:
        `output_path`, unchanged.

    Raises:
        SystemExit: If a toolkit binary cannot be found, or with a command's
            exit code if it fails.

    Note:
        - `ngram_order` is passed to `text2idngram` only. `idngram2lm` is run
          without `-n` and uses its own default order.
        - Paths are interpolated unquoted into shell command strings, so
          paths containing spaces or shell metacharacters break the commands.
    """  # noqa: DOC502
    text2wfreq = find_tool("text2wfreq", root)
    wfreq2vocab = find_tool("wfreq2vocab", root)
    text2idngram = find_tool("text2idngram", root)
    idngram2lm = find_tool("idngram2lm", root)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        text_file = tmp / "train.txt"
        wfreq_file = tmp / "train.wfreq"
        vocab_file = tmp / "train.vocab"
        idngram_file = tmp / "train.idngram"

        text_file.write_text(text)
        print("  computing word frequencies...")
        _run(f"{text2wfreq} < {text_file} > {wfreq_file}")
        print("  building vocabulary...")
        _run(f"{wfreq2vocab} < {wfreq_file} > {vocab_file}")
        print(f"  counting {ngram_order}-grams...")
        _run(
            f"{text2idngram} -vocab {vocab_file} "
            f"-n {ngram_order} -temp {tmp}"
            f"< {text_file} > {idngram_file}"
        )
        print("  building arpa language model...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _run(
            f"{idngram2lm} -idngram {idngram_file} "
            f"-vocab {vocab_file} -arpa {output_path}"
        )

    return output_path


def _run(cmd: str):
    """Run a shell command and exit the process if it fails.

    The command runs with `shell=True` and its stdout and stderr are
    captured. On success both are discarded. On failure the command line and
    its stderr are printed to stdout.

    Args:
        cmd: Shell command line, including any redirections.

    Raises:
        SystemExit: With the command's exit code if it is nonzero.
    """  # noqa: DOC502
    ret = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )
    if ret.returncode != 0:
        print(f"  Command failed: {cmd}")
        if ret.stderr:
            print(f"  {ret.stderr.strip()}")
        sys.exit(ret.returncode)


def main():
    """Run `sphinx.sh lm`: build a language model for a split or a corpus.

    Parses `corpus`, the optional `split`, `--all`, `-n/--order` and
    `-o/--output`; loads the corpus; chooses the output path (a relative
    `--output` is resolved against `SPHINX_ROOT`); extracts the text; builds
    the model with `build_lm`; and prints the `corpus.yml` and
    `experiment.yml` snippets that select it.

    Raises:
        SystemExit: On invalid arguments, if both or neither of `split` and
            `--all` are given (the latter with code 1 after printing help),
            if the split is not defined, or if text extraction or the toolkit
            fails.
        NameError: Always in per-split mode, because `extract_text` is not
            defined.

    Warning:
        The per-split path calls `extract_text`, which is not defined, so
        every `lm.py <corpus> <split>` invocation raises `NameError` after
        argument validation. `extract_text_from_split` exists but is unused.

    Note:
        The closing hints use `Path.relative_to`, which raises `ValueError`
        when `--output` lies outside `SPHINX_ROOT`, or outside the corpus
        directory if the corpus has no `lm` key.
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(
        description="build a trigram language model from training transcripts"
    )
    parser.add_argument("corpus", help="Corpus name")
    parser.add_argument(
        "split",
        nargs="?",
        help="Split name"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "build LM from full corpus transcript "
            "(corpus.yml full_transcripts)"
        ),
    )
    parser.add_argument(
        "-n", "--order",
        type=int,
        default=3,
        help="N-gram order (default: 3)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path (default: corpus/<name>/lm/<split>.arpa)",
    )
    args = parser.parse_args()
    root = get_sphinx_root()
    corpus = load_corpus(args.corpus, root)

    if args.all and args.split:
        err("cannot use both a split name and --all")
    if not args.all and not args.split:
        parser.print_help()
        sys.exit(1)

    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = root / output_path
    elif args.all:
        output_path = corpus["_dir"] / "lm" / f"{args.corpus}.arpa"
    else:
        output_path = corpus["_dir"] / "lm" / f"{args.split}.arpa"

    if args.all:
        print(f"building {args.order}-gram LM for {args.corpus} (full corpus)")
        text = extract_text_from_full(corpus, root)
    else:
        splits = corpus.get("splits", {})
        if args.split not in splits:
            available = ", ".join(splits.keys())
            err(f"Split '{args.split}' not found. Available: {available}")
        print(f"building {args.order}-gram LM for {args.corpus}/{args.split}")
        text = extract_text(corpus, args.split, root)

    line_count = text.count("\n")

    print(f"  output: {output_path}")
    print(f"  extracting text from transcripts...")
    print(f"  {line_count} sentences")

    build_lm(text, output_path, root, args.order)

    print(f"\ndone. language model written to {output_path}")
    if not corpus.get("lm"):
        print(f"to use as the default LM for this corpus, add to corpus.yml:")
        print(f"  lm: {output_path.relative_to(corpus['_dir'])}")
    print(f"to override in an experiment, add to experiment.yml:")
    print(f"  decode:")
    print(f"    lm: {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()
