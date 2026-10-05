#!/usr/bin/env python3
"""Scaffold a new numbered experiment directory from a config template.

Invoked as `sphinx.sh new [-t CORPUS] [-l]`. Creates
`experiments/<NNN>/` (the next free number, zero-padded to three digits)
and copies a template into it as `experiment.yml`. With `-t CORPUS` the
template is `corpus/<CORPUS>/experiment.yml.template` if it exists;
otherwise, and without `-t`, it is the generic `experiment.yml.template`
at the project root. With `-l`, the corpora are listed instead.

Files read: `experiment.yml.template`, `corpus/*/corpus.yml` (existence
only) and `corpus/*/experiment.yml.template`. Files written:
`experiments/<NNN>/experiment.yml`.

Environment variables read: `SPHINX_ROOT`.

Usage: `new_experiment.py [-t CORPUS] [-l]`
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from lib.asr_util import err, get_sphinx_root

# TODO media wiki stuff

@dataclass
class Corpus:
    """Summary of one corpus, as listed by `sphinx.sh new -l`.

    Attributes:
        name: Corpus directory name under `corpus/`.
        has_template: Whether `corpus/<name>/experiment.yml.template` exists.
    """
    name: str
    has_template: bool

@dataclass
class ConfigTemplate:
    """An experiment template file and the label shown when it is used.

    Attributes:
        path: Path of the template file to copy.
        label: Human-readable description printed after creation, such as
            the corpus name or `generic`.
    """
    path: Path
    label: str


def get_corpora_list(corpus_dir: Path) -> list[Corpus]:
    """List the corpora that have a `corpus.yml`.

    Args:
        corpus_dir: The `corpus/` directory under the project root.

    Returns:
        One `Corpus` per subdirectory containing `corpus.yml`, sorted by
            name. Empty if `corpus_dir` does not exist.
    """
    corpora = []
    if not corpus_dir.is_dir():
        return corpora
    for path in sorted(corpus_dir.iterdir()):
        if path.is_dir() and (path / "corpus.yml").is_file():
            has_template = (path / "experiment.yml.template").is_file()
            corpora.append(Corpus(path.name, has_template))
    return corpora


def next_experiment_number(experiments_dir: Path) -> int:
    """Return the next unused experiment number.

    Args:
        experiments_dir: The `experiments/` directory.

    Returns:
        One more than the largest entry name in `experiments_dir` that parses
            as an integer, or 1 if there is none or the directory does not
            exist. Gaps in the numbering are not reused.
    """
    if not experiments_dir.is_dir():
        return 1
    existing = []
    for path in experiments_dir.iterdir():
        try:
            existing.append(int(path.name))
        except ValueError:
            continue
    return max(existing, default=0) + 1


def print_corpora_list(corpora_list: list[Corpus], templates_only=False):
    """Print corpora, marking those that ship an experiment template.

    Args:
        corpora_list: Corpora to print, in order.
        templates_only (bool): If true, print only corpora that have a
            template. `main` never sets it.
    """
    print("Available corpora:")
    for corpus in corpora_list:
        if not corpus.has_template and templates_only:
            continue
        marker = " (template available)" if corpus.has_template else ""
        print(f"  {corpus.name}{marker}")


def get_corpus_template(
        template_name: str | None,
        default_path: Path,
        corpus_path: Path
) -> ConfigTemplate | None:
    """Choose the experiment template for a corpus.

    Args:
        template_name: Corpus name given with `-t`, or `None` or empty for
            the generic template.
        default_path: Path of the generic `experiment.yml.template` at the
            project root.
        corpus_path: The corpus directory, `corpus/<template_name>`.

    Returns:
        `None` if `corpus_path` is not a directory. Otherwise the generic
            template when `template_name` is empty, the corpus's own
            `experiment.yml.template` when it exists, or else the generic
            template with a label noting that no corpus template was found.
    """
    if not corpus_path.is_dir():
        return None

    if not template_name:
        return ConfigTemplate(path=default_path, label="generic")

    template_path = corpus_path / "experiment.yml.template"
    template_label = ""
    if template_path.is_file():
        template_label = template_name
    else:
        template_path = default_path
        template_label = f"generic (no template found for {template_name}"

    return ConfigTemplate(path=template_path, label=template_label)


def make_experiment_dir(
        num: int,
        exp_dir: Path,
        root_dir: Path,
        config_template: ConfigTemplate):
    """Create `experiments/<NNN>/` and copy the template into it.

    The directory is created with its parents if needed. If it already
    exists, its `experiment.yml` is overwritten. The template is copied with
    `shutil.copy2`, which preserves file metadata.

    Args:
        num: Experiment number, formatted as three zero-padded digits.
        exp_dir: The `experiments/` directory.
        root_dir: Project root, used to print a relative path.
        config_template: The template to copy.
    """
    exp_name = f"{num:03d}"
    exp_subdir = exp_dir / exp_name
    exp_subdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_template.path, exp_subdir / "experiment.yml")
    rel_dir = exp_subdir.relative_to(root_dir)
    print(f"Created experiment {exp_name} at {rel_dir}/")
    print(f"  Template: {config_template.label}")


def main():
    """Run `sphinx.sh new`: create the next numbered experiment.

    Raises:
        SystemExit: With code 0 after `--list`; with code 1 if `SPHINX_ROOT`
            is unset, if `--list` finds no corpora, or if the chosen template
            file does not exist (via `err`); with code 2 on argparse errors.
        AttributeError: If `-t` names a corpus whose directory does not
            exist.

    Warning:
        An unknown corpus passed with `-t` makes `get_corpus_template` return
        `None`, and the next statement reads `config_template.path`, so the
        script fails with `AttributeError` instead of a clear error message.

    Note:
        The "template not found" message lacks an f-string prefix, so it
        prints `{config_template.path}` literally.
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(
        description="Create a new experiment directory with a config template."
    )
    parser.add_argument(
        "-t", "--template",
        metavar="CORPUS",
        help="use the experiment template from a specific corpus",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        dest="list_corpora",
        help="list available corpora"
    )
    args = parser.parse_args()

    root = get_sphinx_root()
    corpus_dir = root / "corpus"
    experiments_dir = root / "experiments"
    root_template = root / "experiment.yml.template"
    corpora = get_corpora_list(corpus_dir)

    # list
    if args.list_corpora:
        if not corpora:
            err("No corpora found in corpus/")
        print_corpora_list(corpora)
        sys.exit(0)

    # template
    if args.template:
        config_template = get_corpus_template(
            args.template,
            root_template,
            corpus_dir / args.template)
    else:
        config_template = ConfigTemplate(path=root_template, label="generic")

    if not config_template.path.is_file():
        err("template not found at {config_template.path}")

    # make exp directory
    num = next_experiment_number(experiments_dir)
    make_experiment_dir(num, experiments_dir, root, config_template)


if __name__ == "__main__":
    main()
