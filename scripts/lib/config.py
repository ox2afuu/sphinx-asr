"""Load experiment and corpus YAML and render `sphinx_train.cfg`.

This is the configuration layer behind `sphinx.sh setup` (through
`setup.py`). `train.py` also uses `load_experiment`, and `feats.py` and
`lm.py` use `load_corpus`. It is a library module and is never run
directly.

The central flow is:

1. `load_experiment` reads `<exp_dir>/experiment.yml`, validates its
   shape, and attaches each referenced `corpus/<name>/corpus.yml` as a
   `_corpus` key (with `_dir` set to the corpus directory). The
   `splits: [...]` shorthand is expanded into one `train.corpora` entry
   per split.
2. `generate_sphinx_train_cfg` reads the vendored
   `vendor/sphinxtrain/etc/sphinx_train.cfg` template, fills in path
   placeholders, and rewrites Perl `$VAR = value;` assignments with
   corpus-derived defaults followed by the experiment's `sphinxtrain:`
   overrides.

Files read:

- `<exp_dir>/experiment.yml`
- `$SPHINX_ROOT/corpus/<name>/corpus.yml`
- `$SPHINX_ROOT/vendor/sphinxtrain/etc/sphinx_train.cfg` (template)

Files written: none. The rendered configuration is returned as a string
and `setup.py` writes it to `<exp_dir>/etc/sphinx_train.cfg`.

Environment variables: none are read here. Callers pass the project root,
obtained from `SPHINX_ROOT` with `lib.asr_util.get_sphinx_root`.
"""

import platform
import re
from pathlib import Path

import yaml

from .asr_util import err


def load_yaml(path: Path) -> dict:
    """Load a YAML file whose top-level value must be a mapping.

    Args:
        path: Path of the YAML file to read.

    Returns:
        The parsed mapping. An empty file, or one containing only `null`,
            yields an empty dict.

    Raises:
        ValueError: If the file is not valid YAML. The original
            `yaml.YAMLError` is chained as the cause.
        TypeError: If the top-level YAML value is not a mapping (for example
            a list or a scalar).
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


def validate_experiment(experiment: dict, exp_yml: Path):
    """Check the structure of a parsed `experiment.yml` and exit on errors.

    All problems are collected first and reported together in one message,
    so they can be fixed in a single edit. The checks are structural only:

    - `train` is a mapping and `train.corpora` is a non-empty list of
      mappings, each with `name` and either `split` or `splits`.
    - `decode` is a mapping and `decode.corpus` is a mapping with `name`
      and `split`.
    - `sphinxtrain`, if present, is a mapping.

    Corpus and split names are not resolved here (`load_experiment` does
    that), and the types of the `split` and `splits` values are not checked.

    Args:
        experiment: The parsed contents of `experiment.yml`.
        exp_yml: Path of the file, used only in error messages.

    Raises:
        SystemExit: If `experiment` is not a dict or any structural check
            fails (exit code 1, via `err`).
    """  # noqa: DOC502
    errors = []
    # top level
    if not isinstance(experiment, dict):
        err(f"{exp_yml}: file is empty or malformed")

    # train section
    train = experiment.get("train")
    if not isinstance(train, dict):
        errors.append("missing 'train:' section")
    else:
        corpora = train.get("corpora")
        if not isinstance(corpora, list) or len(corpora) == 0:
            errors.append("train.corpora: must be a non-empty list")
        else:
            for i, entry in enumerate(corpora):
                if not isinstance(entry, dict):
                    errors.append(
                        f"train.corpora[{i}]: must be a mapping, not "
                        f"{type(entry).__name__}"
                    )
                    continue
                if "name" not in entry:
                    errors.append(f"train.corpora[{i}]: missing 'name'")
                if "split" not in entry and "splits" not in entry:
                    errors.append(
                        f"train.corpora[{i}]: missing 'split' or 'splits'"
                    )

    # decode section
    decode = experiment.get("decode")
    if not isinstance(decode, dict):
        errors.append("missing 'decode:' section")
    else:
        corpus = decode.get("corpus")
        if not isinstance(corpus, dict):
            errors.append("decode.corpus: must be a mapping")
        else:
            if "name" not in corpus:
                errors.append("decode.corpus: missing 'name'")
            if "split" not in corpus:
                errors.append("decode.corpus: missing 'split'")

    st = experiment.get("sphinxtrain")
    if st is not None and not isinstance(st, dict):
        errors.append("sphinxtrain: must be a mapping of key: value pairs")

    if errors:
        msg = f"{exp_yml}: configuration errors:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        err(msg)


def load_corpus(corpus_name: str, sphinx_root: Path) -> dict:
    """Load `corpus/<corpus_name>/corpus.yml` and record its directory.

    Args:
        corpus_name: Directory name of the corpus under
            `<sphinx_root>/corpus/`.
        sphinx_root: Project root, normally from `get_sphinx_root`.

    Returns:
        The parsed `corpus.yml` mapping with an extra `_dir` key holding the
            corpus directory (`sphinx_root / "corpus" / corpus_name`).

    Raises:
        FileNotFoundError: If the corpus has no `corpus.yml`.

    Note:
        No `name` key is added. Callers that read `corpus["name"]`
        (`feats.py`, `lm.py`) rely on `corpus.yml` declaring it. YAML errors
        from `load_yaml` (`ValueError`, `TypeError`) propagate unchanged.
    """
    corpus_dir = sphinx_root / "corpus" / corpus_name
    corpus_yml = corpus_dir / "corpus.yml"

    if not corpus_yml.is_file():
        raise FileNotFoundError(
            f"corpus.yml not found for {corpus_dir.name} at {corpus_yml}"
        )

    data = load_yaml(corpus_yml)
    data["_dir"] = corpus_dir
    return data


def load_experiment(exp_dir: Path, sphinx_root: Path) -> dict:
    """Load `experiment.yml` and resolve its corpus references.

    After `validate_experiment` passes, each `train.corpora` entry is loaded
    with `load_corpus` and expanded into one entry per split, of the form
    `{"name": ..., "split": ..., "_corpus": ...}`. If an entry has both
    `splits` and `split`, the single `split` is added to the list. The
    `decode.corpus` mapping receives a `_corpus` key in place. Every split
    name is checked against its corpus's `splits` mapping.

    Args:
        exp_dir: Experiment directory containing `experiment.yml`.
        sphinx_root: Project root, used to locate `corpus/<name>/corpus.yml`.

    Returns:
        The parsed experiment mapping, with `train.corpora` replaced by the
            expanded per-split list and `decode.corpus["_corpus"]` set. Other
            keys (for example `sphinxtrain` and `pipeline`) are unchanged.

    Raises:
        FileNotFoundError: If `experiment.yml` or a referenced `corpus.yml`
            does not exist.
        ValueError: If a YAML file is malformed, or a split name is not
            defined by its corpus.
        TypeError: If a YAML file's top-level value is not a mapping.
        SystemExit: If `validate_experiment` rejects the structure.

    Warning:
        - When an entry has both `splits: [...]` and `split:`, the `split`
          value is appended to the user's `splits` list object itself, so the
          parsed YAML data is mutated in place.
        - Overlap between the training and decode data is not checked. The
          same corpus split can be listed under both `train.corpora` and
          `decode.corpus` without any error or warning.
    """  # noqa: DOC502
    exp_yml = exp_dir / "experiment.yml"
    if not exp_yml.is_file():
        raise FileNotFoundError(f"experiment.yml not found at {exp_yml}")

    experiment = load_yaml(exp_yml)
    validate_experiment(experiment, exp_yml)

    # resolve training corpus references
    raw_corpora = experiment.get("train", {}).get("corpora", [])
    expanded = []
    for entry in raw_corpora:
        corpus = load_corpus(entry["name"], sphinx_root)

        splits = entry.get("splits", [])
        if "split" in entry:
            splits.append(entry["split"])

        for split_name in splits:
            _validate_split(entry["name"], split_name, corpus)
            expanded.append({
                "name": entry["name"],
                "split": split_name,
                "_corpus": corpus
            })
    experiment["train"]["corpora"] = expanded

    # resolve decode corpus reference
    decode_corpus = experiment.get("decode", {}).get("corpus", {})
    if decode_corpus and decode_corpus.get("name"):
        corpus = load_corpus(decode_corpus["name"], sphinx_root)
        _validate_split(decode_corpus["name"], decode_corpus["split"], corpus)
        decode_corpus["_corpus"] = corpus

    return experiment


def generate_sphinx_train_cfg(
        exp_dir: Path,
        experiment: dict,
        sphinx_root: Path,
) -> str:
    """Render the `sphinx_train.cfg` text for an experiment.

    Reads the vendored template and applies, in order:

    1. Literal placeholder replacement: `___DB_NAME___` (the experiment
       directory name), `___BASE_DIR___` (the resolved experiment
       directory), `___SPHINXTRAIN_DIR___` and `___SPHINXTRAIN_BIN_DIR___`
       (`bin/<platform.machine()>` under the project root).
    2. Perl assignment rewriting with `_apply_overrides`, from a mapping
       built in this order: front-end settings of the first training corpus
       (`CFG_WAVFILE_EXTENSION`, `CFG_WAVFILE_TYPE`, `CFG_WAVFILE_SRATE`,
       `CFG_NUM_FILT`, `CFG_LO_FILT`, `CFG_HI_FILT`, `CFG_CMN`); the fixed
       values `CFG_CASEDSYMBOLS = 'yes'` and `CFG_G2P_MODEL = 'no'`;
       `CFG_WAVFILES_DIR`, `CFG_FEATFILES_DIR` and `DEC_CFG_FEATFILES_DIR`
       set to the project root (fileids are relative to it); the decode
       language model; the decode fileids and transcription paths under
       `<exp_dir>/etc/`; and finally every key of the experiment's
       `sphinxtrain:` mapping, which overrides all of the above.

    Args:
        exp_dir: Experiment directory. It is resolved to an absolute path and
            its final component becomes the database name.
        experiment: Experiment mapping as returned by `load_experiment`.
        sphinx_root: Project root. It is resolved to an absolute path.

    Returns:
        The complete contents of `sphinx_train.cfg`. Nothing is written to
            disk.

    Raises:
        FileNotFoundError: If the vendored template is missing.
        SystemExit: If no language model is defined by `decode.lm` or by the
            decode corpus's `lm` key (exit code 1, via `err`).

    Warning:
        - Front-end settings (`audio_format`, `audio_type`, `sample_rate`,
          `num_filt`, `lo_filt`, `hi_filt`, `cmn`) come from
          `train.corpora[0]` only. Other training corpora with a different
          front end are neither detected nor reconciled, and the decode
          corpus's settings are not consulted.
        - Overrides whose key has no `$KEY = ...;` line (active or commented
          out) in the template are dropped silently, so a misspelt
          `sphinxtrain:` key has no effect and produces no message.
    """  # noqa: DOC502
    template_path = (
        sphinx_root / "vendor" / "sphinxtrain" / "etc" / "sphinx_train.cfg"
    )
    if not template_path.is_file():
        raise FileNotFoundError(
            f"sphinx_train.cfg template not found at {template_path}"
        )

    cfg = template_path.read_text()
    exp_dir = exp_dir.resolve()
    sphinx_root = sphinx_root.resolve()

    db_name = exp_dir.name
    sphinxtrain_dir = str(sphinx_root / "vendor" / "sphinxtrain")
    bin_dir = str(sphinx_root / "bin" / platform.machine())

    cfg = cfg.replace("___DB_NAME___", db_name)
    cfg = cfg.replace("___BASE_DIR___", str(exp_dir))
    cfg = cfg.replace("___SPHINXTRAIN_DIR___", sphinxtrain_dir)
    cfg = cfg.replace("___SPHINXTRAIN_BIN_DIR___", bin_dir)

    overrides = {}

    # per-corpus settings
    train_corpora = experiment.get("train", {}).get("corpora", [])
    if train_corpora:
        primary = train_corpora[0].get("_corpus", {})
        overrides["CFG_WAVFILE_EXTENSION"] = primary.get(
            "audio_format", "wav"
        )
        overrides["CFG_WAVFILE_TYPE"] = primary.get("audio_type", "mswav")
        overrides["CFG_WAVFILE_SRATE"] = float(
            primary.get("sample_rate", 16000)
        )

        overrides["CFG_NUM_FILT"] = int(primary.get("num_filt", 25))
        overrides["CFG_LO_FILT"] = int(primary.get("lo_filt", 130))
        overrides["CFG_HI_FILT"] = int(primary.get("hi_filt", 6800))
        overrides["CFG_CMN"] = primary.get("cmn", "live")

    # default global overrides
    overrides["CFG_CASEDSYMBOLS"] = "yes"
    overrides["CFG_G2P_MODEL"] = "no"

    # filepaths
    overrides["CFG_WAVFILES_DIR"] = str(sphinx_root)
    overrides["CFG_FEATFILES_DIR"] = str(sphinx_root)
    overrides["DEC_CFG_FEATFILES_DIR"] = str(sphinx_root)

    # LM path
    decode_cfg = experiment.get("decode", {})
    lm_path = _resolve_lm_path(decode_cfg, sphinx_root)
    if not lm_path:
        err(
            "no language model defined. Either:\n"
            " - add 'lm: lm/<file>.arpa' to corpus.yml\n"
            " - add 'lm: corpus/<name>/lm/<file.arpa>' to experiment.yml "
            "under decode:\n"
            " - run 'sphinx lm <corpus> --all' to generate one"
        )

    overrides["DEC_CFG_LANGUAGEMODEL"] = lm_path

    overrides["DEC_CFG_LISTOFFILES"] = (
        f"{exp_dir}/etc/{db_name}_decode.fileids"
    )
    overrides["DEC_CFG_TRANSCRIPTFILE"] = (
        f"{exp_dir}/etc/{db_name}_decode.transcription"
    )

    user_overrides = experiment.get("sphinxtrain", {})
    if user_overrides:
        overrides.update(user_overrides)

    cfg = _apply_overrides(cfg, overrides)
    return cfg

##############################################################################
# Internal helpers
##############################################################################

def _validate_split(corpus_name: str, split_name: str, corpus: dict):
    """Ensure that a split name is defined in a corpus's `splits` mapping.

    Args:
        corpus_name: Corpus name, used only in the error message.
        split_name: Split name to look up.
        corpus: Parsed `corpus.yml` mapping.

    Raises:
        ValueError: If `split_name` is not a key of `corpus["splits"]`. The
            message lists the available split names.
    """
    splits = corpus.get("splits", {})
    if split_name not in splits:
        available = ", ".join(splits.keys())
        raise ValueError(
            f"Split '{split_name}' not found in corpus '{corpus_name}'. "
            f"Available: {available}"
        )


def _resolve_lm_path(decode_cfg: dict, sphinx_root: Path) -> str | None:
    """Resolve the language model path used for decoding.

    `decode.lm` from `experiment.yml` takes precedence and is interpreted
    relative to the project root. Otherwise the decode corpus's `lm` key
    from `corpus.yml` is used, relative to that corpus's directory. Neither
    path is checked for existence.

    Args:
        decode_cfg: The experiment's `decode` mapping, after
            `load_experiment` has set `decode_cfg["corpus"]["_corpus"]`.
        sphinx_root: Project root, used to anchor `decode.lm`.

    Returns:
        The language model path as a string, or `None` if neither source
            defines one.
    """
    # experiment.yml override
    explicit_lm = decode_cfg.get("lm")
    if explicit_lm:
        return str(sphinx_root / explicit_lm)

    # corpus default
    corpus = decode_cfg.get("corpus", {}).get("_corpus")
    if corpus:
        lm_rel = corpus.get("lm", "")
        if lm_rel:
            return str(corpus["_dir"] / lm_rel)
    return None


def _apply_overrides(cfg: str, overrides: dict) -> str:
    r"""Rewrite Perl `$KEY = value;` assignments in `sphinx_train.cfg` text.

    Each value is converted with `_to_perl_value`, then:

    1. If an active (uncommented) assignment `$KEY = ...;` exists, every such
       line is rewritten, keeping its leading whitespace and any trailing
       `# comment`.
    2. Otherwise, if a commented-out assignment `# $KEY = ...;` exists, the
       first such line is uncommented and rewritten.
    3. Otherwise the key is skipped.

    Matching is line-based and the old value is matched non-greedily up to
    the first `;` that is followed only by optional whitespace and an
    optional comment.

    Args:
        cfg: Full text of a `sphinx_train.cfg` file.
        overrides: Mapping of Perl variable names (without the `$`) to
            Python values (`bool`, `int`, `float`, or anything convertible
            with `str`). Applied in iteration order.

    Returns:
        The rewritten configuration text.

    Raises:
        re.error: If a converted value contains a backslash sequence that is
            invalid in a `re.sub` replacement string (for example `\d`).

    Warning:
        - Keys with no matching line in the template are dropped without any
          message.
        - Values are spliced unescaped into the `re.sub` replacement string,
          so backslash sequences in a value (`\1`, `\g<0>`, `\n`, ...) are
          interpreted as replacement escapes instead of being copied
          literally, or raise `re.error`.
    """  # noqa: DOC502
    for key, value in overrides.items():
        perl_value = _to_perl_value(value)

        # look for uncommented vars first
        pattern = re.compile(
            r"^(\s*\$" + re.escape(key) + r"\s*=\s*)(.+?)(;\s*(?:#.*)?)$",
            re.MULTILINE,
        )

        if pattern.search(cfg):
            cfg = pattern.sub(rf"\g<1>{perl_value}\g<3>", cfg)
            continue

        # look for commented out vars
        comment_pattern = re.compile(
            r"^(\s*)#\s*(\$"
            + re.escape(key)
            + r"\s*=\s*)(.+?)(;\s*(?:#.*)?)$",
            re.MULTILINE,
        )

        if comment_pattern.search(cfg):
            cfg = comment_pattern.sub(
                rf"\g<1>\g<2>{perl_value}\g<4>", cfg, count=1
            )

    return cfg


def _to_perl_value(value) -> str:
    """Convert a Python value into a Perl literal for `sphinx_train.cfg`.

    The rules are applied in order:

    1. `bool` becomes `'yes'` or `'no'` (checked before `int`, because
       `bool` is a subclass of `int`).
    2. `int` and `float` are emitted bare, via `str()`.
    3. Any other value is converted with `str()`. If the result contains
       `$`, it is wrapped in double quotes so that Perl interpolates
       variables such as `$CFG_BASE_DIR`.
    4. Strings that look numeric (optional `-`, digits, optional `.` and
       digits, optional exponent, e.g. `1e-80`) are emitted bare.
    5. Everything else is wrapped in single quotes.

    Args:
        value (Any): Value from the override mapping, typically parsed from
            YAML.

    Returns:
        The Perl source text for the value.

    Warning:
        Quote characters inside the value are not escaped. A string that
        contains `'` (or `"` when it also contains `$`) produces a malformed
        Perl literal in `sphinx_train.cfg`.

    Examples:
        >>> _to_perl_value(True)
        "'yes'"
        >>> _to_perl_value(4000)
        '4000'
        >>> _to_perl_value(16000.0)
        '16000.0'
        >>> _to_perl_value("1e-80")
        '1e-80'
        >>> _to_perl_value("4000")
        '4000'
        >>> _to_perl_value("$CFG_BASE_DIR/feat")
        '"$CFG_BASE_DIR/feat"'
        >>> _to_perl_value("mswav")
        "'mswav'"
        >>> _to_perl_value("it's")
        "'it's'"
    """
    if isinstance(value, bool):
        return "'yes'" if value else "'no'"
    if isinstance(value, (int, float)):
        return str(value)

    s = str(value)

    # strings containing $ need double quotes
    if "$" in s:
        return f'"{s}"'

    # numeric strings (example: 1e-80)
    if re.match(r"^-?\d+\.?\d*(e[+-]?\d+)?$", s, re.IGNORECASE):
        return s

    # everything else is single quotes
    return f"'{s}'"
