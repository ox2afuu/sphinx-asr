"""Characterization tests for ``scripts/lib/config.py``.

These tests pin the current behaviour of the pure (or tmp_path-only) logic
that turns ``experiment.yml`` into ``sphinx_train.cfg``. They describe what
the code does today, not what it should do; known defects are asserted in
``test_known_defects.py`` as strict xfails instead.
"""

from pathlib import Path

import pytest
from lib.config import (
    _apply_overrides,
    _resolve_lm_path,
    _to_perl_value,
    _validate_split,
    generate_sphinx_train_cfg,
    load_corpus,
    load_yaml,
    validate_experiment,
)

# ------------------------------------------------------------- _to_perl_value


def test_bool_becomes_perl_yes_no():
    """Booleans are rendered as quoted Perl yes/no strings.

    Given:
        A Python ``True`` and ``False``; baseline: sphinxtrain expects
        ``'yes'`` / ``'no'`` literals.
    When:
        ``_to_perl_value`` is called on each.
    Then:
        ``True`` renders as ``'yes'`` and ``False`` as ``'no'``.
    """
    rendered = (_to_perl_value(True), _to_perl_value(False))

    assert rendered == ("'yes'", "'no'")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0"), (4000, "4000"), (-3, "-3"), (16000.0, "16000.0"), (1e-80, "1e-80")],
)
def test_numbers_are_rendered_bare(value, expected):
    """Ints and floats are rendered unquoted via ``str``.

    Given:
        An int or float; expected: its ``str()`` form, with no quotes.
    When:
        ``_to_perl_value`` is called on it.
    Then:
        The result equals the bare ``str()`` of the number.
    """
    rendered = _to_perl_value(value)

    assert rendered == expected


@pytest.mark.parametrize("value", ["1e-80", "-2.5", "4000", "1.", "3E+5"])
def test_numeric_strings_are_rendered_bare(value):
    """Strings that look like plain or exponent numbers are left unquoted.

    Given:
        A string matching ``-?digits[.digits][e[+-]digits]``; baseline:
        YAML values like ``1e-80`` load as strings and must stay numeric.
    When:
        ``_to_perl_value`` is called on it.
    Then:
        The string is returned unchanged, with no quotes.
    """
    rendered = _to_perl_value(value)

    assert rendered == value


def test_leading_dot_number_string_is_single_quoted():
    """A numeric string without a leading digit is treated as text.

    Given:
        The string ``".5"``; baseline: the numeric regex requires a
        leading digit.
    When:
        ``_to_perl_value`` is called on it.
    Then:
        It is single-quoted as ``'.5'``.
    """
    rendered = _to_perl_value(".5")

    assert rendered == "'.5'"


def test_string_with_dollar_is_double_quoted():
    """Strings containing ``$`` are double-quoted so Perl interpolates them.

    Given:
        The string ``$CFG_BASE_DIR/feat``; expected: Perl variable
        interpolation must still work.
    When:
        ``_to_perl_value`` is called on it.
    Then:
        It is wrapped in double quotes.
    """
    rendered = _to_perl_value("$CFG_BASE_DIR/feat")

    assert rendered == '"$CFG_BASE_DIR/feat"'


def test_other_values_are_single_quoted_strings():
    """Plain strings and ``Path`` objects are single-quoted via ``str``.

    Given:
        The string ``live`` and ``Path("/srv/x")``; baseline: any
        non-bool, non-number, non-``$``, non-numeric value.
    When:
        ``_to_perl_value`` is called on each.
    Then:
        They render as ``'live'`` and ``'/srv/x'``.
    """
    rendered = (_to_perl_value("live"), _to_perl_value(Path("/srv/x")))

    assert rendered == ("'live'", "'/srv/x'")


# ----------------------------------------------------------- _apply_overrides


def test_override_replaces_active_assignment_and_keeps_comment():
    """An active ``$VAR = ...;`` line has only its value replaced.

    Given:
        A cfg line ``  $CFG_B = "x";  # note``; baseline: indentation and
        the trailing comment are part of the template.
    When:
        ``_apply_overrides`` sets ``CFG_B`` to ``"y"``.
    Then:
        The line becomes ``  $CFG_B = 'y';  # note``.
    """
    cfg = '  $CFG_B = "x";  # note\n'

    result = _apply_overrides(cfg, {"CFG_B": "y"})

    assert result == "  $CFG_B = 'y';  # note\n"


def test_override_replaces_every_active_occurrence():
    """All active assignments of the same variable are rewritten.

    Given:
        A cfg assigning ``$CFG_A`` twice; baseline: the active-line
        substitution has no ``count`` limit.
    When:
        ``_apply_overrides`` sets ``CFG_A`` to 9.
    Then:
        Both lines read ``$CFG_A = 9;``.
    """
    cfg = "$CFG_A = 1;\n$CFG_A = 2;\n"

    result = _apply_overrides(cfg, {"CFG_A": 9})

    assert result == "$CFG_A = 9;\n$CFG_A = 9;\n"


def test_override_uncomments_only_first_commented_assignment():
    """With no active line, the first commented-out assignment is enabled.

    Given:
        Two commented ``$CFG_C`` lines and no active one; expected: only
        the first is uncommented (``count=1``).
    When:
        ``_apply_overrides`` sets ``CFG_C`` to 5.
    Then:
        The first line becomes ``$CFG_C = 5;`` and the second is unchanged.
    """
    cfg = "# $CFG_C = 3;\n#$CFG_C = 4;\n"

    result = _apply_overrides(cfg, {"CFG_C": 5})

    assert result == "$CFG_C = 5;\n#$CFG_C = 4;\n"


def test_active_assignment_wins_over_commented_one():
    """If an active line exists, commented copies are left alone.

    Given:
        An active ``$CFG_A = 1;`` followed by ``# $CFG_A = 2;``; baseline:
        the active pattern is tried first.
    When:
        ``_apply_overrides`` sets ``CFG_A`` to 9.
    Then:
        Only the active line changes.
    """
    cfg = "$CFG_A = 1;\n# $CFG_A = 2;\n"

    result = _apply_overrides(cfg, {"CFG_A": 9})

    assert result == "$CFG_A = 9;\n# $CFG_A = 2;\n"


def test_override_key_is_matched_literally():
    """Keys are regex-escaped and must match the whole variable name.

    Given:
        A cfg with ``$CFG_AB = 1;`` and an override for ``CFG_A``;
        expected: ``CFG_A`` is not a prefix match for ``CFG_AB``.
    When:
        ``_apply_overrides`` is applied.
    Then:
        The cfg is unchanged.
    """
    cfg = "$CFG_AB = 1;\n"

    result = _apply_overrides(cfg, {"CFG_A": 2})

    assert result == cfg


# ------------------------------------------------- _validate_split / lm path


def test_validate_split_accepts_known_split():
    """A split listed in the corpus passes validation silently.

    Given:
        A corpus with splits ``train`` and ``test``; expected: no error.
    When:
        ``_validate_split`` is called for ``train``.
    Then:
        It returns ``None``.
    """
    corpus = {"splits": {"train": {}, "test": {}}}

    result = _validate_split("demo", "train", corpus)

    assert result is None


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


def test_resolve_lm_path_prefers_explicit_lm_under_root():
    """An explicit ``decode.lm`` is resolved relative to SPHINX_ROOT.

    Given:
        ``decode.lm`` of ``lm/x.arpa`` and a corpus that also has an LM;
        baseline: the experiment value takes priority.
    When:
        ``_resolve_lm_path`` is called with root ``/r``.
    Then:
        It returns ``/r/lm/x.arpa``.
    """
    decode_cfg = {
        "lm": "lm/x.arpa",
        "corpus": {"_corpus": {"lm": "lm/c.arpa", "_dir": Path("/c")}},
    }

    result = _resolve_lm_path(decode_cfg, Path("/r"))

    assert result == "/r/lm/x.arpa"


def test_resolve_lm_path_falls_back_to_corpus_lm():
    """Without ``decode.lm``, the corpus LM is resolved against its dir.

    Given:
        A decode corpus whose ``corpus.yml`` has ``lm: lm/c.arpa`` and
        ``_dir`` ``/c``; expected: the corpus default is used.
    When:
        ``_resolve_lm_path`` is called.
    Then:
        It returns ``/c/lm/c.arpa``.
    """
    decode_cfg = {"corpus": {"_corpus": {"lm": "lm/c.arpa", "_dir": Path("/c")}}}

    result = _resolve_lm_path(decode_cfg, Path("/r"))

    assert result == "/c/lm/c.arpa"


def test_resolve_lm_path_returns_none_without_any_lm():
    """With neither an explicit nor a corpus LM the result is ``None``.

    Given:
        A decode corpus with no ``lm`` key; expected: no path.
    When:
        ``_resolve_lm_path`` is called.
    Then:
        It returns ``None``.
    """
    decode_cfg = {"corpus": {"_corpus": {"_dir": Path("/c")}}}

    result = _resolve_lm_path(decode_cfg, Path("/r"))

    assert result is None


# ---------------------------------------------------------- validate_experiment

VALID_EXPERIMENT = {
    "train": {"corpora": [{"name": "librispeech", "splits": ["train-clean-100"]}]},
    "decode": {"corpus": {"name": "librispeech", "split": "test-clean"}},
    "sphinxtrain": {"CFG_N_TIED_STATES": 4000},
}


def test_validate_experiment_accepts_minimal_valid_config():
    """A config with train.corpora and decode.corpus validates silently.

    Given:
        A minimal valid experiment mapping; expected: no error output.
    When:
        ``validate_experiment`` is called.
    Then:
        It returns ``None``.
    """
    result = validate_experiment(VALID_EXPERIMENT, Path("experiment.yml"))

    assert result is None


def test_validate_experiment_reports_all_errors_then_exits(capsys):
    """Every structural problem is listed in one message before exiting.

    Given:
        An experiment whose corpus entry lacks name and split, with no
        decode section and a non-mapping sphinxtrain; expected: four
        errors reported together.
    When:
        ``validate_experiment`` is called.
    Then:
        The process exits with status 1 and stderr lists each problem.
    Raises:
        SystemExit: from ``asr_util.err``.
    """
    experiment = {"train": {"corpora": [{}]}, "sphinxtrain": ["x"]}

    with pytest.raises(SystemExit) as exc_info:
        validate_experiment(experiment, Path("experiment.yml"))

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "train.corpora[0]: missing 'name'" in stderr
    assert "train.corpora[0]: missing 'split' or 'splits'" in stderr
    assert "missing 'decode:' section" in stderr
    assert "sphinxtrain: must be a mapping" in stderr


# ------------------------------------------------------- load_yaml / load_corpus


def test_load_yaml_empty_file_returns_empty_dict(tmp_path):
    """An empty YAML file loads as ``{}`` rather than ``None``.

    Given:
        An empty file; baseline: ``yaml.safe_load`` returns ``None``.
    When:
        ``load_yaml`` reads it.
    Then:
        The result is ``{}``.
    """
    path = tmp_path / "empty.yml"
    path.write_text("")

    result = load_yaml(path)

    assert result == {}


def test_load_yaml_rejects_non_mapping(tmp_path):
    """A YAML document that is not a mapping is rejected.

    Given:
        A file containing a YAML list; expected: only mappings allowed.
    When:
        ``load_yaml`` reads it.
    Then:
        A ``TypeError`` naming the file is raised.
    Raises:
        TypeError: because the top-level value is a list.
    """
    path = tmp_path / "list.yml"
    path.write_text("- a\n- b\n")

    with pytest.raises(TypeError, match="list.yml"):
        load_yaml(path)


def test_load_yaml_wraps_syntax_errors(tmp_path):
    """Malformed YAML is re-raised as ``ValueError`` naming the file.

    Given:
        A file with an unterminated flow mapping; expected: a friendly
        error rather than a raw ``yaml.YAMLError``.
    When:
        ``load_yaml`` reads it.
    Then:
        The message mentions ``invalid YAML syntax``.
    Raises:
        ValueError: wrapping the underlying ``yaml.YAMLError``.
    """
    path = tmp_path / "bad.yml"
    path.write_text("a: {b: 1\n")

    with pytest.raises(ValueError, match="invalid YAML syntax"):
        load_yaml(path)


def test_load_corpus_adds_dir_key(tmp_path):
    """``load_corpus`` returns corpus.yml data plus the resolved ``_dir``.

    Given:
        ``corpus/demo/corpus.yml`` under a fake SPHINX_ROOT; expected: the
        YAML keys plus ``_dir``.
    When:
        ``load_corpus("demo", root)`` is called.
    Then:
        ``_dir`` is the corpus directory and YAML values are preserved.
    """
    corpus_dir = tmp_path / "corpus" / "demo"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "corpus.yml").write_text("sample_rate: 8000\n")

    result = load_corpus("demo", tmp_path)

    assert result == {"sample_rate": 8000, "_dir": corpus_dir}


def test_load_corpus_missing_file_raises(tmp_path):
    """A corpus without ``corpus.yml`` raises ``FileNotFoundError``.

    Given:
        A fake SPHINX_ROOT with no ``corpus/`` tree; expected: failure.
    When:
        ``load_corpus("nope", root)`` is called.
    Then:
        The error message names the corpus.
    Raises:
        FileNotFoundError: because ``corpus/nope/corpus.yml`` is absent.
    """
    with pytest.raises(FileNotFoundError, match="nope"):
        load_corpus("nope", tmp_path)


# ------------------------------------------------- generate_sphinx_train_cfg

TEMPLATE = """\
$CFG_DB_NAME = '___DB_NAME___';
$CFG_BASE_DIR = '___BASE_DIR___';
$CFG_WAVFILE_SRATE = 8000.0;
$CFG_N_TIED_STATES = 1000;
# $DEC_CFG_LANGUAGEMODEL = 'none';
"""


def _fake_root(tmp_path: Path) -> Path:
    """Create a fake SPHINX_ROOT holding a minimal sphinx_train.cfg template.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The resolved fake root.
    """
    etc = tmp_path / "vendor" / "sphinxtrain" / "etc"
    etc.mkdir(parents=True)
    (etc / "sphinx_train.cfg").write_text(TEMPLATE)
    return tmp_path.resolve()


def test_generate_cfg_applies_placeholders_corpus_and_user_overrides(tmp_path):
    """Placeholders, corpus values and user overrides are layered in order.

    Given:
        A fake template, a corpus with ``sample_rate: 16000`` and user
        ``sphinxtrain`` overrides for the sample rate and tied states;
        baseline: user overrides beat corpus-derived values.
    When:
        ``generate_sphinx_train_cfg`` renders experiment ``exp/007``.
    Then:
        DB name and base dir are substituted, the user sample rate wins,
        tied states are overridden, and the commented LM line is enabled
        with the resolved path.
    """
    root = _fake_root(tmp_path)
    exp_dir = root / "exp" / "007"
    experiment = {
        "train": {"corpora": [{"_corpus": {"sample_rate": 16000}}]},
        "decode": {"lm": "lm/a.arpa"},
        "sphinxtrain": {"CFG_WAVFILE_SRATE": 22050, "CFG_N_TIED_STATES": 4000},
    }

    cfg = generate_sphinx_train_cfg(exp_dir, experiment, root)

    assert "$CFG_DB_NAME = '007';" in cfg
    assert f"$CFG_BASE_DIR = '{exp_dir}';" in cfg
    assert "$CFG_WAVFILE_SRATE = 22050;" in cfg
    assert "$CFG_N_TIED_STATES = 4000;" in cfg
    assert f"$DEC_CFG_LANGUAGEMODEL = '{root}/lm/a.arpa';" in cfg


def test_generate_cfg_without_language_model_exits(tmp_path, capsys):
    """Rendering without any language model aborts with guidance.

    Given:
        A fake template and an experiment with no ``decode.lm`` and no
        corpus LM; expected: setup refuses to continue.
    When:
        ``generate_sphinx_train_cfg`` is called.
    Then:
        It exits with status 1 and stderr explains how to add an LM.
    Raises:
        SystemExit: from ``asr_util.err``.
    """
    root = _fake_root(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        generate_sphinx_train_cfg(root / "exp" / "001", {"decode": {}}, root)

    assert exc_info.value.code == 1
    assert "no language model defined" in capsys.readouterr().err
