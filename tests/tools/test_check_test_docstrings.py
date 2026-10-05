"""Tests for ``tools/check_test_docstrings.py``, the GIVEN/WHEN/THEN checker.

These tests were written before the checker was implemented. Each one feeds
a small, synthetic test module to the checker and asserts the exact rule
codes it reports.
"""

import textwrap

import check_test_docstrings as ctd

GOOD = '''
def test_ok():
    """Something is rendered.

    Given:
        An input; baseline: the default output.
    When:
        The function is called.
    Then:
        The output matches.
    """
    assert True
'''


def codes(source: str) -> list[str]:
    """Run the checker on dedented source and return the rule codes found.

    Args:
        source: Python source code of a synthetic test module.

    Returns:
        The rule codes reported, in report order.
    """
    return [f.code for f in ctd.check_source(textwrap.dedent(source))]


def write(tmp_path, name: str, source: str):
    """Write dedented source to a file under ``tmp_path``.

    Args:
        tmp_path: Directory to write into.
        name: File name to create.
        source: Python source to write.

    Returns:
        The path of the written file.
    """
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


# ---------------------------------------------------------------- happy path


def test_well_formed_test_has_no_findings():
    """A complete GIVEN/WHEN/THEN docstring produces no findings.

    Given:
        A module with one test whose docstring has a summary line and
        non-empty Given (with a baseline), When and Then sections;
        expected: the checker accepts it.
    When:
        ``check_source`` is run on the module.
    Then:
        No findings are reported.
    """
    result = codes(GOOD)

    assert result == []


def test_non_test_functions_are_ignored():
    """Helpers, fixtures and non-Test classes are not checked.

    Given:
        A module with an undocumented helper, an undocumented method
        ``test_x`` on a class not named ``Test*`` and an undocumented
        nested ``test_inner``; baseline: only collected tests are checked.
    When:
        ``check_source`` is run on the module.
    Then:
        No findings are reported.
    """
    source = '''
    def helper():
        pass

    class Helper:
        def test_x(self):
            pass

    def outer():
        def test_inner():
            pass
    '''

    result = codes(source)

    assert result == []


# --------------------------------------------------------------------- TDS001


def test_missing_docstring_is_tds001():
    """A module-level test with no docstring is reported as TDS001.

    Given:
        A ``test_*`` function with no docstring; expected: exactly one
        finding, since no other rule can run without a docstring.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS001.
    """
    result = codes("def test_x():\n    assert True\n")

    assert result == ["TDS001"]


def test_method_on_test_class_is_checked():
    """Methods named ``test_*`` on ``Test*`` classes are checked.

    Given:
        A class ``TestThing`` with an undocumented method ``test_y``;
        baseline: pytest collects it, so the checker must too.
    When:
        ``check_source`` is run on the module.
    Then:
        A TDS001 finding is reported for the method.
    """
    source = '''
    class TestThing:
        """Docstring."""

        def test_y(self):
            assert True
    '''

    result = codes(source)

    assert result == ["TDS001"]


def test_async_test_is_checked():
    """``async def test_*`` functions are checked like sync ones.

    Given:
        An undocumented ``async def test_a``; expected: treated as a test.
    When:
        ``check_source`` is run on the module.
    Then:
        A TDS001 finding is reported.
    """
    result = codes("async def test_a():\n    assert True\n")

    assert result == ["TDS001"]


# --------------------------------------------------------------------- TDS002


def test_summary_without_period_is_tds002():
    """A summary line that does not end in a period is TDS002.

    Given:
        A docstring whose first line is ``No period here``; baseline: the
        rest of the docstring is valid.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS002.
    """
    source = GOOD.replace("Something is rendered.", "No period here")

    result = codes(source)

    assert result == ["TDS002"]


def test_docstring_starting_with_section_is_tds002():
    """A docstring that opens directly with ``Given:`` lacks a summary.

    Given:
        A docstring whose first line is the ``Given:`` header; expected:
        sections are otherwise complete.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS002.
    """
    source = '''
    def test_x():
        """Given:
            Input; baseline: none.
        When:
            Called.
        Then:
            Works.
        """
    '''

    result = codes(source)

    assert result == ["TDS002"]


# --------------------------------------------------------------------- TDS003


def test_missing_when_and_then_is_tds003_each():
    """Each missing required section is reported separately as TDS003.

    Given:
        A docstring with a summary and only a Given section; baseline:
        When and Then are both required.
    When:
        ``check_source`` is run on the module.
    Then:
        Two TDS003 findings are reported, one naming When and one Then.
    """
    source = '''
    def test_x():
        """Summary.

        Given:
            Input; baseline: none.
        """
    '''

    findings = ctd.check_source(textwrap.dedent(source))

    assert [f.code for f in findings] == ["TDS003", "TDS003"]
    assert "When" in findings[0].message
    assert "Then" in findings[1].message


# --------------------------------------------------------------------- TDS004


def test_then_before_when_is_tds004():
    """Sections in the wrong order are reported as TDS004.

    Given:
        A docstring with Given, Then, When in that order; expected order
        is Given, When, Then.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS004.
    """
    source = '''
    def test_x():
        """Summary.

        Given:
            Input; baseline: none.
        Then:
            Works.
        When:
            Called.
        """
    '''

    result = codes(source)

    assert result == ["TDS004"]


def test_raises_before_then_is_tds004():
    """Raises must come after Then.

    Given:
        A test asserting ``ValueError`` whose Raises section precedes
        Then; baseline: Raises is otherwise correct.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS004.
    """
    source = '''
    import pytest

    def test_x():
        """Summary.

        Given:
            Input; baseline: none.
        When:
            Called.
        Raises:
            ValueError: always.
        Then:
            It raises.
        """
        with pytest.raises(ValueError):
            int("x")
    '''

    result = codes(source)

    assert result == ["TDS004"]


# --------------------------------------------------------------------- TDS005


def test_empty_section_body_is_tds005():
    """A section header followed by no text is TDS005.

    Given:
        A docstring whose When header has no body; baseline: every other
        section is filled in.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS005 and it names the When section.
    """
    source = '''
    def test_x():
        """Summary.

        Given:
            Input; baseline: none.
        When:
        Then:
            Works.
        """
    '''

    findings = ctd.check_source(textwrap.dedent(source))

    assert [f.code for f in findings] == ["TDS005"]
    assert "When" in findings[0].message


# --------------------------------------------------------------------- TDS006


def test_pytest_raises_without_raises_section_is_tds006():
    """A ``with pytest.raises`` test must document a Raises section.

    Given:
        A test using ``with pytest.raises(ValueError)`` and a docstring
        with no Raises section; expected: one finding.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS006.
    """
    source = GOOD.replace(
        "    assert True",
        "    with pytest.raises(ValueError):\n        int('x')",
    )

    result = codes("import pytest\n" + source)

    assert result == ["TDS006"]


def test_raises_section_missing_one_exception_is_tds006():
    """Every exception asserted must be named in the Raises section.

    Given:
        A test asserting ``pytest.raises((KeyError, yaml.YAMLError))``
        whose Raises section only names ``KeyError``; baseline: dotted
        names are matched on their last component.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS006 and it names ``YAMLError``.
    """
    source = '''
    import pytest
    import yaml

    def test_x():
        """Summary.

        Given:
            Input; baseline: none.
        When:
            Called.
        Then:
            It raises.
        Raises:
            KeyError: on a missing key.
        """
        pytest.raises((KeyError, yaml.YAMLError), dict().pop, "k")
    '''

    findings = ctd.check_source(textwrap.dedent(source))

    assert [f.code for f in findings] == ["TDS006"]
    assert "YAMLError" in findings[0].message


def test_assert_raises_without_raises_section_is_tds006():
    """``self.assertRaises`` counts as an exception assertion.

    Given:
        A ``Test*`` method calling ``self.assertRaises(TypeError, ...)``
        with no Raises section; expected: one finding.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS006.
    """
    source = '''
    class TestX:
        """Docstring."""

        def test_x(self):
            """Summary.

            Given:
                Input; baseline: none.
            When:
                Called.
            Then:
                It raises.
            """
            self.assertRaises(TypeError, len, 1)
    '''

    result = codes(source)

    assert result == ["TDS006"]


def test_documented_raises_has_no_findings():
    """A Raises section naming every asserted exception is accepted.

    Given:
        A test asserting ``pytest.raises(SystemExit)`` with a Raises
        section naming ``SystemExit``; expected: no findings.
    When:
        ``check_source`` is run on the module.
    Then:
        No findings are reported.
    """
    source = '''
    import pytest

    def test_x():
        """Summary.

        Given:
            Input; baseline: none.
        When:
            Called.
        Then:
            It exits.
        Raises:
            SystemExit: from ``sys.exit``.
        """
        with pytest.raises(SystemExit, match="1"):
            raise SystemExit(1)
    '''

    result = codes(source)

    assert result == []


# --------------------------------------------------------------------- TDS007


def test_raises_section_without_assertion_is_tds007():
    """A Raises section with no exception assertion in the body is TDS007.

    Given:
        A test documenting ``Raises: ValueError`` whose body only does a
        plain ``assert``; baseline: sections are otherwise valid.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS007.
    """
    source = GOOD.replace(
        '        The output matches.\n    """',
        '        The output matches.\n    Raises:\n        ValueError: x.\n'
        '    """',
    )

    result = codes(source)

    assert result == ["TDS007"]


# --------------------------------------------------------------------- TDS008


def test_given_without_baseline_is_tds008():
    """A Given section that states no baseline is TDS008.

    Given:
        A Given body of ``An input.`` with neither "baseline" nor
        "expected"; baseline: all other sections valid.
    When:
        ``check_source`` is run on the module.
    Then:
        The only finding is TDS008.
    """
    source = GOOD.replace("An input; baseline: the default output.", "An input.")

    result = codes(source)

    assert result == ["TDS008"]


def test_given_with_expected_satisfies_tds008():
    """The word "expected" (any case) satisfies the baseline rule.

    Given:
        A Given body containing ``Expected`` but not "baseline";
        baseline: TDS008 accepts either word.
    When:
        ``check_source`` is run on the module.
    Then:
        No findings are reported.
    """
    source = GOOD.replace(
        "An input; baseline: the default output.",
        "An input. Expected: the default output.",
    )

    result = codes(source)

    assert result == []


# ------------------------------------------------------------ output and CLI


def test_finding_format_is_path_line_code_message():
    """Findings render as ``path:line: CODE message``.

    Given:
        A missing-docstring test on line 3 of a file named ``t.py``;
        expected: the def line is reported.
    When:
        The finding is converted with ``str``.
    Then:
        The text starts with ``t.py:3: TDS001 ``.
    """
    findings = ctd.check_source("\n\ndef test_x():\n    pass\n", "t.py")

    text = str(findings[0])

    assert text.startswith("t.py:3: TDS001 ")


def test_main_exits_zero_on_clean_file(tmp_path, capsys):
    """The CLI exits 0 and prints nothing for a clean file.

    Given:
        A file containing only a well-formed test; expected: exit 0.
    When:
        ``main`` is called with that path.
    Then:
        The return code is 0 and stdout is empty.
    """
    path = write(tmp_path, "test_ok.py", GOOD)

    rc = ctd.main([str(path)])

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_exits_one_and_prints_findings(tmp_path, capsys):
    """The CLI exits 1 and prints each finding when there are violations.

    Given:
        A file with one undocumented test; expected: one finding.
    When:
        ``main`` is called with that path.
    Then:
        The return code is 1 and stdout has one ``path:1: TDS001`` line.
    """
    path = write(tmp_path, "test_bad.py", "def test_x():\n    pass\n")

    rc = ctd.main([str(path)])

    out = capsys.readouterr().out.splitlines()
    assert rc == 1
    assert len(out) == 1
    assert out[0].startswith(f"{path}:1: TDS001 ")


def test_main_exits_two_on_syntax_error(tmp_path, capsys):
    """The CLI exits 2 when a file cannot be parsed.

    Given:
        One clean file and one file with a syntax error; expected: the
        parse failure dominates the exit code.
    When:
        ``main`` is called with both paths.
    Then:
        The return code is 2 and the bad path is reported on stderr.
    """
    good = write(tmp_path, "test_ok.py", GOOD)
    bad = write(tmp_path, "test_broken.py", "def test_x(:\n")

    rc = ctd.main([str(good), str(bad)])

    assert rc == 2
    assert str(bad) in capsys.readouterr().err


def test_main_recurses_into_directories(tmp_path, capsys):
    """A directory argument is expanded to the ``*.py`` files under it.

    Given:
        A directory containing ``sub/test_bad.py`` with an undocumented
        test; expected: the nested file is found.
    When:
        ``main`` is called with the directory.
    Then:
        The return code is 1 and the nested file is reported.
    """
    (tmp_path / "sub").mkdir()
    bad = write(tmp_path, "sub/test_bad.py", "def test_x():\n    pass\n")

    rc = ctd.main([str(tmp_path)])

    assert rc == 1
    assert str(bad) in capsys.readouterr().out
