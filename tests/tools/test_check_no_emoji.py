"""Tests for ``tools/check_no_emoji.py``."""

import check_no_emoji as cne


def test_plain_text_and_typographic_symbols_are_allowed():
    """ASCII, arrows and the multiplication sign are not emoji.

    Given:
        Text with ``->``, a Unicode right arrow and ``5 \u00d7 2``;
        expected: none of these are flagged.
    When:
        ``find_emoji`` scans it.
    Then:
        No hits are returned.
    """
    text = "build -> bin/ \u2192 done, 5 \u00d7 2\n"

    hits = cne.find_emoji(text)

    assert hits == []


def test_emoji_is_reported_with_position():
    """Each emoji is reported with its 1-based line and column.

    Given:
        A second line ``ok \U0001f680`` (rocket); baseline: first line clean.
    When:
        ``find_emoji`` scans it.
    Then:
        One hit at line 2, column 4 for the rocket.
    """
    text = "clean\nok \U0001f680\n"

    hits = cne.find_emoji(text)

    assert hits == [(2, 4, "\U0001f680")]


def test_main_skips_binary_and_flags_emoji(tmp_path, capsys):
    """The CLI ignores binary files and fails on emoji in text files.

    Given:
        A binary file containing NUL bytes and a text file with a check
        mark dingbat; expected: only the text file is reported.
    When:
        ``main`` is called with both paths.
    Then:
        The exit status is 1 and only the text file appears in output.
    """
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\x00\xf0\x9f\x9a\x80")
    text = tmp_path / "notes.md"
    text.write_text("done \u2705\n", encoding="utf-8")

    rc = cne.main([str(binary), str(text)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "notes.md:1:6: EMJ001" in out
    assert "blob.bin" not in out
