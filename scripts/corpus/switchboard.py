"""Switchboard corpus adapter.

Loaded with `get_adapter("switchboard")` by `sphinx.sh setup` and
`sphinx.sh lm`. Switchboard transcripts are segment lists with one line
per utterance (utterance id, start time, end time, text) and use
bracketed markup for noises, partial words and laughter. `clean_text`
reduces that markup to the word sequence used for training, keeping three
noise events as filler tokens.

The adapter assumes the audio is already segmented into one file per
utterance, named after the utterance id, in the audio directory. The
start and end times in the transcript are ignored.

Files read: the split's transcript file (`transcripts` in the split's
`corpus.yml` entry). Files written: none. Environment variables: none;
the project root is inferred from the corpus directory.

Attributes:
    FILLERS: Lower-case names of the bracketed events kept as filler
        tokens.
    LAUGHTER_WORD: Matches `[laughter-<word>]`, case-insensitively, and
        captures the word.
    BRACKET_TAG: Matches any `[...]` annotation.
    ANGLE_TAG: Matches `<<...>>` annotations.
    MULTI_SPACE: Matches whitespace runs, for collapsing.
"""

import re
from pathlib import Path
from typing import Iterator

FILLERS = {"noise", "laughter", "vocalized-noise"}

# regex
LAUGHTER_WORD = re.compile(r"\[laughter-(\S+)\]", re.IGNORECASE)
BRACKET_TAG = re.compile(r"\[[^\]]+\]")
ANGLE_TAG = re.compile(r"<<[^>]+>>")
MULTI_SPACE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Normalise the text of one Switchboard transcript segment.

    The transformations are applied in order:

    1. `[laughter-<word>]` becomes `[laughter] <word>` (the tag is matched
       case-insensitively).
    2. The filler tags `[noise]`, `[laughter]` and `[vocalized-noise]` are
       protected with temporary placeholders. Only these exact lower-case
       spellings are protected.
    3. Every other `[...]` tag (for example `[t]`, `[d]`, `[ou]`, or an
       upper-case `[NOISE]`) is deleted, as are `<<...>>` annotations.
    4. The protected fillers are restored as `[NOISE]`, `[LAUGHTER]` and
       `[VOCALIZED-NOISE]`.
    5. The string is upper-cased, and whitespace runs are collapsed to
       single spaces with no leading or trailing space.

    Args:
        text: Raw text of one segment (the fourth field of a transcript
            line).

    Returns:
        The cleaned text. It is empty if the segment contained only removed
            annotations.

    Examples:
        >>> clean_text("[laughter-yes] i [noise] think [t] <<unclear>> so")
        '[LAUGHTER] YES I [NOISE] THINK SO'
        >>> clean_text("[silence]")
        ''
    """
    text = LAUGHTER_WORD.sub(r"[laughter] \1", text)

    # replace fillers with placeholders so they dont get messed up
    for filler in FILLERS:
        text = text.replace(f"[{filler}]", f"__{filler.upper()}__")

    text = BRACKET_TAG.sub("", text)
    text = ANGLE_TAG.sub("", text)

    # restore fillers
    for filler in FILLERS:
        text = text.replace(f"__{filler.upper()}__", f"[{filler.upper()}]")

    text = text.upper()
    text = MULTI_SPACE.sub(" ", text).strip()

    return text


def get_utterances(
        corpus_dir: Path,
        split_name: str,
        split_cfg: dict,
        corpus_cfg: dict
) -> Iterator[tuple[str, str]]:
    """Yield `(fileid, text)` pairs for one Switchboard split.

    The split's transcript file is read line by line. Each non-blank line is
    split on whitespace into at most four fields (utterance id, start time,
    end time, text). Lines with fewer than four fields are skipped, the text
    is passed through `clean_text`, and segments whose cleaned text is empty
    are skipped.

    The project root is taken to be `corpus_dir.parent.parent`; that is, the
    corpus is assumed to live at `<SPHINX_ROOT>/corpus/<name>/`.

    Args:
        corpus_dir: Corpus directory (the `_dir` key set by `load_corpus`).
        split_name: Split name, used only in the error message.
        split_cfg: The split's mapping from `corpus.yml`. `transcripts`
            (a path relative to `corpus_dir`) is required; `audio`
            optionally overrides the corpus-level audio directory.
        corpus_cfg: The full `corpus.yml` mapping. Its `audio_dir` is used
            when the split has no `audio` key.

    Yields:
        Pairs `(fileid, text)`. `fileid` is `<audio_dir>/<utt_id>` relative
            to the project root, without an extension, and `text` is the
            output of `clean_text`.

    Raises:
        FileNotFoundError: If the transcript file does not exist. Because
            this is a generator, it is raised on the first `next()` call.
    """
    trans_path = corpus_dir / split_cfg["transcripts"]
    if not trans_path.is_file():
        raise FileNotFoundError(
            f"transcript file not found: {trans_path}\n"
            f"Have you downloaded the '{split_name}' split?"
        )

    sphinx_root = corpus_dir.parent.parent
    audio_rel = split_cfg.get("audio") or corpus_cfg.get("audio_dir")
    audio_dir = corpus_dir / audio_rel

    with open(trans_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # utt_id start_time end_time text
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue

            utt_id = parts[0]
            text = clean_text(parts[3])
            if not text:
                continue

            fileid = (audio_dir / utt_id).relative_to(sphinx_root)

            yield str(fileid), text
