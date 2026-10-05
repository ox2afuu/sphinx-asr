#!/usr/bin/env python3
"""LibriSpeech corpus adapter.

Loaded with `get_adapter("librispeech")` by `sphinx.sh setup` and
`sphinx.sh lm`. LibriSpeech stores each chapter as a directory
`<speaker>/<chapter>/` holding the audio files and one
`<speaker>-<chapter>.trans.txt` transcript whose lines have the form
`<utt_id> <TEXT>`. The text is already upper-case and unpunctuated, so it
is passed through unchanged.

Files read: every `*.trans.txt` below the split's audio directory.
Files written: none. Environment variables: none; the project root is
inferred from the corpus directory.
"""

from pathlib import Path
from typing import Iterator


def get_utterances(
        corpus_dir: Path,
        split_name: str,
        split_cfg: dict,
        corpus_cfg: dict
) -> Iterator[tuple[str, str]]:
    """Yield `(fileid, text)` pairs for one LibriSpeech split.

    The split's audio directory is `corpus_dir / split_cfg["audio"]`, or
    `corpus_dir / split_name` when `audio` is not set. Every `*.trans.txt`
    file below it is read in sorted path order, and each non-blank line is
    split at its first space into an utterance id and the text.

    The project root is taken to be `corpus_dir.parent.parent`; that is, the
    corpus is assumed to live at `<SPHINX_ROOT>/corpus/<name>/`.

    Args:
        corpus_dir: Corpus directory (the `_dir` key set by `load_corpus`).
        split_name: Split name. Also the default audio subdirectory.
        split_cfg: The split's mapping from `corpus.yml`. Only the optional
            `audio` key is used.
        corpus_cfg: The full `corpus.yml` mapping. Unused by this adapter;
            accepted for interface compatibility.

    Yields:
        Pairs `(fileid, text)`. `fileid` is `<chapter_dir>/<utt_id>`
            relative to the project root, without an extension, and `text` is
            the transcript text exactly as it appears in the file.

    Raises:
        FileNotFoundError: If the split's audio directory does not exist.
            Because this is a generator, it is raised on the first `next()`
            call, not when the function is called.
    """
    audio_dir = corpus_dir / split_cfg.get("audio", split_name)

    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}\n"
            f"Have you downloaded the '{split_name}' split?"
        )

    for trans_file in sorted(audio_dir.rglob("*.trans.txt")):
        utt_dir = trans_file.parent
        sphinx_root = corpus_dir.parent.parent

        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                utt_id, text = line.split(" ", 1)
                fileid = (utt_dir / utt_id).relative_to(sphinx_root)

                yield str(fileid), text
