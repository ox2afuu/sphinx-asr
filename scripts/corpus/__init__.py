#!/usr/bin/env python3
"""Corpus adapter registry and interface.

Each supported corpus format has an adapter module
`scripts/corpus/<name>.py`, where `<name>` matches the corpus directory
`corpus/<name>/`. An adapter hides a corpus's transcript format and
directory layout behind one generator, which `setup.py`
(`sphinx.sh setup`) and `lm.py` (`sphinx.sh lm`) consume through
`get_adapter`.

The required interface is:

```python
def get_utterances(corpus_dir, split_name, split_cfg, corpus_cfg)
        -> Iterator[tuple[str, str]]
```

It must yield `(fileid, text)` pairs. `fileid` is the audio path relative
to `SPHINX_ROOT` with no extension (SphinxTrain appends `.mfc` and looks
under `CFG_FEATFILES_DIR`, which is set to the project root); `text` is
the cleaned, space-separated word sequence. `librispeech.py` is the
simplest adapter and `switchboard.py` shows transcript cleaning.

To add a corpus, create `corpus/<name>/corpus.yml` and
`scripts/corpus/<name>.py` implementing `get_utterances`.

This module reads no files and no environment variables itself.
"""

import importlib
from pathlib import Path


def get_adapter(corpus_name: str):
    """Import and return the adapter module for a corpus.

    The module is imported as `corpus.<corpus_name>`, which requires the
    `scripts/` directory to be on `sys.path`. That holds when a script in
    `scripts/` is run directly, as `sphinx.sh` does.

    Args:
        corpus_name: Corpus name, matching both `corpus/<name>/` and
            `scripts/corpus/<name>.py`.

    Returns:
        adapter (types.ModuleType): The imported adapter module, which
            provides `get_utterances`.

    Raises:
        ModuleNotFoundError: If the import fails with `ModuleNotFoundError`.
            The message always says that no adapter was found, even when the
            adapter file exists but one of its own imports is missing.
    """
    try:
        return importlib.import_module(f"corpus.{corpus_name}")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"No corpus adapter found for '{corpus_name}'. "
            f"Expected: scripts/corpus/{corpus_name}.py"
        )
