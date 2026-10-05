#!/usr/bin/env python3
"""Generate the SphinxTrain inputs for an experiment from `experiment.yml`.

Invoked as `sphinx.sh setup <exp_dir>`, which runs `setup.py <exp_dir>`.
A relative `<exp_dir>` is resolved against `SPHINX_ROOT`, not the current
directory. The database name `<db>` is the final component of
`<exp_dir>`.

Steps, in order:

1. Create the SphinxTrain directory layout under `<exp_dir>`.
2. Write training and decode fileids and transcriptions with the corpus
   adapters.
3. Build a pruned pronunciation dictionary for every word in the
   training and decode transcripts, drop utterances containing
   out-of-vocabulary (OOV) words from both sets, and write the phone
   list and filler dictionary. No G2P is run.
4. Render `etc/feat.params` from the vendored template.
5. Render `etc/sphinx_train.cfg` with `lib.config`.

Files read:

- `<exp_dir>/experiment.yml` and each referenced `corpus/<name>/corpus.yml`
- the corpus transcripts, through the adapters
- the master lexicon (`train.dict` in `experiment.yml`, else the first
  training corpus's `dict`)
- `vendor/sphinxtrain/etc/feat.params` and
  `vendor/sphinxtrain/etc/sphinx_train.cfg` (templates)

Files written, all under `<exp_dir>/etc/`:

- `<db>_train.fileids`, `<db>_train.transcription`
- `<db>_decode.fileids`, `<db>_decode.transcription`
- `<db>.dic`, `<db>.phone`, `<db>.filler`
- `<db>_train.oov`, only when OOV words exist
- `feat.params`, `sphinx_train.cfg`

Environment variables read: `SPHINX_ROOT`.

Usage: `setup.py <experiment_dir>`
"""

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from lib.asr_util import get_sphinx_root
from lib.config import generate_sphinx_train_cfg, load_experiment

from corpus import get_adapter


@dataclass
class Dictionary:
    """Pruned pronunciation dictionary built from a master lexicon.

    Instances are created with `Dictionary.build` and consumed by
    `generate_dictionary`.

    Attributes:
        entries: Dictionary lines in SphinxTrain format, `WORD PH1 PH2 ...`,
            in sorted word order. Alternative pronunciations follow the first
            as `WORD(2)`, `WORD(3)`, ... in lexicon order. Stress digits are
            removed from the phones.
        phones: Every phone used in `entries`, without stress digits.
        oov: Requested words that have no entry in the lexicon.
    """
    entries: list[str]
    phones: set[str]
    oov: set[str] # words not found in lexicon

    @staticmethod
    def build(lexicon_path: Path, words: set[str]) -> "Dictionary":
        """Build a pruned dictionary for a set of words from a master lexicon.

        The lexicon is read as whitespace-separated lines `WORD PH1 PH2 ...`.
        Blank lines and lines with fewer than two fields are skipped, and
        repeated headwords accumulate as alternative pronunciations. One
        trailing digit is stripped from every phone (`AH0` becomes `AH`). Word
        lookup is exact and case-sensitive.

        Args:
            lexicon_path: Path of the master lexicon (a CMUdict-style file).
            words: Words to look up, normally the transcript vocabulary with
                filler tokens removed.

        Returns:
            The pruned dictionary.

        Note:
            Headwords written with a variant suffix in the lexicon, such as
            CMUdict's `WORD(2)`, are separate keys that no transcript word
            matches, so those variants are not included.
        """
        stress_re = re.compile(r"\d$")  # strip the stress numbers

        available = {}
        with open(lexicon_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) < 2:
                    continue

                word = fields[0]
                phones = fields[1:]

                if word not in available:
                    available[word] = []
                available[word].append(phones)

        entries = []
        phones = set()
        oov = set()

        for word in sorted(words):
            if word in available:
                for i, raw_phones in enumerate(available[word]):
                    clean = [stress_re.sub("", p) for p in raw_phones]
                    phones.update(clean)
                    entry_word = word if i == 0 else f"{word}({i + 1})"
                    entries.append(f"{entry_word} {' '.join(clean)}")
            else:
                oov.add(word)

        return Dictionary(entries=entries, phones=phones, oov=oov)


def _resolve_dict_path(experiment: dict, root: Path) -> Path | None:
    """Locate the master lexicon for an experiment.

    `train.dict` from `experiment.yml`, interpreted relative to the project
    root, is used if it names an existing file. If it is set but missing, a
    warning is printed and the lookup falls through. Otherwise the `dict`
    key of the first training corpus's `corpus.yml` is used, relative to that
    corpus's directory.

    Args:
        experiment: Experiment mapping as returned by `load_experiment`.
        root: Project root (`SPHINX_ROOT`).

    Returns:
        The lexicon path, or `None` if no candidate exists as a file.

    Note:
        Only `train.corpora[0]` is consulted for the corpus default. The
        `dict` keys of other training corpora are ignored.
    """
    # experiment.yml override
    explicit = experiment.get("train", {}).get("dict")
    if explicit:
        path = root / explicit
        if path.is_file():
            return path
        print(f"  Warning: explicit dict not found: {path}")

    # corpus default
    train_corpora = experiment.get("train", {}).get("corpora", [])
    if train_corpora:
        corpus = train_corpora[0].get("_corpus", {})
        dict_rel = corpus.get("dict", "")
        if dict_rel:
            path = corpus["_dir"] / dict_rel
            if path.is_file():
                return path
    return None


def filter_oov_utterances(
        exp_dir: Path,
        db_name: str,
        oov: set[str],
        prefix: str
) -> int:
    """Drop utterances containing OOV words from a fileids/transcription pair.

    Reads `<exp_dir>/etc/<db_name>_<prefix>.fileids` and the matching
    `.transcription` file, keeps only the lines whose transcript words (the
    text between `<s>` and `</s>`) are all absent from `oov`, and rewrites
    both files in place. This stands in for G2P: utterances with unknown
    words are removed instead of receiving generated pronunciations.

    Args:
        exp_dir: Experiment directory.
        db_name: Database name (the experiment directory name).
        oov: Out-of-vocabulary words to filter on. Matching is exact and
            case-sensitive.
        prefix: Which set to filter, `"train"` or `"decode"`.

    Returns:
        The number of utterances removed. 0 is returned without touching any
            file if the fileids file does not exist.

    Raises:
        FileNotFoundError: If the fileids file exists but the transcription
            file does not.

    Warning:
        Fileids and transcription lines are paired with `zip()`, which stops
        at the shorter list. If the two files have different line counts,
        the extra lines of the longer file are silently dropped when the
        files are rewritten, and they are not counted as removed.
    """  # noqa: DOC502
    fileids_path = exp_dir / "etc" / f"{db_name}_{prefix}.fileids"
    trans_path = exp_dir / "etc" / f"{db_name}_{prefix}.transcription"

    if not fileids_path.is_file():
        return 0

    fileids = fileids_path.read_text().splitlines()
    trans = trans_path.read_text().splitlines()

    kept_fileids = []
    kept_trans = []
    removed = 0

    for fid, tr, in zip(fileids, trans):
        words = tr.split("</s>")[0].split("<s>")[-1].split()
        if oov.isdisjoint(words):
            kept_fileids.append(fid)
            kept_trans.append(tr)
        else:
            removed += 1

    fileids_path.write_text("\n".join(kept_fileids) + "\n")
    trans_path.write_text("\n".join(kept_trans) + "\n")

    return removed


def generate_dictionary(
        exp_dir: Path,
        db_name: str,
        experiment: dict,
        root: Path,
        words: set[str]):
    """Write the pruned dictionary, phone list and filler dictionary.

    The given words, minus `<s>`, `</s>`, `<sil>` and every filler word
    declared in any training corpus's `fillers:` mapping, are looked up in
    the master lexicon found by `_resolve_dict_path`. Then:

    - `<db>.dic` receives the pruned dictionary entries.
    - If any words are OOV, they are written to `<db>_train.oov` and
      `filter_oov_utterances` is run on both the train and the decode sets.
    - `<db>.phone` receives every dictionary phone plus `SIL` and each
      corpus filler phone, sorted.
    - `<db>.filler` maps `<s>`, `</s>` and `<sil>` to `SIL`, followed by
      every corpus filler word and its phone.

    Args:
        exp_dir: Experiment directory. Files are written to `<exp_dir>/etc/`.
        db_name: Database name (the experiment directory name).
        experiment: Experiment mapping as returned by `load_experiment`.
        root: Project root (`SPHINX_ROOT`).
        words: Union of the words in the training and decode transcripts, as
            returned by `generate_utterances`.

    Warning:
        - The dictionary is built from the training and decode words
          together, and OOV filtering is also applied to the decode set.
          Every word of the remaining decode utterances is therefore in the
          dictionary, so the test-set OOV rate is 0% by construction, and
          decode utterances containing words missing from the lexicon are
          excluded from scoring.
        - If no lexicon is found, only a warning is printed and the function
          returns without writing `.dic`, `.phone` or `.filler`. `setup`
          still completes and reports success.
        - The OOV file is named `<db>_train.oov` but lists OOV words from
          both the training and the decode transcripts.

    Note:
        Fillers are read once per `train.corpora` entry, and entries are per
        split. When several entries share a corpus (for example two splits
        of the same corpus), that corpus's filler words are written to
        `<db>.filler` once per entry.
    """
    train_corpora = experiment.get("train", {}).get("corpora", [])

    # exlude filler words from main dict
    filler_words = {"<s>", "</s>", "<sil>"}
    for entry in train_corpora:
        for word in entry["_corpus"].get("fillers", {}).keys():
            filler_words.add(word)
    words = words - filler_words

    print(f"Building dictionary ({len(words)} unique words)...")
    dict_path = _resolve_dict_path(experiment, root)

    if dict_path is None:
        print("  Warning: no dictionary found, skipping dict/phone/filler generation")
        return

    print(f"  Lexicon: {dict_path}")
    dictionary = Dictionary.build(dict_path, words)

    # save pruned dictionary
    dic_file = exp_dir / "etc" / f"{db_name}.dic"
    with open(dic_file, "w") as f:
        for entry in dictionary.entries:
            f.write(f"{entry}\n")
    print(f"  Written: {dic_file.name} ({len(dictionary.entries)} entries)")

    # save OOV words
    if dictionary.oov:
        oov_file = exp_dir / "etc" / f"{db_name}_train.oov"
        with open(oov_file, "w") as f:
            for word in sorted(dictionary.oov):
                f.write(f"{word}\n")
        print(
            f"  Warning: {len(dictionary.oov)} words not in dictionary, "
            f"written to {oov_file.name}")

        train_removed = filter_oov_utterances(exp_dir, db_name, dictionary.oov, "train")
        decode_removed = filter_oov_utterances(exp_dir, db_name, dictionary.oov, "decode")
        if train_removed or decode_removed:
            print(
                f"  Filtered {train_removed} train, "
                f"{decode_removed} decode utterances with OOV words"
            )

    # save phone list
    dictionary.phones.add("SIL")
    # corpus specific fillers:
    for entry in train_corpora:
        for phone in entry["_corpus"].get("fillers", {}).values():
            dictionary.phones.add(phone)

    phone_file = exp_dir / "etc" / f"{db_name}.phone"
    with open(phone_file, "w") as f:
        for phone in sorted(dictionary.phones):
            f.write(f"{phone}\n")
    print(f"  Written: {phone_file.name} ({len(dictionary.phones)} phones)")

    # save filler dictionary
    filler_file = exp_dir / "etc" / f"{db_name}.filler"
    with open(filler_file, "w") as f:
        f.write("<s> SIL\n")
        f.write("</s> SIL\n")
        f.write("<sil> SIL\n")
        # corpus specific fillers
        for entry in train_corpora:
            corpus_fillers = entry["_corpus"].get("fillers", {})
            for word, phone in corpus_fillers.items():
                f.write(f"{word} {phone}\n")
    print(f"  Written: {filler_file.name}")


def setup_directories(exp_dir: Path):
    """Create the SphinxTrain working directories under an experiment.

    Creates `etc`, `feat`, `logdir`, `bwaccumdir`, `model_parameters`,
    `model_architecture`, `qmanager`, `result` and `presult`, with parents.
    Existing directories are left untouched.

    Args:
        exp_dir: Experiment directory.
    """
    print("creating directory structure...")
    dir_list = [
        "etc", "feat", "logdir", "bwaccumdir",
        "model_parameters", "model_architecture",
        "qmanager", "result", "presult"
    ]
    for d in dir_list:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)


def generate_utterances(
        exp_dir: Path,
        db_name: str,
        experiment: dict
) -> set[str]:
    """Write fileids and transcriptions for the training and decode sets.

    Calls `generate_train_utterances` and then `generate_decode_utterances`.

    Args:
        exp_dir: Experiment directory. Files are written to `<exp_dir>/etc/`.
        db_name: Database name (the experiment directory name).
        experiment: Experiment mapping as returned by `load_experiment`.

    Returns:
        The union of the whitespace-separated tokens of all training and
            decode transcripts, including filler tokens.
    """
    train_words = generate_train_utterances(exp_dir, db_name, experiment)
    decode_words = generate_decode_utterances(exp_dir, db_name, experiment)
    return train_words | decode_words


def generate_train_utterances(
        exp_dir: Path,
        db_name: str,
        experiment: dict
) -> set[str]:
    """Write `<db>_train.fileids` and `<db>_train.transcription`.

    Every `train.corpora` entry (one per split after `load_experiment`) is
    read through its corpus adapter, and utterances are written in entry
    order. Each fileids line is the adapter's fileid, and each transcription
    line is `<s> TEXT </s> (fileid)`. Both files are truncated first. The
    utterance count per entry is printed.

    Args:
        exp_dir: Experiment directory. Files are written to `<exp_dir>/etc/`.
        db_name: Database name (the experiment directory name).
        experiment: Experiment mapping as returned by `load_experiment`.

    Returns:
        The set of whitespace-separated tokens in all training transcripts,
            including filler tokens.
    """
    print("Generating training fileids and transcriptions...")
    words = set()

    fileids_path = exp_dir / "etc" / f"{db_name}_train.fileids"
    trans_path = exp_dir / "etc" / f"{db_name}_train.transcription"

    with open(fileids_path, "w") as fids, open(trans_path, "w") as trans:
        for entry in experiment.get("train", {}).get("corpora", []):
            corpus_data = entry["_corpus"]
            split_cfg = corpus_data["splits"][entry["split"]]

            print(f"  {entry['name']}/{entry['split']}...")
            adapter = get_adapter(entry["name"])
            count = 0

            adapter_utt = adapter.get_utterances(
                corpus_data["_dir"],
                entry["split"],
                split_cfg,
                corpus_data
            )
            for fileid, text in adapter_utt:
                fids.write(f"{fileid}\n")
                trans.write(f"<s> {text} </s> ({fileid})\n")
                words.update(text.split())
                count += 1
            print(f"    {count} utterances")
    return words

def generate_decode_utterances(
        exp_dir: Path,
        db_name: str,
        experiment: dict
) -> set[str]:
    """Write `<db>_decode.fileids` and `<db>_decode.transcription`.

    Reads `decode.corpus` through its adapter and uses the same line formats
    as `generate_train_utterances`. If no decode corpus is configured,
    nothing is written and an empty set is returned (`validate_experiment`
    normally rejects such a configuration earlier).

    Args:
        exp_dir: Experiment directory. Files are written to `<exp_dir>/etc/`.
        db_name: Database name (the experiment directory name).
        experiment: Experiment mapping as returned by `load_experiment`.

    Returns:
        The set of whitespace-separated tokens in the decode transcripts.
    """
    print("Generating decode fileids and transcriptions...")
    words = set()
    decode_cfg = experiment.get("decode", {}).get("corpus", {})

    if not decode_cfg or not decode_cfg.get("name"):
        return words

    corpus_data = decode_cfg["_corpus"]
    split_name = decode_cfg["split"]
    split_cfg = corpus_data["splits"][split_name]

    adapter = get_adapter(decode_cfg["name"])
    count = 0
    fileids_path = exp_dir / "etc" / f"{db_name}_decode.fileids"
    trans_path = exp_dir / "etc" / f"{db_name}_decode.transcription"
    with open(fileids_path, "w") as fids, open(trans_path, "w") as trans:
        adapter_utt = adapter.get_utterances(
            corpus_data["_dir"],
            split_name,
            split_cfg,
            corpus_data
        )
        for fileid, text in adapter_utt:
            fids.write(f"{fileid}\n")
            trans.write(f"<s> {text} </s> ({fileid})\n")
            words.update(text.split())
            count += 1
    print(f"  {decode_cfg['name']}/{split_name}: {count} utterances")

    return words


def generate_feat_params(exp_dir: Path, experiment: dict, root: Path):
    """Render `<exp_dir>/etc/feat.params` from the vendored template.

    The `__CFG_*__` placeholders in `vendor/sphinxtrain/etc/feat.params` are
    replaced with `lo_filt`, `hi_filt`, `num_filt`, `cmn` and `sample_rate`
    from the first training corpus's `corpus.yml` (defaults 130, 6800, 25,
    `live` and 16000), and with fixed values for the rest: `-transform dct`,
    `-lifter 22`, `-feat 1s_c_d_dd`, `-agc none` and `-varnorm no`. The
    `-svspec` line is filled from `sphinxtrain.CFG_SVSPEC` if that is set,
    and removed otherwise.

    Args:
        exp_dir: Experiment directory.
        experiment: Experiment mapping as returned by `load_experiment`.
        root: Project root (`SPHINX_ROOT`).

    Warning:
        - `feat.params` is built from `corpus.yml` only. `sphinxtrain:`
          overrides in `experiment.yml` for the same front-end settings (for
          example `CFG_NUM_FILT`, `CFG_LO_FILT`, `CFG_HI_FILT`, `CFG_CMN`,
          `CFG_WAVFILE_SRATE` or `CFG_FEATURE`) reach `sphinx_train.cfg` but
          not `feat.params`. `CFG_SVSPEC` is the only exception.
        - If the template is missing, a warning is printed and no
          `feat.params` is written.
    """
    src = root / "vendor" / "sphinxtrain" / "etc" / "feat.params"
    dst = exp_dir / "etc" / "feat.params"

    if not src.is_file():
        print(f"Warning: feat.params not found at {src}")
        return

    content = src.read_text()

    train_corpora = experiment.get("train", {}).get("corpora", [])
    if train_corpora:
        primary = train_corpora[0].get("_corpus", {})
    else:
        primary = {}

    replacements = {
        "__CFG_LO_FILT__": str(primary.get("lo_filt", 130)),
        "__CFG_HI_FILT__": str(primary.get("hi_filt", 6800)),
        "__CFG_NUM_FILT__": str(primary.get("num_filt", 25)),
        "__CFG_TRANSFORM__": "dct",
        "__CFG_LIFTER__": "22",
        "__CFG_FEATURE__": "1s_c_d_dd",
        "__CFG_AGC__": "none",
        "__CFG_CMN__": primary.get("cmn", "live"),
        "__CFG_VARNORM__": "no",
        "__CFG_WAVFILE_SRATE__": str(int(primary.get("sample_rate", 16000)))
    }

    # if users set CFG_SVSPEC, then set svspec to that in feat.params,
    # otherwise just remove it
    svspec = experiment.get("sphinxtrain", {}).get("CFG_SVSPEC", "")
    if svspec:
        content = content.replace("__CFG_SVSPEC__", svspec)
    else:
        content = re.sub(r"^-svspec\s.*\n?", "", content, flags=re.MULTILINE)

    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    dst.write_text(content)
    print("Generated feat.params")


def generate_config(exp_dir: Path, experiment: dict, root: Path):
    """Write `<exp_dir>/etc/sphinx_train.cfg`.

    Renders the file with `lib.config.generate_sphinx_train_cfg` and
    overwrites any existing copy.

    Args:
        exp_dir: Experiment directory.
        experiment: Experiment mapping as returned by `load_experiment`.
        root: Project root (`SPHINX_ROOT`).

    Raises:
        FileNotFoundError: If the vendored `sphinx_train.cfg` template is
            missing.
        SystemExit: If no decode language model is configured.
    """  # noqa: DOC502
    print("Generating sphinx_train.cfg...")
    cfg = generate_sphinx_train_cfg(exp_dir, experiment, root)
    cfg_file = exp_dir / "etc" / "sphinx_train.cfg"
    cfg_file.write_text(cfg)
    print(f"  Written: {cfg_file}")


def main():
    """Run `sphinx.sh setup`: generate all SphinxTrain inputs for an experiment.

    Parses `exp_dir` from the command line, resolves it against
    `SPHINX_ROOT` if it is relative, loads the experiment, and runs
    `setup_directories`, `generate_utterances`, `generate_dictionary`,
    `generate_feat_params` and `generate_config` in that order. Progress is
    printed to stdout. Errors from loading YAML or reading corpora propagate
    as uncaught exceptions.

    Raises:
        SystemExit: On invalid command-line arguments, if `SPHINX_ROOT` is
            unset, if `experiment.yml` fails validation, or if no decode
            language model is configured.
    """  # noqa: DOC502
    parser = argparse.ArgumentParser(
        description="Set up a SphinxTrain experiment from experiment.yml."
    )
    parser.add_argument(
        "exp_dir",
        type=Path,
        help="Path to the experiment directory (e.g. experiments/001)",
    )
    args = parser.parse_args()

    root = get_sphinx_root()
    exp_dir = args.exp_dir
    if not exp_dir.is_absolute():
        exp_dir = root / exp_dir

    db_name = exp_dir.name

    print(f"Loading experiment from {exp_dir}/experiment.yml")
    experiment = load_experiment(exp_dir, root)

    setup_directories(exp_dir)
    words = generate_utterances(exp_dir, db_name, experiment)
    generate_dictionary(exp_dir, db_name, experiment, root, words)
    generate_feat_params(exp_dir, experiment, root)
    generate_config(exp_dir, experiment, root)
    print(f"\nSetup complete for experiment {db_name}.")


if __name__ == "__main__":
    main()
