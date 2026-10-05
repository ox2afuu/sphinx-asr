# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`sphinx-asr` is a thin wrapper around vendored CMU speech tools:

- `vendor/sphinxtrain` — acoustic model training (Perl + C).
- `vendor/pocketsphinx` — decoding binaries.
- `vendor/cmu_toolkit` — CMU-Cambridge SLM toolkit (for building ARPA language models).

The Python code under `scripts/` generates the config/fileids/transcriptions those tools expect, then orchestrates the Perl pipeline steps. A pytest suite (`tests/`) covers the pure config logic and known defects, and docstrings are linted (see "Tests, docstrings and hooks" below); end-to-end correctness is still validated by actually running a train/decode on a corpus.

## Common commands

```
make                       # build vendor/ → bin/<arch>/ (one-time; required before anything else)
make clean                 # drop build trees + bin/<arch>/
sudo make link             # symlink ./sphinx.sh to /usr/local/bin/sphinx

./sphinx.sh new [-t CORPUS] [-l]   # scaffold experiments/NNN/ from template
./sphinx.sh setup <exp_dir>        # generate etc/* (fileids, transcription, dict, sphinx_train.cfg, feat.params)
./sphinx.sh feats <corpus> <split> # pre-extract .mfc features; 'all' for every split; -j <n> for parallelism
./sphinx.sh lm <corpus> <split>    # build ARPA LM from a split (or --all for corpus full_transcripts)
./sphinx.sh train <exp_dir>        # run the 13-step sphinxtrain pipeline
./sphinx.sh decode <exp_dir>       # run pocketsphinx batch decode + scoring
```

Unit tests: see "Tests, docstrings and hooks" below. The end-to-end smoke test is `sphinx new -t librispeech && sphinx setup experiments/NNN && sphinx train experiments/NNN && sphinx decode experiments/NNN`.

First invocation of `sphinx.sh` auto-creates `.venv/` and `pip install`s `requirements.txt` (`pyyaml`, `numpy`, `scipy`). All Python scripts require `SPHINX_ROOT` to be set; `sphinx.sh` sets it for you, so prefer invoking scripts through it rather than calling `python scripts/*.py` directly.

## Tests, docstrings and hooks

Full standard: `docs/DOCSTRINGS.md`. Dev tooling runs via `uv`/`uvx --no-project`, never inside `.venv/` (runtime only; `requirements.txt` stays the airgap source of truth, `pyproject.toml` just mirrors it and configures tools).

```
uv run --no-project --with pytest --with-requirements requirements.txt python -m pytest -q   # all tests
uv run --no-project --with pytest --with-requirements requirements.txt python -m pytest -q tests/scripts/test_config.py::test_bool_becomes_perl_yes_no   # one test
uvx ruff@0.16.10 check --select D,DOC,F --preview scripts tools                              # Google docstrings
uv run --no-project python tools/check_test_docstrings.py tests                              # GIVEN/WHEN/THEN checker
```

- **Application code** (`scripts/`, `tools/`): Google-style docstrings, enforced by ruff `D` (convention google) + preview `DOC` (pydoclint), so Args/Returns/Raises must match the code. List only explicitly raised exceptions under `Raises:`.
- **Tests** (`tests/`): docstrings use `Given:` (inputs + a stated baseline/expected), `When:` (one action), `Then:` (every assertion), optional `Raises:`; checked by `tools/check_test_docstrings.py` (rules TDS001-TDS008).
- `tests/conftest.py` puts `scripts/`, `scripts/lib/` and `tools/` on `sys.path`; import config as `from lib.config import ...` (it uses a relative import).
- Known review defects are `xfail(strict=True)` tests in `tests/scripts/test_known_defects.py` asserting the correct behaviour. Fixing one makes it XPASS and fail: delete its marker. Note #6: `_apply_overrides()` currently drops `sphinxtrain:` keys that are absent from the template.
- **Hooks** (`lefthook.yml`, run `lefthook install`): pre-commit checks only staged files (ruff on `scripts/`+`tools/`, TDS checker on `tests/`, no-emoji, fast pytest), a ratchet so touched files must comply; pre-push checks the whole tree and runs the full suite.

## Architecture

### Entry point and environment

`sphinx.sh` is the only user-facing entry point. It:

1. Creates/uses `.venv/` for Python deps.
2. Exports `SPHINX_ROOT`, `SPHINXTRAIN_DIR`, `SPHINXTRAIN_BIN_DIR`, and prepends `bin/$(uname -m)` + `.venv/bin` to `PATH`.
3. Dispatches by subcommand. `train` and `decode` go through `run_script()`, which submits to Torque via `qsub` if `qsub` is on `PATH`, otherwise runs locally. The Torque path calls `scripts/sphinx_job.sh`, which **is not yet present in the repo** — the Torque branch is explicitly untested.

Every Python script reads `SPHINX_ROOT` via `lib.asr_util.get_sphinx_root()`; do not assume `os.getcwd()`.

### Config pipeline: experiment.yml → sphinx_train.cfg

This is the central flow. `scripts/lib/config.py::load_experiment()`:

1. Loads `<exp_dir>/experiment.yml`, validates required sections (`train.corpora`, `decode.corpus`).
2. For each entry in `train.corpora`, loads `corpus/<name>/corpus.yml`, attaches it as `_corpus` with `_dir` set to the corpus path. Expands the `splits: [...]` shorthand into one entry per split. Validates that each split exists in the corpus.
3. Same resolution for `decode.corpus`.

`generate_sphinx_train_cfg()` then reads `vendor/sphinxtrain/etc/sphinx_train.cfg` as a template and substitutes:

- Path placeholders: `___DB_NAME___`, `___BASE_DIR___`, `___SPHINXTRAIN_DIR___`, `___SPHINXTRAIN_BIN_DIR___`.
- Perl variable assignments via `_apply_overrides()` — a regex that rewrites `$VAR = ...;` lines (both active and commented-out). Ordering matters: `experiment.yml` `sphinxtrain:` entries override defaults and corpus-derived values.
- Types: `_to_perl_value()` decides quoting — bools become `'yes'/'no'`, numbers stay bare, strings with `$` get double-quoted, numeric strings (e.g. `1e-80`) stay bare, everything else gets single quotes. Pass numeric config as int/float in YAML, not as strings, to avoid unwanted quoting.

`feat.params` has a separate template mechanism — `__CFG_*__` placeholders get replaced with corpus-derived values in `setup.py::generate_feat_params()`.

### Corpus adapters

Each corpus format has an adapter module at `scripts/corpus/<name>.py` (loaded dynamically by `scripts/corpus/__init__.py::get_adapter()`). The required interface is a single generator:

```python
def get_utterances(corpus_dir, split_name, split_cfg, corpus_cfg) -> Iterator[(fileid, text)]
```

`fileid` must be a path **relative to `SPHINX_ROOT`** (not to the corpus dir) with no extension — sphinxtrain appends `.mfc` and looks under `$CFG_FEATFILES_DIR` (which is set to `SPHINX_ROOT`). Look at `librispeech.py` as the simplest example and `switchboard.py` for transcript-cleaning conventions (filler tokens, bracket tags, etc).

Adding a new corpus = drop `corpus/<name>/corpus.yml` + implement `scripts/corpus/<name>.py`. `corpus.yml` fields consumed by the pipeline: `audio_format`, `audio_type`, `sample_rate`, `num_filt`, `lo_filt`, `hi_filt`, `cmn`, `dict`, `lm`, `splits`, `audio_dir`, `full_transcripts`, `fillers`.

### OOV handling

Setup does **not** run G2P. Instead, `setup.py::filter_oov_utterances()` drops any utterance whose transcript contains a word missing from the lexicon, and writes the OOV word list to `etc/<db>_train.oov`. If results look sparse after `setup`, check the OOV file — this is the usual cause.

### Training step runner

`scripts/train.py` declares `STEPS: list[TrainingStep]` (the 00.verify → 90.deleted_interpolation pipeline). `run_step()`:

- `perl scripts/<step>/...` via `subprocess.Popen`, streaming stdout/stderr live.
- Non-zero exit is **not** automatically fatal if `step.artifacts` all exist and are non-empty — this is specifically to tolerate MLLT's spurious exit-1-on-convergence behavior.
- On clean exit, `validate_step()` scans the step's log dir for `fatal_patterns` (`"0 frames"`, `"FATAL"`) and exits, or for `warning_patterns` (`"failed"`, `"Aborting"`) and warns.
- Skipped steps (those whose output mentions `"Skipped"`) bypass validation.

`--from-step` does substring matching against the step name (e.g. `--from-step 30` resumes at `30.cd_hmm_untied`).

MLLT seed: `pipeline.mllt_init` in `experiment.yml` can be `eye` (deterministic, default), `random`, or an integer. Non-`eye` values are exported as `SPHINXASR_MLLT_SEED` for the Perl scripts to consume.

### Decoding

`scripts/decode.py` invokes `vendor/sphinxtrain/scripts/decode/slave.pl`. Progress is polled on a side thread by counting lines in `result/<db>-<i>-<npart>.match` files (`DEC_CFG_NPART` partitioning, read back from `sphinx_train.cfg`). Old `*.match` / `*.matchseg` are deleted up-front so the counter starts at zero.

WER/SER are extracted by grepping the slave.pl output for `"WORD ERROR"` / `"SENTENCE ERROR"` lines.

### Language model building

`scripts/lm.py` chains the CMU SLM toolkit binaries in a temp dir: `text2wfreq | wfreq2vocab | text2idngram | idngram2lm`. Input text comes from `get_adapter(...).get_utterances()` — either a specific split or the corpus's `full_transcripts` file (via `--all`). Each sentence is wrapped in `<s> ... </s>`.

## Cluster deployment context

When `sphinx-asr` runs on the ivaliceCluster (1 head `dalmasca` + 5 workers
on a DeskPi Super6c, Debian 13 Trixie, aarch64), several invariants change
vs. the laptop-single-node model this file otherwise assumes:

- **`SPHINX_ROOT=/srv/ivalice/sphinx-asr`** is an NFSv4 mount exported from
  the head. All 6 nodes see the same tree, so fileids (which are paths
  relative to `SPHINX_ROOT`) resolve identically everywhere. **Without
  NFS, the cluster is unusable** — every worker would have a different view
  of `etc/`, `model_parameters/`, `logdir/`, `bwaccumdir/`, and `result/`.

- **`.venv/` is pre-populated by Ansible** from bundled aarch64 wheels
  (`assets/sphinx-wheels/` at the ivaliceCluster super-repo). The
  auto-create logic in `sphinx.sh:13-18` must **not** run at runtime because
  the cluster is airgapped. If `.venv/` is missing on a fresh deploy, fix
  the Ansible role, don't fall back to `pip install`.

- **`bin/aarch64/` is built once on the head** by Ansible (a single `make`
  invocation, ~30 min) and served via NFS. Workers never rebuild.

- **Slurm is the de facto job scheduler.** Set
  `CFG_QUEUE_TYPE: "Queue::Slurm"` + `CFG_QUEUE_NAME: "default"` in
  `experiment.yml` to dispatch training partitions via `sbatch`. Use
  `CFG_NPART: 10` (5 workers × 2 partitions/node) by default —
  `CFG_NPART: 20` saturates all 20 cores but risks OOM on
  `50.cd_hmm_tied`. Decoding bypasses the Perl Queue layer entirely:
  `decode.py` submits `sbatch --array=1-DEC_CFG_NPART` directly.

- **`scripts/perl/Queue/Slurm.pm`** is the Slurm queue adapter, loaded via
  `PERL5LIB="$SPHINX_ROOT/scripts/perl:$SPHINXTRAIN_DIR/scripts/lib:..."`.
  It lives outside `vendor/` so vendor refreshes don't lose the change.
  API matches `Queue::PBS` (submit_job/query_job/waitfor_job/cancel_job);
  translation is sbatch/scontrol/sacct/scancel.

- **All Slurm stdout/stderr** land under `<exp_dir>/slurm-logs/`.
  Sphinxtrain's per-step logs stay at `<exp_dir>/logdir/NN.step/` as
  before. Use `sphinx status <exp_dir>` for squeue/sacct summary and
  `sphinx cancel <exp_dir>` to cancel all jobs for one experiment.

- **sphinx-asr is tracked as a submodule by the ivaliceCluster super-repo.**
  All code changes here happen on feature branches in this repo and land
  via PR; ivaliceCluster then bumps its submodule pointer in a separate
  commit. Do not commit into this repo from an ivaliceCluster session.

Cross-references:
- `ivaliceCluster/docs/uml/02-sphinx-asr-asis/` — today's class model,
  sequences, and CLI dispatch.
- `ivaliceCluster/docs/uml/03-cmu-sphinx-domain/` — acoustic model
  lifecycle, Baum-Welch, decoder, corpus class model, E2E dataflow.
- `ivaliceCluster/docs/uml/04-slurm-integration/` — proposed integration
  design reviewed before Phase 2/3 code lands.

## Conventions to follow

- **Never hardcode paths relative to cwd.** Use `get_sphinx_root()` and the corpus `_dir`. `train.py` and `decode.py` do `os.chdir(exp_dir)` before running Perl.
- **`fileid` paths in etc/\*.fileids are relative to `SPHINX_ROOT`**, not to the corpus or the experiment. Get this wrong and sphinxtrain silently reports "0 frames".
- **Do not add Perl config overrides as Python strings when they're numeric** — let `_to_perl_value()` pick the quoting. For example, set `CFG_N_TIED_STATES: 4000` (int), not `"4000"`.
- **New config flags**: if you need a new `sphinxtrain:` key in `experiment.yml`, just add it — `_apply_overrides()` already handles arbitrary `$VAR = ...;` substitution in the template. No code change required unless you want a non-string-literal default.
- **New training step**: append a `TrainingStep` to `STEPS` in `train.py`. Point `log_dir` at the subdir of `logdir/` it produces; include any `artifacts` that must exist to consider the step successful.
