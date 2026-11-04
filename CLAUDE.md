# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Codabench competition whose evaluation **is** a benchopt benchmark. Participants
predict RNA-seq read counts for (gene × run) observations of bacteria and must
generalize to unseen runs, genes and species. The same `benchmark/` code runs
locally and on the Codabench worker, so a change to the benchmark changes what
the platform scores.

## Commands

Run from the repo root. Dependencies are in `requirements.txt` (benchopt comes
from the GitHub main tarball, not PyPI).

```bash
pip install -r requirements.txt

# Data (goes to benchmark/data/ or $BENCHOPT_DATA_HOME); add --force to rebuild
benchopt prepare benchmark -d "BacterialExpression[mode=debug]"   # one E. coli genome
benchopt prepare benchmark -d "BacterialExpression[mode=train]"   # all public data, ~1.1 GB

# Run a solver (BENCHOPT_DEBUG makes solver errors raise, as on the platform)
BENCHOPT_DEBUG=true benchopt run benchmark -d "BacterialExpression[mode=debug]" \
    -s solution/submission.py --no-cache --no-plot
benchopt run benchmark -d Simulated -s Constant -s BaselineNN \
    -r 1 --no-cache --no-plot --no-html --no-display   # no-data smoke test

# Tests
benchopt test benchmark --skip-install                       # benchopt design tests
pytest benchmark/benchmark_utils/test_nb_loss.py             # metric unit tests
pytest benchmark/benchmark_utils/test_nb_loss.py -k matches_scipy   # one test

# Lint (what CI runs)
flake8 . --max-line-length=79 --exclude=.git,worktrees,__cache__

# Bundle and full ingestion -> scoring run on offline demo data (mirrors CI)
export BENCHOPT_DATA_HOME=$PWD/benchmark/data
python tools/setup_data.py --demo
python tools/create_bundle.py        # -> bundle.zip
python -c "import zipfile; zipfile.ZipFile('bundle.zip').extractall('_bundle')"
benchopt prepare benchmark --config phases/main/input_data/config.yaml
python _bundle/ingestion_program/ingestion.py --submission-dir _bundle/solution \
    --input-data _bundle/main_phase/input_data --output-dir ingestion_res
python _bundle/scoring_program/scoring.py --prediction-dir ingestion_res \
    --output-dir scoring_res
```

CI (`.github/workflows/test.yml`) runs on Python 3.12 with CPU torch, then the
lint, design tests, Simulated smoke run and the end-to-end bundle run above, and
also builds `tools/Dockerfile`.

## Architecture

**Evaluation flow.** `ingestion_program/ingestion.py` strips the competition-only
`scoring` key from `phases/main/input_data/config.yaml` (benchopt rejects
unknown keys) and shells out to `benchopt run <benchmark> -s submission.py`.
`scoring_program/scoring.py` only maps columns of the resulting
`results.parquet` to leaderboard keys using `scoring.columns` from that config,
and fails if a listed column is missing or NaN. All metric computation happens
in `benchmark/objective.py`.

**Objective owns the labels.** `Dataset.get_data` hands the objective the train
frame plus `test_sets` (region name → features, counts). `get_objective` passes
only train data, `samples_df`, `genomes_dir` and meta to the solver. The solver
trains during `run`, and `evaluate_result` calls `model.predict` on each region
itself. Test counts never reach solver code. `sampling_strategy = "run_once"`:
each solver runs once, with no convergence curve.

**Submission contract** (`benchmark/benchmark_utils/base_solver.py`): a
submission is `class Solver(GeneCompetSolver)` implementing `get_model(meta)`
and `fit(model, train_df, samples_df, genomes_dir)`. The returned model must
have `predict(test_df, samples_df, genomes_dir)` returning an `(N, 2)` array
`[mu, sigma]` with `mu >= 0` and finite `sigma > 0` (checked by
`check_predictions`). `sigma` is the NB dispersion `r` (variance
`mu + mu²/sigma`), not a standard deviation. benchopt loads `submission.py` by
path, so **a submission must be a single file**: it can import
`benchmark_utils.*` but not sibling `.py` files. Non-code files are read from
`meta["submission_dir"]`. The built-in baselines in `benchmark/solvers/` follow
the same contract and are selected by their `name` (`Constant`, `BaselineNN`,
`BaselineCatBoost`, `BaselineAdditive`).

**Regions and metrics** (`benchmark/datasets/bacterial_expression.py`). Held-out
observations fall into region A (gene and run both seen, never together), B
(new run), C (new gene) or D (new gene in a new run). Splits group genes by
homology and operon proximity to prevent leakage. `prepare` drops held-out rows
that the training data contradicts. Metrics are emitted as
`<metric>_<region>` plus `<metric>_challenge`, the **mean over A, B and C
only** (D is reported but not ranked). The ranking metric is NB-NLL with each
observation capped at `NLL_CAP` (`benchmark_utils/nb_loss.py`).

**Dataset modes.** `debug` and `train` fit on `Train` rows and score on the
`eval_*` regions. `full` (the platform phase) fits on train+eval and scores on
`test_*` regions, each split into a public half and a hidden `_private` half via
`is_private`. That produces the `*_private` columns that `config.yaml` maps to
hidden leaderboard keys. The `full` release is in a restricted Zenodo record
(`BACTERIA_DATA_TOKEN`).

**Data layout.** `prepare` downloads raw files to `get_data_path("bacteria_raw")`,
then writes `bacteria/<mode>/{train,A,B,C,D}.parquet` (observations with the
`log_length_ratio` and `log_size_factor` offsets joined in) plus the two
annotation CSVs. Genomes go to a shared `bacteria/genomes/<genome_id>.gbff`.
Join keys: `(genome_id, feature_id)` for genes and `(genome_id, run_accession)`
for runs. The gene table's `Chr` matches either a GenBank record's VERSION or
its LOCUS name, and `Chr`/`Start`/`End`/`Strand` can hold `;`-joined values.

**Simulated dataset has a different schema.** `datasets/simulated.py` (numpy and
pandas only) joins samples on `Run`, keeps gene columns in the observations,
and passes `genomes_dir=None`. Shared code such as `benchmark_utils/features.py`
(`FeatureBuilder`) must handle both schemas, because `benchopt test` and CI only
run on Simulated. `benchmark/test_config.py` skips the real dataset in
`benchopt test`.

**Bundle and deployment.** `tools/create_bundle.py` zips the config, pages,
programs and `solution/`, and puts the whole `benchmark/` inside the phase
`input_data`, so benchmark changes need no Docker image rebuild. Data is never
bundled: the worker mounts it read-only at `$BENCHOPT_DATA_HOME=/app/data`.
`requirements.txt` is the participants' dependency contract, because nothing
is installed at submission time. Adding a runtime dependency means updating
`requirements.txt` and rebuilding the image (`docker_image` in
`competition.yaml`). The platform has a 1 h execution limit per submission.

## Conventions

- flake8 with a 79-character line limit.
- Commit messages start with a prefix: `ENH`, `DOC`, and so on.
