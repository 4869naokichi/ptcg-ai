# ptcg-ai

Pokemon Trading Card Game agent workspace for Kaggle-style submissions.

## Layout

```text
src/ptcg_ai/        Reusable agent, deck, evaluation, and training code
submission/         Kaggle entry point, deck file, and simulator bindings
scripts/            Local commands for evaluation and packaging
tests/              Fast regression checks
data/               Competition data and sample submission files
```

`submission/main.py` is intentionally small. Most behavior lives in
`src/ptcg_ai/`, then `scripts/build_submission.py` copies the package into
`submission/` and creates an archive.

## Quick Checks

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/evaluate.py --games 2
```

## Build A Submission

```bash
.venv/bin/python scripts/build_submission.py
```

The build script copies `src/ptcg_ai/` to `submission/ptcg_ai/` and creates
`submission.tar.gz`.

## Development Direction

The current policy is a simple score-based rule policy. The intended path is:

1. Improve rule scores and deck-specific heuristics.
2. Run local self-play to catch regressions.
3. Save game logs and derive legal-action features.
4. Tune rule weights or train a small action-scoring model.
