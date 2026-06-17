# Repository Guide

## Project Shape

- Keep reusable logic under `src/ptcg_ai/`.
- Keep Kaggle-facing files under `submission/`.
- Treat `submission/ptcg_ai/` and `submission.tar.gz` as generated output from `scripts/build_submission.py`.
- Keep local utilities in `scripts/`.
- Keep fast checks in `tests/`.

## Development Flow

1. Change policy, feature, deck, or evaluation code in `src/ptcg_ai/`.
2. Run unit checks with `python -m unittest discover -s tests`.
3. Run a small self-play smoke test with `python scripts/evaluate.py --games 2`.
4. Build a Kaggle package with `python scripts/build_submission.py`.

## Coding Notes

- The competition runtime calls `submission/main.py::agent(obs_dict)`.
- `submission/main.py` should stay thin; route logic through `ptcg_ai.submission`.
- Add new decision logic as a policy class under `src/ptcg_ai/agent/`.
- Prefer score-based policies over one-off conditionals so later tuning and learning can reuse the same interface.
- Keep generated archives, virtual environments, credentials, logs, and model checkpoints out of Git.
