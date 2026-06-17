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
.venv/bin/python scripts/visualize.py --max-steps 200
```

## Build A Submission

```bash
.venv/bin/python scripts/build_submission.py
```

The build script copies `src/ptcg_ai/` to `submission/ptcg_ai/` and creates
`submission.tar.gz`.

The current default submission deck is a compact Crustle deck. It keeps a
non-ex attacker as the main win condition so the agent is not hard-walled by
Crustle-style damage prevention mirrors, while preserving the older Abomasnow
deck under `ptcg_ai.decks.ABOMASNOW_DECK` for experiments.

## Local Visualizer

```bash
.venv/bin/python scripts/visualize.py --player0 rule --player1 random --output outputs/visualizer/game.html
```

The visualizer writes an HTML replay. Open the generated file in a browser to
inspect board state, selections, and logs step by step. Use the play/pause
button for automatic playback. Without optional card images, the HTML is
self-contained.

Card images are optional and local-only. If `data/card_images/721.jpg` or
`data/card_images/721.png` exists, the replay will use it for card ID `721`.

To fetch images for the current deck:

```bash
.venv/bin/python scripts/download_card_images.py --deck-only
.venv/bin/python scripts/visualize.py
```

Downloaded images are stored under `data/card_images/` and are ignored by Git.
By default the downloader uses direct `images.pokemontcg.io` URLs for known set
codes. Use `--source api` or `--source auto` to try API-based matching for cards
that do not have a direct set-code mapping.

## Self-Play Training

```bash
.venv/bin/python scripts/collect_self_play.py --games 1000
.venv/bin/python scripts/train_action_model.py
.venv/bin/python scripts/evaluate.py --player0 learned --player1 random
```

Self-play logs are written to `outputs/training/self_play.jsonl`, and the
trained action scorer is written to `outputs/models/action_model.npz`. Both are
local artifacts and ignored by Git. The first trainer is an imitation-style
pairwise linear scorer over legal actions; it is intentionally small so it can
serve as the bridge from rule-based play to later reinforcement learning.

## Development Direction

The current policy is a simple score-based rule policy. The intended path is:

1. Improve rule scores and deck-specific heuristics.
2. Run local self-play to catch regressions.
3. Save game logs and derive legal-action features.
4. Tune rule weights or train a small action-scoring model.
