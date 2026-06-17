# Benchmark opponents

Sample agents from the competition's Code page, used as virtual opponents to
estimate ladder performance (the mirror is ~50/50 and not informative).

The notebooks themselves are not committed (third-party code). Re-pull them:

```bash
kaggle kernels pull kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck   -p benchmarks/a-sample-rule-based-agent-mega-lucario-ex-deck
kaggle kernels pull kiyotah/a-sample-rule-based-agent-dragapult-ex-deck      -p benchmarks/a-sample-rule-based-agent-dragapult-ex-deck
kaggle kernels pull kiyotah/a-sample-rule-based-agent-iono-s-deck            -p benchmarks/a-sample-rule-based-agent-iono-s-deck
kaggle kernels pull kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck -p benchmarks/a-sample-rule-based-agent-mega-abomasnow-ex-deck
```

Then run the benchmark / capture a replay:

```bash
python scripts/benchmark.py --policy rule --games 40
python scripts/benchmark.py --policy rule --visualize-vs a-sample-rule-based-agent-iono-s-deck --seat 0
```

`scripts/benchmark.py` reconstructs each opponent's deck from the `Name = id  # xN`
constants documented in its agent cell, loads the agent function, and plays it on
both seats. Opponents whose deck is not documented that way are skipped.
