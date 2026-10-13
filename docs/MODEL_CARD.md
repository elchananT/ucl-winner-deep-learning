# Model card: Champions League match-outcome model

## Summary

* **Task.** Given two clubs and the stage, output P(home win), P(draw), P(away win) for the **90-minute result** of a UEFA Champions League match, using only information from earlier matches.
* **Model shipped.** Average of five small Keras MLPs (17 standardised features + shared team embeddings), followed by one fitted temperature. Artifact: `models/ucl_mlp/` (~0.5 MB). Inference: `ucl-predict "Real Madrid" "Manchester City" --stage knockout` or `ucl.predict.Predictor`.
* **Honest headline.** The network is **not better than** a tuned logistic regression or gradient boosting on the same features. It is slightly better than a logistic regression on Elo alone. The gain comes from engineered form features, not from the learned embeddings.

## Intended use and non-use

Intended for education, methodology demonstration and as a transparent baseline for football forecasting. **Not** intended for betting, trading or any decision with financial consequences: it has not been compared with bookmaker odds, which are the real benchmark.

## Data

Open CC0 data, see [DATA_CARD.md](DATA_CARD.md). Chronological split by season: train 1992/93-2016/17 (2,880 matches), validation 2017/18-2020/21 (494), test 2021/22-2025/26 (753).

## Features (all computed from strictly earlier match days)

Elo rating of both clubs (goal-difference-weighted updates, country-based prior for debutants), exponentially weighted form (points, goal difference), smoothed goals scored/conceded, log number of past matches, and context flags (neutral venue, knockout, second leg, final). `tests/test_leakage.py` replaces the future with random results and verifies that all earlier features are bit-identical. Clubs unseen in training receive an `<UNK>` embedding.

## Training

Adam, L2, dropout, early stopping on validation log-loss with best-weights restore; hyper-parameters from a 12-trial random search on validation data (selected: one hidden layer of 16, 16-d embeddings, dropout 0.5, L2 0.01, learning rate 3e-3, batch 128). Five seeds (2026-2030). CPU only, TensorFlow 2.21 / Keras 3.15, deterministic ops; five networks train in about 11 s. The full pipeline reproduces **bit-identical metrics** from a fresh environment (checked).

## Results (test seasons 2021/22-2025/26, 753 matches; all models temperature-scaled on validation)

| Model | Log-loss (95% CI) | Brier | RPS | Accuracy | ECE |
|---|---|---|---|---|---|
| Uniform | 1.0986 | 0.6667 | 0.2470 | - | - |
| Base rates | 1.0439 (1.021-1.068) | 0.6288 | 0.2374 | 48.9% | 0.007 |
| Elo + logistic regression | 0.9441 (0.905-0.985) | 0.5543 | 0.2006 | 56.8% | 0.033 |
| All features + logistic regression | 0.9343 (0.899-0.972) | 0.5494 | 0.1977 | 57.6% | 0.029 |
| Gradient boosting | 0.9332 (0.897-0.972) | 0.5481 | 0.1979 | 58.7% | 0.018 |
| **MLP + team embeddings (5-seed ensemble)** | **0.9323** (0.898-0.969) | 0.5487 | 0.1980 | 57.1% | 0.026 |

Paired bootstrap of log-loss, MLP minus baseline: vs Elo logistic regression -0.0118 (95% CI -0.0232 to -0.0006); vs all-features logistic regression -0.0020 (-0.0070 to +0.0029); vs gradient boosting -0.0009 (-0.0112 to +0.0098). Single networks: 0.9324 +/- 0.0027 over seeds.

**Ablation** (5 seeds each, test log-loss): full 0.9324; without embeddings 0.9336 (within noise); without form features 0.9426 (+0.010); embeddings and context only, no Elo 0.9637.

**Tournament simulation** (3 held-out classic-format seasons, 20,000 simulated tournaments each): stage-reach Brier 0.1273 (uniform) vs 0.1007-0.1014 for the four informed models (no separation). Probability given to the actual champion by the network: 8.0%, 8.6%, 11.6%; mean log score 2.37 vs 3.44 for uniform and 2.20 for Elo logistic regression. Three trophies are anecdotal.

Temperature scaling is a mixed bag: fitted temperatures are 1.05-1.18, test log-loss moves by less than 0.005, and ECE gets worse for 3 of 4 models.

## Limitations and risks

* Small data (about 2,900 training matches): deep models have little to exploit, and per-club embeddings are estimated from a few dozen matches.
* Information set limited to Champions League results; no domestic form, squad data, injuries or odds.
* Test intervals are wide; differences below roughly 0.01 log-loss are not detectable. Validation and test sometimes rank models differently.
* Simulation simplifications: frozen ratings, goal margins sampled from the empirical distribution, coin-flip extra time/penalties, no head-to-head tie-breaks, classic 32-team format only.
* Regime changes (closed-doors season, Lisbon bubble, league phase from 2024/25) are only partly encoded.
* Development disclosure: an untuned prototype was run once and its test score seen. Every reported hyper-parameter, architecture and calibration choice was then made on validation data only.
* Ratings in the shipped artifact are as of 2026-05-30 and are not updated unless the pipeline is re-run.
