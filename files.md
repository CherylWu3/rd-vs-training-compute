# File Summary

## Scripts (`scripts/`)

- **analysis.py** — Main analysis script. Compares R&D compute spending (from IPO filings) vs. final training run compute for MiniMax and Zhipu. Converts FLOP estimates to dollar costs via Monte Carlo simulation over MFU, GPU-hour price, and per-model FLOP uncertainty. Produces Marimekko charts and training-vs-R&D scatter plots.
- **image_regression.py** — Regression of image generation model training compute vs. parameters × resolution. Uses bootstrap + MC sampling to predict training FLOP for Image-01 (unknown param count). Outputs regression plot and prediction histograms.
- **video_regression.py** — Regression of video generation model training compute vs. parameters × resolution. Uses bootstrap + MC sampling to predict training FLOP for Hailuo-01, Hailuo-02, and CogVideoX. Outputs regression plot and prediction histograms.
- **speech_regression.py** — Regression of speech/audio model training compute vs. parameters × sample rate. Uses bootstrap + MC sampling to predict training FLOP for Speech-01, Speech-01 Turbo, Speech-02, and Speech-02 Turbo. Outputs regression plot and prediction histograms.

## Data (`data/`)

- **all_ai_models.csv** — Epoch's Notable AI Models dataset. Source data for the regression scripts; contains model metadata, training compute, parameters, hardware info, etc.
- **image_models.csv** — Curated subset of image generation models with parameters, training compute, and resolution. Used by `image_regression.py`.
- **video_models.csv** — Curated subset of video generation models with parameters, training compute, and resolution. Used by `video_regression.py`.
- **hk_ipo_financials_one_sheet.xlsx** — MiniMax Hong Kong IPO financial data (R&D spending, revenue, etc.).
- **zhipu_ipo_financial_metrics.xlsx** — Zhipu IPO financial data.

## Documentation (root)

- **assumptions.md** — Documents key assumptions: FLOP-to-cost conversion parameters, MFU ranges, GPU pricing, post-training overhead, and per-model FLOP bounds.
- **caveats.md** — Known issues and caveats (arithmetic errors in Epoch notes, lognormal tail behavior, distribution mismatch between regression and MC scripts).

## Output (`output/`)

- **per_model_training_cost.xlsx** — Per-model training cost estimates (output of `analysis.py`).
- **training_vs_total_compute.png** — Scatter plot of training run compute vs. total R&D compute for MiniMax and Zhipu.
- **training_vs_rd_annualized.png** — Training run compute vs. annualized R&D compute.
- **marimekko_compute.html** — Interactive Marimekko chart of compute allocation.
- **marimekko_compute_openai.html** — Marimekko chart including OpenAI data.
- **marimekko_compute_openai_no_hailuo.html** — Marimekko chart including OpenAI, excluding Hailuo models.
- **marimekko_training_compute.html** / **marimekko_training_compute.png** — Marimekko chart of training compute only.
- **marimekko_training_compute_no_hailuo.html** — Training compute Marimekko, excluding Hailuo models.
- **image_resolution_regression.png** — Image model regression fit plot.
- **video_resolution_regression.png** — Video model regression fit plot.
- **speech_regression.png** — Speech model regression fit plot.
- **image01_prediction.png** / **image01_prediction_histogram.png** — Image-01 FLOP prediction and histogram.
- **cogvideox_prediction.png** / **cogvideox_prediction_histogram.png** — CogVideoX FLOP prediction and histogram.
- **hailuo01_prediction.png** / **hailuo01_prediction_histogram.png** — Hailuo-01 FLOP prediction and histogram.
- **hailuo02_prediction.png** / **hailuo02_prediction_histogram.png** — Hailuo-02 FLOP prediction and histogram.
- **speech01_prediction.png** / **speech01_prediction_histogram.png** — Speech-01 prediction and histogram.
- **speech01_turbo_prediction.png** / **speech01_turbo_prediction_histogram.png** — Speech-01 Turbo prediction and histogram.
- **speech02_prediction.png** / **speech02_prediction_histogram.png** — Speech-02 prediction and histogram.
- **speech02_turbo_prediction.png** / **speech02_turbo_prediction_histogram.png** — Speech-02 Turbo prediction and histogram.
