# Steam Popularity Project

This project analyzes Steam game popularity using included local CSV files for monthly player counts, monthly reviews, prices, discounts, free-to-play status, and genre/type groupings.

## How to Run

From the project root, run:

```bash
python3 run_project.py
```

This runs the full pipeline for analysis that:

1. repairs app ID and price matching
2. merge and clean datasets
3. runs modeling
4. regenerate outputs, figures, and pushes it to `results_summary.md`

In other words, the runner calls:

```bash
python3 scripts/repair_appid_price_matching.py
python3 scripts/merge_clean_data.py
python3 scripts/run_modeling.py
```

## Important Note About Data Fetching

`steam_reviews_fetcher.py` is included for reference because it was used during data collection, but it is **not** run by default.

The final project should be reproducible from the included CSV files. Did not make ALL CSV files regenerable as doing so can break because of rate limits, API changes, page structure changes, network issues, or Steam title changes over time.

## Main Outputs

- `Data/merged_steam_data.csv`
- `Data/modeling_dataset.csv`
- `Data/model_results.csv`
- `Data/model_comparison_results.csv`
- `Data/review_timing_comparison_results.csv`
- `Data/random_forest_feature_importance.csv`
- `Data/cluster_summary.csv`
- `results_summary.md`
- `figures/`

## Requirements

Install all of project dependencies with:

```bash
pip install -r requirements.txt
```
