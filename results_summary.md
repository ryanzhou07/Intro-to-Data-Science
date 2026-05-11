# Results Summary

## Dataset

- Modeling rows: 4,549
- Games: 50
- App IDs: 50
- Date range: 2012-07-01 to 2026-04-01
- Target: `log_monthly_avg_players`
- Update-history status: No usable saved update-history CSV/parquet file was found; update frequency is documented as planned future work.
- Static review note: Static total review counts look like current all-time totals, so they were saved in the dataset but excluded from the main model features to avoid future-information leakage.

## Missing Data Decisions

- Time-varying price/review fields were forward-filled within each game only.
- Remaining missing `monthly_num_reviews` values were filled with 0.
- Remaining missing `discount_percent` and `discount_active` values were filled with 0.
- Remaining missing `positive_review_percent` values were filled with the game median, then the dataset median if needed.
- Remaining missing `price` values were filled with the game median, then 0 if no price was available.
- Missingness flags were kept for price, monthly reviews, positive-review percent, and static review totals.

## Features Used

- `months_since_release`
- `price`
- `log_price`
- `discount_percent`
- `discount_active`
- `is_free_to_play`
- `monthly_num_reviews`
- `log_monthly_num_reviews`
- `positive_review_percent`
- `weighted_review`
- `price_missing_flag`
- `monthly_reviews_missing_flag`
- `positive_review_percent_missing_flag`
- `Action`
- `Adventure`
- `Casual`
- `Early Access`
- `Indie`
- `Massively Multiplayer`
- `RPG`
- `Racing`
- `Simulation`
- `Sports`
- `Strategy`
- `Co-op`
- `Multi-player`
- `Online Co-op`
- `Single-player`
- `cluster_1`
- `cluster_2`
- `cluster_3`
- `cluster_4`
- `cluster_5`
- `cluster_6`
- `cluster_7`
- `cluster_8`
- `cluster_9`

## Model Performance

| model             |   r2_mean |   r2_std |   mae_mean |   mae_std |   rmse_mean |   rmse_std |
|:------------------|----------:|---------:|-----------:|----------:|------------:|-----------:|
| Mean baseline     |    -0.025 |    0.024 |      1.091 |     0.149 |       1.659 |      0.165 |
| Linear Regression |     0.251 |    0.244 |      1.057 |     0.243 |       1.396 |      0.253 |
| Random Forest     |     0.452 |    0.156 |      0.792 |     0.152 |       1.201 |      0.199 |

Best model by mean GroupKFold R²: **Random Forest**.

## Top 10 Random Forest Feature Importances

| feature                 |   importance |
|:------------------------|-------------:|
| monthly_num_reviews     |       0.2870 |
| log_monthly_num_reviews |       0.2862 |
| price_missing_flag      |       0.0764 |
| months_since_release    |       0.0350 |
| positive_review_percent |       0.0340 |
| Single-player           |       0.0323 |
| Online Co-op            |       0.0317 |
| Early Access            |       0.0290 |
| Simulation              |       0.0227 |
| price                   |       0.0197 |

## Cluster Summary

|   cluster_id |   games |     rows |   avg_log_players |   avg_players |   avg_price |   free_to_play_share |
|-------------:|--------:|---------:|------------------:|--------------:|------------:|---------------------:|
|        4.000 |   5.000 |  534.000 |            11.751 |    251896.848 |       8.728 |                0.934 |
|        1.000 |  19.000 | 1419.000 |            10.567 |     91005.211 |      13.861 |                0.522 |
|        5.000 |   6.000 |  492.000 |             9.768 |     28656.400 |      20.717 |                0.000 |
|        6.000 |   4.000 |  411.000 |             9.767 |     22561.394 |      15.102 |                0.034 |
|        8.000 |   3.000 |  421.000 |             9.643 |     30662.850 |      26.607 |                0.000 |
|        2.000 |   8.000 |  708.000 |             9.130 |     15766.138 |      23.963 |                0.154 |
|        3.000 |   1.000 |  112.000 |             9.057 |     17537.530 |       0.000 |                1.000 |
|        9.000 |   3.000 |  349.000 |             8.970 |     18632.276 |      21.266 |                0.000 |
|        7.000 |   1.000 |  103.000 |             8.757 |      7719.763 |       0.000 |                1.000 |

## Paper-Ready Findings

- Random Forest achieved the strongest predictive performance with mean R² = 0.452, MAE = 0.792, and RMSE = 1.201. This suggests that nonlinear relationships among time, pricing, review, and genre/type features explain player activity better than a purely linear model.
- Linear Regression produced mean R² = 0.251 and MAE = 1.057. This gives a useful simple baseline, but its lower performance suggests the associations are not well captured by one straight-line model.
- The most important Random Forest feature was `monthly_num_reviews`. This indicates that the model relied heavily on that variable when predicting log monthly players, but it does not prove causality.
- Review-related variables appeared among the model inputs and should be interpreted as associations with player activity. Because same-month reviews and players are measured in the same period, the direction of influence is ambiguous.
- Pricing features are useful for describing associations with popularity, but the model should not be described as estimating pricing elasticity because it does not isolate causal price responses.

## Limitations

- No usable update-history CSV was found, so update frequency/support strategy could not be included in the final models.
- Price data is incomplete for some games/months, and missing price values were imputed with conservative within-game medians or 0.
- Static review totals were available for only 38 games and were not used as main predictive features because they appear to be current all-time totals, which can leak future information into earlier months.
- The sample includes top games, so results may not generalize to typical or low-popularity Steam games.
- The results are correlational and should not be described as causal.