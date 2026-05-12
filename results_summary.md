# Results Summary

## Cleaned Modeling Data

- Modeling rows: 4,549
- Games: 50
- App IDs: 50
- Date range: 2012-07-01 to 2026-04-01
- Target: `log_monthly_avg_players`
- Update-history status: No usable saved update-history CSV/parquet file was found; update frequency is documented as planned future work.

## Features Used

- `months_since_release`
- `log_price`
- `discount_percent`
- `is_free_to_play`
- `log_monthly_num_reviews`
- `positive_review_percent`
- `weighted_review`
- `price_missing_flag`
- `monthly_reviews_missing_flag`
- `positive_review_percent_missing_flag`

## Model Performance

| model             |   r2_mean |   r2_std |   mae_mean |   mae_std |   rmse_mean |   rmse_std |
|:------------------|----------:|---------:|-----------:|----------:|------------:|-----------:|
| Mean baseline     |    -0.025 |    0.024 |      1.091 |     0.149 |       1.659 |      0.165 |
| Linear Regression |     0.545 |    0.112 |      0.761 |     0.124 |       1.094 |      0.151 |
| Random Forest     |     0.580 |    0.061 |      0.741 |     0.092 |       1.061 |      0.145 |

## Genre/Cluster Feature Comparison

| feature_set           | model             |   r2_mean |   r2_std |   mae_mean |   mae_std |   rmse_mean |   rmse_std |
|:----------------------|:------------------|----------:|---------:|-----------:|----------:|------------:|-----------:|
| no_genre_or_cluster   | Linear Regression |     0.545 |    0.112 |      0.761 |     0.124 |       1.094 |      0.151 |
| no_genre_or_cluster   | Mean baseline     |    -0.025 |    0.024 |      1.091 |     0.149 |       1.659 |      0.165 |
| no_genre_or_cluster   | Random Forest     |     0.580 |    0.061 |      0.741 |     0.092 |       1.061 |      0.145 |
| with_cluster_features | Linear Regression |     0.482 |    0.105 |      0.842 |     0.095 |       1.170 |      0.131 |
| with_cluster_features | Mean baseline     |    -0.025 |    0.024 |      1.091 |     0.149 |       1.659 |      0.165 |
| with_cluster_features | Random Forest     |     0.517 |    0.132 |      0.780 |     0.103 |       1.125 |      0.146 |

## Review Timing Comparison

| feature_set        | model             |   r2_mean |   r2_std |   mae_mean |   mae_std |   rmse_mean |   rmse_std |
|:-------------------|:------------------|----------:|---------:|-----------:|----------:|------------:|-----------:|
| lagged_reviews     | Linear Regression |     0.534 |    0.122 |      0.773 |     0.134 |       1.111 |      0.185 |
| lagged_reviews     | Mean baseline     |    -0.025 |    0.024 |      1.091 |     0.149 |       1.659 |      0.165 |
| lagged_reviews     | Random Forest     |     0.560 |    0.064 |      0.758 |     0.086 |       1.086 |      0.149 |
| same_month_reviews | Linear Regression |     0.545 |    0.112 |      0.761 |     0.124 |       1.094 |      0.151 |
| same_month_reviews | Mean baseline     |    -0.025 |    0.024 |      1.091 |     0.149 |       1.659 |      0.165 |
| same_month_reviews | Random Forest     |     0.580 |    0.061 |      0.741 |     0.092 |       1.061 |      0.145 |

## Top 10 Random Forest Feature Importances

| feature                              |   importance |
|:-------------------------------------|-------------:|
| log_monthly_num_reviews              |       0.5724 |
| price_missing_flag                   |       0.1192 |
| positive_review_percent              |       0.0790 |
| months_since_release                 |       0.0627 |
| is_free_to_play                      |       0.0601 |
| log_price                            |       0.0429 |
| weighted_review                      |       0.0293 |
| discount_percent                     |       0.0223 |
| positive_review_percent_missing_flag |       0.0068 |
| monthly_reviews_missing_flag         |       0.0054 |

## Top Linear Regression Standardized Coefficients

| feature                              |   coefficient |   abs_coefficient |
|:-------------------------------------|--------------:|------------------:|
| log_monthly_num_reviews              |        1.5494 |            1.5494 |
| positive_review_percent_missing_flag |        0.7729 |            0.7729 |
| price_missing_flag                   |       -0.4858 |            0.4858 |
| is_free_to_play                      |        0.1950 |            0.1950 |
| positive_review_percent              |       -0.1495 |            0.1495 |
| log_price                            |       -0.0710 |            0.0710 |
| months_since_release                 |        0.0666 |            0.0666 |
| weighted_review                      |       -0.0385 |            0.0385 |
| monthly_reviews_missing_flag         |        0.0141 |            0.0141 |
| discount_percent                     |       -0.0031 |            0.0031 |

These coefficients show linear association strength after standardizing the features. Interpret them cautiously because some predictors are related, especially `log_monthly_num_reviews`, `positive_review_percent`, and `weighted_review`.

## Cluster Summary

|   cluster_id |   games |   rows |   avg_log_players |   median_players |   avg_log_monthly_reviews |   avg_positive_review_percent |   avg_log_price |   free_to_play_share |   discount_month_share | common_genre_type_flags                                                            | example_games                                                                |
|-------------:|--------:|-------:|------------------:|-----------------:|--------------------------:|------------------------------:|----------------:|---------------------:|-----------------------:|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|
|            4 |       5 |    534 |            11.751 |        97890.020 |                     9.433 |                         0.807 |           1.110 |                0.934 |                  0.418 | Action (100%), Multi-player (100%), Adventure (60%), Massively Multiplayer (40%)   | Counter-Strike 2, NARAKA: BLADEPOINT, PUBG: BATTLEGROUNDS, Street Fighter™ 6 |
|            1 |      19 |   1419 |            10.567 |        36264.080 |                     8.161 |                         0.758 |           1.550 |                0.522 |                  0.353 | Action (100%), Multi-player (100%), Co-op (95%), Online Co-op (84%)                | Apex Legends™, Battlefield™ 6, Call of Duty®, Dead by Daylight               |
|            5 |       6 |    492 |             9.768 |        19787.225 |                     8.136 |                         0.921 |           2.762 |                0.000 |                  0.459 | Adventure (100%), RPG (100%), Co-op (100%), Multi-player (100%)                    | ARK: Survival Ascended, Baldur's Gate 3, Don't Starve Together, Palworld     |
|            6 |       4 |    411 |             9.767 |        20835.400 |                     7.956 |                         0.938 |           2.498 |                0.034 |                  0.625 | Indie (100%), Simulation (100%), Co-op (100%), Multi-player (100%)                 | Bongo Cat, Euro Truck Simulator 2, Garry's Mod, Satisfactory                 |
|            8 |       3 |    421 |             9.643 |        19266.800 |                     7.443 |                         0.734 |           2.946 |                0.000 |                  0.622 | Action (100%), Adventure (100%), Massively Multiplayer (100%), Co-op (100%)        | Black Desert, DayZ, Rust                                                     |
|            2 |       8 |    708 |             9.130 |        13032.110 |                     7.397 |                         0.883 |           2.610 |                0.154 |                  0.534 | Single-player (100%), Simulation (62%), Strategy (38%), Indie (25%)                | BeamNG.drive, Cyberpunk 2077, Geometry Dash, Hearts of Iron IV               |
|            3 |       1 |    112 |             9.057 |        15687.305 |                     7.342 |                         0.821 |           0.000 |                1.000 |                  0.000 | Adventure (100%), Casual (100%), Early Access (100%), Massively Multiplayer (100%) | VRChat                                                                       |
|            9 |       3 |    349 |             8.970 |        15209.370 |                     7.497 |                         0.907 |           2.924 |                0.000 |                  0.570 | Indie (100%), RPG (100%), Simulation (100%), Multi-player (100%)                   | Mount & Blade II: Bannerlord, Project Zomboid, Stardew Valley                |
|            7 |       1 |    103 |             8.757 |         7216.970 |                     7.134 |                         0.761 |           0.000 |                1.000 |                  0.000 | Action (100%), Massively Multiplayer (100%), Simulation (100%), Strategy (100%)    | World of Warships                                                            |
