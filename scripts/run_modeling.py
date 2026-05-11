"""Build the modeling dataset, run the models, and save plots."""

from pathlib import Path
import os
import tempfile

cache_root = Path(tempfile.gettempdir()) / "steam_project_mpl_cache"

os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdgcache"))
os.environ.setdefault("MPLBACKEND", "Agg")

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "merged_steam_data.csv"
MODELING_PATH = ROOT / "Data" / "modeling_dataset.csv"
FIGURES_DIR = ROOT / "figures"
SUMMARY_PATH = ROOT / "results_summary.md"
RESULTS_PATH = ROOT / "Data" / "model_results.csv"
IMPORTANCE_PATH = ROOT / "Data" / "random_forest_feature_importance.csv"


GENRE_COLS = [
    "Action",
    "Adventure",
    "Casual",
    "Early Access",
    "Indie",
    "Massively Multiplayer",
    "RPG",
    "Racing",
    "Simulation",
    "Sports",
    "Strategy",
    "Co-op",
    "Multi-player",
    "Online Co-op",
    "Single-player",
]


def rmse(y_true, y_pred):
    """
    Calculate root mean squared error.

    Parameters:
    - y_true: Actual target values
    - y_pred: Predicted target values

    Returns:
    - RMSE as a float
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_grouped_models(X, y, groups):
    """
    Run baseline, linear regression, and random forest with GroupKFold.

    Parameters:
    - X: Feature matrix
    - y: Target values
    - groups: App IDs used to keep each game in one fold

    Returns:
    - DataFrame with fold-level model results
    - DataFrame with Random Forest predictions
    """
    n_groups = groups.nunique()
    n_splits = min(5, n_groups)
    cv = GroupKFold(n_splits=n_splits)

    models = {
        "Mean baseline": None,
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=3,
        ),
    }

    rows = []
    rf_predictions = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in models.items():
            if model is None:
                pred = np.repeat(y_train.mean(), len(y_test))
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "r2": r2_score(y_test, pred),
                    "mae": mean_absolute_error(y_test, pred),
                    "rmse": rmse(y_test, pred),
                    "test_rows": len(y_test),
                    "test_games": groups.iloc[test_idx].nunique(),
                }
            )

            if model_name == "Random Forest":
                rf_predictions.append(
                    pd.DataFrame(
                        {
                            "actual": y_test.values,
                            "predicted": pred,
                            "app_id": groups.iloc[test_idx].values,
                        }
                    )
                )

    results = pd.DataFrame(rows)
    rf_pred_df = pd.concat(rf_predictions, ignore_index=True)
    return results, rf_pred_df


def save_correlation_heatmap(df, feature_cols):
    """
    Save a heatmap for the main numeric variables.

    Parameters:
    - df: Modeling dataset
    - feature_cols: Feature columns used for modeling

    Returns:
    - None
    """
    corr_cols = [
        "log_monthly_avg_players",
        "months_since_release",
        "price",
        "log_price",
        "discount_percent",
        "discount_active",
        "is_free_to_play",
        "monthly_num_reviews",
        "log_monthly_num_reviews",
        "positive_review_percent",
        "weighted_review",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns and c in feature_cols + ["log_monthly_avg_players"]]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), cmap="vlag", center=0, annot=False, square=True)
    plt.title("Correlation Heatmap of Main Numeric Variables")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_feature_importance(model, feature_cols):
    """
    Save Random Forest feature importances.

    Parameters:
    - model: Trained Random Forest model
    - feature_cols: Feature names used by the model

    Returns:
    - DataFrame containing feature importances
    """
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    top = importance.head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close()
    return importance


def save_predicted_vs_actual(pred_df):
    """
    Save the predicted-versus-actual plot.

    Parameters:
    - pred_df: DataFrame with actual and predicted values

    Returns:
    - None
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.45, s=18)
    lo = min(pred_df["actual"].min(), pred_df["predicted"].min())
    hi = max(pred_df["actual"].max(), pred_df["predicted"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    plt.title("Random Forest Predicted vs. Actual Player Activity")
    plt.xlabel("Actual log monthly average players")
    plt.ylabel("Predicted log monthly average players")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=200)
    plt.close()


def save_cluster_summary(df):
    """
    Save a simple cluster summary table and plot.

    Parameters:
    - df: Modeling dataset with cluster labels

    Returns:
    - DataFrame containing cluster-level summary statistics
    """
    summary = (
        df.groupby("cluster_id")
        .agg(
            games=("Game", "nunique"),
            rows=("Game", "size"),
            avg_log_players=("log_monthly_avg_players", "mean"),
            avg_players=("monthly_avg_players", "mean"),
            avg_price=("price", "mean"),
            free_to_play_share=("is_free_to_play", "mean"),
        )
        .reset_index()
        .sort_values("avg_log_players", ascending=False)
    )
    summary.to_csv(ROOT / "Data" / "cluster_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(summary["cluster_id"].astype(str), summary["avg_log_players"])
    plt.title("Average Player Activity by Genre/Type Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Average log monthly players")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster_summary.png", dpi=200)
    plt.close()
    return summary


def save_example_time_series(df):
    """
    Save a player-count time series plot for a few example games.

    Parameters:
    - df: Modeling dataset with monthly player counts

    Returns:
    - None
    """
    examples = ["Counter-Strike 2", "PUBG: BATTLEGROUNDS", "ELDEN RING", "Stardew Valley", "Apex Legends™"]
    sample = df[df["Game"].isin(examples)].copy()

    plt.figure(figsize=(11, 6))
    for game, g in sample.groupby("Game"):
        g = g.sort_values("date")
        plt.plot(g["date"], g["monthly_avg_players"], label=game, linewidth=1.7)
    plt.title("Monthly Average Players Over Time for Example Games")
    plt.xlabel("Date")
    plt.ylabel("Monthly average players")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "example_player_count_over_time.png", dpi=200)
    plt.close()


def write_summary(df, feature_cols, results, importance, cluster_summary, update_note, leakage_note):
    """
    Write a short markdown summary of the modeling results.

    Parameters:
    - df: Final modeling dataset
    - feature_cols: Feature columns used for modeling
    - results: Fold-level model result table
    - importance: Random Forest feature importance table
    - cluster_summary: Cluster-level summary table
    - update_note: Note about update-history availability
    - leakage_note: Note about excluded static review totals

    Returns:
    - DataFrame containing average model performance
    """
    summary = results.groupby("model").agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
    )
    # keep the table easy to read
    order = ["Mean baseline", "Linear Regression", "Random Forest"]
    summary = summary.loc[[m for m in order if m in summary.index]]
    best_model = summary["r2_mean"].idxmax()

    lines = []
    lines.append("# Results Summary\n")
    lines.append("## Dataset\n")
    lines.append(f"- Modeling rows: {len(df):,}")
    lines.append(f"- Games: {df['Game'].nunique()}")
    lines.append(f"- App IDs: {df['app_id'].nunique()}")
    lines.append(f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    lines.append(f"- Target: `log_monthly_avg_players`")
    lines.append(f"- Update-history status: {update_note}")
    lines.append(f"- Static review note: {leakage_note}\n")

    lines.append("## Missing Data Decisions\n")
    lines.append("- Time-varying price/review fields were forward-filled within each game only.")
    lines.append("- Remaining missing `monthly_num_reviews` values were filled with 0.")
    lines.append("- Remaining missing `discount_percent` and `discount_active` values were filled with 0.")
    lines.append("- Remaining missing `positive_review_percent` values were filled with the game median, then the dataset median if needed.")
    lines.append("- Remaining missing `price` values were filled with the game median, then 0 if no price was available.")
    lines.append("- Missingness flags were kept for price, monthly reviews, positive-review percent, and static review totals.\n")

    lines.append("## Features Used\n")
    lines.extend([f"- `{c}`" for c in feature_cols])
    lines.append("")

    lines.append("## Model Performance\n")
    lines.append(summary.to_markdown(floatfmt=".3f"))
    lines.append("")
    lines.append(f"Best model by mean GroupKFold R²: **{best_model}**.")
    lines.append("")

    lines.append("## Top 10 Random Forest Feature Importances\n")
    lines.append(importance.head(10).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Cluster Summary\n")
    lines.append(cluster_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Paper-Ready Findings\n")
    rf = summary.loc["Random Forest"]
    lr = summary.loc["Linear Regression"]
    top_feature = importance.iloc[0]["feature"]
    lines.append(
        f"- Random Forest achieved the strongest predictive performance with mean R² = {rf['r2_mean']:.3f}, "
        f"MAE = {rf['mae_mean']:.3f}, and RMSE = {rf['rmse_mean']:.3f}. This suggests that nonlinear "
        "relationships among time, pricing, review, and genre/type features explain player activity better than a purely linear model."
    )
    lines.append(
        f"- Linear Regression produced mean R² = {lr['r2_mean']:.3f} and MAE = {lr['mae_mean']:.3f}. "
        "This gives a useful simple baseline, but its lower performance suggests the associations are not well captured by one straight-line model."
    )
    lines.append(
        f"- The most important Random Forest feature was `{top_feature}`. This indicates that the model relied heavily on that variable when predicting log monthly players, but it does not prove causality."
    )
    lines.append(
        "- Review-related variables appeared among the model inputs and should be interpreted as associations with player activity. Because same-month reviews and players are measured in the same period, the direction of influence is ambiguous."
    )
    lines.append(
        "- Pricing features are useful for describing associations with popularity, but the model should not be described as estimating pricing elasticity because it does not isolate causal price responses."
    )
    lines.append("")

    lines.append("## Limitations\n")
    lines.append("- No usable update-history CSV was found, so update frequency/support strategy could not be included in the final models.")
    lines.append("- Price data is incomplete for some games/months, and missing price values were imputed with conservative within-game medians or 0.")
    lines.append("- Static review totals were available for only 38 games and were not used as main predictive features because they appear to be current all-time totals, which can leak future information into earlier months.")
    lines.append("- The sample includes top games, so results may not generalize to typical or low-popularity Steam games.")
    lines.append("- The results are correlational and should not be described as causal.")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    """
    Run the full modeling workflow. Entry-point for the script.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    (ROOT / "Data").mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["app_id", "date"]).copy()

    print("Loaded merged dataset")
    print("  shape:", df.shape)
    print("  games:", df["Game"].nunique())
    print("  app IDs:", df["app_id"].nunique())
    print("  date range:", df["date"].min(), "to", df["date"].max())

    update_files = list(ROOT.glob("**/*update*")) + list(ROOT.glob("**/*changelist*"))
    usable_update_files = [p for p in update_files if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}]
    update_note = "No usable saved update-history CSV/parquet file was found; update frequency is documented as planned future work."
    if usable_update_files:
        update_note = "Potential update files were found, but this script did not merge them automatically: " + ", ".join(
            str(p.relative_to(ROOT)) for p in usable_update_files
        )
    print("\nUpdate-history search")
    print(" ", update_note)

    # flag rows that need imputation
    df["price_missing_flag"] = df["price"].isna().astype(int)
    df["monthly_reviews_missing_flag"] = df["monthly_num_reviews"].isna().astype(int)
    df["positive_review_percent_missing_flag"] = df["positive_review_percent"].isna().astype(int)
    df["static_reviews_missing_flag"] = df["total_review_count"].isna().astype(int)

    time_cols = [
        "price",
        "discount_percent",
        "discount_active",
        "monthly_num_reviews",
        "positive_review_percent",
        "weighted_review",
        "total_review_count",
    ]
    time_cols = [c for c in time_cols if c in df.columns]

    print("\nMissing values before imputation")
    print(df[time_cols].isna().sum().to_string())

    df[time_cols] = df.groupby("app_id", group_keys=False)[time_cols].ffill()

    # simple fills after within-game ffill
    if "monthly_num_reviews" in df:
        df["monthly_num_reviews"] = df["monthly_num_reviews"].fillna(0)
    if "discount_percent" in df:
        df["discount_percent"] = df["discount_percent"].fillna(0)
    if "discount_active" in df:
        df["discount_active"] = df["discount_active"].fillna(0)

    game_pos_median = df.groupby("app_id")["positive_review_percent"].transform("median")
    dataset_pos_median = df["positive_review_percent"].median()
    df["positive_review_percent"] = df["positive_review_percent"].fillna(game_pos_median).fillna(dataset_pos_median)

    game_price_median = df.groupby("app_id")["price"].transform("median")
    df["price"] = df["price"].fillna(game_price_median).fillna(0)

    df["total_review_count"] = df["total_review_count"].fillna(0)
    df["weighted_review"] = np.log1p(df["monthly_num_reviews"].clip(lower=0)) * df["positive_review_percent"].fillna(0)
    df["log_price"] = np.log1p(df["price"].clip(lower=0))
    df["log_monthly_num_reviews"] = np.log1p(df["monthly_num_reviews"].clip(lower=0))
    df["log_total_review_count"] = np.log1p(df["total_review_count"].clip(lower=0))
    df["log_monthly_avg_players"] = np.log1p(df["monthly_avg_players"].clip(lower=0))

    print("\nMissing values after imputation")
    print(df[time_cols].isna().sum().to_string())

    leakage_note = (
        "Static total review counts look like current all-time totals, so they were saved in the dataset "
        "but excluded from the main model features to avoid future-information leakage."
    )

    # turn cluster_id into model columns
    cluster_dummies = pd.get_dummies(df["cluster_id"].astype(str), prefix="cluster", dtype=int)
    df = pd.concat([df, cluster_dummies], axis=1)

    base_features = [
        "months_since_release",
        "price",
        "log_price",
        "discount_percent",
        "discount_active",
        "is_free_to_play",
        "monthly_num_reviews",
        "log_monthly_num_reviews",
        "positive_review_percent",
        "weighted_review",
        "price_missing_flag",
        "monthly_reviews_missing_flag",
        "positive_review_percent_missing_flag",
    ]
    feature_cols = base_features + GENRE_COLS + list(cluster_dummies.columns)
    feature_cols = [c for c in feature_cols if c in df.columns]

    modeling = df.dropna(subset=["log_monthly_avg_players", "app_id"]).copy()
    modeling[feature_cols] = modeling[feature_cols].apply(pd.to_numeric, errors="coerce")

    remaining_missing = modeling[feature_cols].isna().sum().sort_values(ascending=False)
    if remaining_missing.sum() > 0:
        print("\nRemaining missing feature values filled with 0")
        print(remaining_missing[remaining_missing > 0].to_string())
        modeling[feature_cols] = modeling[feature_cols].fillna(0)

    modeling.to_csv(MODELING_PATH, index=False)

    print("\nModeling dataset")
    print("  shape:", modeling.shape)
    print("  games:", modeling["Game"].nunique())
    print("  app IDs:", modeling["app_id"].nunique())
    print("  target:", "log_monthly_avg_players")
    print("  feature count:", len(feature_cols))
    print("  features:")
    for c in feature_cols:
        print("   -", c)

    X = modeling[feature_cols]
    y = modeling["log_monthly_avg_players"]
    groups = modeling["app_id"]

    results, rf_pred_df = evaluate_grouped_models(X, y, groups)
    results.to_csv(RESULTS_PATH, index=False)
    print("\nCross-validation results by model")
    print(
        results.groupby("model")
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .to_string()
    )

    # final rf for feature importances
    final_rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3,
    )
    final_rf.fit(X, y)

    save_correlation_heatmap(modeling, feature_cols)
    importance = save_feature_importance(final_rf, feature_cols)
    save_predicted_vs_actual(rf_pred_df)
    cluster_summary = save_cluster_summary(modeling)
    save_example_time_series(modeling)

    summary_table = write_summary(
        modeling,
        feature_cols,
        results,
        importance,
        cluster_summary,
        update_note,
        leakage_note,
    )

    print("\nTop 10 Random Forest feature importances")
    print(importance.head(10).to_string(index=False))
    print("\nSaved files")
    print(" ", MODELING_PATH.relative_to(ROOT))
    print(" ", RESULTS_PATH.relative_to(ROOT))
    print(" ", IMPORTANCE_PATH.relative_to(ROOT))
    print(" ", SUMMARY_PATH.relative_to(ROOT))
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(" ", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
