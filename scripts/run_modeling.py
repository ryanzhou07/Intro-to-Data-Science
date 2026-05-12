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
MODEL_COMPARISON_PATH = ROOT / "Data" / "model_comparison_results.csv"
REVIEW_TIMING_PATH = ROOT / "Data" / "review_timing_comparison_results.csv"
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


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate root mean squared error.

    Parameters:
    - y_true: Actual target values
    - y_pred: Predicted target values

    Returns:
    - RMSE as a float
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_grouped_models(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run baseline, linear regression, and random forest with GroupKFold.

    GroupKFold keeps rows from the same game together, so the same app does not appear in both training and test data.

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


def save_correlation_heatmap(df: pd.DataFrame, feature_cols: list) -> None:
    """
    Save a heatmap for the final modeling variables.

    Parameters:
    - df: Modeling dataset
    - feature_cols: Feature columns used for modeling

    Returns:
    - None
    """
    corr_cols = [
        "log_monthly_avg_players",
        "months_since_release",
        "log_price",
        "discount_percent",
        "is_free_to_play",
        "log_monthly_num_reviews",
        "positive_review_percent",
        "weighted_review",
        "price_missing_flag",
        "monthly_reviews_missing_flag",
        "positive_review_percent_missing_flag",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns and c in feature_cols + ["log_monthly_avg_players"]]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), cmap="vlag", center=0, annot=False, square=True)
    plt.title("Correlation Heatmap of Final Modeling Variables")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

def save_feature_importance(model: RandomForestRegressor, feature_cols: list) -> pd.DataFrame:
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


def save_predicted_vs_actual(pred_df: pd.DataFrame):
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


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-related features for modeling and summaries.

    Parameters:
    - df: DataFrame containing merged game-month data

    Returns:
    - DataFrame with added price feature columns
    """
    df = df.copy()

    # mark rows where price data was missing before imputation
    df["price_missing_flag"] = df["price"].isna().astype(int)

    # split out free vs paid games so the price column is easier to interpret
    df["is_paid"] = 1 - df["is_free_to_play"]
    df["has_price_history"] = 1 - df["price_missing_flag"]

    # discounts get their own flag so price and sale activity are separate signals
    df["is_discounted"] = (df["discount_percent"].fillna(0) > 0).astype(int)

    # price buckets are mostly for summaries and paper wording
    df["price_bucket"] = pd.cut(
        df["price"].fillna(-1),
        bins=[-1.01, -0.01, 0, 10, 30, float("inf")],
        labels=["missing", "free", "low", "mid", "high"],
    )
    df["discount_bucket"] = pd.cut(
        df["discount_percent"].fillna(-1),
        bins=[-1.01, -0.01, 0, 0.25, 0.50, float("inf")],
        labels=["missing", "none", "small", "medium", "large"],
    )
    return df


def most_common_flags(group: pd.DataFrame, cols: list, max_flags=4) -> str:
    """
    Summarize the most common genre/type flags in a cluster.

    Parameters:
    - group: DataFrame for one cluster
    - cols: Genre/type indicator columns
    - max_flags: Number of labels to keep

    Returns:
    - Comma-separated genre/type labels
    """
    rates = group[cols].mean().sort_values(ascending=False)
    common = rates[rates > 0].head(max_flags)
    return ", ".join(f"{name} ({share:.0%})" for name, share in common.items())


def example_games(group, max_games=4):
    """
    Pick a few example games from a cluster.

    Parameters:
    - group: DataFrame for one cluster
    - max_games: Number of example titles to keep

    Returns:
    - Comma-separated game names
    """
    games = sorted(group["Game"].dropna().unique())[:max_games]
    return ", ".join(games)


def save_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Save a compact cluster summary table and plot.

    Parameters:
    - df: Modeling dataset with cluster labels

    Returns:
    - DataFrame containing cluster-level summary statistics
    """
    # summarize clusters so the paper can describe what the labels mean
    game_level = df.sort_values("date").drop_duplicates("Game").copy()

    summary = (
        df.groupby("cluster_id")
        .agg(
            games=("Game", "nunique"),
            rows=("Game", "size"),
            avg_log_players=("log_monthly_avg_players", "mean"),
            median_players=("monthly_avg_players", "median"),
            avg_log_monthly_reviews=("log_monthly_num_reviews", "mean"),
            avg_positive_review_percent=("positive_review_percent", "mean"),
            avg_log_price=("log_price", "mean"),
            free_to_play_share=("is_free_to_play", "mean"),
            discount_month_share=("is_discounted", "mean"),
        )
        .reset_index()
        .sort_values("avg_log_players", ascending=False)
    )
    cluster_notes = []
    for cluster_id, group in game_level.groupby("cluster_id"):
        cluster_notes.append(
            {
                "cluster_id": cluster_id,
                "common_genre_type_flags": most_common_flags(group, GENRE_COLS),
                "example_games": example_games(group),
            }
        )
    cluster_notes = pd.DataFrame(cluster_notes)
    summary = summary.merge(cluster_notes, on="cluster_id", how="left")
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

def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Average model metrics across folds.

    Parameters:
    - results: Fold-level model result table

    Returns:
    - DataFrame with mean and standard deviation for each metric
    """
    summary = results.groupby("model").agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
    )
    order = ["Mean baseline", "Linear Regression", "Random Forest"]
    return summary.loc[[m for m in order if m in summary.index]]


def summarize_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """
    Average model metrics by feature set.

    Parameters:
    - results: Fold-level comparison result table

    Returns:
    - DataFrame with mean and standard deviation for each feature set/model
    """
    return (
        results.groupby(["feature_set", "model"])
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .reset_index()
        .sort_values(["feature_set", "model"])
    )


def save_example_time_series(df: pd.DataFrame) -> None:
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


def write_summary(df: pd.DataFrame, feature_cols: list, results: pd.DataFrame, comparison_summary: pd.DataFrame, review_timing_summary: pd.DataFrame, importance: pd.DataFrame, cluster_summary: pd.DataFrame, update_note: str) -> pd.DataFrame:
    """
    Write a compact markdown summary of cleaned data and model results.

    Parameters:
    - df: Final modeling dataset
    - feature_cols: Feature columns used for modeling
    - results: Fold-level model result table
    - comparison_summary: Model comparison with and without cluster features
    - review_timing_summary: Model comparison using same-month vs lagged reviews
    - importance: Random Forest feature importance table
    - cluster_summary: Cluster-level summary table
    - update_note: Note about update-history availability

    Returns:
    - DataFrame containing average model performance
    """
    summary = summarize_results(results)

    lines = []
    lines.append("# Results Summary\n")

    lines.append("## Cleaned Modeling Data\n")
    lines.append(f"- Modeling rows: {len(df):,}")
    lines.append(f"- Games: {df['Game'].nunique()}")
    lines.append(f"- App IDs: {df['app_id'].nunique()}")
    lines.append(f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    lines.append("- Target: `log_monthly_avg_players`")
    lines.append(f"- Update-history status: {update_note}")
    lines.append("")

    lines.append("## Features Used\n")
    lines.extend([f"- `{c}`" for c in feature_cols])
    lines.append("")

    lines.append("## Model Performance\n")
    lines.append(summary.to_markdown(floatfmt=".3f"))
    lines.append("")

    lines.append("## Genre/Cluster Feature Comparison\n")
    lines.append(comparison_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Review Timing Comparison\n")
    lines.append(review_timing_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Top 10 Random Forest Feature Importances\n")
    lines.append(importance.head(10).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Cluster Summary\n")
    lines.append(cluster_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

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

    # build simple price and free-to-play differentiators before filling values
    df = build_price_features(df)

    # flag review rows that need imputation
    df["monthly_reviews_missing_flag"] = df["monthly_num_reviews"].isna().astype(int)
    df["positive_review_percent_missing_flag"] = df["positive_review_percent"].isna().astype(int)

    time_cols = [
        "price",
        "discount_percent",
        "discount_active",
        "monthly_num_reviews",
        "positive_review_percent",
        "weighted_review",
    ]
    time_cols = [c for c in time_cols if c in df.columns]

    print("\nMissing values before imputation")
    print(df[time_cols].isna().sum().to_string())

    # use simple fills so we do not borrow info from future months
    # missing flags let the model know which values were unavailable
    if "monthly_num_reviews" in df:
        df["monthly_num_reviews"] = df["monthly_num_reviews"].fillna(0)
    df["price"] = df["price"].fillna(0)
    df["positive_review_percent"] = df["positive_review_percent"].fillna(0)
    if "discount_percent" in df:
        df["discount_percent"] = df["discount_percent"].fillna(0)
    if "discount_active" in df:
        df["discount_active"] = df["discount_active"].fillna(0)

    df["log_price"] = np.log1p(df["price"].clip(lower=0))
    df["log_monthly_num_reviews"] = np.log1p(df["monthly_num_reviews"].clip(lower=0))
    df["weighted_review"] = df["log_monthly_num_reviews"] * df["positive_review_percent"]
    df["log_monthly_avg_players"] = np.log1p(df["monthly_avg_players"].clip(lower=0))
    df["is_discounted"] = (df["discount_percent"].fillna(0) > 0).astype(int)

    df = df.sort_values(["app_id", "date"]).copy()
    df["monthly_num_reviews_lag1"] = df.groupby("app_id")["monthly_num_reviews"].shift(1)
    df["log_monthly_num_reviews_lag1"] = df.groupby("app_id")["log_monthly_num_reviews"].shift(1)
    df["positive_review_percent_lag1"] = df.groupby("app_id")["positive_review_percent"].shift(1)
    df["weighted_review_lag1"] = df.groupby("app_id")["weighted_review"].shift(1)
    lag_cols = [
        "monthly_num_reviews_lag1",
        "log_monthly_num_reviews_lag1",
        "positive_review_percent_lag1",
        "weighted_review_lag1",
    ]
    df[lag_cols] = df[lag_cols].fillna(0)

    print("\nMissing values after imputation")
    print(df[time_cols].isna().sum().to_string())

    # use the existing genre/type clusters as the broad category signal
    # raw genre flags stay in the data for summaries, but not as separate model inputs
    cluster_dummies = pd.get_dummies(df["cluster_id"].astype(str), prefix="cluster", dtype=int)
    df = pd.concat([df, cluster_dummies], axis=1)

    final_features = [
        "months_since_release",
        "log_price",
        "discount_percent",
        "is_free_to_play",
        "log_monthly_num_reviews",
        "positive_review_percent",
        "weighted_review",
        "price_missing_flag",
        "monthly_reviews_missing_flag",
        "positive_review_percent_missing_flag",
    ]
    lagged_features = [
        "months_since_release",
        "log_price",
        "discount_percent",
        "is_free_to_play",
        "log_monthly_num_reviews_lag1",
        "positive_review_percent_lag1",
        "weighted_review_lag1",
        "price_missing_flag",
        "monthly_reviews_missing_flag",
        "positive_review_percent_missing_flag",
    ]
    no_genre_features = [c for c in final_features if c in df.columns]
    lagged_features = [c for c in lagged_features if c in df.columns]
    cluster_features = list(cluster_dummies.columns)
    feature_cols = no_genre_features
    feature_cols = [c for c in feature_cols if c in df.columns]

    modeling = df.dropna(subset=["log_monthly_avg_players", "app_id"]).copy()
    comparison_feature_cols = sorted(set(feature_cols + lagged_features + cluster_features))
    modeling[comparison_feature_cols] = modeling[comparison_feature_cols].apply(pd.to_numeric, errors="coerce")

    remaining_missing = modeling[comparison_feature_cols].isna().sum().sort_values(ascending=False)
    if remaining_missing.sum() > 0:
        print("\nRemaining missing feature values filled with 0")
        print(remaining_missing[remaining_missing > 0].to_string())
        modeling[comparison_feature_cols] = modeling[comparison_feature_cols].fillna(0)

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

    y = modeling["log_monthly_avg_players"]
    groups = modeling["app_id"]

    comparison_frames = []

    # compare a model with no genre/cluster signal against the model with clusters
    for feature_set, cols in {
        "no_genre_or_cluster": no_genre_features,
        "with_cluster_features": no_genre_features + cluster_features,
    }.items():
        X_compare = modeling[cols]
        fold_results, _ = evaluate_grouped_models(X_compare, y, groups)
        fold_results["feature_set"] = feature_set
        comparison_frames.append(fold_results)

    comparison_results = pd.concat(comparison_frames, ignore_index=True)
    comparison_results.to_csv(MODEL_COMPARISON_PATH, index=False)
    comparison_summary = summarize_comparison(comparison_results)

    review_timing_frames = []
    for feature_set, cols in {
        "same_month_reviews": feature_cols,
        "lagged_reviews": lagged_features,
    }.items():
        X_compare = modeling[cols]
        fold_results, _ = evaluate_grouped_models(X_compare, y, groups)
        fold_results["feature_set"] = feature_set
        review_timing_frames.append(fold_results)

    review_timing_results = pd.concat(review_timing_frames, ignore_index=True)
    review_timing_results.to_csv(REVIEW_TIMING_PATH, index=False)
    review_timing_summary = summarize_comparison(review_timing_results)

    X = modeling[feature_cols]
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
        comparison_summary,
        review_timing_summary,
        importance,
        cluster_summary,
        update_note,
    )

    print("\nTop 10 Random Forest feature importances")
    print(importance.head(10).to_string(index=False))
    print("\nSaved files")
    print(" ", MODELING_PATH.relative_to(ROOT))
    print(" ", RESULTS_PATH.relative_to(ROOT))
    print(" ", MODEL_COMPARISON_PATH.relative_to(ROOT))
    print(" ", REVIEW_TIMING_PATH.relative_to(ROOT))
    print(" ", IMPORTANCE_PATH.relative_to(ROOT))
    print(" ", SUMMARY_PATH.relative_to(ROOT))
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(" ", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
