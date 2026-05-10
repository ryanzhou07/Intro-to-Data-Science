"""
Merge and clean the Steam final project datasets.

The monthly player-count file stays as the base table, and the other review, cluster, free-to-play, and price features are merged onto it.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
REVIEWS_DIR = ROOT / "Reviews"
PRICE_DIR = DATA_DIR / "Prices"
OUT_PATH = DATA_DIR / "merged_steam_data.csv"


def clean_money(series):
    """
    Convert price-like values into numeric dollar amounts.

    Parameters:
    - series: Series containing price strings or numeric values

    Returns:
    - Numeric Series with invalid values set to NaN
    """
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def normalize_name(value):
    """
    Normalize a game name for app ID matching.

    Parameters:
    - value: Raw game name

    Returns:
    - Cleaned name string used for matching
    """
    value = str(value).casefold()
    value = value.replace("™", "").replace("®", "")
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def print_merge_report(name, before_rows, after_df, key_cols, match_col, new_cols):
    """
    Print basic diagnostics for a merge step.

    Parameters:
    - name: Name of the merge being reported
    - before_rows: Number of rows before the merge
    - after_df: DataFrame after the merge
    - key_cols: Columns used to show unmatched examples
    - match_col: Column used to count matched rows
    - new_cols: Columns added by the merge

    Returns:
    - None
    """
    matched = after_df[match_col].notna().sum() if match_col in after_df.columns else 0
    print(f"\n{name}")
    print(f"  rows before: {before_rows:,}")
    print(f"  rows after:  {len(after_df):,}")
    print(f"  matched rows: {matched:,} ({matched / len(after_df):.1%})")
    missing_counts = after_df[new_cols].isna().sum().sort_values(ascending=False)
    print("  missing values in added columns:")
    print(missing_counts.to_string())
    unmatched = after_df[after_df[match_col].isna()][key_cols].drop_duplicates().head(10)
    if len(unmatched):
        print("  example unmatched keys:")
        print(unmatched.to_string(index=False))


def load_f2p_games():
    """
    Load the free-to-play game list.

    Returns:
    - Set of game names marked as free-to-play
    """
    # f2p.csv is messy, so just read the raw rows
    raw = pd.read_csv(ROOT / "f2p.csv", header=None)
    values = raw.stack().dropna().astype(str).str.strip()
    values = values[~values.isin(["0", "Counter-Strike 2"])]
    games = set(values)
    games.add("Counter-Strike 2")
    return games


def build_appid_mapping(popularity_games):
    """
    Build app ID mappings for games in the popularity dataset.

    Parameters:
    - popularity_games: Iterable of game names from the popularity file

    Returns:
    - DataFrame with game names and repaired app IDs
    """
    lookup = pd.read_csv(ROOT / "complete_steam_lookup_2026.csv")
    lookup["name_norm"] = lookup["name"].str.casefold()
    lookup["name_simple"] = lookup["name"].map(normalize_name)

    mapping = pd.DataFrame({"Game": sorted(popularity_games)})
    mapping["name_norm"] = mapping["Game"].str.casefold()
    mapping = mapping.merge(
        lookup[["appid", "name", "name_norm"]],
        on="name_norm",
        how="left",
    ).drop(columns=["name_norm"])

    missing_exact = mapping["appid"].isna()
    if missing_exact.any():
        simple = pd.DataFrame({"Game": mapping.loc[missing_exact, "Game"]})
        simple["name_simple"] = simple["Game"].map(normalize_name)
        simple = simple.merge(
            lookup[["appid", "name", "name_simple"]],
            on="name_simple",
            how="left",
            suffixes=("", "_simple"),
        )
        mapping.loc[missing_exact, "appid"] = simple["appid"].to_numpy()
        mapping.loc[missing_exact, "name"] = simple["name"].to_numpy()

    # fallback if the lookup name does not line up exactly
    review_map = pd.read_csv(REVIEWS_DIR / "game_appid_mapping.csv")
    review_map = review_map.rename(columns={"game_name": "Game", "appid": "appid_review"})
    mapping = mapping.merge(review_map[["Game", "appid_review", "status"]], on="Game", how="left")
    mapping["app_id"] = mapping["appid"].combine_first(mapping["appid_review"])
    mapping["app_id"] = pd.to_numeric(mapping["app_id"], errors="coerce").astype("Int64")

    missing = mapping[mapping["app_id"].isna()]
    print("\nApp ID mapping")
    print(f"  games in popularity data: {len(mapping)}")
    print(f"  games with app_id: {mapping['app_id'].notna().sum()}")
    if len(missing):
        print("  games still missing app_id:")
        print(missing[["Game"]].to_string(index=False))

    return mapping[["Game", "app_id"]]


def load_price_events():
    """
    Load SteamDB price event CSVs.

    Returns:
    - DataFrame containing price history and discount features
    """
    frames = []
    for path in sorted(PRICE_DIR.glob("steamdb_chart_*.csv")):
        app_id = int(path.stem.split("_")[-1])
        price = pd.read_csv(path)
        price["app_id"] = app_id
        price["date"] = pd.to_datetime(price["DateTime"], errors="coerce").dt.normalize()
        price["price"] = clean_money(price["Final price"])
        price["historical_low"] = clean_money(price["Historical Low"])
        price = price.dropna(subset=["date"])
        frames.append(price[["app_id", "date", "price", "historical_low"]])

    if not frames:
        return pd.DataFrame(columns=["app_id", "date", "price", "historical_low"])

    price_events = pd.concat(frames, ignore_index=True).sort_values(["app_id", "date"])
    price_events["original_price"] = price_events.groupby("app_id")["price"].transform("max")
    price_events["discount_percent"] = np.where(
        price_events["original_price"] > 0,
        (price_events["original_price"] - price_events["price"]) / price_events["original_price"],
        0,
    )
    price_events["discount_percent"] = price_events["discount_percent"].clip(lower=0)
    price_events["discount_active"] = (price_events["discount_percent"] > 0).astype(int)
    price_events["price_change_indicator"] = (
        price_events.groupby("app_id")["price"].diff().fillna(0).ne(0).astype(int)
    )
    return price_events


def main():
    """
    Run the full merge and feature-building workflow. Entry-point for the script.
    """
    popularity = pd.read_csv(DATA_DIR / "monthy_player_count.csv")
    popularity = popularity.drop(columns=[c for c in popularity.columns if c.startswith("Unnamed")])
    popularity = popularity.rename(
        columns={
            "Date": "date",
            "Month_Since_Release": "months_since_release",
            "Avg_Players": "monthly_avg_players",
            "Log_Players": "log_monthly_avg_players_existing",
        }
    )
    popularity["date"] = pd.to_datetime(popularity["date"], errors="coerce")
    popularity = popularity.dropna(subset=["date"])

    app_map = build_appid_mapping(popularity["Game"].unique())
    df = popularity.merge(app_map, on="Game", how="left")
    df["app_id"] = df["app_id"].astype("Int64")

    print("\nBase popularity data")
    print(f"  rows: {len(df):,}")
    print(f"  games: {df['Game'].nunique()}")
    print(f"  date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  duplicate Game-date rows: {df.duplicated(['Game', 'date']).sum()}")

    # pull monthly review data
    reviews = pd.read_csv(DATA_DIR / "Reviews.csv")
    reviews = reviews.drop(columns=[c for c in reviews.columns if c.startswith("Unnamed")])
    reviews = reviews.rename(
        columns={
            "recommendations_up": "monthly_positive_reviews",
            "recommendations_down": "monthly_negative_reviews",
            "Percent_Positive": "legacy_percent_positive",
        }
    )
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews["monthly_num_reviews"] = (
        reviews["monthly_positive_reviews"].fillna(0) + reviews["monthly_negative_reviews"].fillna(0)
    )
    reviews["positive_review_percent"] = np.where(
        reviews["monthly_num_reviews"] > 0,
        reviews["monthly_positive_reviews"] / reviews["monthly_num_reviews"],
        np.nan,
    )
    review_cols = [
        "monthly_positive_reviews",
        "monthly_negative_reviews",
        "monthly_num_reviews",
        "positive_review_percent",
    ]
    before = len(df)
    df = df.merge(reviews[["Game", "date"] + review_cols], on=["Game", "date"], how="left")
    print_merge_report("Monthly review merge", before, df, ["Game", "date"], "monthly_num_reviews", review_cols)

    # pull static review totals
    review_stats = pd.read_csv(REVIEWS_DIR / "game_review_stats.csv")
    review_stats = review_stats.rename(
        columns={
            "appid": "app_id",
            "total_reviews": "total_review_count",
            "total_positive": "total_positive_reviews",
            "total_negative": "total_negative_reviews",
        }
    )
    review_stats["app_id"] = pd.to_numeric(review_stats["app_id"], errors="coerce").astype("Int64")
    review_stats["overall_positive_review_percent"] = np.where(
        review_stats["total_review_count"] > 0,
        review_stats["total_positive_reviews"] / review_stats["total_review_count"],
        np.nan,
    )
    static_review_cols = [
        "total_review_count",
        "total_positive_reviews",
        "total_negative_reviews",
        "overall_positive_review_percent",
        "review_score",
    ]
    before = len(df)
    df = df.merge(review_stats[["app_id"] + static_review_cols], on="app_id", how="left")
    print_merge_report("Static review-summary merge", before, df, ["Game", "app_id"], "total_review_count", static_review_cols)

    # add clusters and genre/type flags
    clusters = pd.read_csv(DATA_DIR / "clusters.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    details = pd.read_csv(DATA_DIR / "cluster_details.csv")
    before = len(df)
    df = df.merge(clusters, on="Game", how="left")
    print_merge_report("Cluster-label merge", before, df, ["Game"], "cluster_id", ["cluster_id"])
    before = len(df)
    detail_cols = [c for c in details.columns if c != "Game"]
    df = df.merge(details, on="Game", how="left")
    print_merge_report("Genre/type binary merge", before, df, ["Game"], detail_cols[0], detail_cols)

    # add f2p status
    f2p_games = load_f2p_games()
    df["is_free_to_play"] = df["Game"].isin(f2p_games).astype(int)

    # attach the last known price before each month
    price_events = load_price_events()
    price_cols = [
        "price",
        "historical_low",
        "original_price",
        "discount_percent",
        "discount_active",
        "price_change_indicator",
    ]
    before = len(df)
    left = df.sort_values(["app_id", "date"]).copy()
    right = price_events.sort_values(["app_id", "date"]).copy()
    merged_parts = []
    for app_id, group in left.groupby("app_id", dropna=False):
        if pd.isna(app_id):
            for col in price_cols:
                group[col] = np.nan
            merged_parts.append(group)
            continue

        price_group = right[right["app_id"] == int(app_id)]
        if price_group.empty:
            for col in price_cols:
                group[col] = np.nan
            merged_parts.append(group)
            continue

        merged_parts.append(
            pd.merge_asof(
                group.sort_values("date"),
                price_group[["date"] + price_cols].sort_values("date"),
                on="date",
                direction="backward",
            )
        )

    df = pd.concat(merged_parts, ignore_index=True).sort_values(["Game", "date"])
    print_merge_report("Price as-of merge", before, df, ["Game", "app_id", "date"], "price", price_cols)

    # fill price state inside each game only
    state_cols = ["price", "historical_low", "original_price", "discount_percent", "discount_active"]
    df[state_cols] = df.groupby("app_id", dropna=False)[state_cols].ffill()
    df["price_change_indicator"] = df["price_change_indicator"].fillna(0)
    df.loc[df["is_free_to_play"].eq(1) & df["price"].isna(), "price"] = 0
    df.loc[df["is_free_to_play"].eq(1) & df["original_price"].isna(), "original_price"] = 0
    df.loc[df["is_free_to_play"].eq(1) & df["discount_percent"].isna(), "discount_percent"] = 0
    df.loc[df["is_free_to_play"].eq(1) & df["discount_active"].isna(), "discount_active"] = 0

    df["avg_price_over_time"] = df.groupby("app_id", dropna=False)["price"].transform("mean")
    df["weighted_review"] = np.log1p(df["monthly_num_reviews"].fillna(0)) * df["positive_review_percent"].fillna(0)
    df["log_monthly_avg_players"] = np.log1p(df["monthly_avg_players"].clip(lower=0))
    df["log_monthly_num_reviews"] = np.log1p(df["monthly_num_reviews"].fillna(0).clip(lower=0))
    df["log_total_review_count"] = np.log1p(df["total_review_count"].fillna(0).clip(lower=0))
    df["log_price"] = np.log1p(df["price"].fillna(0).clip(lower=0))

    print("\nFinal missing-value counts for key fields")
    key_fields = [
        "app_id",
        "monthly_avg_players",
        "price",
        "monthly_num_reviews",
        "positive_review_percent",
        "total_review_count",
        "cluster_id",
        "is_free_to_play",
    ]
    print(df[key_fields].isna().sum().sort_values(ascending=False).to_string())

    print("\nRows with app_id missing after repair")
    missing_app = df[df["app_id"].isna()][["Game"]].drop_duplicates()
    print(missing_app.to_string(index=False) if len(missing_app) else "  none")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved merged dataset to {OUT_PATH.relative_to(ROOT)}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
