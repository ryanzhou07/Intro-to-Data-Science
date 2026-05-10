"""Repair game app IDs and check which games have price files.

keep existing IDs, use a small manual override list, try exact normalized title matches, and only accept very high-confidence fuzzy matches.
"""

from pathlib import Path
import difflib
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
REVIEWS_DIR = ROOT / "Reviews"
PRICE_DIR = DATA_DIR / "Prices"


def normalize_title(title):
    """
    Normalize a game title for matching against Steam lookup data.

    Parameters:
    - title: Raw game title

    Returns:
    - Cleaned title string used for matching
    """
    if pd.isna(title):
        return ""

    title = str(title)
    title = unicodedata.normalize("NFKD", title)
    title = title.replace("’", "'").replace("‘", "'")
    title = title.replace("“", '"').replace("”", '"')
    title = title.replace("™", "").replace("®", "").replace("©", "")
    title = title.lower()
    title = title.replace("&", " and ")

    edition_words = [
        "standard edition",
        "deluxe edition",
        "ultimate edition",
        "gold edition",
        "complete edition",
        "game of the year edition",
        "goty edition",
        "definitive edition",
        "legacy edition",
        "enhanced edition",
    ]
    for word in edition_words:
        title = title.replace(word, " ")

    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


# checked against the local Steam lookup
MANUAL_APPID_OVERRIDES = {
    "Tom Clancy’s Rainbow Six Siege": 359550,
    "Tom Clancy's Rainbow Six Siege": 359550,
    "Rainbow Six Siege": 359550,
    "Apex Legends": 1172470,
    "Apex Legends™": 1172470,
    "Overwatch": 2357570,
    "Don’t Starve Together": 322330,
    "Don't Starve Together": 322330,
    "Baldur’s Gate 3": 1086940,
    "Baldur's Gate 3": 1086940,
    "Mount & Blades II: Bannerlord": 261550,
    "Mount & Blade II: Bannerlord": 261550,
    "Monster Hunger: World": 582010,
    "Monster Hunter: World": 582010,
    "Rocket League": 252950,
    "Rocket League®": 252950,
    "Garry’s Mod": 4000,
    "Garry's Mod": 4000,
    "Sid Meier’s Civilization VI": 289070,
    "Sid Meier's Civilization VI": 289070,
    "Sid Meier's Civilization® VII": 1295660,
    # reviews used "Legacy"; the project data uses the current Steam title
    "Grand Theft Auto V Legacy": 3240220,
    "Grand Theft Auto V Enhanced": 3240220,
}


def build_lookup():
    """
    Load the Steam lookup file and add normalized names.

    Returns:
    - DataFrame containing Steam app IDs, names, and normalized names
    """
    lookup = pd.read_csv(ROOT / "complete_steam_lookup_2026.csv")
    lookup["normalized_name"] = lookup["name"].map(normalize_title)
    return lookup


def choose_exact_match(norm_name, lookup):
    """
    Find an exact match on normalized title.

    Parameters:
    - norm_name: Normalized project game title
    - lookup: Steam lookup DataFrame

    Returns:
    - Matching lookup row, or None if no match exists
    """
    matches = lookup[lookup["normalized_name"] == norm_name]
    if matches.empty:
        return None
    # if there are duplicates, keep the lowest appid
    row = matches.sort_values("appid").iloc[0]
    return row


def choose_fuzzy_match(norm_name, lookup):
    """
    Find the closest fuzzy title match
    Confidence score is returned based on string similarity as inconsistency in naming is NOT fun :(

    Parameters:
    - norm_name: Normalized project game title
    - lookup: Steam lookup DataFrame

    Returns:
    - Matching lookup row and score, or None and 0
    """
    names = lookup["normalized_name"].dropna().unique().tolist()
    match = difflib.get_close_matches(norm_name, names, n=1, cutoff=0)
    if not match:
        return None, 0
    matched_norm = match[0]
    score = difflib.SequenceMatcher(None, norm_name, matched_norm).ratio() * 100
    row = lookup[lookup["normalized_name"] == matched_norm].sort_values("appid").iloc[0]
    return row, float(score)


def repair_one(game_name, current_appid, current_status, lookup):
    """
    Repair the app ID for one game if possible.

    Parameters:
    - game_name: Project game title
    - current_appid: Existing app ID from the mapping file
    - current_status: Existing lookup status
    - lookup: Steam lookup DataFrame

    Returns:
    - Dictionary with match details and decision
    """
    norm = normalize_title(game_name)

    if pd.notna(current_appid):
        appid = int(current_appid)
        steam_row = lookup[lookup["appid"] == appid]
        steam_name = steam_row.iloc[0]["name"] if len(steam_row) else ""
        return {
            "original_game": game_name,
            "current_app_id": appid,
            "current_status": current_status,
            "best_matched_steam_name": steam_name,
            "suggested_app_id": appid,
            "match_confidence": 100.0,
            "reason": "existing_app_id_kept",
            "decision": "accepted_existing",
        }

    if game_name in MANUAL_APPID_OVERRIDES:
        appid = MANUAL_APPID_OVERRIDES[game_name]
        steam_row = lookup[lookup["appid"] == appid]
        steam_name = steam_row.iloc[0]["name"] if len(steam_row) else ""
        return {
            "original_game": game_name,
            "current_app_id": pd.NA,
            "current_status": current_status,
            "best_matched_steam_name": steam_name,
            "suggested_app_id": appid,
            "match_confidence": 100.0,
            "reason": "manual override verified in lookup",
            "decision": "accepted_manual_override",
        }

    exact = choose_exact_match(norm, lookup)
    if exact is not None:
        return {
            "original_game": game_name,
            "current_app_id": pd.NA,
            "current_status": current_status,
            "best_matched_steam_name": exact["name"],
            "suggested_app_id": int(exact["appid"]),
            "match_confidence": 100.0,
            "reason": "exact normalized title match",
            "decision": "accepted_exact",
        }

    fuzzy, fuzzy_score = choose_fuzzy_match(norm, lookup)
    if fuzzy is not None and fuzzy_score >= 95:
        decision = "accepted_high_confidence_fuzzy"
        suggested_appid = int(fuzzy["appid"])
    elif fuzzy is not None and fuzzy_score >= 85:
        decision = "needs_manual_review"
        suggested_appid = int(fuzzy["appid"])
    else:
        decision = "unresolved"
        suggested_appid = pd.NA

    return {
        "original_game": game_name,
        "current_app_id": pd.NA,
        "current_status": current_status,
        "best_matched_steam_name": fuzzy["name"] if fuzzy is not None else "",
        "suggested_app_id": suggested_appid,
        "match_confidence": fuzzy_score,
        "reason": "fuzzy title match" if fuzzy is not None else "no reasonable match",
        "decision": decision,
    }


def price_diagnostic_row(game, appid, popularity):
    """
    Check whether a game has a usable price CSV.

    Parameters:
    - game: Project game title
    - appid: Steam app ID
    - popularity: Monthly player-count DataFrame

    Returns:
    - Dictionary with price-file diagnostics
    """
    if pd.isna(appid):
        return {
            "Game": game,
            "app_id": pd.NA,
            "price_file_found": False,
            "price_rows": 0,
            "first_price_date": pd.NaT,
            "last_price_date": pd.NaT,
            "missing_reason": "missing_app_id",
        }

    path = PRICE_DIR / f"steamdb_chart_{int(appid)}.csv"
    if not path.exists():
        return {
            "Game": game,
            "app_id": int(appid),
            "price_file_found": False,
            "price_rows": 0,
            "first_price_date": pd.NaT,
            "last_price_date": pd.NaT,
            "missing_reason": "no_price_csv",
        }

    try:
        price = pd.read_csv(path)
    except Exception:
        return {
            "Game": game,
            "app_id": int(appid),
            "price_file_found": True,
            "price_rows": 0,
            "first_price_date": pd.NaT,
            "last_price_date": pd.NaT,
            "missing_reason": "empty_or_unreadable_price_csv",
        }

    if price.empty:
        missing_reason = "empty_price_csv"
        first = last = pd.NaT
    elif "DateTime" not in price.columns:
        missing_reason = "date_parse_failed"
        first = last = pd.NaT
    else:
        dates = pd.to_datetime(price["DateTime"], errors="coerce")
        if dates.notna().sum() == 0:
            missing_reason = "date_parse_failed"
            first = last = pd.NaT
        else:
            first = dates.min()
            last = dates.max()
            pop_dates = popularity[popularity["Game"] == game]["Date"]
            pop_dates = pd.to_datetime(pop_dates, errors="coerce")
            if len(pop_dates) and first > pop_dates.max():
                missing_reason = "no_price_before_popularity_date"
            else:
                missing_reason = "ok"

    return {
        "Game": game,
        "app_id": int(appid),
        "price_file_found": True,
        "price_rows": len(price),
        "first_price_date": first,
        "last_price_date": last,
        "missing_reason": missing_reason,
    }


def main():
    """
    Repair app IDs and save price coverage diagnostics. Entry-point for the script.
    """
    lookup = build_lookup()
    mapping = pd.read_csv(REVIEWS_DIR / "game_appid_mapping.csv")
    popularity = pd.read_csv(DATA_DIR / "monthy_player_count.csv")
    popularity_games = pd.DataFrame({"Game": sorted(popularity["Game"].unique())})

    before_matched = mapping["appid"].notna().sum()
    before_not_found = (mapping["status"] == "not_found").sum()
    before_price_ids = set(pd.to_numeric(mapping["appid"], errors="coerce").dropna().astype(int))
    available_price_ids = {int(p.stem.split("_")[-1]) for p in PRICE_DIR.glob("steamdb_chart_*.csv")}
    before_price_matched = len(before_price_ids & available_price_ids)

    repaired_rows = [
        repair_one(row["game_name"], row["appid"], row["status"], lookup)
        for _, row in mapping.iterrows()
    ]
    repaired = pd.DataFrame(repaired_rows)
    repaired["appid"] = pd.to_numeric(repaired["suggested_app_id"], errors="coerce").astype("Int64")
    repaired["status"] = repaired["decision"].map(
        lambda d: "success" if str(d).startswith("accepted") else "needs_review"
    )

    repaired_out = repaired[
        [
            "original_game",
            "appid",
            "status",
            "current_status",
            "best_matched_steam_name",
            "match_confidence",
            "reason",
            "decision",
        ]
    ].rename(columns={"original_game": "game_name"})

    unresolved = repaired[repaired["decision"].isin(["needs_manual_review", "unresolved"])].copy()

    # check price coverage using popularity game names
    pop_repaired_rows = [
        repair_one(row["Game"], pd.NA, "popularity_game", lookup)
        for _, row in popularity_games.iterrows()
    ]
    pop_repaired = pd.DataFrame(pop_repaired_rows)

    price_rows = [
        price_diagnostic_row(row["original_game"], row["suggested_app_id"], popularity)
        for _, row in pop_repaired.iterrows()
    ]
    price_diag = pd.DataFrame(price_rows)

    after_matched = repaired["suggested_app_id"].notna().sum()
    after_not_found = len(repaired) - after_matched
    after_price_ids = set(pd.to_numeric(repaired["suggested_app_id"], errors="coerce").dropna().astype(int))
    after_price_matched = len(after_price_ids & available_price_ids)

    pop_after_matched = pop_repaired["suggested_app_id"].notna().sum()
    pop_price_matched = price_diag["price_file_found"].sum()

    DATA_DIR.mkdir(exist_ok=True)
    repaired_out.to_csv(DATA_DIR / "repaired_game_appid_mapping.csv", index=False)
    price_diag.to_csv(DATA_DIR / "price_file_diagnostics.csv", index=False)
    unresolved.to_csv(DATA_DIR / "unresolved_game_matches.csv", index=False)

    print("Original game_appid_mapping.csv repair diagnostics")
    print(f"Before repair app IDs matched: {before_matched} / {len(mapping)}")
    print(f"Before repair not_found: {before_not_found} / {len(mapping)}")
    print(f"Before repair price CSVs matched: {before_price_matched} / {len(mapping)}")
    print(f"After repair app IDs matched: {after_matched} / {len(mapping)}")
    print(f"After repair not_found/unresolved: {after_not_found} / {len(mapping)}")
    print(f"After repair price CSVs matched: {after_price_matched} / {len(mapping)}")

    print("\nPopularity dataset diagnostics")
    print(f"Popularity app IDs matched after repair: {pop_after_matched} / {len(popularity_games)}")
    print(f"Popularity price CSVs found after repair: {pop_price_matched} / {len(popularity_games)}")

    print("\nRepair table for originally not_found rows")
    cols = [
        "original_game",
        "current_app_id",
        "current_status",
        "best_matched_steam_name",
        "suggested_app_id",
        "match_confidence",
        "reason",
        "decision",
    ]
    print(repaired[repaired["current_status"] == "not_found"][cols].to_string(index=False))

    print("\nPrice file diagnostics by popularity game")
    print(price_diag.to_string(index=False))

    print("\nGames still unresolved")
    if unresolved.empty:
        print("none")
    else:
        print(unresolved[cols].to_string(index=False))

    print("\nSaved outputs")
    print("Data/repaired_game_appid_mapping.csv")
    print("Data/price_file_diagnostics.csv")
    print("Data/unresolved_game_matches.csv")


if __name__ == "__main__":
    main()
