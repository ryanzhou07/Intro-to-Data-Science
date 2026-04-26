import requests
import pandas as pd
import time
from typing import Optional, Dict, List
import argparse

def find_appid(game_name: str, lookup_df: pd.DataFrame) -> Optional[int]:
    """
    Find AppID from the lookup file by matching game name
    """
    game_name_clean = game_name.strip()
    
    # Try exact match first (case insensitive)
    exact_match = lookup_df[lookup_df['name'].str.lower() == game_name_clean.lower()]
    if not exact_match.empty:
        return int(exact_match.iloc[0]['appid'])
    
    # Try partial match
    partial_match = lookup_df[lookup_df['name'].str.contains(game_name_clean, case=False, na=False, regex=False)]
    if not partial_match.empty:
        # Prefer closer matches
        for idx, row in partial_match.iterrows():
            if game_name_clean.lower() in row['name'].lower():
                return int(row['appid'])
        return int(partial_match.iloc[0]['appid'])
    
    return None
 
def get_reviews(appid: int, num_reviews: int = 100, filter_type: str = 'recent', 
                language: str = 'english', day_range: int = 365) -> Dict:
    """
    Fetch reviews for a specific AppID using Steam's Review API
    
    Parameters:
    - appid: Steam App ID
    - num_reviews: Number of reviews to fetch per request (max 100)
    - filter_type: 'recent', 'updated', or 'all'
    - language: Language filter (e.g., 'english', 'all')
    - day_range: Number of days to look back (for 'recent' filter)
    
    Returns dictionary with review data
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"
    
    params = {
        'json': 1,
        'filter': filter_type,
        'language': language,
        'day_range': day_range,
        'num_per_page': min(num_reviews, 100),
        'review_type': 'all',
        'purchase_type': 'all'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f" Error: {e}")
        return None
 
def get_all_reviews_paginated(appid: int, max_reviews: int = 1000, 
                               filter_type: str = 'recent') -> List[Dict]:
    """
    Fetch multiple pages of reviews (up to max_reviews)
    """
    url = f"https://store.steampowered.com/appreviews/{appid}"
    all_reviews = []
    cursor = "*"
    
    while len(all_reviews) < max_reviews:
        params = {
            'json': 1,
            'filter': filter_type,
            'language': 'english',
            'num_per_page': 100,
            'cursor': cursor,
            'review_type': 'all',
            'purchase_type': 'all'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                break
            
            reviews = data.get('reviews', [])
            if not reviews:
                break
            
            all_reviews.extend(reviews)
            cursor = data.get('cursor')
            
            if not cursor or cursor == "*":
                break
                
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"  Pagination error: {e}")
            break
    
    return all_reviews[:max_reviews]
 
def parse_review_data(reviews: List[Dict], game_name: str, appid: int) -> List[Dict]:
    """Extract relevant info from review data"""
    parsed_reviews = []
    
    for review in reviews:
        parsed_reviews.append({
            'game_name': game_name,
            'appid': appid,
            'recommendationid': review.get('recommendationid'),
            'author_steamid': review.get('author', {}).get('steamid'),
            'author_playtime_hours': round(review.get('author', {}).get('playtime_forever', 0) / 60, 2),
            'author_num_reviews': review.get('author', {}).get('num_reviews', 0),
            'voted_up': review.get('voted_up'),
            'votes_up': review.get('votes_up'),
            'votes_funny': review.get('votes_funny'),
            'weighted_vote_score': review.get('weighted_vote_score'),
            'comment_count': review.get('comment_count'),
            'review_text': review.get('review', '').replace('\n', ' ').replace('\r', ''),
            'timestamp_created': pd.to_datetime(review.get('timestamp_created'), unit='s'),
            'timestamp_updated': pd.to_datetime(review.get('timestamp_updated'), unit='s'),
            'written_during_early_access': review.get('written_during_early_access'),
            'received_for_free': review.get('received_for_free'),
            'steam_purchase': review.get('steam_purchase', True)
        })
    
    return parsed_reviews
 
def get_review_summary(appid: int) -> Optional[Dict]:
    """Get review summary statistics"""
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {'json': 1, 'num_per_page': 0}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') == 1:
            summary = data.get('query_summary', {})
            return {
                'total_reviews': summary.get('total_reviews', 0),
                'total_positive': summary.get('total_positive', 0),
                'total_negative': summary.get('total_negative', 0),
                'review_score': summary.get('review_score', 0),
                'review_score_desc': summary.get('review_score_desc', 'N/A')
            }
    except Exception:
        pass
    
    return None
 
def main():
    parser = argparse.ArgumentParser(description='Fetch Steam reviews for games in game_list.csv')
    parser.add_argument('--reviews-per-game', type=int, default=100, 
                       help='Number of reviews to fetch per game (default: 100, max: 1000)')
    parser.add_argument('--filter', choices=['recent', 'all', 'updated'], default='recent',
                       help='Review filter type (default: recent)')
    parser.add_argument('--delay', type=float, default=1.5,
                       help='Delay between requests in seconds (default: 1.5)')
    
    args = parser.parse_args()
    
    
    print("\n" + "="*70)
    print("STEAM REVIEWS FETCHER")
    print("="*70)
    
    # Load files
    print("\nLoading files...")
    try:
        games_df = pd.read_csv("game_list.csv")
        lookup_df =pd.read_csv("complete_steam_lookup_2026.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure these files are in the same directory:")
        print(f"  - {GAME_LIST_FILE}")
        print(f"  - {LOOKUP_FILE}")
        return
    
    print(f"Loaded {len(games_df)} games to process")
    print(f"Loaded {len(lookup_df):,} games in Steam catalog")
    print(f"\nSettings:")
    print(f"  - Reviews per game: {args.reviews_per_game}")
    print(f"  - Filter: {args.filter}")
    print(f"  - Delay: {args.delay}s")
    
    # Results storage
    all_reviews = []
    game_mapping = []
    game_stats = []
    
    print("\n" + "="*70)
    print("FETCHING REVIEWS")
    print("="*70 + "\n")
    
    for idx, row in games_df.iterrows():
        game_name = row['Game']
        print(f"[{idx+1}/{len(games_df)}] {game_name}")
        
        # Find AppID
        appid = find_appid(game_name, lookup_df)
        
        if appid is None:
            print(f"  Not found in Steam catalog\n")
            game_mapping.append({
                'game_name': game_name,
                'appid': None,
                'status': 'not_found'
            })
            continue
        
        print(f"  AppID: {appid}")
        
        # Get summary stats
        stats = get_review_summary(appid)
        if stats:
            print(f"  {stats['total_reviews']:,} total reviews ({stats['review_score_desc']})")
            stats['game_name'] = game_name
            stats['appid'] = appid
            game_stats.append(stats)
        
        # Fetch reviews
        if args.reviews_per_game > 100:
            # Use pagination for more than 100 reviews
            print(f"  Fetching {args.reviews_per_game} reviews (paginated)...")
            reviews_raw = get_all_reviews_paginated(appid, max_reviews=args.reviews_per_game, 
                                                    filter_type=args.filter)
        else:
            # Single request
            review_data = get_reviews(appid, num_reviews=args.reviews_per_game, 
                                     filter_type=args.filter)
            reviews_raw = review_data.get('reviews', []) if review_data and review_data.get('success') else []
        
        if reviews_raw:
            reviews = parse_review_data(reviews_raw, game_name, appid)
            print(f"  Fetched {len(reviews)} reviews")
            all_reviews.extend(reviews)
            
            game_mapping.append({
                'game_name': game_name,
                'appid': appid,
                'status': 'success',
                'reviews_fetched': len(reviews)
            })
        else:
            print(f"  No reviews available")
            game_mapping.append({
                'game_name': game_name,
                'appid': appid,
                'status': 'no_reviews',
                'reviews_fetched': 0
            })
        
        # Rate limiting
        time.sleep(args.delay)
        print()
    
    # Save results
    print("="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    if all_reviews:
        reviews_df = pd.DataFrame(all_reviews)
        reviews_df.to_csv('steam_reviews.csv', index=False)
        print(f"Saved {len(reviews_df):,} reviews to 'steam_reviews.csv'")
    else:
        print("No reviews were collected")
    
    mapping_df = pd.DataFrame(game_mapping)
    mapping_df.to_csv('game_appid_mapping.csv', index=False)
    print(f"Saved game mapping to 'game_appid_mapping.csv'")
    
    if game_stats:
        stats_df = pd.DataFrame(game_stats)
        stats_df.to_csv('game_review_stats.csv', index=False)
        print(f"Saved review statistics to 'game_review_stats.csv'")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total games processed: {len(games_df)}")
    
    status_counts = mapping_df['status'].value_counts()
    print(f"Found and fetched: {status_counts.get('success', 0)}")
    print(f"Found but no reviews: {status_counts.get('no_reviews', 0)}")
    print(f"Not found in catalog: {status_counts.get('not_found', 0)}")
    print(f"\nTotal reviews collected: {len(all_reviews):,}")
    
    if all_reviews:
        reviews_df = pd.DataFrame(all_reviews)
        print(f"\nReview breakdown:")
        print(f"  Positive: {reviews_df['voted_up'].sum():,}")
        print(f"  Negative: {(~reviews_df['voted_up']).sum():,}")
        print(f"   Avg playtime: {reviews_df['author_playtime_hours'].mean():.1f} hours")
        print(f"  Date range: {reviews_df['timestamp_created'].min().date()} to {reviews_df['timestamp_created'].max().date()}")
 
if __name__ == "__main__":
    main()
