# config/config.py

# Collection settings
COLLECTION_SETTINGS = {
    'recent_tracks_limit': 200,  # Maximum allowed by Last.fm API
    'top_items_limit': 50,      # Get more top items
    'recent_pages': 10,         # Start with 10 pages (2000 tracks) for recent history
    'time_periods': ['overall', '7day', '1month', '3month', '6month', '12month']  # Include all time periods
}