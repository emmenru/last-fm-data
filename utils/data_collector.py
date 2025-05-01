import time
from tqdm.notebook import tqdm
from typing import Dict, List, Any, Optional


class DataCollector:
    """Last.fm data collector"""

    def __init__(self, lastfm_api, username):
        """
        Initialize the data collector with API access and username.
        
        Parameters:
        -----------
        lastfm_api : LastFMAPI
            Instance of the LastFMAPI class for making API calls
        username : str
            Last.fm username to collect data for
        """
        self.api = lastfm_api
        self.username = username
        self.cache = {'artists': {}, 'albums': {}, 'tracks': {}, 'tags': {}}

    def _extract_images(self, images):
        """
        Extract and standardize image URLs from Last.fm's image list format.

        Last.fm API returns images as a list of objects with 'size' and '#text' properties.
        This method converts that format into a standardized dictionary with consistent
        keys that match our database schema.

        Parameters:
        -----------
        images : list or None
            List of image objects from Last.fm API, or None if no images available

        Returns:
        --------
        dict
            Dictionary with standardized image URL keys:
            - image_small: URL to small image
            - image_medium: URL to medium image
            - image_large: URL to large image

            Empty strings are used for any missing image sizes.
        """
        if not images:
            return {'image_small': '', 'image_medium': '', 'image_large': ''}
        
        result = {}
        for img in images:
            if img.get('#text'):
                if img.get('size') == 'small':
                    result['image_small'] = img.get('#text', '')
                elif img.get('size') == 'medium':
                    result['image_medium'] = img.get('#text', '')
                elif img.get('size') == 'large':
                    result['image_large'] = img.get('#text', '')
        
        # Ensure all fields exist
        for size in ['image_small', 'image_medium', 'image_large']:
            if size not in result:
                result[size] = ''
                
        return result

    def _safe_call(self, method_name, *args, **kwargs):
        """
        Safely call an API method with comprehensive error handling.

        This helper method attempts to call the specified Last.fm API method
        and handles any exceptions that might occur, preventing them from
        crashing the data collection process.

        Parameters:
        -----------
        method_name : str
            Name of the API method to call (e.g., 'get_artist_info')
        *args : tuple
            Positional arguments to pass to the method
        **kwargs : dict
            Keyword arguments to pass to the method

        Returns:
        --------
        Any or None
            The data returned by the API method if successful, or None if an error occurs
        """
        try:
            method = getattr(self.api, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            print(f"Error calling {method_name}: {e}")
            return None

    def get_user_recent_tracks(self, limit=50, pages=1):
        """
        Fetches the user's recently played tracks from Last.fm.

        Parameters:
        -----------
        limit : int, default=50
            Maximum number of tracks to fetch per page
        pages : int, default=1
            Number of pages to fetch (each page contains 'limit' tracks)

        Returns:
        --------
        list
            List of dictionaries containing recently played tracks with:
            - track_name: Name of the track
            - artist_name: Name of the artist
            - album_name: Name of the album (if available)
            - listened_at: Unix timestamp when the track was played
            - is_now_playing: Boolean indicating if the track is currently playing
            - mbid: MusicBrainz ID for the track (if available)
        """
        all_tracks = []
        for page in tqdm(range(1, pages + 1), desc="Fetching recent tracks"):
            tracks = self._safe_call(
                'get_user_recent_tracks', 
                self.username, 
                limit=limit, 
                page=page
            )
            if not tracks:
                continue
                
            for track in tracks:
                all_tracks.append({
                    'track_name': track['name'],
                    'artist_name': track['artist']['name'],
                    'album_name': track.get('album', {}).get('#text', ''),
                    'listened_at': int(track.get('date', {}).get('uts', time.time())),
                    'is_now_playing': track.get('@attr', {}).get('nowplaying') == 'true',
                    'mbid': track.get('mbid', '')
                })
                
        print(f"Collected {len(all_tracks)} listening history records")
        return all_tracks

    def get_user_top_items(self, item_type, periods=['7day'], limit=50):
        """
        Fetches a user's top artists, albums, or tracks for specified time periods.

        Parameters:
        -----------
        item_type : str
            Type of items to fetch - must be 'artists', 'albums', or 'tracks'
        periods : list, default=['7day']
            Time periods to fetch data for. Options: '7day', '1month', '3month', 
            '6month', '12month', 'overall'
        limit : int, default=50
            Maximum number of items to fetch per time period

        Returns:
        --------
        list
            List of dictionaries containing the user's top items with rank, 
            playcount, and identifying information
        """
        method_map = {
            'artists': 'get_user_top_artists',
            'albums': 'get_user_top_albums',
            'tracks': 'get_user_top_tracks'
        }
        
        if item_type not in method_map:
            return []

        result = []
        for period in periods:
            items = self._safe_call(
                method_map[item_type], 
                self.username, 
                period=period, 
                limit=limit
            )
            if not items:
                continue
                
            for idx, item in enumerate(items):
                entry = {
                    'rank': idx + 1,
                    'playcount': int(item['playcount']),
                    'time_period': period,
                    'mbid': item.get('mbid', '')
                }
                
                # Add entity-specific fields
                if item_type == 'artists':
                    entry['artist_name'] = item['name']
                elif item_type == 'albums':
                    entry['album_name'] = item['name']
                    entry['artist_name'] = item['artist']['name']
                elif item_type == 'tracks':
                    entry['track_name'] = item['name']
                    entry['artist_name'] = item['artist']['name']
                    
                result.append(entry)
                
        return result

    def get_entity_details(self, entity_type, **kwargs):
        """
        Fetches detailed information about a specific artist, album, or track from Last.fm.

        This method is a unified gateway to retrieve comprehensive details about music entities.
        It handles caching to avoid redundant API calls and collects related information like
        tags and similar entities in a single call.

        Parameters:
        -----------
        entity_type : str
            Type of entity to fetch. Must be one of: 'artist', 'album', or 'track'
        **kwargs : dict
            Entity identification parameters:
            - For artists: artist_name (str)
            - For albums: artist_name (str), album_name (str)
            - For tracks: artist_name (str), track_name (str)

        Returns:
        --------
        dict or None
            Dictionary containing all available information about the entity, including:
            - Basic metadata (name, mbid, url, etc.)
            - Image URLs (image_small, image_medium, image_large)
            - Stats (listeners, playcount)
            - Associated tags
            - Related entities (similar artists or similar tracks)
            - Entity-specific data (e.g., bio for artists, duration for tracks)

            Returns None if the entity cannot be found or an error occurs.

        Notes:
        ------
        Results are cached by entity type and name to avoid redundant API calls.
        This method may make multiple API calls for a single entity to gather
        all related information (tags, similar entities, etc.).
        """
        # Create a cache key from entity type and params
        cache_key = f"{entity_type}_{'.'.join(kwargs.values())}"

        if cache_key in self.cache.get(f"{entity_type}s", {}):
            return self.cache[f"{entity_type}s"][cache_key]

        # Get base entity info based on entity type
        info_method = f"get_{entity_type}_info"

        if entity_type == 'artist':
            data = self._safe_call(info_method, kwargs.get('artist_name', ''))
        elif entity_type == 'album':
            data = self._safe_call(
                info_method, 
                kwargs.get('artist_name', ''), 
                kwargs.get('album_name', '')
            )
        elif entity_type == 'track':
            data = self._safe_call(
                info_method, 
                kwargs.get('artist_name', ''), 
                kwargs.get('track_name', '')
            )
        else:
            return None

        if not data:
            return None

        # Extract entity data from response if needed
        if entity_type in data:
            data = data[entity_type]

        # Extract image URLs
        images = self._extract_images(data.get('image', []))
        data.update(images)

        # Get tags based on entity type
        tags_method = f"get_{entity_type}_top_tags"

        if entity_type == 'artist':
            tags = self._safe_call(tags_method, kwargs.get('artist_name', ''))
        elif entity_type == 'album':
            tags = self._safe_call(
                tags_method, 
                kwargs.get('artist_name', ''),
                kwargs.get('album_name', '')
            )
        elif entity_type == 'track':
            tags = self._safe_call(
                tags_method, 
                kwargs.get('artist_name', ''),
                kwargs.get('track_name', '')
            )
        else:
            tags = None

        data['tags'] = []
        if tags:
            data['tags'] = [
                {'name': t['name'], 'count': int(t.get('count', 0))} 
                for t in tags
            ]
            # Cache tag info
            for tag in data['tags']:
                self.cache['tags'][tag['name']] = {'name': tag['name']}

        # Get entity-specific related data
        if entity_type == 'artist':
            similar = self._safe_call(
                'get_artist_similar', 
                kwargs.get('artist_name', ''), 
                10
            )
            data['similar_artists'] = []
            if similar:
                data['similar_artists'] = [
                    {'name': a['name'], 'match': float(a.get('match', 0))}
                    for a in similar
                ]

        elif entity_type == 'track':
            similar = self._safe_call(
                'get_track_similar', 
                kwargs.get('artist_name', ''),
                kwargs.get('track_name', ''),
                10
            )
            data['similar_tracks'] = []
            if similar:
                data['similar_tracks'] = [
                    {
                        'name': t['name'],
                        'artist': t.get('artist', {}),
                        'match': float(t.get('match', 0))
                    }
                    for t in similar
                ]

        # Cache the results
        self.cache.setdefault(f"{entity_type}s", {})[cache_key] = data
        return data

    def collect_library_data(
        self, 
        recent_tracks_limit=50, 
        top_items_limit=50,
        recent_pages=1, 
        time_periods=['7day']
    ):
        """
        Collects a complete dataset of the user's Last.fm library for ETL processing.

        This function executes the full data collection pipeline using a two-pass approach:
        
        First pass:
        1. Fetches user's listening history and top items (artists, albums, tracks)
        2. Identifies all unique entities (artists, albums, tracks) from that history
        3. Collects detailed metadata for each unique entity
        4. Builds relationship data (tags, similarities) between entities
        
        Second pass:
        5. Identifies album references in track data that weren't collected in the first pass
        6. Collects detailed metadata for these missing albums
        7. Updates relationship data with new album information
        
        This two-pass approach ensures maximum coverage of album data by capturing albums
        that are referenced by tracks but might not appear in the user's top albums or
        recent listening history.

        Parameters:
        -----------
        recent_tracks_limit : int, default=50
            Number of recent tracks to fetch per page
        top_items_limit : int, default=50
            Number of top items (artists, albums, tracks) to fetch per time period
        recent_pages : int, default=1
            Number of pages of recent tracks to fetch (each page has 'recent_tracks_limit' items)
        time_periods : list, default=['7day']
            Time periods for top items (options: '7day', '1month', '3month', '6month', 
            '12month', 'overall')

        Returns:
        --------
        dict
            A dictionary containing all collected data organized by entity type:
            - 'listening_history': Recent tracks the user has listened to
            - 'top_artists', 'top_albums', 'top_tracks': User's top items by time period
            - 'artists', 'albums', 'tracks', 'tags': Detailed entity information
            - 'artist_tags', 'album_tags', 'track_tags': Tag relationships
            - 'artist_similar', 'track_similar': Similarity relationships
        """
        # Initialize result structure
        result = {
            'listening_history': [], 'top_artists': [], 'top_albums': [], 'top_tracks': [],
            'artists': [], 'albums': [], 'tracks': [], 'tags': [],
            'artist_tags': [], 'album_tags': [], 'track_tags': [],
            'artist_similar': [], 'track_similar': []
        }

        # Collect listening history and top items
        result['listening_history'] = self.get_user_recent_tracks(
            limit=recent_tracks_limit, pages=recent_pages
        )
        result['top_artists'] = self.get_user_top_items(
            'artists', periods=time_periods, limit=top_items_limit
        )
        result['top_albums'] = self.get_user_top_items(
            'albums', periods=time_periods, limit=top_items_limit
        )
        result['top_tracks'] = self.get_user_top_items(
            'tracks', periods=time_periods, limit=top_items_limit
        )

        # Extract unique entities
        unique_artists = {
            item['artist_name'] for item in 
            result['listening_history'] + result['top_artists'] + 
            result['top_albums'] + result['top_tracks']
        }
        
        # Get artist details
        print(f"Collecting details for {len(unique_artists)} artists...")
        for artist_name in tqdm(unique_artists, desc="Artists"):
            artist_data = self.get_entity_details('artist', artist_name=artist_name)
            if not artist_data:
                continue
                
            # Add artist
            result['artists'].append({
                'name': artist_name,
                'mbid': artist_data.get('mbid', ''),
                'url': artist_data.get('url', ''),
                'image_small': artist_data.get('image_small', ''),
                'image_medium': artist_data.get('image_medium', ''),
                'image_large': artist_data.get('image_large', ''),
                'listeners': int(artist_data.get('stats', {}).get('listeners', 0)),
                'playcount': int(artist_data.get('stats', {}).get('playcount', 0)),
                'bio_summary': artist_data.get('bio', {}).get('summary', ''),
                'bio_content': artist_data.get('bio', {}).get('content', '')
            })

            # Add artist tags
            for tag in artist_data.get('tags', []):
                result['artist_tags'].append({
                    'artist_name': artist_name,
                    'tag_name': tag['name'],
                    'count': tag.get('count', 0)
                })
                
                # Add tag if new
                if tag['name'] not in [t.get('name') for t in result['tags']]:
                    result['tags'].append({'name': tag['name']})

            # Add similar artists
            for similar in artist_data.get('similar_artists', []):
                result['artist_similar'].append({
                    'artist_name': artist_name,
                    'similar_artist': similar['name'],
                    'match_score': float(similar.get('match', 0))
                })

        # Get unique albums - FIRST PASS album collection
        unique_albums = {
            (item['artist_name'], item['album_name']) 
            for item in result['listening_history'] + result['top_albums'] 
            if item.get('album_name')
        }

        # Get album details
        print(f"Collecting details for {len(unique_albums)} albums (first pass)...")
        for artist_name, album_name in tqdm(unique_albums, desc="Albums"):
            album_data = self.get_entity_details(
                'album', 
                artist_name=artist_name, 
                album_name=album_name
            )
            if not album_data:
                continue
                
            # Add album
            result['albums'].append({
                'name': album_name,
                'artist': artist_name,
                'mbid': album_data.get('mbid', ''),
                'url': album_data.get('url', ''),
                'image_small': album_data.get('image_small', ''),
                'image_medium': album_data.get('image_medium', ''),
                'image_large': album_data.get('image_large', ''),
                'listeners': int(album_data.get('listeners', 0)),
                'playcount': int(album_data.get('playcount', 0))
            })

            # Add album tags
            for tag in album_data.get('tags', []):
                result['album_tags'].append({
                    'artist_name': artist_name,
                    'album_name': album_name,
                    'tag_name': tag['name'],
                    'count': tag.get('count', 0)
                })

        # Get unique tracks
        unique_tracks = {
            (item['artist_name'], item['track_name']) 
            for item in result['listening_history'] + result['top_tracks']
        }

        # Get track details
        print(f"Collecting details for {len(unique_tracks)} tracks...")
        for artist_name, track_name in tqdm(unique_tracks, desc="Tracks"):
            track_data = self.get_entity_details(
                'track', 
                artist_name=artist_name, 
                track_name=track_name
            )
            if not track_data:
                continue
                
            # Add track
            result['tracks'].append({
                'name': track_name,
                'artist': artist_name,
                'mbid': track_data.get('mbid', ''),
                'url': track_data.get('url', ''),
                'duration': int(track_data.get('duration', 0)),
                'listeners': int(track_data.get('listeners', 0)),
                'playcount': int(track_data.get('playcount', 0)),
                'album_name': track_data.get('album', {}).get('title', '')
            })

            # Add track tags
            for tag in track_data.get('tags', []):
                result['track_tags'].append({
                    'artist_name': artist_name,
                    'track_name': track_name,
                    'tag_name': tag['name'],
                    'count': tag.get('count', 0)
                })

            # Add similar tracks
            for similar in track_data.get('similar_tracks', []):
                similar_artist = similar.get('artist', {})
                similar_artist_name = (
                    similar_artist.get('name', '') 
                    if isinstance(similar_artist, dict) 
                    else ''
                )
                
                result['track_similar'].append({
                    'track_name': track_name,
                    'artist_name': artist_name,
                    'similar_track': similar['name'],
                    'similar_artist': similar_artist_name,
                    'match_score': float(similar.get('match', 0))
                })

        # SECOND PASS album collection to capture albums referenced in tracks
        # Extract all album references from tracks
        track_album_refs = set()
        for track in result['tracks']:
            if track.get('album_name') and track.get('artist'):
                track_album_refs.add((track['artist'], track['album_name']))
        
        # Find which albums are missing from our collection
        existing_albums = {(album['artist'], album['name']) for album in result['albums']}
        missing_albums = track_album_refs - existing_albums
        
        if missing_albums:
            print(f"Found {len(missing_albums)} album references from tracks that weren't collected in first pass")
            print("Collecting details for these missing albums (second pass)...")
            
            for artist_name, album_name in tqdm(missing_albums, desc="Missing Albums"):
                # Skip empty album names
                if not album_name.strip():
                    continue
                    
                album_data = self.get_entity_details(
                    'album', 
                    artist_name=artist_name, 
                    album_name=album_name
                )
                if not album_data:
                    continue
                    
                # Add the missing album
                result['albums'].append({
                    'name': album_name,
                    'artist': artist_name,
                    'mbid': album_data.get('mbid', ''),
                    'url': album_data.get('url', ''),
                    'image_small': album_data.get('image_small', ''),
                    'image_medium': album_data.get('image_medium', ''),
                    'image_large': album_data.get('image_large', ''),
                    'listeners': int(album_data.get('listeners', 0)),
                    'playcount': int(album_data.get('playcount', 0))
                })
                
                # Add album tags
                for tag in album_data.get('tags', []):
                    result['album_tags'].append({
                        'artist_name': artist_name,
                        'album_name': album_name,
                        'tag_name': tag['name'],
                        'count': tag.get('count', 0)
                    })
                
        print(f"Data collection complete:")
        print(f"- {len(result['listening_history'])} history records")
        print(f"- {len(result['artists'])} artists")
        print(f"- {len(result['albums'])} albums (including {len(missing_albums)} from second pass)")
        print(f"- {len(result['tracks'])} tracks")
        print(f"- {len(result['tags'])} tags")

        return result