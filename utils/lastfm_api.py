import time 
import logging 
from typing import Dict, List, Any, Optional
import requests


class LastFMAPI:
    """A wrapper for the Last.fm API providing easy access to music data."""

    def __init__(self, api_key: str):
        """
        Initialize the Last.fm API wrapper.
        
        Parameters:
        -----------
        api_key : str
            Last.fm API key for authentication
        """
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        self.last_request_time = 0  # For improved rate limiting

    def _make_request(self, method: str, params: Dict, timeout: int = 5) -> Optional[Dict]:
        """
        Make the actual HTTP request to the Last.fm API with adaptive rate limiting.

        Parameters:
        -----------
        method : str
            Last.fm API method name
        params : dict
            Parameters to send with the request
        timeout : int, default=5
            Timeout in seconds for the request

        Returns:
        --------
        dict or None
            JSON response from the API or None if request failed
        """
        # Add common parameters
        params.update({
            'method': method,
            'api_key': self.api_key,
            'format': 'json'
        })

        # Adaptive rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.2:  # Minimum 5 requests per second
            time.sleep(0.2 - elapsed)

        try:
            # Add User-Agent header to improve connection reliability
            headers = {'User-Agent': 'LastFM ETL Pipeline/1.0'}

            # Log the request start time
            start_time = time.time()

            # Use 5-second timeout
            response = requests.get(self.base_url, params=params, headers=headers, timeout=timeout)

            # Calculate request duration
            duration = time.time() - start_time

            # Log duration for monitoring
            if duration > 3:  # Log if request takes more than 3 seconds
                logging.info(f"Slow request to {method}: {duration:.2f} seconds")
            elif duration > 1:  # Also log moderately slow requests
                logging.debug(f"Moderate request to {method}: {duration:.2f} seconds")

            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.Timeout as e:
            # Log timeout errors specifically
            logging.warning(f"Timeout error for {method} after {timeout} seconds: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Common case: resource not found
                logging.info(f"Resource not found for {method}: {e}")
                return None
            else:
                # Log other HTTP errors
                logging.error(f"HTTP error for {method}: {e}")
                raise
        except requests.exceptions.RequestException as e:
            # Log general request errors
            logging.error(f"Request failed for {method}: {e}")
            return None
    
    def _extract_data(self, response: Optional[Dict], result_path: List[str]) -> Any:
        """
        Extract specific data from the nested response.
        
        Parameters:
        -----------
        response : dict or None
            API response to extract data from
        result_path : list
            List of keys to follow in the nested JSON
            
        Returns:
        --------
        Any
            The extracted data or None if not found
        """
        if response is None:
            return None
            
        result = response
        # Navigate through the nested JSON following the result_path
        for key in result_path:
            if result is None:
                return None
            result = result.get(key, None)
        
        return result

    def _call_endpoint(self, method: str, result_path: List[str], **params) -> Any:
        """
        Call an endpoint and extract specific data from the nested response.
        
        Parameters:
        -----------
        method : str
            Last.fm API method to call
        result_path : list
            Path to extract from the response
        **params : dict
            Parameters to send with the API request
            
        Returns:
        --------
        Any
            The extracted data, empty list for missing list results,
            or None for missing single items
        """
        response = self._make_request(method, params)
        result = self._extract_data(response, result_path)
        
        # Handle common result patterns
        if result_path and result is None:
            return []  # Empty list for list results
        return result

    # ========== TRACK METHODS ==========

    def get_track_info(self, artist: str, track: str) -> Optional[Dict]:
        """
        Get detailed information about a track.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        track : str
            Name of the track
            
        Returns:
        --------
        dict or None
            Track information or None if not found
        """
        return self._call_endpoint(
            'track.getInfo', 
            [], 
            artist=artist, 
            track=track
        )
    
    def get_track_similar(
        self, 
        artist: str, 
        track: str, 
        limit: int = 10
    ) -> List[Dict]:
        """
        Get similar tracks to a given track.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        track : str
            Name of the track
        limit : int, default=10
            Maximum number of similar tracks to return
            
        Returns:
        --------
        list
            List of similar tracks
        """
        return self._call_endpoint(
            'track.getSimilar', 
            ['similartracks', 'track'],
            artist=artist, 
            track=track, 
            limit=limit
        ) or []

    def get_track_top_tags(self, artist: str, track: str) -> List[Dict]:
        """
        Get top tags for a track.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        track : str
            Name of the track
            
        Returns:
        --------
        list
            List of top tags for the track
        """
        return self._call_endpoint(
            'track.getTopTags', 
            ['toptags', 'tag'],
            artist=artist, 
            track=track
        ) or []

    # ========== ARTIST METHODS ==========

    def get_artist_info(self, artist: str) -> Dict:
        """
        Get detailed information about an artist.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
            
        Returns:
        --------
        dict
            Artist information
        """
        return self._call_endpoint('artist.getInfo', [], artist=artist)

    def get_artist_similar(self, artist: str, limit: int = 10) -> List[Dict]:
        """
        Get similar artists.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        limit : int, default=10
            Maximum number of similar artists to return
            
        Returns:
        --------
        list
            List of similar artists
        """
        return self._call_endpoint(
            'artist.getSimilar', 
            ['similarartists', 'artist'],
            artist=artist, 
            limit=limit
        ) or []

    def get_artist_top_tracks(self, artist: str, limit: int = 10) -> List[Dict]:
        """
        Get top tracks for an artist.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        limit : int, default=10
            Maximum number of tracks to return
            
        Returns:
        --------
        list
            List of the artist's top tracks
        """
        return self._call_endpoint(
            'artist.getTopTracks', 
            ['toptracks', 'track'],
            artist=artist, 
            limit=limit
        ) or []

    def get_artist_top_albums(self, artist: str, limit: int = 10) -> List[Dict]:
        """
        Get top albums for an artist.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        limit : int, default=10
            Maximum number of albums to return
            
        Returns:
        --------
        list
            List of the artist's top albums
        """
        return self._call_endpoint(
            'artist.getTopAlbums', 
            ['topalbums', 'album'],
            artist=artist, 
            limit=limit
        ) or []

    def get_artist_top_tags(self, artist: str) -> List[Dict]:
        """
        Get top tags for an artist.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
            
        Returns:
        --------
        list
            List of the artist's top tags
        """
        return self._call_endpoint(
            'artist.getTopTags', 
            ['toptags', 'tag'],
            artist=artist
        ) or []

    # ========== ALBUM METHODS ==========

    def get_album_info(self, artist: str, album: str) -> Dict:
        """
        Get detailed information about an album.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        album : str
            Name of the album
            
        Returns:
        --------
        dict
            Album information
        """
        return self._call_endpoint(
            'album.getInfo', 
            [],
            artist=artist, 
            album=album
        )

    def get_album_top_tags(self, artist: str, album: str) -> List[Dict]:
        """
        Get top tags for an album.
        
        Parameters:
        -----------
        artist : str
            Name of the artist
        album : str
            Name of the album
            
        Returns:
        --------
        list
            List of the album's top tags
        """
        return self._call_endpoint(
            'album.getTopTags', 
            ['toptags', 'tag'],
            artist=artist, 
            album=album
        ) or []

    # ========== TAG METHODS ==========

    def get_tag_info(self, tag: str) -> Dict:
        """
        Get information about a tag.
        
        Parameters:
        -----------
        tag : str
            Name of the tag
            
        Returns:
        --------
        dict
            Tag information
        """
        return self._call_endpoint('tag.getInfo', [], tag=tag)

    def get_tag_similar(self, tag: str) -> List[Dict]:
        """
        Get similar tags.
        
        Parameters:
        -----------
        tag : str
            Name of the tag
            
        Returns:
        --------
        list
            List of similar tags
        """
        return self._call_endpoint(
            'tag.getSimilar', 
            ['similartags', 'tag'],
            tag=tag
        ) or []

    def get_tag_top_artists(self, tag: str, limit: int = 10) -> List[Dict]:
        """
        Get top artists for a tag.
        
        Parameters:
        -----------
        tag : str
            Name of the tag
        limit : int, default=10
            Maximum number of artists to return
            
        Returns:
        --------
        list
            List of top artists for the tag
        """
        return self._call_endpoint(
            'tag.getTopArtists', 
            ['topartists', 'artist'],
            tag=tag, 
            limit=limit
        ) or []

    def get_tag_top_tracks(self, tag: str, limit: int = 10) -> List[Dict]:
        """
        Get top tracks for a tag.
        
        Parameters:
        -----------
        tag : str
            Name of the tag
        limit : int, default=10
            Maximum number of tracks to return
            
        Returns:
        --------
        list
            List of top tracks for the tag
        """
        return self._call_endpoint(
            'tag.getTopTracks', 
            ['tracks', 'track'],
            tag=tag, 
            limit=limit
        ) or []

    def get_tag_top_albums(self, tag: str, limit: int = 10) -> List[Dict]:
        """
        Get top albums for a tag.
        
        Parameters:
        -----------
        tag : str
            Name of the tag
        limit : int, default=10
            Maximum number of albums to return
            
        Returns:
        --------
        list
            List of top albums for the tag
        """
        return self._call_endpoint(
            'tag.getTopAlbums', 
            ['albums', 'album'],
            tag=tag, 
            limit=limit
        ) or []

    # ========== USER METHODS ==========

    def get_user_info(self, username: str) -> Dict:
        """
        Get information about a user.
        
        Parameters:
        -----------
        username : str
            Last.fm username
            
        Returns:
        --------
        dict
            User information
        """
        return self._call_endpoint('user.getInfo', [], user=username)

    def get_user_recent_tracks(
        self, 
        username: str, 
        limit: int = 50, 
        page: int = 1
    ) -> List[Dict]:
        """
        Get recent tracks listened to by a user.
        
        Parameters:
        -----------
        username : str
            Last.fm username
        limit : int, default=50
            Maximum number of tracks to return per page
        page : int, default=1
            Page number to fetch
            
        Returns:
        --------
        list
            List of recently played tracks
        """
        return self._call_endpoint(
            'user.getRecentTracks', 
            ['recenttracks', 'track'],
            user=username, 
            limit=limit, 
            page=page, 
            extended=1
        ) or []

    def get_user_top_artists(
        self, 
        username: str, 
        period: str = 'overall', 
        limit: int = 50
    ) -> List[Dict]:
        """
        Get top artists for a user.
        
        Parameters:
        -----------
        username : str
            Last.fm username
        period : str, default='overall'
            Time period. Options: overall, 7day, 1month, 3month, 6month, 12month
        limit : int, default=50
            Maximum number of artists to return
            
        Returns:
        --------
        list
            List of the user's top artists
        """
        return self._call_endpoint(
            'user.getTopArtists', 
            ['topartists', 'artist'],
            user=username, 
            period=period, 
            limit=limit
        ) or []

    def get_user_top_albums(
        self, 
        username: str, 
        period: str = 'overall', 
        limit: int = 50
    ) -> List[Dict]:
        """
        Get top albums for a user.
        
        Parameters:
        -----------
        username : str
            Last.fm username
        period : str, default='overall'
            Time period. Options: overall, 7day, 1month, 3month, 6month, 12month
        limit : int, default=50
            Maximum number of albums to return
            
        Returns:
        --------
        list
            List of the user's top albums
        """
        return self._call_endpoint(
            'user.getTopAlbums', 
            ['topalbums', 'album'],
            user=username, 
            period=period, 
            limit=limit
        ) or []

    def get_user_top_tracks(
        self, 
        username: str, 
        period: str = 'overall', 
        limit: int = 50
    ) -> List[Dict]:
        """
        Get top tracks for a user.
        
        Parameters:
        -----------
        username : str
            Last.fm username
        period : str, default='overall'
            Time period. Options: overall, 7day, 1month, 3month, 6month, 12month
        limit : int, default=50
            Maximum number of tracks to return
            
        Returns:
        --------
        list
            List of the user's top tracks
        """
        return self._call_endpoint(
            'user.getTopTracks', 
            ['toptracks', 'track'],
            user=username, 
            period=period, 
            limit=limit
        ) or []

    def get_user_loved_tracks(
        self, 
        username: str, 
        limit: int = 50, 
        page: int = 1
    ) -> List[Dict]:
        """
        Get tracks loved by a user.
        
        Parameters:
        -----------
        username : str
            Last.fm username
        limit : int, default=50
            Maximum number of loved tracks to return per page
        page : int, default=1
            Page number to fetch
            
        Returns:
        --------
        list
            List of the user's loved tracks
        """
        return self._call_endpoint(
            'user.getLovedTracks', 
            ['lovedtracks', 'track'],
            user=username, 
            limit=limit, 
            page=page
        ) or []

    def get_user_library_artists(
        self, 
        username: str, 
        limit: int = 50, 
        page: int = 1
    ) -> List[Dict]:
        """
        Get all artists in a user's library.
        
        Parameters:
        -----------
        username : str
            Last.fm username
        limit : int, default=50
            Maximum number of artists to return per page
        page : int, default=1
            Page number to fetch
            
        Returns:
        --------
        list
            List of artists in the user's library
        """
        return self._call_endpoint(
            'library.getArtists', 
            ['artists', 'artist'],
            user=username, 
            limit=limit, 
            page=page
        ) or []