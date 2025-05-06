import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


class DatabaseHelper:
    """Helper class to manage SQLite database operations for Last.fm data."""
    
    def __init__(self, db_path: str):
        """
        Initialize the database helper with the path to the SQLite database.
        
        Parameters:
        -----------
        db_path : str
            Path to the SQLite database file
        """
        self.db_path = db_path
        
    def _get_connection(self) -> sqlite3.Connection:
        """
        Create and return a database connection with Row factory enabled.
        
        Returns a SQLite connection where query results can be accessed both by 
        index (row[0]) and column name (row['name']), making data handling more 
        flexible and readable.
    
        Returns:
        --------
        sqlite3.Connection
            Connection to the SQLite database with Row factory configured
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  
        return conn
        
    def _find_matching_album_id(
        self, 
        artist_name: str, 
        album_name: str, 
        id_map: Dict[str, Dict]
    ) -> Optional[int]:
        """
        Try various methods to match an album name to its ID.
        
        Parameters:
        -----------
        artist_name : str
            Name of the artist
        album_name : str
            Name of the album to find
        id_map : dict
            Dictionary mapping entity names to their IDs
            
        Returns:
        --------
        int or None
            Album ID if a match is found, None otherwise
        """
        if not album_name:
            return None
            
        # 1. Try exact match first
        album_key = (artist_name, album_name)
        if album_key in id_map['albums']:
            return id_map['albums'][album_key]
        
        # 2. Try case-insensitive match
        for (a_name, a_title), a_id in id_map['albums'].items():
            if a_name.lower() == artist_name.lower() and a_title.lower() == album_name.lower():
                return a_id
        
        # 3. Try substring matching
        for (a_name, a_title), a_id in id_map['albums'].items():
            if a_name == artist_name:
                # Check if album_name is contained within a_title or vice versa
                if album_name.lower() in a_title.lower() or a_title.lower() in album_name.lower():
                    return a_id
        
        # No match found
        return None
    
    def initialize_database(
        self, 
        schema_file: Optional[str] = None, 
        schema_sql: Optional[str] = None
    ) -> bool:
        """
        Initialize the database with tables defined in a schema file or SQL string.
        
        Parameters:
        -----------
        schema_file : str, optional
            Path to a SQL file containing the database schema
        schema_sql : str, optional
            SQL string containing the database schema
            
        Returns:
        --------
        bool
            True if initialization was successful, False otherwise
        """
        conn = self._get_connection()
        
        try:
            if schema_file:
                # Read schema from file
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
            
            if schema_sql:
                conn.executescript(schema_sql)
                conn.commit()
                print("Database initialized successfully")
                return True
            else:
                print("No schema provided for initialization")
                return False
                
        except Exception as e:
            print(f"Error initializing database: {e}")
            return False
        finally:
            conn.close()
    
    def _insert_entity(
        self, 
        conn: sqlite3.Connection, 
        table_name: str, 
        data: Dict[str, Any], 
        id_field: str, 
        check_fields: List[str]
    ) -> Optional[int]:
        """
        Generic method to insert an entity into the database with proper error handling.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        table_name : str
            Name of the table to insert into
        data : dict
            Data to insert
        id_field : str
            Name of the ID field for the table
        check_fields : list
            Fields to use for checking if the entity already exists
            
        Returns:
        --------
        int or None
            ID of the inserted or existing entity, or None if insertion failed
        """
        try:
            cursor = conn.cursor()
            
            # Build the check query
            check_conditions = " AND ".join([f"{field} = ?" for field in check_fields])
            check_values = [data[field] for field in check_fields]
            
            # Check if entity already exists
            cursor.execute(
                f"SELECT {id_field} FROM {table_name} WHERE {check_conditions}",
                check_values
            )
            existing = cursor.fetchone()
            
            if existing:
                return existing[id_field]
            
            # Build the insert query
            fields = [k for k in data.keys() if k in data]
            placeholders = ", ".join(["?"] * len(fields))
            field_names = ", ".join(fields)
            
            # Insert new entity
            cursor.execute(
                f"INSERT INTO {table_name} ({field_names}) VALUES ({placeholders})",
                [data.get(field, None) for field in fields]
            )
            
            return cursor.lastrowid
            
        except Exception as e:
            entity_name = data.get('name', 'unknown')
            print(f"Error inserting {table_name} {entity_name}: {e}")
            return None
    
    def _insert_relationship(
        self, 
        conn: sqlite3.Connection, 
        table_name: str, 
        data: Dict[str, Any], 
        key_fields: List[str]
    ) -> bool:
        """
        Generic method to insert or update a relationship with proper error handling.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        table_name : str
            Name of the relationship table
        data : dict
            Relationship data to insert
        key_fields : list
            Fields that make up the primary key
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            cursor = conn.cursor()
            
            # Build the check query
            check_conditions = " AND ".join([f"{field} = ?" for field in key_fields])
            check_values = [data[field] for field in key_fields]
            
            # Check if relationship already exists
            cursor.execute(
                f"SELECT 1 FROM {table_name} WHERE {check_conditions}",
                check_values
            )
            
            if cursor.fetchone():
                # Build update query for non-key fields
                update_fields = [f for f in data.keys() if f not in key_fields]
                if update_fields:
                    update_stmt = ", ".join([f"{field} = ?" for field in update_fields])
                    update_values = [data[field] for field in update_fields]
                    
                    # Add key values for WHERE clause
                    where_values = [data[field] for field in key_fields]
                    
                    cursor.execute(
                        f"UPDATE {table_name} SET {update_stmt} WHERE {check_conditions}",
                        update_values + where_values
                    )
            else:
                # Build insert query
                fields = list(data.keys())
                placeholders = ", ".join(["?"] * len(fields))
                field_names = ", ".join(fields)
                
                cursor.execute(
                    f"INSERT INTO {table_name} ({field_names}) VALUES ({placeholders})",
                    [data[field] for field in fields]
                )
            
            return True
            
        except Exception as e:
            print(f"Error inserting/updating {table_name} relationship: {e}")
            return False
    
    def insert_artist(
        self, 
        conn: sqlite3.Connection, 
        artist_data: Dict[str, Any]
    ) -> Optional[int]:
        """
        Insert artist data and return the artist_id.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        artist_data : dict
            Artist data to insert
            
        Returns:
        --------
        int or None
            ID of the inserted artist, or None if insertion failed
        """
        return self._insert_entity(
            conn, 
            "artists", 
            artist_data, 
            "artist_id", 
            ["name"]
        )
    
    def insert_album(
        self, 
        conn: sqlite3.Connection, 
        album_data: Dict[str, Any], 
        artist_id: int
    ) -> Optional[int]:
        """
        Insert album data and return the album_id.

        # ... (docstring as before) ...
        """
        # Ensure artist_id is included in data
        album_data['artist_id'] = artist_id

        # Remove the 'artist' key if it exists
        if 'artist' in album_data:
            del album_data['artist']

        return self._insert_entity(
            conn, 
            "albums", 
            album_data, 
            "album_id", 
            ["name", "artist_id"]
        )
      
    def insert_track(
      self, 
      conn: sqlite3.Connection, 
      track_data: Dict[str, Any]
  ) -> Optional[int]:
      """
      Insert track data and return the track_id.

      Parameters:
      -----------
      conn : sqlite3.Connection
          Database connection
      track_data : dict
          Track data to insert, must include artist_id and optionally album_id

      Returns:
      --------
      int or None
          ID of the inserted track, or None if insertion failed
      """
      return self._insert_entity(
          conn, 
          "tracks", 
          track_data, 
          "track_id", 
          ["name", "artist_id"]
      )

    def insert_tag(
        self, 
        conn: sqlite3.Connection, 
        tag_data: Dict[str, Any]
    ) -> Optional[int]:
        """
        Insert tag data and return the tag_id.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        tag_data : dict
            Tag data to insert
            
        Returns:
        --------
        int or None
            ID of the inserted tag, or None if insertion failed
        """
        return self._insert_entity(
            conn, 
            "tags", 
            tag_data, 
            "tag_id", 
            ["name"]
        )
    
    def insert_listening_history(self, conn: sqlite3.Connection, history_data: Dict[str, Any], 
                               track_id: int) -> Optional[int]:
        """Insert listening history record with better duplicate checking."""
        try:
            cursor = conn.cursor()

            # Convert Unix timestamp to SQLite datetime format
            from datetime import datetime
            listened_at_datetime = datetime.fromtimestamp(history_data['listened_at']).strftime('%Y-%m-%d %H:%M:%S')

            # Look for exact match on both track_id and timestamp
            cursor.execute(
                "SELECT history_id FROM user_listening_history WHERE track_id = ? AND listened_at = ?",
                (track_id, listened_at_datetime)  # Use the converted datetime
            )
            existing = cursor.fetchone()

            if existing:
                # Return existing ID if we already have this exact scrobble
                return existing[0]

            # Otherwise proceed with insertion
            cursor.execute('''
                INSERT INTO user_listening_history 
                (track_id, listened_at, is_now_playing)
                VALUES (?, ?, ?)
            ''', (
                track_id,
                listened_at_datetime,  # Use the converted datetime
                history_data['is_now_playing']
            ))

            return cursor.lastrowid

        except Exception as e:
            print(f"Error inserting listening history: {e}")
            return None
    
    def insert_artist_tag(
        self, 
        conn: sqlite3.Connection, 
        artist_id: int, 
        tag_id: int, 
        count: int
    ) -> bool:
        """
        Insert artist-tag relationship.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        artist_id : int
            ID of the artist
        tag_id : int
            ID of the tag
        count : int
            Count/weight of the tag for this artist
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        return self._insert_relationship(
            conn,
            "artist_tags",
            {"artist_id": artist_id, "tag_id": tag_id, "count": count},
            ["artist_id", "tag_id"]
        )
    
    def insert_album_tag(
        self, 
        conn: sqlite3.Connection, 
        album_id: int, 
        tag_id: int, 
        count: int
    ) -> bool:
        """
        Insert album-tag relationship.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        album_id : int
            ID of the album
        tag_id : int
            ID of the tag
        count : int
            Count/weight of the tag for this album
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        return self._insert_relationship(
            conn,
            "album_tags",
            {"album_id": album_id, "tag_id": tag_id, "count": count},
            ["album_id", "tag_id"]
        )
    
    def insert_track_tag(
        self, 
        conn: sqlite3.Connection, 
        track_id: int, 
        tag_id: int, 
        count: int
    ) -> bool:
        """
        Insert track-tag relationship.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        track_id : int
            ID of the track
        tag_id : int
            ID of the tag
        count : int
            Count/weight of the tag for this track
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        return self._insert_relationship(
            conn,
            "track_tags",
            {"track_id": track_id, "tag_id": tag_id, "count": count},
            ["track_id", "tag_id"]
        )
    
    def insert_artist_similar(
        self, 
        conn: sqlite3.Connection, 
        artist_id: int, 
        similar_artist_id: int, 
        match_score: float
    ) -> bool:
        """
        Insert artist similarity relationship.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        artist_id : int
            ID of the artist
        similar_artist_id : int
            ID of the similar artist
        match_score : float
            Similarity score between 0 and 1
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        return self._insert_relationship(
            conn,
            "artist_similar",
            {
                "artist_id": artist_id, 
                "similar_artist_id": similar_artist_id, 
                "match_score": match_score
            },
            ["artist_id", "similar_artist_id"]
        )
    
    def insert_track_similar(
        self, 
        conn: sqlite3.Connection, 
        track_id: int, 
        similar_track_id: int, 
        match_score: float
    ) -> bool:
        """
        Insert track similarity relationship.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        track_id : int
            ID of the track
        similar_track_id : int
            ID of the similar track
        match_score : float
            Similarity score between 0 and 1
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        return self._insert_relationship(
            conn,
            "track_similar",
            {
                "track_id": track_id, 
                "similar_track_id": similar_track_id, 
                "match_score": match_score
            },
            ["track_id", "similar_track_id"]
        )

    def insert_user_top_entity(
        self, 
        conn: sqlite3.Connection, 
        table_name: str, 
        entity_id_field: str, 
        entity_id: int, 
        time_period: str, 
        rank: int, 
        playcount: int
    ) -> Optional[int]:
        """
        Insert a user's top entity (artist, album, or track).

        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        table_name : str
            Name of the top entity table (should be singular: "artist", "album", "track")
        entity_id_field : str
            Name of the entity ID field
        entity_id : int
            ID of the entity
        time_period : str
            Time period for the ranking
        rank : int
            Ranking position
        playcount : int
            Number of plays in the time period

        Returns:
        --------
        int or None
            ID of the inserted record, or None if insertion failed
        """
        # Fix: Table name and ID field should use singular form
        table = f"user_top_{table_name}s"  # Pluralize for table name
        id_field = f"user_top_{table_name}_id"  # Keep singular for ID field

        data = {
            entity_id_field: entity_id,
            "time_period": time_period,
            "rank": rank,
            "playcount": playcount
        }
        return self._insert_entity(
            conn,
            table,
            data,
            id_field,
            [entity_id_field, "time_period"]
        )
    
    def insert_user_top_artist(
        self, 
        conn: sqlite3.Connection, 
        artist_id: int, 
        time_period: str, 
        rank: int, 
        playcount: int
    ) -> Optional[int]:
        """
        Insert user's top artist.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        artist_id : int
            ID of the artist
        time_period : str
            Time period for the ranking
        rank : int
            Ranking position
        playcount : int
            Number of plays in the time period
            
        Returns:
        --------
        int or None
            ID of the inserted record, or None if insertion failed
        """
        return self.insert_user_top_entity(
            conn, "artist", "artist_id", artist_id, time_period, rank, playcount
        )
    
    def insert_user_top_album(
        self, 
        conn: sqlite3.Connection, 
        album_id: int, 
        time_period: str, 
        rank: int, 
        playcount: int
    ) -> Optional[int]:
        """
        Insert user's top album.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        album_id : int
            ID of the album
        time_period : str
            Time period for the ranking
        rank : int
            Ranking position
        playcount : int
            Number of plays in the time period
            
        Returns:
        --------
        int or None
            ID of the inserted record, or None if insertion failed
        """
        return self.insert_user_top_entity(
            conn, "album", "album_id", album_id, time_period, rank, playcount
        )
    
    def insert_user_top_track(
        self, 
        conn: sqlite3.Connection, 
        track_id: int, 
        time_period: str, 
        rank: int, 
        playcount: int
    ) -> Optional[int]:
        """
        Insert user's top track.
        
        Parameters:
        -----------
        conn : sqlite3.Connection
            Database connection
        track_id : int
            ID of the track
        time_period : str
            Time period for the ranking
        rank : int
            Ranking position
        playcount : int
            Number of plays in the time period
            
        Returns:
        --------
        int or None
            ID of the inserted record, or None if insertion failed
        """
        return self.insert_user_top_entity(
            conn, "track", "track_id", track_id, time_period, rank, playcount
        )
    
    def process_collected_data(
        self, 
        collected_data: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[Dict[str, int]]:
        """
        Process collected data and insert into database.
        
        This method handles the entire ETL process for loading collected Last.fm data
        into the database. It establishes relationships between entities and ensures
        proper transaction management.
        
        Parameters:
        -----------
        collected_data : dict
            Dictionary containing all collected Last.fm data
            
        Returns:
        --------
        dict or None
            Statistics of inserted data counts, or None if processing failed
        """
        print("process_collected_data called")
        conn = self._get_connection()
        
        try:
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Dictionary to store entity IDs for relationship building
            id_map = {
                'artists': {},
                'albums': {},
                'tracks': {},
                'tags': {}
            }
            
            # 1. Insert Tags first (they're referenced by other entities)
            print("Inserting tags...")
            for tag_data in collected_data['tags']:
                tag_id = self.insert_tag(conn, tag_data)
                if tag_id:
                    id_map['tags'][tag_data['name']] = tag_id
            
            # 2. Insert Artists
            print("Inserting artists...")
            for artist_data in collected_data['artists']:
                artist_id = self.insert_artist(conn, artist_data)
                if artist_id:
                    id_map['artists'][artist_data['name']] = artist_id
            
            # 3. Insert Albums (needs artist IDs)
            #print("Inserting albums...")
            #for album_data in collected_data['albums']:
            #    artist_name = album_data['artist']
            #    if artist_name in id_map['artists']:
            #        artist_id = id_map['artists'][artist_name]
            #        album_id = self.insert_album(conn, album_data, artist_id)
            #        if album_id:
            #            id_map['albums'][(artist_name, album_data['name'])] = album_id
            # 3. Insert Albums (needs artist IDs)
            print("Inserting albums...")
            for album_data in collected_data['albums']:
                artist_name = album_data['artist']
                if artist_name in id_map['artists']:
                    artist_id = id_map['artists'][artist_name]

                    # Add these logging lines
                    #print("Album data about to be inserted:", album_data)
                    logging.info(f"Album data about to be inserted: {album_data}")

                    album_id = self.insert_album(conn, album_data, artist_id)
                    if album_id:
                        id_map['albums'][(artist_name, album_data['name'])] = album_id
                    else:
                        print(f"Error inserting albums {album_data['name']}")
                        logging.error(f"Error inserting albums {album_data['name']}")
            
            # 4. Insert Tracks (needs artist and album IDs)
            print("Inserting tracks...")
            for track_data in collected_data['tracks']:
                artist_name = track_data['artist']
                if artist_name in id_map['artists']:
                    artist_id = id_map['artists'][artist_name]

                    # Create database-ready track data
                    db_track_data = {
                        "name": track_data["name"],
                        "artist_id": artist_id,
                        "mbid": track_data.get("mbid", ""),
                        "url": track_data.get("url", ""),
                        "duration": track_data.get("duration", 0),
                        "listeners": track_data.get("listeners", 0),
                        "playcount": track_data.get("playcount", 0)
                    }

                    # Try to find album ID with flexible matching
                    album_name = track_data.get('album_name', '')
                    album_id = self._find_matching_album_id(artist_name, album_name, id_map)
                    if album_id:
                        db_track_data["album_id"] = album_id
                        print(f"Found album ID {album_id} for '{track_data['name']}' - album '{album_name}'")
                    else:
                        print(f"No matching album found for '{track_data['name']}' - album '{album_name}'")

                    track_id = self.insert_track(conn, db_track_data)
                    if track_id:
                        id_map['tracks'][(artist_name, track_data['name'])] = track_id

                    # Create database-ready track data
                    db_track_data = {
                        "name": track_data["name"],
                        "artist_id": artist_id,
                        "mbid": track_data.get("mbid", ""),
                        "url": track_data.get("url", ""),
                        "duration": track_data.get("duration", 0),
                        "listeners": track_data.get("listeners", 0),
                        "playcount": track_data.get("playcount", 0)
                    }

                    # Try to find album ID if available
                    if album_name != 'NO_ALBUM' and (artist_name, album_name) in id_map['albums']:
                        db_track_data["album_id"] = id_map['albums'][(artist_name, album_name)]

                    track_id = self.insert_track(conn, db_track_data)
                    if track_id:
                        id_map['tracks'][(artist_name, track_data['name'])] = track_id
            
            # 5. Insert Relationships
            # 5.1 Artist-Tag relationships
            print("Inserting artist-tag relationships...")
            for rel in collected_data['artist_tags']:
                if (rel['artist_name'] in id_map['artists'] and 
                        rel['tag_name'] in id_map['tags']):
                    self.insert_artist_tag(
                        conn, 
                        id_map['artists'][rel['artist_name']], 
                        id_map['tags'][rel['tag_name']], 
                        rel['count']
                    )
            
            # 5.2 Album-Tag relationships
            print("Inserting album-tag relationships...")
            for rel in collected_data['album_tags']:
                album_key = (rel['artist_name'], rel['album_name'])
                if album_key in id_map['albums'] and rel['tag_name'] in id_map['tags']:
                    self.insert_album_tag(
                        conn, 
                        id_map['albums'][album_key], 
                        id_map['tags'][rel['tag_name']], 
                        rel['count']
                    )
            
            # 5.3 Track-Tag relationships
            print("Inserting track-tag relationships...")
            for rel in collected_data['track_tags']:
                track_key = (rel['artist_name'], rel['track_name'])
                if track_key in id_map['tracks'] and rel['tag_name'] in id_map['tags']:
                    self.insert_track_tag(
                        conn, 
                        id_map['tracks'][track_key], 
                        id_map['tags'][rel['tag_name']], 
                        rel['count']
                    )
            
            # 5.4 Artist similarity relationships
            print("Inserting artist similarity relationships...")
            for rel in collected_data['artist_similar']:
                if (rel['artist_name'] in id_map['artists'] and 
                        rel['similar_artist'] in id_map['artists']):
                    self.insert_artist_similar(
                        conn, 
                        id_map['artists'][rel['artist_name']], 
                        id_map['artists'][rel['similar_artist']], 
                        rel['match_score']
                    )
            
            # 5.5 Track similarity relationships
            print("Inserting track similarity relationships...")
            for rel in collected_data['track_similar']:
                track_key = (rel['artist_name'], rel['track_name'])
                similar_key = (rel['similar_artist'], rel['similar_track'])
                
                if track_key in id_map['tracks'] and similar_key in id_map['tracks']:
                    self.insert_track_similar(
                        conn, 
                        id_map['tracks'][track_key], 
                        id_map['tracks'][similar_key], 
                        rel['match_score']
                    )
            
            # 6. Insert user's listening history
            print("Inserting listening history...")
            for history_item in collected_data['listening_history']:
                track_key = (history_item['artist_name'], history_item['track_name'])
                if track_key in id_map['tracks']:
                    self.insert_listening_history(
                        conn,
                        history_item,
                        id_map['tracks'][track_key]
                    )
            
            # 7. Insert user's top entities
            # 7.1 Top artists
            print("Inserting top artists...")
            for top_artist in collected_data['top_artists']:
                if top_artist['artist_name'] in id_map['artists']:
                    self.insert_user_top_artist(
                        conn,
                        id_map['artists'][top_artist['artist_name']],
                        top_artist['time_period'],
                        top_artist['rank'],
                        top_artist['playcount']
                    )
            
            # 7.2 Top albums
            print("Inserting top albums...")
            for top_album in collected_data['top_albums']:
                album_key = (top_album['artist_name'], top_album['album_name'])
                if album_key in id_map['albums']:
                    self.insert_user_top_album(
                        conn,
                        id_map['albums'][album_key],
                        top_album['time_period'],
                        top_album['rank'],
                        top_album['playcount']
                    )
            
            # 7.3 Top tracks
            print("Inserting top tracks...")
            for top_track in collected_data['top_tracks']:
                track_key = (top_track['artist_name'], top_track['track_name'])
                if track_key in id_map['tracks']:
                    self.insert_user_top_track(
                        conn,
                        id_map['tracks'][track_key],
                        top_track['time_period'],
                        top_track['rank'],
                        top_track['playcount']
                    )
            
            # Commit all changes
            conn.commit()
            print("All data successfully inserted into database")
            
            # Return stats
            return {
                'tags': len(id_map['tags']),
                'artists': len(id_map['artists']),
                'albums': len(id_map['albums']),
                'tracks': len(id_map['tracks']),
                'history_items': len(collected_data['listening_history']),
            }
            
        except Exception as e:
            conn.rollback()
            print(f"Error processing collected data: {e}")
            return None
            
        finally:
            conn.close()