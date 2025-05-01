-- Schema for Last.fm Database

-- Artists table
CREATE TABLE IF NOT EXISTS artists (
    artist_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    mbid TEXT,
    url TEXT,
    image_small TEXT,
    image_medium TEXT,
    image_large TEXT,
    listeners INTEGER,
    playcount INTEGER,
    bio_summary TEXT,
    bio_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Albums table
CREATE TABLE IF NOT EXISTS albums (
    album_id INTEGER PRIMARY KEY,
    artist_id INTEGER,  -- Foreign Key to artists table
    name TEXT NOT NULL,
    mbid TEXT,
    url TEXT,
    image_small TEXT,
    image_medium TEXT,
    image_large TEXT,
    listeners INTEGER,
    playcount INTEGER,
    release_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artist_id) REFERENCES artists (artist_id)
);


-- Tracks table
CREATE TABLE IF NOT EXISTS tracks (
    track_id INTEGER PRIMARY KEY,
    artist_id INTEGER,  -- Foreign Key to artists table
    album_id INTEGER,   -- Foreign Key to albums table
    name TEXT NOT NULL,
    mbid TEXT,
    url TEXT,
    duration INTEGER,
    listeners INTEGER,
    playcount INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
    FOREIGN KEY (album_id) REFERENCES albums (album_id)
);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT,
    reach INTEGER,
    taggings INTEGER,
    wiki_summary TEXT,
    wiki_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Artist Tags junction table
CREATE TABLE IF NOT EXISTS artist_tags (
    artist_id INTEGER,
    tag_id INTEGER,
    count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (artist_id, tag_id),
    FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
    FOREIGN KEY (tag_id) REFERENCES tags (tag_id)
);

-- Album Tags junction table
CREATE TABLE IF NOT EXISTS album_tags (
    album_id INTEGER,
    tag_id INTEGER,
    count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (album_id, tag_id),
    FOREIGN KEY (album_id) REFERENCES albums (album_id),
    FOREIGN KEY (tag_id) REFERENCES tags (tag_id)
);

-- Track Tags junction table
CREATE TABLE IF NOT EXISTS track_tags (
    track_id INTEGER,
    tag_id INTEGER,
    count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (track_id, tag_id),
    FOREIGN KEY (track_id) REFERENCES tracks (track_id),
    FOREIGN KEY (tag_id) REFERENCES tags (tag_id)
);

-- Artist Similar junction table
CREATE TABLE IF NOT EXISTS artist_similar (
    artist_id INTEGER,
    similar_artist_id INTEGER,
    match_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (artist_id, similar_artist_id),
    FOREIGN KEY (artist_id) REFERENCES artists (artist_id),
    FOREIGN KEY (similar_artist_id) REFERENCES artists (artist_id)
);

-- Track Similar junction table
CREATE TABLE IF NOT EXISTS track_similar (
    track_id INTEGER,
    similar_track_id INTEGER,
    match_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (track_id, similar_track_id),
    FOREIGN KEY (track_id) REFERENCES tracks (track_id),
    FOREIGN KEY (similar_track_id) REFERENCES tracks (track_id)
);

-- User Listening History table
CREATE TABLE IF NOT EXISTS user_listening_history (
    history_id INTEGER PRIMARY KEY,
    track_id INTEGER,
    listened_at TIMESTAMP,
    is_now_playing BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks (track_id)
);

-- User Top Artists table
CREATE TABLE IF NOT EXISTS user_top_artists (
    user_top_artist_id INTEGER PRIMARY KEY,
    artist_id INTEGER,
    time_period TEXT,
    rank INTEGER,
    playcount INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artist_id) REFERENCES artists (artist_id)
);

-- User Top Albums table
CREATE TABLE IF NOT EXISTS user_top_albums (
    user_top_album_id INTEGER PRIMARY KEY,
    album_id INTEGER,
    time_period TEXT,
    rank INTEGER,
    playcount INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (album_id) REFERENCES albums (album_id)
);

-- User Top Tracks table
CREATE TABLE IF NOT EXISTS user_top_tracks (
    user_top_track_id INTEGER PRIMARY KEY,
    track_id INTEGER,
    time_period TEXT,
    rank INTEGER,
    playcount INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks (track_id)
);