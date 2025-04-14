CREATE TABLE channels (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    url TEXT,
    created_at TIMESTAMP
);

CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT REFERENCES channels(id),
    title TEXT,
    description TEXT,
    upload_date DATE,
    duration INT,
    view_count BIGINT,
    like_count BIGINT,
    categories TEXT[],
    language TEXT,
    url TEXT,
    extracted_at TIMESTAMP
);

CREATE TABLE transcript_chunks (
    id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES videos(id),
    channel_id TEXT REFERENCES channels(id),
    chunk_index INT,
    start_time FLOAT,
    end_time FLOAT,
    text TEXT
);

