import psycopg2
from psycopg2.extras import execute_values
import logging
from datetime import datetime

# Database config - matches docker-compose.yml settings
DB_CONFIG = {
    "dbname": "think_bro",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

logger = logging.getLogger(__name__)

def get_conn():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.Error as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def insert_video_metadata(meta):
    if not meta:
        logger.warning("Received empty metadata, skipping insertion")
        return
        
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO videos (id, channel_id, title, description, upload_date, duration, 
                                    view_count, like_count, categories, language, url, extracted_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    view_count = EXCLUDED.view_count,
                    like_count = EXCLUDED.like_count,
                    extracted_at = EXCLUDED.extracted_at;
            """, (
                meta.get("id"),
                meta.get("channel_id"),
                meta.get("title", ""),
                meta.get("description", ""),
                meta.get("upload_date"),
                meta.get("duration", 0),
                meta.get("view_count", 0),
                meta.get("like_count", 0),
                meta.get("categories", []),
                meta.get("language", "en"),
                meta.get("webpage_url", ""),
                datetime.utcnow()
            ))
        logger.info(f"Inserted video metadata for {meta.get('id')}")

    except psycopg2.Error as e:
        logger.error(f"Failed to insert video metadata: {e}")
        raise

def insert_transcript_chunks(chunks):
    if not chunks:
        logger.warning("Received empty chunks list, skipping insertion")
        return
        
    try:
        with get_conn() as conn, conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO transcript_chunks 
                    (id, video_id, channel_id, chunk_index, start_time, end_time, text)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    text = EXCLUDED.text;
            """, chunks)
        logger.info(f"Inserted {len(chunks)} transcript chunks")
    except psycopg2.Error as e:
        logger.error(f"Failed to insert transcript chunks: {e}")
        raise

def insert_channel_metadata(channel_data):
    if not channel_data:
        logger.warning("Received empty channel data, skipping insertion")
        return
        
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO channels (id, title, description, url, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    url = EXCLUDED.url;
            """, (
                channel_data.get("id"),
                channel_data.get("title", ""),
                channel_data.get("description", ""),
                channel_data.get("url", ""),
                datetime.utcnow()
            ))
        logger.info(f"Inserted/updated channel metadata for {channel_data.get('id')}")

    except psycopg2.Error as e:
        logger.error(f"Failed to insert channel metadata: {e}")
        raise

def check_video_metadata(video_id):
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT extracted_at 
                FROM videos 
                WHERE id = %s
            """, (video_id,))
            result = cur.fetchone()
            return result[0] if result else None
    except psycopg2.Error as e:
        logger.error(f"Failed to check video metadata: {e}")
        raise

def check_channel_metadata(channel_id):
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT created_at 
                FROM channels 
                WHERE id = %s
            """, (channel_id,))
            result = cur.fetchone()
            return result[0] if result else None
    except psycopg2.Error as e:
        logger.error(f"Failed to check channel metadata: {e}")
        raise
