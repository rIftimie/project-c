import os
import json
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm
from datetime import datetime, timedelta
from ..storage.db_client import (
    insert_video_metadata, 
    insert_transcript_chunks, 
    insert_channel_metadata, 
    check_video_metadata,
    check_channel_metadata
)

# === CONFIG ===
DATA_DIR = "data"
CHROMA_HOST = "0.0.0.0"  # Docker container host
CHROMA_PORT = "8000"      # Docker container port
COLLECTION_NAME = "project_c"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 100  # Number of words per chunk

# Ensure logs directory exists
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# === SETUP LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "embedding_progress.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === SETUP EMBEDDING MODEL ===
logger.info(f"Loading embedding model: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

# === SETUP CHROMA CLIENT ===
logger.info(f"Connecting to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}")
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# === PROCESS ALL TRANSCRIPTS ===
def process_channel(channel_id):
    channel_dir = os.path.join(DATA_DIR, channel_id)
    transcriptions_dir = os.path.join(channel_dir, "transcriptions")
    metadata_dir = os.path.join(channel_dir, "metadata")
    channel_metadata_dir = os.path.join(channel_dir, "channel_info")
    
    if not os.path.exists(transcriptions_dir) or not os.path.exists(metadata_dir):
        logger.warning(f"Missing required directories for channel {channel_id}")
        return
    
    # Check if channel was recently processed (within last 24 hours)
    last_processed = check_channel_metadata(channel_id)
    channel_needs_update = True
    
    if last_processed and datetime.utcnow() - last_processed < timedelta(hours=24):
        logger.info(f"SKIPPING channel {channel_id} - already processed within last 24 hours")
        channel_needs_update = False
    
    # Process channel metadata if needed
    if channel_needs_update:
        channel_meta_path = os.path.join(channel_metadata_dir, "channel_metadata.json")
        if os.path.exists(channel_meta_path):
            try:
                with open(channel_meta_path) as f:
                    channel_meta = json.load(f)
                    channel_meta["id"] = channel_id
                    insert_channel_metadata(channel_meta)
            except Exception as e:
                logger.error(f"Error processing channel metadata for {channel_id}: {e}")
        else:
            logger.warning(f"No channel metadata file found for {channel_id}")
            insert_channel_metadata({"id": channel_id})
    
    transcript_files = [f for f in os.listdir(transcriptions_dir) if f.endswith(".json")]
    logger.info(f"Found {len(transcript_files)} transcript files for channel {channel_id}")
    
    for filename in tqdm(transcript_files, desc=f"Processing {channel_id}"):
        video_id = filename.replace(".json", "")
        logger.info(f"Processing video: {video_id}")
        
        # Check if video was recently processed (within last 24 hours)
        last_processed = check_video_metadata(video_id)
        if last_processed and datetime.utcnow() - last_processed < timedelta(hours=24):
            logger.info(f"SKIPPING video {video_id} - already processed within last 24 hours")
            continue

        # Load metadata first
        metadata_path = os.path.join(metadata_dir, video_id + ".json")
        if not os.path.exists(metadata_path):
            logger.warning(f"MISSING metadata for {video_id}, skipping.")
            continue
            
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
                meta["channel_id"] = channel_id
                meta["id"] = video_id
        except Exception as e:
            logger.error(f"Error loading metadata for {video_id}: {e}")
            continue
            
        # Insert metadata into Postgres
        try:
            insert_video_metadata(meta)
        except Exception as e:
            logger.error(f"Failed to insert metadata for {video_id}: {e}")
            continue
        
        # Load transcript
        transcript_path = os.path.join(transcriptions_dir, filename)
        try:
            with open(transcript_path) as f:
                transcript = json.load(f)
        except Exception as e:
            logger.error(f"Error loading transcript for {video_id}: {e}")
            continue
        
        segments = transcript["segments"]
        chunks = []
        ids = []
        metadatas = []
        texts = []
        
        current_chunk = ""
        start_time = None
        
        for i, seg in enumerate(segments):
            if not start_time:
                start_time = seg["start"]
            current_chunk += seg["text"].strip() + " "
            
            if len(current_chunk.split()) >= CHUNK_SIZE or i == len(segments) - 1:
                end_time = seg["end"]
                texts.append(current_chunk.strip())
                metadatas.append({
                    "video_id": video_id,
                    "channel_id": channel_id,
                    "start": float(start_time),
                    "end": float(end_time),
                    "title": meta.get("title", ""),
                    "channel": meta.get("uploader", ""),
                    "published": meta.get("upload_date", ""),
                    "url": meta.get("webpage_url", "")
                })
                ids.append(f"{channel_id}-{video_id}-{i}")
                current_chunk = ""
                start_time = None
        
        # Store into Postgres transcript_chunks
        pg_chunks = []
        for idx, (txt, meta_data) in enumerate(zip(texts, metadatas)):
            pg_chunks.append((
                ids[idx],
                meta_data["video_id"],
                meta_data["channel_id"],
                idx,
                meta_data["start"],
                meta_data["end"],
                txt
            ))
            
        try:
            insert_transcript_chunks(pg_chunks)
        except Exception as e:
            logger.error(f"Failed to insert transcript chunks for {video_id}: {e}")
            continue
        
        # Store in Chroma
        try:
            # Add in smaller batches to avoid potential memory issues
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                collection.add(
                    documents=texts[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end]
                )
            logger.info(f"Stored {len(texts)} chunks for {video_id}")
        except Exception as e:
            logger.error(f"Error storing chunks in Chroma for {video_id}: {e}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    logger.info("Starting transcript embedding process")
    
    # Process each channel directory
    for channel_id in os.listdir(DATA_DIR):
        channel_path = os.path.join(DATA_DIR, channel_id)
        if os.path.isdir(channel_path):
            logger.info(f"Processing channel: {channel_id}")
            process_channel(channel_id)
    
    logger.info("Transcript embedding process completed")
