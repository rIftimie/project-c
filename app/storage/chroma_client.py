from chromadb import HttpClient, Settings
import time
import sys
from tabulate import tabulate
import json
import argparse

def get_chroma_client(max_retries=3, retry_delay=2):
    """Create and return a ChromaDB client connected to the local container."""
    for attempt in range(max_retries):
        try:
            settings = Settings(
                chroma_server_host="0.0.0.0",
                chroma_server_http_port=8000,
                anonymized_telemetry=False
            )
            
            client = HttpClient(
                host="0.0.0.0",
                port=8000,
                settings=settings
            )
            # Test the connection
            client.heartbeat()
            print("Successfully connected to ChromaDB")
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect to ChromaDB after {max_retries} attempts.")
                print(f"Error: {str(e)}")
                print("\nPlease ensure that:")
                print("1. The Docker container is running (docker-compose up -d)")
                print("2. Port 8000 is available and not blocked")
                print("3. The environment variables are correctly set")
                sys.exit(1)

def list_collections(client):
    """List all collections in the database."""
    try:
        collections = client.list_collections()
        if not collections:
            print("No collections found in the database.")
        else:
            print("\nExisting collections:")
            for idx, collection in enumerate(collections, 1):
                print(f"{idx}. {collection.name}")
        return collections
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        raise

def get_collection_stats(collection):
    """Get basic statistics about a collection."""
    try:
        count = collection.count()
        return {
            "name": collection.name,
            "total_documents": count
        }
    except Exception as e:
        print(f"Error getting collection stats: {str(e)}")
        return None

def display_collection_contents(collection, limit=10):
    """Display the contents of the collection with pagination."""
    try:
        # Get total count
        total = collection.count()
        if total == 0:
            print("\nCollection is empty.")
            return

        print(f"\nTotal documents in collection: {total}")
        
        # Query all documents with their metadata
        results = collection.get(
            limit=limit,
            include=['metadatas', 'documents', 'embeddings']
        )

        if not results['ids']:
            print("No results found.")
            return

        # Prepare table data
        table_data = []
        for idx, (doc_id, metadata, text) in enumerate(zip(
            results['ids'], 
            results['metadatas'], 
            results['documents']
        ), 1):
            # Truncate text if too long
            truncated_text = text[:100] + "..." if len(text) > 100 else text
            
            # Format metadata for display
            metadata_str = json.dumps(metadata, indent=2)
            
            table_data.append([
                idx,
                doc_id,
                truncated_text,
                metadata_str
            ])

        # Display results in a table
        print("\nSample of collection contents (first {limit} documents):")
        print(tabulate(
            table_data,
            headers=['#', 'ID', 'Text Content', 'Metadata'],
            tablefmt='grid'
        ))

        if total > limit:
            print(f"\nShowing {limit} of {total} total documents.")
            print("Use a different limit to see more documents.")

    except Exception as e:
        print(f"Error displaying collection contents: {str(e)}")
        raise

def analyze_collection(client, collection_name="project_c"):
    """Analyze and display detailed information about a specific collection."""
    try:
        # Try to get the collection, create it if it doesn't exist
        try:
            collection = client.get_collection(collection_name)
            print(f"\nFound existing collection: {collection_name}")
        except Exception as e:
            print(f"\nCollection {collection_name} not found. Creating new collection...")
            collection = client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Get basic stats
        stats = get_collection_stats(collection)
        if stats:
            print("\nCollection Statistics:")
            print(f"Name: {stats['name']}")
            print(f"Total Documents: {stats['total_documents']}")

        # Display sample contents
        display_collection_contents(collection)

    except Exception as e:
        print(f"Error analyzing collection: {str(e)}")
        raise

def search_collection(collection, query, n_results=10):
    """Search the collection for relevant content."""
    try:
        print(f"\nSearching for: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents"]
        )

        if not results["documents"][0]:
            print("No results found.")
            return

        print(f"\nFound {len(results['documents'][0])} relevant segments:\n")
        
        # Prepare table data
        table_data = []
        for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            # Format timestamp range
            time_range = f"{meta['start']:.1f}s–{meta['end']:.1f}s"
            
            # Create YouTube timestamp URL
            video_url = meta.get('url', '')
            if video_url:
                timestamp_url = f"{video_url}&t={int(meta['start'])}"
            else:
                timestamp_url = "N/A"
            
            table_data.append([
                i + 1,
                meta.get('title', 'Unknown'),
                time_range,
                text,
                timestamp_url
            ])

        # Display results in a table
        print(tabulate(
            table_data,
            headers=['#', 'Video Title', 'Time Range', 'Content', 'Link'],
            tablefmt='grid',
            maxcolwidths=[5, 30, 15, 50, 30]
        ))

    except Exception as e:
        print(f"Error searching collection: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='ChromaDB Client Tool')
    parser.add_argument('--search', type=str, help='Search query for the collection')
    parser.add_argument('--limit', type=int, default=10, help='Number of results to return')
    args = parser.parse_args()

    # Connect to the Chroma server
    client = get_chroma_client()
    
    if args.search:
        # If search query provided, perform search
        collection = client.get_collection("project_c")
        search_collection(collection, args.search, args.limit)
    else:
        # Otherwise show collection info
        print("\nListing all collections:")
        list_collections(client)
        
        print("\nAnalyzing project_c collection:")
        analyze_collection(client)

if __name__ == "__main__":
    main()