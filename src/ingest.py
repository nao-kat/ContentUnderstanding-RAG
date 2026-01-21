"""CLI tool for ingesting videos into the search index."""
import argparse
import sys
from pathlib import Path
from typing import Optional
from .cu_client import ContentUnderstandingClient
from .cu_parse import parse_cu_response
from .normalize import normalize_to_scene_documents, add_embeddings_to_documents
from .embedding import EmbeddingGenerator
from .search_index import SearchIndexManager


def extract_video_id(video_url: str) -> str:
    """Extract a video ID from the URL.
    
    Args:
        video_url: Video URL
    
    Returns:
        Video ID (filename without extension or last path component)
    """
    # Extract filename from URL
    path_part = video_url.split('?')[0]  # Remove query params
    filename = path_part.split('/')[-1]  # Get last component
    
    # Remove extension
    video_id = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    return video_id or "video"


def ingest_video(
    video_url: str,
    video_id: Optional[str] = None,
    enable_embeddings: bool = True
):
    """Ingest a video: analyze with CU, generate embeddings, and index in Search.
    
    Args:
        video_url: URL to the video (Blob SAS URL recommended)
        video_id: Optional video ID (defaults to extracted from URL)
        enable_embeddings: Whether to generate embeddings (requires embedding config)
    """
    print(f"Starting video ingestion: {video_url}")
    
    # Generate video ID if not provided
    if not video_id:
        video_id = extract_video_id(video_url)
    
    print(f"Video ID: {video_id}")
    
    # Step 1: Analyze video with Content Understanding
    print("\n[1/5] Analyzing video with Azure AI Content Understanding...")
    with ContentUnderstandingClient() as cu_client:
        result = cu_client.analyze_video(
            video_url=video_url,
            enable_segment=True,
            return_details=True
        )
    
    print("✓ Video analysis complete")
    
    # Step 2: Parse segments
    print("\n[2/5] Parsing segments from analysis results...")
    segments = parse_cu_response(result)
    
    if not segments:
        print("✗ No segments found in analysis results")
        sys.exit(1)
    
    print(f"✓ Found {len(segments)} segment(s)")
    
    # Step 3: Normalize to scene documents
    print("\n[3/5] Normalizing segments to scene documents...")
    scene_documents = normalize_to_scene_documents(
        segments=segments,
        video_id=video_id,
        video_url=video_url
    )
    
    print(f"✓ Created {len(scene_documents)} scene document(s)")
    
    # Step 4: Generate embeddings (optional)
    if enable_embeddings:
        print("\n[4/5] Generating embeddings...")
        try:
            embedding_gen = EmbeddingGenerator()
            
            # Extract content texts
            content_texts = [doc.content_text for doc in scene_documents]
            
            # Generate embeddings
            embeddings = embedding_gen.generate_embeddings(content_texts)
            
            # Add embeddings to documents
            scene_documents = add_embeddings_to_documents(scene_documents, embeddings)
            
            print(f"✓ Generated embeddings for {len(embeddings)} document(s)")
        except Exception as e:
            print(f"⚠ Failed to generate embeddings: {e}")
            print("  Continuing without embeddings...")
    else:
        print("\n[4/5] Skipping embeddings (disabled)")
    
    # Step 5: Index in Azure AI Search
    print("\n[5/5] Indexing documents in Azure AI Search...")
    search_manager = SearchIndexManager()
    
    # Create index if it doesn't exist
    created = search_manager.create_index_if_not_exists()
    if created:
        print("  Created new search index")
    else:
        print("  Using existing search index")
    
    # Delete existing documents for this video (if any)
    deleted = search_manager.delete_documents_by_video_id(video_id)
    if deleted > 0:
        print(f"  Deleted {deleted} existing document(s) for video {video_id}")
    
    # Upsert documents
    result = search_manager.upsert_documents(scene_documents)
    
    print(f"✓ Indexed {result['succeeded']} document(s)")
    if result['failed'] > 0:
        print(f"⚠ Failed to index {result['failed']} document(s)")
    
    # Summary
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    print(f"Video ID: {video_id}")
    print(f"Segments: {len(segments)}")
    print(f"Documents indexed: {result['succeeded']}")
    print(f"Search index: {search_manager.index_name}")
    print("\nYou can now query scenes using the /query API endpoint")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest video into search index via Azure AI Content Understanding"
    )
    parser.add_argument(
        "--video_url",
        required=True,
        help="URL to the video (Blob SAS URL recommended)"
    )
    parser.add_argument(
        "--video_id",
        default=None,
        help="Video ID (defaults to extracted from URL)"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    
    args = parser.parse_args()
    
    try:
        ingest_video(
            video_url=args.video_url,
            video_id=args.video_id,
            enable_embeddings=not args.no_embeddings
        )
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
