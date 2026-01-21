"""Azure AI Search index management and operations."""
import json
from pathlib import Path
from typing import List, Dict, Any
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.core.credentials import AzureKeyCredential
from .config import settings
from .normalize import SceneDocument


class SearchIndexManager:
    """Manage Azure AI Search index for video scenes."""
    
    def __init__(
        self,
        endpoint: str = None,
        api_key: str = None,
        index_name: str = None
    ):
        """Initialize the search index manager.
        
        Args:
            endpoint: Azure AI Search endpoint
            api_key: Azure AI Search admin key
            index_name: Name of the search index
        """
        self.endpoint = (endpoint or settings.search_endpoint).rstrip('/')
        self.api_key = api_key or settings.search_api_key
        self.index_name = index_name or settings.search_index_name
        
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
    
    def create_index_if_not_exists(self, schema_path: str = None) -> bool:
        """Create the search index if it doesn't exist.
        
        Args:
            schema_path: Path to index schema JSON file
        
        Returns:
            True if index was created, False if it already existed
        """
        # Check if index exists
        try:
            self.index_client.get_index(self.index_name)
            return False  # Index already exists
        except Exception:
            pass  # Index doesn't exist, create it
        
        # Load schema
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "infra" / "index_schema.json"
        
        with open(schema_path, 'r') as f:
            schema_dict = json.load(f)
        
        # Override index name from config
        schema_dict['name'] = self.index_name
        
        # Create index from schema
        index = SearchIndex.from_dict(schema_dict)
        self.index_client.create_index(index)
        
        return True
    
    def upsert_documents(self, documents: List[SceneDocument]) -> Dict[str, Any]:
        """Upsert scene documents to the search index.
        
        Args:
            documents: List of SceneDocument objects to index
        
        Returns:
            Result summary dictionary
        """
        if not documents:
            return {"succeeded": 0, "failed": 0}
        
        # Convert documents to dictionaries
        doc_dicts = [doc.model_dump(exclude_none=True) for doc in documents]
        
        # Upload documents
        results = self.search_client.upload_documents(documents=doc_dicts)
        
        # Summarize results
        succeeded = sum(1 for r in results if r.succeeded)
        failed = sum(1 for r in results if not r.succeeded)
        
        return {
            "succeeded": succeeded,
            "failed": failed,
            "total": len(documents)
        }
    
    def delete_documents_by_video_id(self, video_id: str) -> int:
        """Delete all documents for a specific video.
        
        Args:
            video_id: Video ID to delete documents for
        
        Returns:
            Number of documents deleted
        """
        # Search for documents with this video_id
        results = self.search_client.search(
            search_text="*",
            filter=f"video_id eq '{video_id}'",
            select=["id"]
        )
        
        # Collect document IDs
        doc_ids = [doc["id"] for doc in results]
        
        if not doc_ids:
            return 0
        
        # Delete documents
        delete_docs = [{"id": doc_id} for doc_id in doc_ids]
        self.search_client.delete_documents(documents=delete_docs)
        
        return len(doc_ids)
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get a single document by ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document dictionary
        """
        return self.search_client.get_document(key=doc_id)
    
    def search(
        self,
        query_text: str = None,
        query_vector: List[float] = None,
        top_k: int = 5,
        video_id: str = None,
        tag_filter: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for scenes using text and/or vector search.
        
        Args:
            query_text: Text query for keyword search
            query_vector: Vector for semantic search
            top_k: Number of results to return
            video_id: Filter by video ID
            tag_filter: Filter by tags (any of these tags)
        
        Returns:
            List of matching documents with scores
        """
        # Build filter expression
        filters = []
        if video_id:
            filters.append(f"video_id eq '{video_id}'")
        if tag_filter:
            tag_conditions = [f"tags/any(t: t eq '{tag}')" for tag in tag_filter]
            filters.append(f"({' or '.join(tag_conditions)})")
        
        filter_expr = " and ".join(filters) if filters else None
        
        # Perform search
        if query_vector:
            # Vector search with optional hybrid
            from azure.search.documents.models import VectorizedQuery
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=query_text or "*",
                vector_queries=[vector_query],
                filter=filter_expr,
                top=top_k,
                select=[
                    "id", "video_id", "video_url",
                    "start_time_ms", "end_time_ms",
                    "tags", "scene_description", "transcript_text"
                ]
            )
        else:
            # Text-only search
            results = self.search_client.search(
                search_text=query_text or "*",
                filter=filter_expr,
                top=top_k,
                select=[
                    "id", "video_id", "video_url",
                    "start_time_ms", "end_time_ms",
                    "tags", "scene_description", "transcript_text"
                ]
            )
        
        # Convert to list with scores
        output = []
        for result in results:
            doc = dict(result)
            # Extract score if available
            score = doc.pop('@search.score', None)
            if score:
                doc['score'] = score
            output.append(doc)
        
        return output
