"""FastAPI application for video scene query."""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .search_index import SearchIndexManager
from .embedding import EmbeddingGenerator
from .config import settings


app = FastAPI(
    title="Video Scene Search API",
    description="Query video scenes indexed from Azure AI Content Understanding",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    """Request model for scene query."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    tag_filter: Optional[List[str]] = Field(default=None, description="Filter by tags")
    video_id: Optional[str] = Field(default=None, description="Filter by specific video ID")


class SceneResult(BaseModel):
    """Single scene result."""
    video_url: str
    video_id: str
    start_time_ms: int
    end_time_ms: int
    tags: List[str]
    scene_description: str
    transcript_text: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for scene query."""
    results: List[SceneResult]
    query: str
    count: int


# Initialize clients lazily
_search_manager = None
_embedding_generator = None


def get_search_manager() -> SearchIndexManager:
    """Get or create search manager."""
    global _search_manager
    if _search_manager is None:
        _search_manager = SearchIndexManager()
    return _search_manager


def get_embedding_generator() -> Optional[EmbeddingGenerator]:
    """Get or create embedding generator."""
    global _embedding_generator
    if _embedding_generator is None:
        try:
            if settings.embedding_endpoint and settings.embedding_key:
                _embedding_generator = EmbeddingGenerator()
        except Exception:
            # Embeddings not configured
            pass
    return _embedding_generator


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Video Scene Search API",
        "endpoints": {
            "query": "/query",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query_scenes(request: QueryRequest):
    """Query video scenes using text and optional filters.
    
    Args:
        request: Query request with text, filters, and top_k
    
    Returns:
        QueryResponse with matching scenes
    """
    try:
        search_manager = get_search_manager()
        embedding_gen = get_embedding_generator()
        
        # Generate query embedding if available
        query_vector = None
        if embedding_gen:
            try:
                query_vector = embedding_gen.generate_embedding(request.query)
            except Exception as e:
                # Fall back to text-only search if embedding fails
                print(f"Warning: Failed to generate embedding: {e}")
        
        # Perform search
        results = search_manager.search(
            query_text=request.query,
            query_vector=query_vector,
            top_k=request.top_k,
            video_id=request.video_id,
            tag_filter=request.tag_filter
        )
        
        # Convert to response model
        scene_results = [
            SceneResult(
                video_url=r.get("video_url", ""),
                video_id=r.get("video_id", ""),
                start_time_ms=r.get("start_time_ms", 0),
                end_time_ms=r.get("end_time_ms", 0),
                tags=r.get("tags", []),
                scene_description=r.get("scene_description", ""),
                transcript_text=r.get("transcript_text", ""),
                score=r.get("score")
            )
            for r in results
        ]
        
        return QueryResponse(
            results=scene_results,
            query=request.query,
            count=len(scene_results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
