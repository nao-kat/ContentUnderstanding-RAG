"""Normalize Content Understanding results to SceneDocument format."""
from typing import List, Optional
from pydantic import BaseModel, Field
from .cu_parse import Segment


class SceneDocument(BaseModel):
    """Represents a single scene document for indexing."""
    
    id: str = Field(..., description="Unique document ID (videoId-seg####)")
    video_id: str = Field(..., description="Video identifier")
    video_url: str = Field(..., description="Video playback URL")
    start_time_ms: int = Field(..., description="Scene start time in milliseconds")
    end_time_ms: int = Field(..., description="Scene end time in milliseconds")
    tags: List[str] = Field(default_factory=list, description="Scene tags")
    scene_description: str = Field(default="", description="Scene description text")
    transcript_text: str = Field(default="", description="Transcript text for this scene")
    content_text: str = Field(default="", description="Combined content (description + transcript)")
    content_vector: Optional[List[float]] = Field(default=None, description="Embedding vector for content_text")
    
    @property
    def duration_ms(self) -> int:
        """Calculate scene duration in milliseconds."""
        return self.end_time_ms - self.start_time_ms


def normalize_to_scene_documents(
    segments: List[Segment],
    video_id: str,
    video_url: str
) -> List[SceneDocument]:
    """Normalize CU segments to SceneDocument format.
    
    Args:
        segments: List of parsed segments from CU
        video_id: Unique identifier for the video
        video_url: URL to access the video
    
    Returns:
        List of SceneDocument objects, one per segment
    """
    scene_documents = []
    
    for idx, segment in enumerate(segments):
        # Generate unique document ID
        doc_id = f"{video_id}-seg{idx:04d}"
        
        # Build content_text by combining description and transcript
        content_parts = []
        if segment.description:
            content_parts.append(segment.description)
        if segment.transcript:
            content_parts.append(segment.transcript)
        
        content_text = "\n".join(content_parts) if content_parts else ""
        
        # Create SceneDocument
        scene_doc = SceneDocument(
            id=doc_id,
            video_id=video_id,
            video_url=video_url,
            start_time_ms=segment.start_time_ms,
            end_time_ms=segment.end_time_ms,
            tags=segment.tags,
            scene_description=segment.description,
            transcript_text=segment.transcript,
            content_text=content_text
        )
        
        scene_documents.append(scene_doc)
    
    return scene_documents


def add_embeddings_to_documents(
    documents: List[SceneDocument],
    embeddings: List[List[float]]
) -> List[SceneDocument]:
    """Add embedding vectors to scene documents.
    
    Args:
        documents: List of SceneDocument objects
        embeddings: List of embedding vectors (must match documents length)
    
    Returns:
        Documents with content_vector populated
    """
    if len(documents) != len(embeddings):
        raise ValueError(
            f"Number of documents ({len(documents)}) must match "
            f"number of embeddings ({len(embeddings)})"
        )
    
    for doc, embedding in zip(documents, embeddings):
        doc.content_vector = embedding
    
    return documents
