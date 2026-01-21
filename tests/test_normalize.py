"""Tests for normalization logic."""
import pytest
from src.cu_parse import Segment
from src.normalize import normalize_to_scene_documents, add_embeddings_to_documents


def test_normalize_basic_segment():
    """Test normalization of a basic segment."""
    segment = Segment(
        start_time_ms=1000,
        end_time_ms=5000,
        description="A person walking in a park",
        tags=["outdoor", "person", "park"],
        transcript="Hello, this is a test."
    )
    
    documents = normalize_to_scene_documents(
        segments=[segment],
        video_id="test-video",
        video_url="https://example.com/video.mp4"
    )
    
    assert len(documents) == 1
    doc = documents[0]
    
    # Check required fields
    assert doc.id == "test-video-seg0000"
    assert doc.video_id == "test-video"
    assert doc.video_url == "https://example.com/video.mp4"
    assert doc.start_time_ms == 1000
    assert doc.end_time_ms == 5000
    assert doc.tags == ["outdoor", "person", "park"]
    assert doc.scene_description == "A person walking in a park"
    assert doc.transcript_text == "Hello, this is a test."
    assert doc.content_text == "A person walking in a park\nHello, this is a test."


def test_normalize_multiple_segments():
    """Test normalization of multiple segments."""
    segments = [
        Segment(
            start_time_ms=0,
            end_time_ms=1000,
            description="Scene 1",
            tags=["intro"],
            transcript="First scene"
        ),
        Segment(
            start_time_ms=1000,
            end_time_ms=2000,
            description="Scene 2",
            tags=["action"],
            transcript="Second scene"
        ),
    ]
    
    documents = normalize_to_scene_documents(
        segments=segments,
        video_id="multi-video",
        video_url="https://example.com/multi.mp4"
    )
    
    assert len(documents) == 2
    assert documents[0].id == "multi-video-seg0000"
    assert documents[1].id == "multi-video-seg0001"
    assert documents[0].start_time_ms == 0
    assert documents[1].start_time_ms == 1000


def test_normalize_empty_fields():
    """Test normalization with empty description and transcript."""
    segment = Segment(
        start_time_ms=0,
        end_time_ms=1000,
        description="",
        tags=["tag1"],
        transcript=""
    )
    
    documents = normalize_to_scene_documents(
        segments=[segment],
        video_id="empty-video",
        video_url="https://example.com/empty.mp4"
    )
    
    assert len(documents) == 1
    doc = documents[0]
    
    assert doc.scene_description == ""
    assert doc.transcript_text == ""
    assert doc.content_text == ""  # Should be empty when both are empty


def test_add_embeddings():
    """Test adding embeddings to documents."""
    segments = [
        Segment(start_time_ms=0, end_time_ms=1000, description="Test 1"),
        Segment(start_time_ms=1000, end_time_ms=2000, description="Test 2"),
    ]
    
    documents = normalize_to_scene_documents(
        segments=segments,
        video_id="test",
        video_url="https://example.com/test.mp4"
    )
    
    # Mock embeddings
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    
    documents = add_embeddings_to_documents(documents, embeddings)
    
    assert documents[0].content_vector == [0.1, 0.2, 0.3]
    assert documents[1].content_vector == [0.4, 0.5, 0.6]


def test_add_embeddings_mismatch():
    """Test that mismatched embeddings raise an error."""
    segments = [
        Segment(start_time_ms=0, end_time_ms=1000, description="Test"),
    ]
    
    documents = normalize_to_scene_documents(
        segments=segments,
        video_id="test",
        video_url="https://example.com/test.mp4"
    )
    
    # Wrong number of embeddings
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],  # Extra embedding
    ]
    
    with pytest.raises(ValueError, match="Number of documents.*must match"):
        add_embeddings_to_documents(documents, embeddings)


def test_document_has_all_required_fields():
    """Test that SceneDocument has all required fields from spec."""
    segment = Segment(
        start_time_ms=0,
        end_time_ms=1000,
        description="Test scene",
        tags=["test"],
        transcript="Test transcript"
    )
    
    documents = normalize_to_scene_documents(
        segments=[segment],
        video_id="spec-test",
        video_url="https://example.com/spec.mp4"
    )
    
    doc = documents[0]
    
    # Required fields from specification
    assert hasattr(doc, 'id')
    assert hasattr(doc, 'video_id')
    assert hasattr(doc, 'video_url')
    assert hasattr(doc, 'start_time_ms')
    assert hasattr(doc, 'end_time_ms')
    assert hasattr(doc, 'tags')
    assert hasattr(doc, 'scene_description')
    assert hasattr(doc, 'transcript_text')
    assert hasattr(doc, 'content_text')
    
    # All fields should have values (even if empty)
    assert doc.id is not None
    assert doc.video_id is not None
    assert doc.video_url is not None
    assert doc.start_time_ms is not None
    assert doc.end_time_ms is not None
    assert doc.tags is not None
    assert doc.scene_description is not None
    assert doc.transcript_text is not None
    assert doc.content_text is not None
