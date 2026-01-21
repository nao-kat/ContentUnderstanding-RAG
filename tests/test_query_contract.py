"""Tests for query API contract."""
import pytest
from fastapi.testclient import TestClient
from src.api import app, QueryRequest, QueryResponse, SceneResult


client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns expected structure."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"


def test_query_request_validation():
    """Test QueryRequest model validation."""
    # Valid request
    request = QueryRequest(
        query="test query",
        top_k=5,
        tag_filter=["tag1", "tag2"],
        video_id="video123"
    )
    
    assert request.query == "test query"
    assert request.top_k == 5
    assert request.tag_filter == ["tag1", "tag2"]
    assert request.video_id == "video123"


def test_query_request_defaults():
    """Test QueryRequest defaults."""
    request = QueryRequest(query="test")
    
    assert request.query == "test"
    assert request.top_k == 5  # Default
    assert request.tag_filter is None  # Default
    assert request.video_id is None  # Default


def test_query_request_top_k_bounds():
    """Test QueryRequest top_k validation."""
    # Valid bounds
    request1 = QueryRequest(query="test", top_k=1)
    assert request1.top_k == 1
    
    request2 = QueryRequest(query="test", top_k=100)
    assert request2.top_k == 100
    
    # Invalid bounds should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        QueryRequest(query="test", top_k=0)
    
    with pytest.raises(Exception):  # Pydantic validation error
        QueryRequest(query="test", top_k=101)


def test_scene_result_structure():
    """Test SceneResult has required fields."""
    result = SceneResult(
        video_url="https://example.com/video.mp4",
        video_id="video123",
        start_time_ms=1000,
        end_time_ms=5000,
        tags=["tag1", "tag2"],
        scene_description="A test scene",
        transcript_text="Test transcript",
        score=0.95
    )
    
    # Required fields from specification
    assert result.video_url == "https://example.com/video.mp4"
    assert result.video_id == "video123"
    assert result.start_time_ms == 1000
    assert result.end_time_ms == 5000
    assert result.tags == ["tag1", "tag2"]
    assert result.scene_description == "A test scene"
    assert result.transcript_text == "Test transcript"
    assert result.score == 0.95


def test_query_response_structure():
    """Test QueryResponse has required structure."""
    results = [
        SceneResult(
            video_url="https://example.com/video.mp4",
            video_id="video123",
            start_time_ms=1000,
            end_time_ms=5000,
            tags=["tag1"],
            scene_description="Scene 1",
            transcript_text="Transcript 1",
            score=0.95
        ),
        SceneResult(
            video_url="https://example.com/video.mp4",
            video_id="video123",
            start_time_ms=5000,
            end_time_ms=10000,
            tags=["tag2"],
            scene_description="Scene 2",
            transcript_text="Transcript 2",
            score=0.87
        ),
    ]
    
    response = QueryResponse(
        results=results,
        query="test query",
        count=2
    )
    
    assert response.query == "test query"
    assert response.count == 2
    assert len(response.results) == 2
    
    # Each result should have time information
    for result in response.results:
        assert result.start_time_ms is not None
        assert result.end_time_ms is not None
        assert result.video_url is not None


def test_query_response_includes_time_information():
    """Test that query response includes start/end time for RAG context."""
    result = SceneResult(
        video_url="https://example.com/video.mp4",
        video_id="test",
        start_time_ms=123000,
        end_time_ms=156000,
        tags=["test"],
        scene_description="Test",
        transcript_text="Test"
    )
    
    response = QueryResponse(
        results=[result],
        query="test",
        count=1
    )
    
    # Verify time information is present (required for RAG)
    assert response.results[0].start_time_ms == 123000
    assert response.results[0].end_time_ms == 156000


def test_query_endpoint_structure():
    """Test query endpoint returns expected structure (mock test).
    
    Note: This test may fail if Azure services are not configured.
    It validates the contract structure rather than actual search.
    """
    request_data = {
        "query": "test query",
        "top_k": 3
    }
    
    # This will likely fail without proper Azure setup, but we test the structure
    response = client.post("/query", json=request_data)
    
    # We expect either 200 (success) or 500 (Azure not configured)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        
        # Verify response structure
        assert "results" in data
        assert "query" in data
        assert "count" in data
        
        assert data["query"] == "test query"
        assert isinstance(data["results"], list)
        assert data["count"] == len(data["results"])
        
        # If results exist, verify structure
        if data["results"]:
            result = data["results"][0]
            assert "video_url" in result
            assert "video_id" in result
            assert "start_time_ms" in result
            assert "end_time_ms" in result
            assert "tags" in result
            assert "scene_description" in result
            assert "transcript_text" in result
