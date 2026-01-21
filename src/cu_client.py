"""Azure AI Content Understanding client for video analysis."""
import httpx
from typing import Dict, Any, Optional
from .config import settings


class ContentUnderstandingClient:
    """Client for Azure AI Content Understanding API."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """Initialize the CU client.
        
        Args:
            endpoint: CU endpoint URL (defaults to settings.cu_endpoint)
            key: CU subscription key (defaults to settings.cu_key)
            api_version: API version (defaults to settings.cu_api_version)
        """
        self.endpoint = (endpoint or settings.cu_endpoint).rstrip('/')
        self.key = key or settings.cu_key
        self.api_version = api_version or settings.cu_api_version
        
        self.client = httpx.Client(
            timeout=600.0,  # Long timeout for video processing
            headers={
                "Ocp-Apim-Subscription-Key": self.key,
                "Content-Type": "application/json"
            }
        )
    
    def analyze_video(
        self,
        video_url: str,
        enable_segment: bool = True,
        return_details: bool = True,
        features: Optional[list] = None
    ) -> Dict[str, Any]:
        """Analyze a video using prebuilt-videoAnalysis.
        
        Args:
            video_url: URL to the video (Blob SAS URL recommended)
            enable_segment: Enable scene segmentation
            return_details: Return detailed analysis results
            features: List of features to analyze (defaults to comprehensive set)
        
        Returns:
            Analysis result dictionary
        """
        if features is None:
            features = [
                "description",
                "transcript",
                "tags",
                "denseCaptions",
                "summary"
            ]
        
        # Construct the analyze endpoint
        url = f"{self.endpoint}/contentunderstanding/analyzers/prebuilt-videoAnalysis"
        
        # Build request body
        body = {
            "url": video_url,
            "features": features,
            "queryFields": []
        }
        
        # Add segment configuration
        if enable_segment:
            body["segmentConfiguration"] = {
                "enableSegment": True
            }
        
        # Add query parameters
        params = {
            "api-version": self.api_version
        }
        
        # Make the analyze request
        response = self.client.post(url, json=body, params=params)
        response.raise_for_status()
        
        # Get operation location from response headers
        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            raise ValueError("No Operation-Location header in response")
        
        # Poll for result
        return self._poll_for_result(operation_location)
    
    def _poll_for_result(self, operation_url: str, max_attempts: int = 120) -> Dict[str, Any]:
        """Poll for analysis result.
        
        Args:
            operation_url: URL to poll for results
            max_attempts: Maximum number of polling attempts (default: 120 = 10 minutes)
        
        Returns:
            Analysis result dictionary
        """
        import time
        
        for attempt in range(max_attempts):
            response = self.client.get(operation_url)
            response.raise_for_status()
            
            result = response.json()
            status = result.get("status")
            
            if status == "succeeded":
                return result
            elif status == "failed":
                error = result.get("error", {})
                raise RuntimeError(f"Analysis failed: {error}")
            elif status in ["notStarted", "running"]:
                # Wait before next poll
                time.sleep(5)
            else:
                raise ValueError(f"Unknown status: {status}")
        
        raise TimeoutError(f"Analysis did not complete after {max_attempts} attempts")
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
