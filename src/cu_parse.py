"""Parser for Azure AI Content Understanding API responses."""
from typing import Dict, Any, List, Optional

# Time conversion constants
SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60
MILLISECONDS_PER_SECOND = 1000


class Segment:
    """Represents a video segment/scene."""
    
    def __init__(
        self,
        start_time_ms: int,
        end_time_ms: int,
        description: str = "",
        tags: Optional[List[str]] = None,
        transcript: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a segment.
        
        Args:
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds
            description: Scene description
            tags: List of tags for this segment
            transcript: Transcript text for this segment
            metadata: Additional metadata
        """
        self.start_time_ms = start_time_ms
        self.end_time_ms = end_time_ms
        self.description = description
        self.tags = tags or []
        self.transcript = transcript
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "description": self.description,
            "tags": self.tags,
            "transcript": self.transcript,
            "metadata": self.metadata
        }


def parse_cu_response(result: Dict[str, Any]) -> List[Segment]:
    """Parse Content Understanding API response to extract segments.
    
    Args:
        result: The API response dictionary
    
    Returns:
        List of Segment objects
    """
    segments = []
    
    # Navigate to the result data
    analyze_result = result.get("result", {}).get("analyzeResult", {})
    
    # Extract video metadata
    videos = analyze_result.get("videos", [])
    if not videos:
        return segments
    
    video_data = videos[0]
    
    # Get segments from the video
    video_segments = video_data.get("segments", [])
    
    # Get descriptions (may be at video level or segment level)
    descriptions = video_data.get("descriptions", [])
    tags_list = video_data.get("tags", [])
    transcripts = video_data.get("transcripts", [])
    
    # Parse each segment
    for seg in video_segments:
        start_ms = _time_to_ms(seg.get("startTime", "00:00:00"))
        end_ms = _time_to_ms(seg.get("endTime", "00:00:00"))
        
        # Find description for this segment
        seg_description = _find_text_for_timerange(
            descriptions, start_ms, end_ms, "content"
        )
        
        # Find tags for this segment
        seg_tags = _find_tags_for_timerange(
            tags_list, start_ms, end_ms
        )
        
        # Find transcript for this segment
        seg_transcript = _find_text_for_timerange(
            transcripts, start_ms, end_ms, "content"
        )
        
        segment = Segment(
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            description=seg_description,
            tags=seg_tags,
            transcript=seg_transcript,
            metadata=seg
        )
        
        segments.append(segment)
    
    # If no segments found but we have video-level data, create one segment
    if not segments and (descriptions or tags_list or transcripts):
        duration_ms = _time_to_ms(video_data.get("duration", "00:00:00"))
        
        all_descriptions = " ".join(d.get("content", "") for d in descriptions)
        all_tags = list(set(t.get("name", "") for t in tags_list if t.get("name")))
        all_transcripts = " ".join(t.get("content", "") for t in transcripts)
        
        segment = Segment(
            start_time_ms=0,
            end_time_ms=duration_ms,
            description=all_descriptions,
            tags=all_tags,
            transcript=all_transcripts
        )
        segments.append(segment)
    
    return segments


def _time_to_ms(time_str: str) -> int:
    """Convert time string (HH:MM:SS or HH:MM:SS.mmm) to milliseconds.
    
    Args:
        time_str: Time string in format HH:MM:SS or HH:MM:SS.mmm
    
    Returns:
        Time in milliseconds
    """
    try:
        # Split by decimal point if present
        if '.' in time_str:
            main_part, ms_part = time_str.split('.')
            ms = int(ms_part.ljust(3, '0')[:3])  # Ensure 3 digits
        else:
            main_part = time_str
            ms = 0
        
        # Split main part into components
        parts = main_part.split(':')
        
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        else:
            return 0
        
        total_ms = (hours * SECONDS_PER_HOUR + minutes * SECONDS_PER_MINUTE + seconds) * MILLISECONDS_PER_SECOND + ms
        return total_ms
    except (ValueError, AttributeError):
        return 0


def _find_text_for_timerange(
    items: List[Dict[str, Any]],
    start_ms: int,
    end_ms: int,
    content_key: str = "content"
) -> str:
    """Find text items that overlap with the given time range.
    
    Args:
        items: List of items with time information
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        content_key: Key to extract content from items
    
    Returns:
        Concatenated text from overlapping items
    """
    texts = []
    
    for item in items:
        item_start = _time_to_ms(item.get("startTime", "00:00:00"))
        item_end = _time_to_ms(item.get("endTime", "00:00:00"))
        
        # Check if item overlaps with segment
        if _ranges_overlap(start_ms, end_ms, item_start, item_end):
            content = item.get(content_key, "")
            if content:
                texts.append(content)
    
    return " ".join(texts)


def _find_tags_for_timerange(
    tags: List[Dict[str, Any]],
    start_ms: int,
    end_ms: int
) -> List[str]:
    """Find tags that overlap with the given time range.
    
    Args:
        tags: List of tag items with time information
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
    
    Returns:
        List of unique tag names
    """
    tag_names = set()
    
    for tag in tags:
        # Tags may have instances with time ranges
        instances = tag.get("instances", [])
        
        if instances:
            for instance in instances:
                inst_start = _time_to_ms(instance.get("startTime", "00:00:00"))
                inst_end = _time_to_ms(instance.get("endTime", "00:00:00"))
                
                if _ranges_overlap(start_ms, end_ms, inst_start, inst_end):
                    tag_name = tag.get("name", "")
                    if tag_name:
                        tag_names.add(tag_name)
                    break
        else:
            # Tag applies to entire video
            tag_name = tag.get("name", "")
            if tag_name:
                tag_names.add(tag_name)
    
    return list(tag_names)


def _ranges_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two time ranges overlap.
    
    Args:
        start1: Start of first range
        end1: End of first range
        start2: Start of second range
        end2: End of second range
    
    Returns:
        True if ranges overlap
    """
    return start1 <= end2 and start2 <= end1
