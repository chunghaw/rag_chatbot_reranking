from typing_extensions import TypedDict
from typing import Optional, List, Dict, Any

class ResearchStateInput(TypedDict):
    """State for the research and podcast generation workflow"""
    # Input fields
    topic: str
    video_url: Optional[str]

class ResearchStateOutput(TypedDict):
    """State for the research and podcast generation workflow"""

    # Final outputs
    report: Optional[str]
    podcast_script: Optional[str]
    podcast_filename: Optional[str]
    
    # ReAct agent insights (optional for output)
    search_method: Optional[str]
    search_iterations_used: Optional[int]
    search_intermediate_steps: Optional[List[Dict[str, Any]]]

class ResearchState(TypedDict):
    """State for the research and podcast generation workflow"""
    # Input fields
    topic: str
    video_url: Optional[str]
    
    # Intermediate results
    search_text: Optional[str]
    search_sources_text: Optional[str]
    video_text: Optional[str]
    
    # ReAct agent details
    search_method: Optional[str]
    search_model_used: Optional[str]
    search_iterations_used: Optional[int]
    search_intermediate_steps: Optional[List[Dict[str, Any]]]
    search_agent_type: Optional[str]
    search_error: Optional[str]
    
    # Final outputs
    report: Optional[str]
    synthesis_text: Optional[str]
    podcast_script: Optional[str]
    podcast_filename: Optional[str]