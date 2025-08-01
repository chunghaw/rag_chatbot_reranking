"""Configuration settings for the research and podcast generation app"""

import os
from dataclasses import dataclass, fields
from typing import Optional, Any
from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """LangGraph Configuration for the deep research agent with LangChain integration."""

    # Model settings - now with OpenAI's native search options and LangChain
    search_model: str = "gpt-4o"  # OpenAI's native search-enabled model
    synthesis_model: str = "gpt-4o"  # Model for synthesis and analysis
    video_model: str = "gpt-4o"  # Model for video transcript analysis
    tts_model: str = "tts-1"  # OpenAI TTS model
    
    # LangChain specific models
    langchain_model: str = "gpt-4o"  # Default model for LangChain operations
    langchain_search_model: str = "gpt-4o"  # Model for LangChain-based search
    
    # Temperature settings for different use cases
    search_temperature: float = 0.0           # Factual search queries
    synthesis_temperature: float = 0.3        # Balanced synthesis
    podcast_script_temperature: float = 0.4   # Creative dialogue
    langchain_temperature: float = 0.3        # LangChain operations
    
    # React Agent Configuration
    react_agent_max_iterations: int = 3       # Maximum search iterations
    react_agent_completeness_threshold: float = 0.8  # Quality threshold to stop searching
    react_agent_enable_reasoning: bool = True # Enable detailed reasoning steps
    react_agent_max_search_terms: int = 2     # Max search terms per iteration
    react_agent_temperature: float = 0.2      # Temperature for reasoning steps
    
    # TTS Configuration
    mike_voice: str = "alloy"  # OpenAI voice for interviewer
    sarah_voice: str = "nova"  # OpenAI voice for expert
    tts_format: str = "wav"    # Audio format
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

