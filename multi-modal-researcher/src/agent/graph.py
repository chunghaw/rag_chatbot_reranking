"""LangGraph implementation of the research and podcast generation workflow"""

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from agent.state import ResearchState, ResearchStateInput, ResearchStateOutput
from agent.utils import (
    display_openai_response, 
    create_podcast_discussion, 
    create_research_report, 
    client,
    perform_web_search,
    get_youtube_transcript
)
from agent.configuration import Configuration
from rich.console import Console
from rich.markdown import Markdown

def search_research_node(state: ResearchState, config: RunnableConfig) -> dict:
    """
    Node that performs web search research using LangChain ReAct agent
    Captures all intermediate steps and agent metadata for transparency
    """
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    
    # Use the ReAct agent web search function
    search_results = perform_web_search(topic)
    
    # Extract the search text and sources from the results
    search_text = search_results.get("search_text", "")
    search_sources_text = search_results.get("sources", "")
    
    # Extract ReAct agent metadata
    method_used = search_results.get("method", "unknown")
    model_used = search_results.get("model_used", "unknown")
    iterations_used = search_results.get("iterations_used", 0)
    intermediate_steps = search_results.get("intermediate_steps", [])
    agent_type = search_results.get("agent_type", "unknown")
    search_error = search_results.get("error", None)
    
    # Ensure we have valid search text
    if not search_text or search_text.strip() == "":
        # Final fallback - basic response
        search_text = f"Unable to research '{topic}' - please check your OpenAI API configuration."
        search_sources_text = "Error occurred during search"
        method_used = "error"
    
    # Ensure search_text is never None or empty
    if not search_text:
        search_text = f"No information could be retrieved for the topic: {topic}"
    
    # Format sources for display
    formatted_sources = search_sources_text or "No sources available"
    
    # Display the response with enhanced ReAct agent details
    console = Console()
    md = Markdown(search_text)
    console.print(md)
    
    # Enhanced display of ReAct agent information
    console.print(f"\n[bold blue]🤖 ReAct Agent Details:[/bold blue]")
    console.print(f"Method: {method_used}")
    console.print(f"Agent Type: {agent_type}")
    console.print(f"Model: {model_used}")
    console.print(f"Iterations: {iterations_used}")
    console.print(f"Steps Captured: {len(intermediate_steps)}")
    
    if search_error:
        console.print(f"[bold red]Error: {search_error}[/bold red]")
    
    if intermediate_steps:
        console.print(f"\n[bold cyan]🔍 Agent Reasoning Steps:[/bold cyan]")
        for i, step in enumerate(intermediate_steps[:3], 1):  # Show first 3 steps
            action = step.get("action", "Unknown action")
            console.print(f"  {i}. {action[:100]}...")
        
        if len(intermediate_steps) > 3:
            console.print(f"  ... and {len(intermediate_steps) - 3} more steps")
    
    console.print(f"\n[bold blue]Sources:[/bold blue] {formatted_sources}")
    
    # Return comprehensive state update including all ReAct agent details
    return {
        "search_text": search_text,
        "search_sources_text": formatted_sources,
        "search_method": method_used,
        "search_model_used": model_used,
        "search_iterations_used": iterations_used,
        "search_intermediate_steps": intermediate_steps,
        "search_agent_type": agent_type,
        "search_error": search_error
    }


def analyze_video_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that analyzes video content if video URL is provided"""
    configuration = Configuration.from_runnable_config(config)
    video_url = state.get("video_url")
    topic = state["topic"]
    
    if not video_url:
        return {"video_text": "No video provided for analysis."}
    
    # Get YouTube transcript
    transcript = get_youtube_transcript(video_url)
    
    if not transcript:
        return {"video_text": "Could not retrieve video transcript. Video may not have captions available."}
    
    # Analyze the transcript with OpenAI
    analysis_prompt = f"""
    Based on the following video transcript, give me an overview of how it relates to this topic: {topic}
    
    Video Transcript:
    {transcript}
    
    Please provide:
    1. Key insights from the video related to the topic
    2. Main arguments or points made
    3. How this information complements or adds to understanding of the topic
    """
    
    video_response = client.chat.completions.create(
        model=configuration.video_model,
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=configuration.search_temperature,
    )
    
    video_text = video_response.choices[0].message.content
    
    # Display the response
    display_openai_response(video_text)
    
    return {"video_text": video_text}

def create_report_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a comprehensive research report"""
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    search_text = state.get("search_text", "")
    video_text = state.get("video_text", "")
    search_sources_text = state.get("search_sources_text", "")
    video_url = state.get("video_url", "")
    
    # Get ReAct agent metadata for transparency
    search_method = state.get("search_method", "unknown")
    search_iterations_used = state.get("search_iterations_used", 0)
    search_intermediate_steps = state.get("search_intermediate_steps", [])
    
    report, synthesis_text = create_research_report(
        topic, search_text, video_text, search_sources_text, video_url, configuration
    )
    
    return {
        "report": report,
        "synthesis_text": synthesis_text,
        # Include ReAct agent insights in final output
        "search_method": search_method,
        "search_iterations_used": search_iterations_used,
        "search_intermediate_steps": search_intermediate_steps
    }


def create_podcast_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Node that creates a podcast discussion"""
    configuration = Configuration.from_runnable_config(config)
    topic = state["topic"]
    search_text = state.get("search_text", "")
    video_text = state.get("video_text", "")
    search_sources_text = state.get("search_sources_text", "")
    video_url = state.get("video_url", "")
    
    # Create unique filename based on topic
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"research_podcast_{safe_topic.replace(' ', '_')}.wav"
    
    podcast_script, podcast_filename = create_podcast_discussion(
        topic, search_text, video_text, search_sources_text, video_url, filename, configuration
    )
    
    return {
        "podcast_script": podcast_script,
        "podcast_filename": podcast_filename
    }


def should_analyze_video(state: ResearchState) -> str:
    """Conditional edge to determine if video analysis should be performed"""
    if state.get("video_url"):
        return "analyze_video"
    else:
        return "create_report"


def create_research_graph() -> StateGraph:
    """Create and return the research workflow graph"""
    
    # Create the graph with configuration schema
    graph = StateGraph(
        ResearchState, 
        input=ResearchStateInput, 
        output=ResearchStateOutput,
        config_schema=Configuration
    )
    
    # Add nodes
    graph.add_node("search_research", search_research_node)
    graph.add_node("analyze_video", analyze_video_node)
    graph.add_node("create_report", create_report_node)
    graph.add_node("create_podcast", create_podcast_node)
    
    # Add edges
    graph.add_edge(START, "search_research")
    graph.add_conditional_edges(
        "search_research",
        should_analyze_video,
        {
            "analyze_video": "analyze_video",
            "create_report": "create_report"
        }
    )
    graph.add_edge("analyze_video", "create_report")
    graph.add_edge("create_report", "create_podcast")
    graph.add_edge("create_podcast", END)
    
    return graph


def create_compiled_graph():
    """Create and compile the research graph"""
    graph = create_research_graph()
    return graph.compile()