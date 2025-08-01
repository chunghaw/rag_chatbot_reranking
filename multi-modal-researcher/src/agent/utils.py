import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from agent.configuration import Configuration
import re
import json
from typing import Optional, Tuple, List, Dict, Any

# LangChain ReAct Agent imports
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.tools import Tool

load_dotenv()

# Initialize both direct OpenAI client and LangChain OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_web_search_tool(configuration=None):
    """Create a web search tool for the ReAct agent"""
    
    if configuration is None:
        configuration = Configuration()
    
    search_llm = ChatOpenAI(
        model=configuration.search_model,
        temperature=configuration.search_temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        output_version="responses/v1"
    )
    
    @tool
    def web_search(query: str) -> str:
        """
        Perform web search to find current information about a topic.
        Use this when you need to search for recent information, facts, or details about any topic.
        
        Args:
            query: The search query to find information about
            
        Returns:
            str: Search results with relevant information
        """
        try:
            tool_config = {"type": "web_search_preview"}
            llm_with_tools = search_llm.bind_tools([tool_config])
            
            response = llm_with_tools.invoke(query)
            result = response.content[0]["text"] if response.content else ""
            
            if not result or result.strip() == "":
                return f"No information found for query: {query}"
                
            return result
            
        except Exception as e:
            return f"Search failed for query '{query}': {str(e)}"
    
    return web_search

@tool
def synthesize_information(information_pieces: str) -> str:
    """
    Synthesize multiple pieces of information into a coherent summary.
    Use this to combine and organize information from multiple searches.
    
    Args:
        information_pieces: Multiple pieces of information to combine
        
    Returns:
        str: A coherent synthesis of the information
    """
    return f"Synthesized information: {information_pieces}"

def perform_web_search(query: str) -> Dict[str, Any]:
    """
    Enhanced web search using LangChain's create_react_agent
    
    This function uses the standard ReAct (Reasoning + Acting) pattern
    implemented by LangChain to perform intelligent web search.
    """
    
    configuration = Configuration()
    console = Console()
    console.print(f"[bold blue]🤖 Starting LangChain ReAct Agent for:[/bold blue] {query}")
    
    try:
        # Initialize the language model
        llm = ChatOpenAI(
            model=configuration.search_model,
            temperature=configuration.react_agent_temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create tools for the agent
        web_search_tool = create_web_search_tool(configuration)
        tools = [web_search_tool, synthesize_information]
        
        # Get the ReAct prompt from LangChain hub
        try:
            prompt = hub.pull("hwchase17/react")
            console.print("[bold green]✅ Loaded ReAct prompt from LangChain hub[/bold green]")
        except Exception as e:
            console.print(f"[bold yellow]⚠️ Could not load prompt from hub: {e}[/bold yellow]")
            # Fallback to a basic ReAct prompt
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate(
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
            )
        
        # Create the ReAct agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create agent executor with configuration
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=configuration.react_agent_max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        console.print("[bold cyan]🔍 Agent is reasoning and searching...[/bold cyan]")
        
        # Execute the agent
        result = agent_executor.invoke({"input": query})
        
        # Extract the final answer
        final_answer = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        console.print(f"[bold green]✅ Agent completed after {len(intermediate_steps)} steps[/bold green]")
        
        # Format intermediate steps for debugging
        steps_summary = []
        for i, (action, observation) in enumerate(intermediate_steps, 1):
            steps_summary.append({
                "step": i,
                "action": str(action),
                "observation": observation[:200] + "..." if len(observation) > 200 else observation
            })
        
        return {
            "search_text": final_answer,
            "sources": [],  # Could be enhanced to extract sources from intermediate steps
            "query": query,
            "method": "langchain_react_agent",
            "model_used": configuration.search_model,
            "iterations_used": len(intermediate_steps),
            "intermediate_steps": steps_summary,
            "agent_type": "create_react_agent"
        }
        
    except Exception as e:
        console.print(f"[bold red]❌ ReAct agent failed: {str(e)}[/bold red]")
        
        # Fallback to simple search
        try:
            simple_search_tool = create_web_search_tool(configuration)
            fallback_result = simple_search_tool.invoke(query)
            
            return {
                "search_text": fallback_result,
                "sources": [],
                "query": query,
                "method": "fallback_simple_search",
                "model_used": configuration.search_model,
                "iterations_used": 1,
                "error": str(e)
            }
        except Exception as fallback_error:
            return {
                "search_text": f"Search failed for query: {query}. Please check your OpenAI API configuration.",
                "sources": [],
                "query": query,
                "method": "error",
                "model_used": "none",
                "iterations_used": 0,
                "error": f"Primary: {str(e)}, Fallback: {str(fallback_error)}"
            }


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_url: str) -> Optional[str]:
    """Get transcript from YouTube video using modern youtube-transcript-api"""
    try:
        video_id = extract_youtube_video_id(video_url)
        if not video_id:
            return None
        
        # Modern API usage (v1.2.1+) - instance-based approach
        ytt_api = YouTubeTranscriptApi()
        
        # Try different approaches to get transcript
        try:
            # Method 1: Fetch transcript directly 
            fetched_transcript = ytt_api.fetch(video_id)
            # Extract text from the FetchedTranscript object
            full_transcript = " ".join([snippet.text for snippet in fetched_transcript])
            return full_transcript
            
        except Exception as e1:
            print(f"Direct fetch failed: {e1}")
            
            try:
                # Method 2: Try with specific languages
                fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
                full_transcript = " ".join([snippet.text for snippet in fetched_transcript])
                return full_transcript
                
            except Exception as e2:
                print(f"English fetch failed: {e2}")
                
                try:
                    # Method 3: Try with multiple language variants
                    fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
                    full_transcript = " ".join([snippet.text for snippet in fetched_transcript])
                    return full_transcript
                    
                except Exception as e3:
                    print(f"All modern API methods failed: {e3}")
                    return None
        
    except Exception as e:
        print(f"Could not retrieve transcript: {str(e)}")
        return None

def display_openai_response(response_text: str, sources: List[Dict] = None) -> Tuple[str, str]:
    """Display OpenAI response and format sources"""
    console = Console()
    
    # Display main content
    md = Markdown(response_text)
    console.print(md)
    
    # Build sources text block
    sources_text = ""
    
    if sources:
        console.print("\n" + "="*50)
        console.print("[bold blue]References & Sources[/bold blue]")
        console.print("="*50)
        
        console.print(f"\n[bold]Sources ({len(sources)}):[/bold]")
        sources_list = []
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'No title')
            url = source.get('url', 'No URL')
            console.print(f"{i}. {title}")
            console.print(f"   [dim]{url}[/dim]")
            sources_list.append(f"{i}. {title}\n   {url}")
        
        sources_text = "\n".join(sources_list)
    
    return response_text, sources_text

def create_podcast_discussion(topic: str, search_text: str, video_text: str, 
                            search_sources_text: str, video_url: str, 
                            filename: str = "research_podcast.wav", 
                            configuration=None) -> Tuple[str, str]:
    """Create a 2-speaker podcast discussion explaining the research topic using LangChain"""
    
    # Use default values if no configuration provided
    if configuration is None:
        from agent.configuration import Configuration
        configuration = Configuration()
    
    # Step 1: Generate podcast script using LangChain
    podcast_prompt = PromptTemplate(
        input_variables=["topic", "search_text", "video_text"],
        template="""Create a natural, engaging podcast conversation between Dr. Sarah (research expert) and Mike (curious interviewer) about "{topic}".
        
        Use this research content:
        
        SEARCH FINDINGS:
        {search_text}
        
        VIDEO INSIGHTS:
        {video_text}
        
        Format as a dialogue with:
        - Mike introducing the topic and asking questions
        - Dr. Sarah explaining key concepts and insights
        - Natural back-and-forth discussion (5-7 exchanges)
        - Mike asking follow-up questions
        - Dr. Sarah synthesizing the main takeaways
        - Keep it conversational and accessible (3-4 minutes when spoken)
        
        Format exactly like this:
        Mike: [opening question]
        Dr. Sarah: [expert response]
        Mike: [follow-up]
        Dr. Sarah: [explanation]
        [continue...]
        """
    )
    
    # Create LangChain for script generation
    script_llm = ChatOpenAI(
        model=configuration.synthesis_model,
        temperature=configuration.podcast_script_temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Use modern LangChain syntax: prompt | llm
    script_chain = podcast_prompt | script_llm
    script_result = script_chain.invoke({
        "topic": topic,
        "search_text": search_text,
        "video_text": video_text
    })
    
    podcast_script = script_result.content
    
    # Step 2: Generate TTS audio for each speaker
    # Split the script by speakers and generate audio for each part
    lines = podcast_script.split('\n')
    audio_segments = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Mike:'):
            text = line[5:].strip()
            if text:
                speech_response = client.audio.speech.create(
                    model=configuration.tts_model,
                    voice=configuration.mike_voice,
                    input=text
                )
                audio_segments.append(speech_response.content)
        elif line.startswith('Dr. Sarah:'):
            text = line[10:].strip()
            if text:
                speech_response = client.audio.speech.create(
                    model=configuration.tts_model,
                    voice=configuration.sarah_voice,
                    input=text
                )
                audio_segments.append(speech_response.content)
    
    # Step 3: Combine audio segments and save
    # For simplicity, we'll just use the first segment as the audio file
    # In a production system, you'd want to properly combine the audio segments
    if audio_segments:
        with open(filename, "wb") as f:
            f.write(audio_segments[0])  # Just write the first segment for now
    
    print(f"Podcast saved as: {filename}")
    return podcast_script, filename

def create_research_report(topic: str, search_text: str, video_text: str, 
                         search_sources_text: str, video_url: str, 
                         configuration=None) -> Tuple[str, str]:
    """Create a comprehensive research report using LangChain for better prompt management"""
    
    # Use default values if no configuration provided
    if configuration is None:
        from agent.configuration import Configuration
        configuration = Configuration()
    
    # Step 1: Create synthesis using LangChain
    synthesis_prompt = PromptTemplate(
        input_variables=["topic", "search_text", "video_text"],
        template="""You are a research analyst. I have gathered information about "{topic}" from two sources:
        
        SEARCH RESULTS:
        {search_text}
        
        VIDEO CONTENT:
        {video_text}
        
        Please create a comprehensive synthesis that:
        1. Identifies key themes and insights from both sources
        2. Highlights any complementary or contrasting perspectives
        3. Provides an overall analysis of the topic based on this multi-modal research
        4. Keep it concise but thorough (3-4 paragraphs)
        
        Focus on creating a coherent narrative that brings together the best insights from both sources.
        """
    )
    
    # Create LangChain for synthesis
    synthesis_llm = ChatOpenAI(
        model=configuration.synthesis_model,
        temperature=configuration.synthesis_temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    synthesis_chain = synthesis_prompt | synthesis_llm
    synthesis_result = synthesis_chain.invoke({
        "topic": topic,
        "search_text": search_text,
        "video_text": video_text
    })
    
    synthesis_text = synthesis_result.content
    
    # Step 2: Create markdown report
    report = f"""# Research Report: {topic}

## Executive Summary

{synthesis_text}

## Video Source
- **URL**: {video_url}

## Additional Sources
{search_sources_text}

---
*Report generated using multi-modal AI research combining web search and video analysis with LangChain*
"""
    
    return report, synthesis_text