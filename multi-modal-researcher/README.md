# Multi-Modal Researcher

This project is a simple research and podcast generation workflow that uses LangGraph with OpenAI's capabilities and LangChain integration. You can pass a research topic and, optionally, a YouTube video URL. The system will then perform research on the topic using web search, analyze the video transcript, combine the insights, and generate a report with citations as well as a short podcast on the topic for you. It takes advantage of OpenAI's capabilities with LangChain integration:

- 🎥 **Video transcript analysis**: Processing of YouTube video transcripts using the YouTube Transcript API
- 🔍 **Web search integration**: LangChain-powered function calling with web search capabilities for real-time information
- 🎙️ **Text-to-speech**: Generate natural conversations with distinct speaker voices using OpenAI's TTS
- 🔗 **LangChain Integration**: Advanced prompt management and chain operations for better AI workflows

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Setup

1. **Navigate to the project**:
```bash
cd multi-modal-researcher
```

2. **Set up environment variables**:
Create a `.env` file in the project root and [add your OpenAI API key](https://platform.openai.com/api-keys):
```bash
OPENAI_API_KEY=your_api_key_here
```

3. **Install dependencies and run the development server**:

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking
```

4. **Access the application**:

LangGraph will open in your browser.

```bash
╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

5. Pass a `topic` and optionally a `video_url`.

Example:
* `topic`: Give me an overview of the idea that LLMs are like a new kind of operating system.
* `video_url`: https://youtu.be/LCEmiRjPEtQ?si=raeMN2Roy5pESNG2

<img width="1604" alt="Screenshot 2025-06-24 at 5 13 31 PM" src="https://github.com/user-attachments/assets/6407e802-8932-4cfb-bdf9-5af96050ee1f" />

Result:

[🔍 See the example report](./example/report/karpathy_os.md)

[▶️ Download the example podcast](./example/audio/karpathy_os.wav)

## Architecture

The system implements a LangGraph workflow with the following nodes:

1. **Search Research Node**: Performs web search using Gemini's Google Search integration
2. **Analyze Video Node**: Analyzes YouTube videos when provided (conditional)
3. **Create Report Node**: Synthesizes findings into a comprehensive markdown report
4. **Create Podcast Node**: Generates a 2-speaker podcast discussion with TTS audio

### Workflow

```
START → search_research → [analyze_video?] → create_report → create_podcast → END
```

The workflow conditionally includes video analysis if a YouTube URL is provided, otherwise proceeds directly to report generation.

### Output

The system generates:

- **Research Report**: Comprehensive markdown report with executive summary and sources
- **Podcast Script**: Natural dialogue between Dr. Sarah (expert) and Mike (interviewer)  
- **Audio File**: Multi-speaker TTS audio file (`research_podcast_*.wav`)

## Configuration

The system supports runtime configuration through the `Configuration` class with enhanced LangChain integration:

### Model Settings
- `search_model`: Model for web search and general tasks (default: "gpt-4o")
- `synthesis_model`: Model for report synthesis (default: "gpt-4o")
- `video_model`: Model for video transcript analysis (default: "gpt-4o")
- `tts_model`: Model for text-to-speech (default: "tts-1")
- `langchain_model`: Default model for LangChain operations (default: "gpt-4o")
- `langchain_search_model`: Model for LangChain-based search (default: "gpt-4o")

### Temperature Settings
- `search_temperature`: Factual search queries (default: 0.0)
- `synthesis_temperature`: Balanced synthesis (default: 0.3)
- `podcast_script_temperature`: Creative dialogue (default: 0.4)
- `langchain_temperature`: LangChain operations (default: 0.3)

### TTS Settings
- `mike_voice`: Voice for interviewer (default: "alloy")
- `sarah_voice`: Voice for expert (default: "nova")
- `tts_format`: Audio format for output (default: "wav")

## Web Search Integration

The system now uses **LangChain's OpenAI integration** with web search capabilities, providing enhanced prompt management and chain operations for better AI workflows.

### 🎯 **LangChain + OpenAI Web Search**

The system leverages LangChain's integration with OpenAI's web search tools:

1. **LangChain Web Search Tool**:
   - Uses `web_search_preview` tool through LangChain
   - Enhanced prompt templates for better search queries
   - Structured chain operations for complex workflows

2. **Fallback Strategy**:
   - Primary: LangChain with OpenAI web search tool
   - Fallback: Direct OpenAI responses API
   - Final fallback: Regular model with training data disclaimer

### ✅ **Simple Setup**

Just need your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**That's it!** No additional API keys or setup required.

### 🔄 **How It Works**

The system uses LangChain's prompt templates and chain operations to:
1. Structure search queries effectively
2. Process multi-modal research content
3. Generate comprehensive reports with proper citation formatting
4. Create engaging podcast dialogues with natural conversation flow

### 🎯 **Benefits of LangChain Integration**

- ✅ **Structured Prompts** - Better prompt management with templates
- ✅ **Chain Operations** - Complex workflows with intermediate steps
- ✅ **Enhanced Error Handling** - Better fallback strategies
- ✅ **Modular Architecture** - Separate concerns for different operations
- ✅ **Type Safety** - Better integration with Pydantic models

### 📊 **Model Configuration**

Configure in `src/agent/configuration.py`:

```python
# Primary search model
langchain_search_model: str = "gpt-4o"
```

### 🔄 **Enhanced Search Architecture**

The current implementation provides a robust multi-layered search approach:
1. **Primary**: LangChain with OpenAI web search tool integration
2. **Fallback**: Direct OpenAI responses API with web search
3. **Final fallback**: Standard model responses with appropriate disclaimers

## Project Structure

```
├── src/agent/
│   ├── state.py           # State definitions (input/output schemas)
│   ├── configuration.py   # Runtime configuration class
│   ├── utils.py          # Utility functions (TTS, report generation)
│   └── graph.py          # LangGraph workflow definition
├── langgraph.json        # LangGraph deployment configuration
├── pyproject.toml        # Python package configuration
└── .env                  # Environment variables
```

## Key Components

### State Management

- **ResearchStateInput**: Input schema (topic, optional video_url)
- **ResearchStateOutput**: Output schema (report, podcast_script, podcast_filename)
- **ResearchState**: Complete state including intermediate results (search_text, search_sources_text, video_text, synthesis_text)

### Utility Functions

- **perform_web_search()**: LangChain-powered web search with OpenAI integration
- **display_openai_response()**: Processes OpenAI responses and formats sources
- **get_youtube_transcript()**: Extracts transcripts from YouTube videos using modern API
- **create_podcast_discussion()**: Generates scripted dialogue and TTS audio using LangChain prompts
- **create_research_report()**: Synthesizes multi-modal research into reports with LangChain templates

### LangChain Integration Features

- **Prompt Templates**: Structured prompts for different tasks (synthesis, podcast generation)
- **Chain Operations**: Sequential processing with intermediate results
- **Model Binding**: Tool binding for web search capabilities
- **Error Handling**: Graceful fallbacks between different search methods

## Deployment

The application is configured for deployment on:

- **Local Development**: Using LangGraph CLI with in-memory storage
- **LangGraph Platform**: Production deployment with persistent storage
- **Self-Hosted**: Using Docker containers

## Dependencies

Core dependencies managed via `pyproject.toml`:

- `langgraph>=0.2.6` - Workflow orchestration
- `langchain>=0.3.19` - LangChain core functionality
- `langchain-openai>=0.3.28` - OpenAI integration for LangChain
- `openai>=1.0.0` - Direct OpenAI API client
- `youtube-transcript-api` - YouTube transcript extraction
- `rich` - Enhanced terminal output
- `python-dotenv` - Environment management
- `fastapi` - API framework support
- `langgraph-sdk>=0.1.57` - LangGraph SDK
- `langgraph-cli` - LangGraph command line interface
- `langgraph-api` - LangGraph API components

### Development Dependencies
- `langgraph-cli[inmem]>=0.1.71` - Development server with in-memory storage
- `pytest>=8.3.5` - Testing framework
- `mypy>=1.11.1` - Type checking
- `ruff>=0.6.1` - Linting and formatting

## License

MIT License - see LICENSE file for details.
