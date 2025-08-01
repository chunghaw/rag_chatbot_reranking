# YouTube Creator Transcript Downloader & RAG Integration

A comprehensive system for downloading videos from specific YouTube creators, generating timed transcripts using OpenAI Whisper, and integrating them into a RAG (Retrieval-Augmented Generation) system for intelligent searching and querying.

## 🎯 Target Creators

The system is configured to process videos from these specific creators:

- **Eczachly_** - Tech and programming content
- **JoeReisData** - Data engineering and analytics  
- **Data With Danny** - Data science and analytics tutorials
- **Matthew Berman** - AI and machine learning content

## 🚀 Features

### Video Download & Processing
- **🎬 Automatic Downloads**: Downloads the most recent 5 videos from each creator
- **🎥 Audio Extraction**: Extracts audio for efficient transcription
- **📁 Organized Storage**: Organizes files by creator name
- **🗑️ Cleanup**: Automatically removes audio files after transcription

### OpenAI Transcript Generation
- **🎤 Whisper Integration**: Uses OpenAI's Whisper model for high-quality transcription
- **✂️ Automatic Splitting**: Automatically splits large audio files (>25 minutes) into manageable chunks
- **⏱️ Timed Segments**: Generates precise timestamps for each segment
- **🌍 Multi-language**: Supports multiple languages (default: English)
- **📝 Dual Format**: Creates both JSON and human-readable transcript files

### RAG Integration
- **🔍 Searchable Segments**: Breaks transcripts into searchable segments
- **📊 Rich Metadata**: Includes creator, video title, YouTube ID, timestamps, and more
- **🔄 Duplicate Prevention**: Prevents duplicate segments using hash-based IDs
- **📈 Progress Tracking**: Real-time progress and detailed reporting

## 📋 Prerequisites

1. **OpenAI API Key**: Required for transcript generation
2. **Python Dependencies**: Install required packages
3. **ffmpeg**: Required for audio splitting (install system-wide)
4. **RAG System**: Your FastAPI RAG system should be running

## 🛠️ Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install ffmpeg** (required for audio splitting):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

3. **Set Environment Variables**:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Verify Installation**:
   ```bash
   python youtube_transcript_downloader.py --help
   python test_audio_splitting.py
   ```

## 📖 Usage

### Step 1: Download Videos & Generate Transcripts

```bash
# Download and transcribe all creators (default: 5 videos each)
python youtube_transcript_downloader.py

# Custom number of videos per creator
python youtube_transcript_downloader.py --max-videos 3

# Process only a specific creator
python youtube_transcript_downloader.py --creator "Eczachly_"

# Custom output directory
python youtube_transcript_downloader.py --output-dir "my_videos"
```

### Step 2: Integrate Transcripts into RAG System

```bash
# Add all transcripts to RAG system
python add_transcripts_to_rag.py

# Process m4a files and generate transcripts
python add_transcripts_to_rag.py --process-m4a

# Process m4a files with custom chunk duration
python add_transcripts_to_rag.py --process-m4a --chunk-duration 20

# Custom transcripts directory
python add_transcripts_to_rag.py --transcripts-dir "my_videos"

# Custom RAG API URL
python add_transcripts_to_rag.py --api-url "http://localhost:8000"
```

### Step 3: Query the RAG System

Once integrated, you can query the transcripts through your RAG chat interface:

- "What did Matthew Berman say about large language models?"
- "Show me segments about data engineering from JoeReisData"
- "What are the latest insights from Data With Danny about analytics?"
- "Find discussions about programming from Eczachly_"

**Enhanced with Channel Metadata**: The system now includes `channel_name` in metadata, providing better source attribution and filtering capabilities.

## 📁 Output Structure

```
creator_videos/
├── Eczachly_/
│   ├── Video Title 1.info.json
│   ├── Video Title 1.jpg
│   └── transcripts/
│       ├── Video_Title_1_transcript.json
│       └── Video_Title_1_readable.txt
├── JoeReisData/
│   └── transcripts/
│       ├── Video_Title_2_transcript.json
│       └── Video_Title_2_readable.txt
├── Data With Danny/
│   └── transcripts/
├── Matthew Berman/
│   └── transcripts/
├── processing_summary.json
└── rag_integration_report.json
```

## 📊 Transcript Formats

### JSON Transcript Structure
```json
{
  "title": "Video Title",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hello everyone, welcome to this video..."
    }
  ],
  "words": [
    {
      "word": "Hello",
      "start": 0.0,
      "end": 0.5
    }
  ],
  "language": "en",
  "duration": 1800.5,
  "generated_at": "2023-12-30T14:30:22"
}
```

### Readable Transcript Format
```
Transcript for: Video Title
Creator: Eczachly_
Generated: 2023-12-30T14:30:22
Duration: 1800.50 seconds
================================================================================

[00:00.00 - 00:05.20] Hello everyone, welcome to this video...

[00:05.20 - 00:12.45] Today we're going to talk about...
```

### RAG Segment Format
Each transcript segment becomes a searchable document in your RAG system:

```json
{
  "text": "[00:05.20 - 00:12.45] Today we're going to talk about...",
  "metadata": {
    "creator": "Eczachly_",
    "channel_name": "Eczachly_",
    "video_title": "Video Title",
    "youtube_id": "dQw4w9WgXcQ",
    "start_time": 5.2,
    "end_time": 12.45,
    "duration": 7.25,
    "segment_type": "transcript_segment",
    "source": "youtube_video",
    "language": "en"
  }
}
```

## ⚙️ Configuration

### Video Download Settings
- **Format**: Audio only (m4a) for efficient transcription
- **Quality**: Best available audio quality
- **Metadata**: Downloads thumbnails, descriptions, and JSON info
- **Organization**: Files organized by creator name

### Transcript Settings
- **Model**: OpenAI Whisper-1
- **Format**: Verbose JSON with word-level timestamps
- **Language**: English (configurable)
- **Segments**: Automatic segment detection
- **Chunk Duration**: Configurable (default: 25 minutes)
- **Auto-Splitting**: Handles files larger than Whisper's limit

### RAG Integration Settings
- **API Endpoint**: http://localhost:8000 (configurable)
- **Segment Size**: Based on Whisper's natural segment detection
- **Metadata**: Rich metadata for filtering and context
- **Duplicate Prevention**: Hash-based primary keys

## 🔧 Advanced Usage

### Custom Creator Configuration
Edit the `CREATORS` dictionary in `youtube_transcript_downloader.py`:

```python
CREATORS = {
    "NewCreator": {
        "url": "https://www.youtube.com/@NewCreator",
        "description": "Description of content"
    }
}
```

### Example Queries with Channel Metadata
Run the example queries script to see how channel_name metadata enhances search:

```bash
python example_queries.py
```

This demonstrates queries like:
- Channel-specific searches: "What did JoeReisData say about data engineering?"
- Cross-channel comparisons: "Compare approaches to Python across all creators"
- Topic-based searches with attribution: "What are the latest insights about AI?"

### Batch Processing
```bash
# Process all creators in sequence
python youtube_transcript_downloader.py

# Process specific creator
python youtube_transcript_downloader.py --creator "Matthew Berman"
```

### Integration with Existing RAG
The system integrates seamlessly with your existing RAG chat interface. Transcripts become searchable documents that can be queried through your chat interface.

## 📈 Performance & Costs

### OpenAI API Costs
- **Whisper Transcription**: ~$0.006 per minute of audio
- **Typical Cost**: ~$0.18 per 30-minute video
- **Batch Processing**: 20 videos ≈ $3.60

### Processing Time
- **Download**: ~2-5 minutes per video (depending on size)
- **Transcription**: ~1-3 minutes per video (depending on length)
- **RAG Integration**: ~30 seconds per video

### Storage Requirements
- **Audio Files**: Temporary (deleted after transcription)
- **Transcripts**: ~50-200KB per video
- **Metadata**: ~10-50KB per video

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   ```bash
   # Add to .env file
   echo "OPENAI_API_KEY=your_key_here" >> .env
   ```

2. **"Channel not found"**
   - Verify creator URLs are correct
   - Check if channels are public and accessible

3. **"Download failed"**
   - Check internet connection
   - Verify video is not age-restricted or private
   - Try with `--max-videos 1` to test

4. **"RAG API connection failed"**
   - Ensure your RAG system is running on http://localhost:8000
   - Check if the `/api/add-document` endpoint is available

5. **"Audio file too large"**
   - The system automatically splits large files using ffmpeg
   - Adjust chunk duration with `--chunk-duration` parameter
   - Ensure ffmpeg is installed: `ffmpeg -version`

6. **"ffmpeg not found"**
   - Install ffmpeg: https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`

### Debug Mode
```bash
# Enable verbose output
python youtube_transcript_downloader.py --max-videos 1
```

## 📊 Monitoring & Reports

### Processing Summary
The system generates detailed reports:

- **`processing_summary.json`**: Overview of all processed videos
- **`rag_integration_report.json`**: RAG integration statistics
- **Console Output**: Real-time progress and status

### Sample Report
```json
{
  "processed_at": "2023-12-30T14:30:22",
  "total_creators": 4,
  "total_videos": 20,
  "creators": {
    "Eczachly_": {
      "videos_processed": 5,
      "videos": [...]
    }
  }
}
```

## 🔒 Legal & Ethical Considerations

⚠️ **Important**: This tool is for personal use and educational purposes only.

- **Respect Copyright**: Only download content you have permission to access
- **YouTube Terms**: Comply with YouTube's Terms of Service
- **Fair Use**: Use transcripts responsibly and ethically
- **Attribution**: Always credit original creators when using their content

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational purposes. Use responsibly and in compliance with all applicable terms of service and copyright laws. 