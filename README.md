# YouTube Creator Transcript RAG System

A sophisticated RAG (Retrieval-Augmented Generation) system that integrates YouTube video transcripts with ChatGPT for intelligent, context-aware conversations about video content.

## 🚀 Features

- 🎬 **YouTube Transcript Integration**: Automatically download and process YouTube video transcripts
- 🤖 **Advanced RAG System**: Retrieval-Augmented Generation using Milvus vector database
- 🎨 **Modern Chat Interface**: Beautiful, responsive design with real-time status monitoring
- 📚 **Multi-format Document Support**: Text, PDF, and Video upload capabilities
- 🔍 **Intelligent Search**: Semantic search across video transcripts with timestamp references
- 📊 **Real-time Status**: Monitor OpenAI API and Milvus connection status
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🔗 **YouTube Link Generation**: Automatic timestamped YouTube URLs in responses

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Milvus        │
│   (HTML/JS/CSS) │◄──►│   Backend       │◄──►│   Vector DB     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │ (GPT + Embed)   │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ YouTube API     │
                       │ (Transcripts)   │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- FFmpeg (for video processing)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (for video processing)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
MILVUS_URI=https://in03-4efcec782ae2f4c.serverless.gcp-us-west1.cloud.zilliz.com
MILVUS_TOKEN=dca9ee30dd6accca68a63953d96a07cf3295cb68d1df55d93823135499762886d4ea0c5cb68b7307f72afce73a991ebc16447360
```

### 4. Start the Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

## 📖 Usage

### Chat Interface

1. **Start a Conversation**: Type your message in the input field and press Enter
2. **View Sources**: RAG sources are displayed with YouTube timestamps when available
3. **YouTube Links**: Click on generated YouTube links to jump to specific timestamps
4. **Clear Chat**: Use the "Clear Chat" button to start a new conversation

### Adding Content to Knowledge Base

#### Text Input
1. Use the left panel "Knowledge Base" section
2. Enter text in the "Add text to the knowledge base" field
3. Click "Add Text"
4. The information will be embedded and stored in Milvus

#### PDF Upload
1. In the same "Knowledge Base" section, use the file upload area
2. Click "Choose File" and select a PDF document
3. Click "Upload PDF"
4. The system will extract text from all pages and add it to the knowledge base

#### Video Upload
1. Use the "Video Upload" section
2. Select a video file (MP4, MOV, MKV)
3. Click "Upload Video"
4. The system will:
   - Extract audio using FFmpeg
   - Transcribe using OpenAI Whisper
   - Process and store the transcript

### YouTube Transcript Integration

The system can automatically process YouTube videos:

1. **Video Processing**: Upload video files for automatic transcription
2. **Transcript Storage**: Transcripts are chunked and stored with metadata
3. **Timestamp References**: Responses include clickable YouTube timestamps
4. **Channel Filtering**: Search within specific YouTube channels

## 🔧 API Endpoints

### Chat
- `POST /api/chat` - Send a message and get AI response with RAG sources

### Document Management
- `POST /api/add-document` - Add a document to the RAG system

### Health Check
- `GET /api/health` - Check API and Milvus connection status

## 🏗️ Technical Architecture

### RAG System

The advanced RAG system works as follows:

1. **Content Ingestion**: 
   - Text documents are processed directly
   - PDFs are extracted and chunked
   - Videos are transcribed using Whisper
   - YouTube transcripts are downloaded and processed

2. **Embedding Generation**: 
   - Uses OpenAI's `text-embedding-3-large` model (3072 dimensions)
   - Generates high-quality embeddings for semantic search

3. **Vector Storage**: 
   - Embeddings stored in Milvus vector database
   - Metadata includes channel name, video title, timestamps
   - Murmur3 hashing for efficient deduplication

4. **Query Processing**: 
   - User queries converted to embeddings
   - Semantic search finds relevant content
   - Channel filtering for targeted results

5. **Response Generation**: 
   - GPT-3.5-turbo generates responses
   - Includes timestamped YouTube links
   - Provides source attribution

### Vector Database Schema

```python
Collection Schema:
- id: INT64 (Primary Key, Murmur3 hash)
- text: VARCHAR (Document text, max 65535 chars)
- embedding: FLOAT_VECTOR (3072 dimensions)
- channel_name: VARCHAR (YouTube channel name)
- metadata: VARCHAR (JSON metadata with video info)
```

### Metadata Structure

```json
{
  "channel_name": "Eczachly_",
  "video_title": "Example Video Title",
  "youtube_id": "dQw4w9WgXcQ",
  "start_time": 120.5,
  "end_time": 180.0,
  "transcript_chunk": "This is the transcript text..."
}
```

## 🎨 UI Features

### Modern Design
- **Gradient Backgrounds**: Beautiful color schemes
- **Glass Morphism**: Translucent elements with blur effects
- **Smooth Animations**: Fade-in effects and hover states
- **Responsive Layout**: Adapts to different screen sizes

### Chat Experience
- **Clear Typography**: Black, non-italic text for optimal readability
- **Message Bubbles**: Distinct user and AI message styles
- **Typing Indicators**: Real-time feedback during AI processing
- **Auto-resize Input**: Dynamic textarea that grows with content
- **Keyboard Shortcuts**: Enter to send, Ctrl+Enter for text ingestion

### Knowledge Base Management
- **Triple Input Methods**: Text area, PDF file upload, and Video file upload
- **Status Feedback**: Success/error messages with auto-dismiss
- **Loading States**: Visual feedback during operations
- **File Validation**: Automatic file type checking
- **Progress Tracking**: Real-time progress bars for video processing

### Connection Monitoring
- **Real-time Status**: Live indicators for API and database connections
- **Color-coded Status**: Green (connected), Yellow (checking), Red (disconnected)
- **Pulsing Animation**: Visual feedback for active connections

## 🔍 Advanced Features

### YouTube Integration
- **Automatic Transcript Download**: Uses youtube-transcript-api
- **Timestamp Preservation**: Maintains video timing information
- **Channel Filtering**: Search within specific YouTube channels
- **Link Generation**: Creates clickable YouTube URLs with timestamps

### Video Processing Pipeline
1. **File Upload**: Accepts MP4, MOV, MKV formats
2. **Audio Extraction**: Uses FFmpeg for audio extraction
3. **Whisper Transcription**: OpenAI Whisper for accurate transcription
4. **Text Processing**: Chunking and embedding generation
5. **Vector Storage**: Milvus storage with metadata

### RAG Enhancement
- **Multi-modal Input**: Text, PDF, and Video support
- **Semantic Search**: Advanced embedding-based retrieval
- **Context Preservation**: Maintains conversation history
- **Source Attribution**: Clear indication of information sources

## 🛠️ Development

### Project Structure

```
week2_rag/
├── main.py                    # FastAPI application
├── vector_service.py          # Milvus integration
├── index.html                 # Modern chat interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── YOUTUBE_TRANSCRIPT_README.md  # YouTube integration guide
├── .env                       # Environment variables
└── venv/                     # Virtual environment
```

### Key Dependencies

- **FastAPI**: Modern web framework
- **OpenAI**: GPT models and embeddings
- **PyMilvus**: Vector database integration
- **yt-dlp**: YouTube video processing
- **youtube-transcript-api**: Transcript extraction
- **FFmpeg**: Video/audio processing
- **LangChain**: Advanced RAG capabilities

### Adding New Features

1. **Backend**: Add new endpoints in `main.py`
2. **Frontend**: Update `index.html` for new functionality
3. **Vector Service**: Modify `vector_service.py` for database changes
4. **Styling**: Update CSS for UI improvements

## 🔧 Configuration

### Milvus Setup

The application uses Zilliz Cloud (managed Milvus):

- **URI**: https://in03-4efcec782ae2f4c.serverless.gcp-us-west1.cloud.zilliz.com
- **Token**: Pre-configured for immediate use
- **Collection**: `youtube_creator_videos`

### OpenAI Configuration

Required API key for:
- **GPT-3.5-turbo**: Chat responses
- **text-embedding-3-large**: Embedding generation
- **Whisper**: Video transcription

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Ensure your API key is valid and has sufficient credits
   - Check rate limits for embedding generation

2. **Milvus Connection Issues**
   - Verify internet connection
   - Check Milvus URI and token configuration

3. **Video Processing Issues**
   - Ensure FFmpeg is installed and in PATH
   - Check video file format compatibility
   - Verify file size limits

4. **YouTube Transcript Issues**
   - Some videos may have disabled transcripts
   - Check video availability and region restrictions

5. **Memory Issues**
   - Large video files may require more memory
   - Consider chunking long transcripts

### Health Check

Access the health endpoint at `/api/health` to check:
- API server status
- Milvus connection status
- OpenAI API availability

## 🔒 Security Notes

- API keys are stored in environment variables
- No persistent storage of chat history on server
- Milvus credentials are securely configured
- File uploads are processed in temporary storage

## 📊 Performance

- **Async Operations**: Non-blocking I/O for better performance
- **Connection Pooling**: Efficient Milvus client usage
- **Embedding Caching**: Reduces API calls for repeated queries
- **Memory Management**: Limits conversation history to prevent issues
- **Chunking Strategy**: Efficient text processing for long documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenAI**: For GPT models and Whisper transcription
- **Zilliz**: For managed Milvus vector database
- **YouTube**: For video content and transcript APIs
- **FFmpeg**: For video/audio processing capabilities

---

**Enjoy your Advanced RAG Chat Assistant! 🎬🤖✨**
