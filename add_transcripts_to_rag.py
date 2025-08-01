#!/usr/bin/env python3
"""
Add YouTube Transcripts to RAG System

Takes the generated transcripts from YouTube creators and adds them to the Milvus RAG system
for searching and querying. Also processes m4a files with Whisper and cleans up transcripts.
"""

import os
import sys
import json
import asyncio
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
RAG_API_URL = "http://localhost:8000"  # Your RAG API endpoint
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


class TranscriptRAGIntegrator:
    def __init__(self, transcripts_dir: str = "creator_videos", api_url: str = RAG_API_URL):
        """Initialize the transcript RAG integrator."""
        self.transcripts_dir = Path(transcripts_dir)
        self.api_url = api_url
        
    async def generate_transcript_from_audio(self, audio_file_path: str, video_title: str, max_chunk_duration: int = 25) -> Optional[Dict]:
        """Generate transcript from m4a file using OpenAI Whisper, with automatic splitting for large files."""
        if not client:
            print(f"❌ OpenAI client not available for {video_title}")
            return None
            
        print(f"🎤 Generating transcript for: {video_title}")
        
        try:
            # Split audio file if needed
            audio_chunks = self.split_audio_file(audio_file_path, max_chunk_duration)
            temp_files_to_cleanup = []
            
            if len(audio_chunks) > 1:
                print(f"📁 Processing {len(audio_chunks)} audio chunks...")
                temp_files_to_cleanup = audio_chunks[1:]  # First chunk is original file
            
            all_segments = []
            all_words = []
            total_duration = 0
            chunk_offset = 0
            
            # Process each chunk
            for i, chunk_file in enumerate(audio_chunks):
                print(f"  🎵 Processing chunk {i+1}/{len(audio_chunks)}: {Path(chunk_file).name}")
                
                try:
                    with open(chunk_file, 'rb') as audio_file:
                        chunk_transcript = await client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="verbose_json"
                        )
                    
                    # Get chunk data
                    chunk_segments = getattr(chunk_transcript, 'segments', [])
                    chunk_words = getattr(chunk_transcript, 'words', [])
                    chunk_duration = getattr(chunk_transcript, 'duration', 0)
                    
                    # Adjust timestamps for chunks after the first
                    if i > 0:
                        for segment in chunk_segments:
                            segment['start'] += chunk_offset
                            segment['end'] += chunk_offset
                        
                        for word in chunk_words:
                            word['start'] += chunk_offset
                            word['end'] += chunk_offset
                    
                    all_segments.extend(chunk_segments)
                    all_words.extend(chunk_words)
                    total_duration += chunk_duration
                    chunk_offset += chunk_duration
                    
                    print(f"    ✅ Chunk {i+1} processed: {chunk_duration:.1f}s")
                    
                except Exception as e:
                    print(f"    ❌ Error processing chunk {i+1}: {e}")
                    continue
            
            # Clean up temporary files
            if temp_files_to_cleanup:
                self.cleanup_temp_files(temp_files_to_cleanup)
            
            # Combine all text from segments
            combined_text = " ".join([segment.get('text', '') for segment in all_segments])
            
            # Process the combined transcript
            processed_transcript = {
                'title': video_title,
                'segments': all_segments,
                'words': all_words,
                'language': getattr(chunk_transcript, 'language', 'en') if 'chunk_transcript' in locals() else 'en',
                'duration': total_duration,
                'text': combined_text,
                'generated_at': asyncio.get_event_loop().time(),
                'chunks_processed': len(audio_chunks)
            }
            
            print(f"✅ Transcript generated successfully for {video_title} ({len(audio_chunks)} chunks)")
            return processed_transcript
            
        except Exception as e:
            print(f"❌ Error generating transcript for {video_title}: {e}")
            return None
    
    async def clean_transcript_segment(self, segment_text: str, context: str = "") -> str:
        """Clean up a transcript segment using LLM."""
        if not client:
            return segment_text
            
        try:
            system_prompt = """You are a transcript cleaning assistant. Your job is to clean up transcript segments to make them more readable and coherent while preserving the original meaning and technical accuracy.

Guidelines:
1. Fix obvious transcription errors (e.g., "um", "uh", "you know" → remove or replace)
2. Improve sentence structure and grammar
3. Maintain technical terms and proper nouns exactly as spoken
4. Keep the original meaning and intent
5. Preserve any timestamps or technical references
6. Make it sound natural and professional

Return only the cleaned text, no explanations."""

            user_prompt = f"""Please clean up this transcript segment:

Original: "{segment_text}"

Context: {context}

Cleaned version:"""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            cleaned_text = response.choices[0].message.content
            return cleaned_text.strip() if cleaned_text else segment_text
            
        except Exception as e:
            print(f"⚠️  Error cleaning segment, using original: {e}")
            return segment_text
    
    def check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg and ffprobe are available."""
        try:
            # Check ffmpeg
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            # Check ffprobe
            subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_audio_duration(self, audio_file_path: str) -> float:
        """Get audio duration in seconds using ffmpeg."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(audio_file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"⚠️  Could not determine audio duration: {e}")
            return 0.0
    
    def split_audio_file(self, audio_file_path: str, max_duration_minutes: int = 25) -> List[str]:
        """Split large audio files into smaller chunks using ffmpeg."""
        print(f"✂️  Splitting audio file: {Path(audio_file_path).name}")
        
        # Check if ffmpeg is available
        if not self.check_ffmpeg_available():
            print(f"❌ ffmpeg not found. Please install ffmpeg to enable audio splitting.")
            print(f"💡 Install ffmpeg: https://ffmpeg.org/download.html")
            return [audio_file_path]
        
        try:
            # Get audio duration using ffmpeg
            duration_seconds = self.get_audio_duration(audio_file_path)
            duration_minutes = duration_seconds / 60
            
            print(f"📊 Audio duration: {duration_minutes:.1f} minutes")
            
            # Check if splitting is needed (Whisper limit is ~25 minutes)
            if duration_minutes <= max_duration_minutes:
                print(f"✅ Audio file is within size limit, no splitting needed")
                return [audio_file_path]
            
            # Calculate chunk duration in seconds
            chunk_duration_seconds = max_duration_minutes * 60
            
            # Create temporary directory for chunks
            temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
            chunk_files = []
            
            print(f"🔪 Splitting into chunks of {max_duration_minutes} minutes each...")
            
            # Calculate number of chunks needed
            num_chunks = int((duration_seconds + chunk_duration_seconds - 1) // chunk_duration_seconds)
            
            # Split audio using ffmpeg
            for i in range(num_chunks):
                start_time = i * chunk_duration_seconds
                end_time = min((i + 1) * chunk_duration_seconds, duration_seconds)
                
                # Create chunk filename
                chunk_file = temp_dir / f"chunk_{i:03d}.m4a"
                
                # Build ffmpeg command
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output files
                    '-i', str(audio_file_path),  # Input file
                    '-ss', str(start_time),  # Start time
                    '-t', str(end_time - start_time),  # Duration
                    '-c', 'copy',  # Copy codec (fast, no re-encoding)
                    '-avoid_negative_ts', 'make_zero',  # Handle negative timestamps
                    str(chunk_file)  # Output file
                ]
                
                # Execute ffmpeg command
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and chunk_file.exists():
                    chunk_files.append(str(chunk_file))
                    chunk_duration = (end_time - start_time) / 60
                    print(f"  📁 Chunk {i+1}: {chunk_duration:.1f} minutes -> {chunk_file.name}")
                else:
                    print(f"  ❌ Failed to create chunk {i+1}: {result.stderr}")
            
            if chunk_files:
                print(f"✅ Split into {len(chunk_files)} chunks")
                return chunk_files
            else:
                print(f"❌ No chunks were created successfully")
                return [audio_file_path]
            
        except Exception as e:
            print(f"❌ Error splitting audio file: {e}")
            return [audio_file_path]  # Return original file if splitting fails
    
    def cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary audio chunk files."""
        for temp_file in temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
                    print(f"🗑️  Cleaned up: {Path(temp_file).name}")
            except Exception as e:
                print(f"⚠️  Could not clean up {temp_file}: {e}")
    
    def find_m4a_files(self) -> List[Dict[str, Any]]:
        """Find all m4a files in the transcripts directory."""
        m4a_files = []
        
        if not self.transcripts_dir.exists():
            print(f"❌ Transcripts directory not found: {self.transcripts_dir}")
            return m4a_files
        
        # Walk through all creator directories
        for creator_dir in self.transcripts_dir.iterdir():
            if creator_dir.is_dir() and creator_dir.name != "__pycache__":
                creator_name = creator_dir.name
                
                print(f"📁 Scanning for m4a files in: {creator_name}")
                
                # Find all m4a files
                for m4a_file in creator_dir.glob("*.m4a"):
                    try:
                        # Extract video title from filename
                        video_title = m4a_file.stem  # Remove .m4a extension
                        
                                                # Check if transcript already exists
                        transcript_file = creator_dir / "transcripts" / f"{video_title}_transcript.json"
                        metadata_file = creator_dir / f"{video_title}.info.json"
                        
                        # Read metadata file to get YouTube video ID
                        youtube_id = None
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r', encoding='utf-8') as f:
                                    metadata_data = json.load(f)
                                    youtube_id = metadata_data.get('id', None)
                                    print(f"  📺 Found YouTube ID: {youtube_id}")
                            except Exception as e:
                                print(f"  ⚠️  Could not read metadata file: {e}")
  
                        file_info = {
                            'creator': creator_name,
                            'channel_name': creator_name,
                            'title': video_title,
                            'youtube_id': youtube_id,
                            'audio_file': str(m4a_file),
                            'transcript_file': str(transcript_file) if transcript_file.exists() else None,
                            'metadata_file': str(metadata_file) if metadata_file.exists() else None,
                            'needs_transcription': not transcript_file.exists()
                        }
                        
                        m4a_files.append(file_info)
                        print(f"  ✅ Found: {video_title}")
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {m4a_file}: {e}")
        
        print(f"\n📊 Total m4a files found: {len(m4a_files)}")
        return m4a_files
        
    def load_transcript_files(self) -> List[Dict[str, Any]]:
        """Load all transcript files from the transcripts directory."""
        transcripts = []
        
        if not self.transcripts_dir.exists():
            print(f"❌ Transcripts directory not found: {self.transcripts_dir}")
            return transcripts
        
        # Walk through all creator directories
        for creator_dir in self.transcripts_dir.iterdir():
            if creator_dir.is_dir() and creator_dir.name != "__pycache__":
                creator_name = creator_dir.name
                transcripts_dir = creator_dir / "transcripts"
                
                if transcripts_dir.exists():
                    print(f"📁 Processing transcripts for: {creator_name}")
                    
                    # Find all transcript JSON files
                    for transcript_file in transcripts_dir.glob("*_transcript.json"):
                        try:
                            with open(transcript_file, 'r', encoding='utf-8') as f:
                                transcript_data = json.load(f)
                            
                            # Extract video info from filename
                            video_title = transcript_data.get('title', 'Unknown')
                            
                            # Try to get YouTube ID from metadata file
                            youtube_id = None
                            metadata_file = creator_dir / f"{video_title}.info.json"
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file, 'r', encoding='utf-8') as f:
                                        metadata_data = json.load(f)
                                        youtube_id = metadata_data.get('id', None)
                                except Exception as e:
                                    print(f"  ⚠️  Could not read metadata file for {video_title}: {e}")
                            
                            transcript_info = {
                                'creator': creator_name,
                                'channel_name': creator_name,  # Add channel_name for consistency
                                'title': video_title,
                                'youtube_id': youtube_id,
                                'transcript_file': str(transcript_file),
                                'data': transcript_data,
                                'duration': transcript_data.get('duration', 0),
                                'language': transcript_data.get('language', 'en'),
                                'generated_at': transcript_data.get('generated_at', '')
                            }
                            
                            transcripts.append(transcript_info)
                            print(f"  ✅ Loaded: {video_title}")
                            
                        except Exception as e:
                            print(f"  ❌ Error loading {transcript_file}: {e}")
        
        print(f"\n📊 Total transcripts loaded: {len(transcripts)}")
        return transcripts
    
    async def create_segments_from_transcript(self, transcript_info: Dict[str, Any], clean_segments: bool = True) -> List[Dict[str, Any]]:
        """Create searchable segments from a transcript with optional cleaning."""
        segments = []
        transcript_data = transcript_info['data']
        
        print(f"📝 Creating segments from transcript: {transcript_info['title']}")
        
        # Process segments with timestamps
        for i, segment in enumerate(transcript_data.get('segments', [])):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            if text:
                # Clean the segment text if requested
                if clean_segments:
                    print(f"  🧹 Cleaning segment {i+1}/{len(transcript_data.get('segments', []))}")
                    context = f"Video: {transcript_info['title']}, Creator: {transcript_info['creator']}"
                    cleaned_text = await self.clean_transcript_segment(text, context)
                else:
                    cleaned_text = text
                
                # Create metadata for this segment
                metadata = {
                    'creator': transcript_info['creator'],
                    'channel_name': transcript_info['creator'],  # Add channel_name for consistency
                    'video_title': transcript_info['title'],
                    'youtube_id': transcript_info['youtube_id'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'segment_type': 'transcript_segment',
                    'source': 'youtube_video',
                    'language': transcript_info['language'],
                    'cleaned': clean_segments
                }
                
                # Format timestamp for display
                start_min, start_sec = divmod(start_time, 60)
                end_min, end_sec = divmod(end_time, 60)
                timestamp = f"{int(start_min):02d}:{int(start_sec):02d}-{int(end_min):02d}:{int(end_sec):02d}"
                
                # Create segment text with timestamp
                segment_text = f"[{timestamp}] {cleaned_text}"
                
                segments.append({
                    'text': segment_text,
                    'metadata': json.dumps(metadata),
                    'raw_text': cleaned_text,
                    'original_text': text,
                    'timestamp': timestamp,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        print(f"✅ Created {len(segments)} segments")
        return segments
    
    async def add_segment_to_rag(self, segment: Dict[str, Any]) -> bool:
        """Add a transcript segment to the RAG system."""
        try:
            response = requests.post(
                f"{self.api_url}/api/add-document",
                json={
                    'text': segment['text'],
                    'metadata': segment['metadata']
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('message') == 'Document added successfully':
                    return True
                elif result.get('message') == 'Document already exists':
                    print(f"  ⚠️  Segment already exists: {segment['timestamp']}")
                    return True
                else:
                    print(f"  ❌ Unexpected response: {result}")
                    return False
            else:
                print(f"  ❌ HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error adding segment: {e}")
            return False
    
    async def process_transcript(self, transcript_info: Dict[str, Any], clean_segments: bool = True) -> Dict[str, Any]:
        """Process a single transcript and add all segments to RAG."""
        print(f"\n🎬 Processing transcript: {transcript_info['title']}")
        print(f"📺 Creator: {transcript_info['creator']}")
        print(f"⏱️  Duration: {transcript_info['duration']:.2f} seconds")
        
        # Create segments from transcript
        segments = await self.create_segments_from_transcript(transcript_info, clean_segments)
        print(f"📝 Created {len(segments)} segments")
        
        # Add segments to RAG
        added_count = 0
        failed_count = 0
        
        for i, segment in enumerate(segments, 1):
            print(f"  [{i}/{len(segments)}] Adding segment: {segment['timestamp']}")
            
            success = await self.add_segment_to_rag(segment)
            if success:
                added_count += 1
            else:
                failed_count += 1
        
        return {
            'creator': transcript_info['creator'],
            'title': transcript_info['title'],
            'total_segments': len(segments),
            'added_segments': added_count,
            'failed_segments': failed_count,
            'success_rate': added_count / len(segments) if segments else 0
        }
    
    async def process_all_transcripts(self) -> Dict[str, Any]:
        """Process all transcripts and add them to the RAG system."""
        print("🚀 Starting Transcript RAG Integration")
        print(f"📁 Transcripts directory: {self.transcripts_dir}")
        print(f"🔗 RAG API URL: {self.api_url}")
        print("=" * 80)
        
        # Load all transcript files
        transcripts = self.load_transcript_files()
        
        if not transcripts:
            print("❌ No transcripts found to process!")
            return {}
        
        # Process each transcript
        results = []
        total_segments = 0
        total_added = 0
        
        for i, transcript_info in enumerate(transcripts, 1):
            print(f"\n[{i}/{len(transcripts)}] Processing transcript...")
            
            result = await self.process_transcript(transcript_info)
            results.append(result)
            
            total_segments += result['total_segments']
            total_added += result['added_segments']
        
        # Create summary
        summary = {
            'processed_at': asyncio.get_event_loop().time(),
            'total_transcripts': len(transcripts),
            'total_segments': total_segments,
            'total_added': total_added,
            'success_rate': total_added / total_segments if total_segments > 0 else 0,
            'results': results
        }
        
        return summary
    
    async def process_m4a_files(self, clean_segments: bool = True, chunk_duration: int = 25) -> Dict[str, Any]:
        """Process m4a files and generate transcripts."""
        print("🎵 Processing m4a files and generating transcripts")
        print("=" * 80)
        
        # Find all m4a files
        m4a_files = self.find_m4a_files()
        
        if not m4a_files:
            print("❌ No m4a files found to process")
            return {}
        
        results = []
        total_processed = 0
        total_transcribed = 0
        
        for i, file_info in enumerate(m4a_files, 1):
            print(f"\n[{i}/{len(m4a_files)}] Processing: {file_info['title']}")
            print(f"📺 Creator: {file_info['creator']}")
            
            # Check if transcript already exists
            if file_info['transcript_file'] and Path(file_info['transcript_file']).exists():
                print(f"✅ Transcript already exists: {file_info['transcript_file']}")
                total_processed += 1
                continue
            
            # Generate transcript from m4a file
            transcript_data = await self.generate_transcript_from_audio(
                file_info['audio_file'], 
                file_info['title'],
                max_chunk_duration=chunk_duration
            )
            
            if transcript_data:
                # Save transcript to file
                creator_dir = Path(file_info['audio_file']).parent
                transcripts_dir = creator_dir / "transcripts"
                transcripts_dir.mkdir(exist_ok=True)
                
                transcript_file = transcripts_dir / f"{file_info['title']}_transcript.json"
                
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Transcript saved to: {transcript_file}")
                
                # Create transcript info for processing
                transcript_info = {
                    'creator': file_info['creator'],
                    'channel_name': file_info['channel_name'],
                    'title': file_info['title'],
                    'youtube_id': file_info.get('youtube_id'),
                    'transcript_file': str(transcript_file),
                    'data': transcript_data,
                    'duration': transcript_data.get('duration', 0),
                    'language': transcript_data.get('language', 'en'),
                    'generated_at': transcript_data.get('generated_at', '')
                }
                
                # Process transcript and add to RAG
                result = await self.process_transcript(transcript_info, clean_segments)
                results.append(result)
                
                total_transcribed += 1
                total_processed += 1
            else:
                print(f"❌ Failed to generate transcript for: {file_info['title']}")
        
        # Create summary
        summary = {
            'processed_at': asyncio.get_event_loop().time(),
            'total_m4a_files': len(m4a_files),
            'total_processed': total_processed,
            'total_transcribed': total_transcribed,
            'results': results
        }
        
        return summary
    
    def save_integration_report(self, summary: Dict[str, Any]):
        """Save integration report to file."""
        report_file = self.transcripts_dir / "rag_integration_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Integration report saved to: {report_file}")
        return str(report_file)
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a summary of the integration results."""
        print("\n" + "=" * 80)
        print("🎉 RAG INTEGRATION COMPLETE!")
        print("=" * 80)
        
        print(f"📊 Total transcripts processed: {summary['total_transcripts']}")
        print(f"📝 Total segments created: {summary['total_segments']}")
        print(f"✅ Total segments added to RAG: {summary['total_added']}")
        print(f"📈 Success rate: {summary['success_rate']:.1%}")
        
        print(f"\n📺 Results by creator:")
        creator_stats = {}
        for result in summary['results']:
            creator = result['creator']
            if creator not in creator_stats:
                creator_stats[creator] = {'videos': 0, 'segments': 0, 'added': 0}
            
            creator_stats[creator]['videos'] += 1
            creator_stats[creator]['segments'] += result['total_segments']
            creator_stats[creator]['added'] += result['added_segments']
        
        for creator, stats in creator_stats.items():
            success_rate = stats['added'] / stats['segments'] if stats['segments'] > 0 else 0
            print(f"  📺 {creator}: {stats['videos']} videos, {stats['segments']} segments, {stats['added']} added ({success_rate:.1%})")
        
        print(f"\n🔍 You can now search these transcripts in your RAG chat interface!")
        print(f"💡 Try queries like:")
        print(f"   • 'What did [Creator] say about [topic]?'")
        print(f"   • 'Show me segments about [specific topic]'")
        print(f"   • 'What are the latest insights from [Creator]?'")


async def main():
    parser = argparse.ArgumentParser(
        description="Add YouTube transcripts to RAG system with Whisper transcription and LLM cleaning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process existing transcripts (default)
  python add_transcripts_to_rag.py
  
  # Process m4a files and generate transcripts
  python add_transcripts_to_rag.py --process-m4a
  
  # Process m4a files without cleaning segments
  python add_transcripts_to_rag.py --process-m4a --no-clean
  
  # Process m4a files with custom chunk duration
  python add_transcripts_to_rag.py --process-m4a --chunk-duration 20
  
  # Custom transcripts directory
  python add_transcripts_to_rag.py --transcripts-dir "my_videos"
  
  # Custom RAG API URL
  python add_transcripts_to_rag.py --api-url "http://localhost:8000"
        """
    )
    
    parser.add_argument('--transcripts-dir', default='creator_videos',
                       help='Directory containing transcript files (default: creator_videos)')
    parser.add_argument('--api-url', default=RAG_API_URL,
                       help=f'RAG API URL (default: {RAG_API_URL})')
    parser.add_argument('--process-m4a', action='store_true',
                       help='Process m4a files and generate transcripts using Whisper')
    parser.add_argument('--no-clean', action='store_true',
                       help='Skip LLM cleaning of transcript segments')
    parser.add_argument('--chunk-duration', type=int, default=25,
                       help='Maximum duration in minutes for audio chunks (default: 25)')
    
    args = parser.parse_args()
    
    # Check OpenAI API key if processing m4a files
    if args.process_m4a and not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY required for m4a processing!")
        print("💡 Add it to your .env file: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    # Check ffmpeg availability if processing m4a files
    if args.process_m4a:
        integrator = TranscriptRAGIntegrator(
            transcripts_dir=args.transcripts_dir,
            api_url=args.api_url
        )
        if not integrator.check_ffmpeg_available():
            print("❌ ffmpeg not found. Audio splitting requires ffmpeg to be installed.")
            print("💡 Install ffmpeg: https://ffmpeg.org/download.html")
            print("   macOS: brew install ffmpeg")
            print("   Ubuntu: sudo apt install ffmpeg")
            print("   Windows: Download from https://ffmpeg.org/download.html")
            sys.exit(1)
        else:
            print("✅ ffmpeg found - audio splitting enabled")
    
    # Initialize integrator
    integrator = TranscriptRAGIntegrator(
        transcripts_dir=args.transcripts_dir,
        api_url=args.api_url
    )
    
    try:
        if args.process_m4a:
            # Process m4a files and generate transcripts
            print("🎵 Processing m4a files with Whisper transcription...")
            print(f"📏 Chunk duration: {args.chunk_duration} minutes")
            summary = await integrator.process_m4a_files(clean_segments=not args.no_clean, chunk_duration=args.chunk_duration)
            
            if summary:
                print(f"\n📊 M4A Processing Summary:")
                print(f"   Total m4a files: {summary['total_m4a_files']}")
                print(f"   Total processed: {summary['total_processed']}")
                print(f"   Total transcribed: {summary['total_transcribed']}")
                
                # Save report
                integrator.save_integration_report(summary)
            else:
                print("❌ No m4a files were processed!")
                sys.exit(1)
        else:
            # Process existing transcripts
            print("📝 Processing existing transcripts...")
            summary = await integrator.process_all_transcripts()
            
            if summary:
                # Save report
                integrator.save_integration_report(summary)
                
                # Print summary
                integrator.print_summary(summary)
            else:
                print("❌ No transcripts were processed!")
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Integration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 