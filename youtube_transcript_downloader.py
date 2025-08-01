#!/usr/bin/env python3
"""
YouTube Creator Video Downloader with OpenAI Transcript Generation

Downloads the most recent 5 videos from specified YouTube creators and generates
timed transcripts using OpenAI's Whisper model.
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import yt_dlp
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY not found in environment variables!")
    sys.exit(1)

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Target YouTube creators
CREATORS = {
    "JoeReisData": {
        "url": "https://www.youtube.com/@JoeReisData", 
        "description": "Data engineering and analytics"
    },
    "Data With Danny": {
        "url": "https://www.youtube.com/@DataWithDanny",
        "description": "Data science and analytics tutorials"
    },
    "Matthew Berman": {
        "url": "https://www.youtube.com/@MatthewBerman",
        "description": "AI and machine learning content"
    }
}


class YouTubeTranscriptDownloader:
    def __init__(self, output_dir: str = "creator_videos", max_videos: int = 5):
        """Initialize the YouTube transcript downloader."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_videos = max_videos
        
        # yt-dlp options for video download
        self.ydl_opts = {
            'format': 'bestaudio[ext=m4a]',  # Audio only for transcription
            'outtmpl': str(self.output_dir / '%(uploader)s/%(title)s.%(ext)s'),
            'writesubtitles': False,  # We'll generate our own
            'writeautomaticsub': False,
            'writethumbnail': True,
            'writedescription': True,
            'writeinfojson': True,
            'ignoreerrors': True,
            'no_warnings': False,
            'quiet': False,
            'playlist_items': f'1-{max_videos}',  # Limit to recent videos
        }
    
    def get_channel_videos(self, channel_url: str, creator_name: str) -> List[Dict]:
        """Get recent videos from a YouTube channel without downloading."""
        print(f"📺 Getting recent videos from {creator_name}...")
        
        info_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'playlist_items': f'1-{self.max_videos}',
        }
        
        videos = []
        try:
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                channel_info = ydl.extract_info(channel_url, download=False)
                
                if 'entries' in channel_info:
                    for entry in channel_info['entries']:
                        if entry:
                            video_info = {
                                'title': entry.get('title', 'Unknown'),
                                'url': entry.get('webpage_url', ''),
                                'upload_date': entry.get('upload_date', ''),
                                'duration': entry.get('duration', 0),
                                'view_count': entry.get('view_count', 0),
                                'description': entry.get('description', ''),
                                'creator': creator_name,
                                'channel_name': creator_name,  # Add channel_name for consistency
                                'id': entry.get('id', ''),
                                'thumbnail': entry.get('thumbnail', ''),
                            }
                            videos.append(video_info)
                
                print(f"✅ Found {len(videos)} videos from {creator_name}")
                
        except Exception as e:
            print(f"❌ Error getting videos from {creator_name}: {e}")
        
        return videos
    
    def download_video(self, video_url: str, creator_name: str) -> Optional[str]:
        """Download a single video and return the file path."""
        print(f"🎥 Downloading: {video_url}")
        
        # Create creator-specific directory
        creator_dir = self.output_dir / creator_name
        creator_dir.mkdir(exist_ok=True)
        
        # Update output template for this download
        download_opts = self.ydl_opts.copy()
        download_opts['outtmpl'] = str(creator_dir / '%(title)s.%(ext)s')
        
        try:
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                # Get video info first
                video_info = ydl.extract_info(video_url, download=False)
                video_title = video_info.get('title', 'Unknown')
                
                # Download the video
                ydl.download([video_url])
                
                # Find the downloaded file
                for file_path in creator_dir.glob(f"{video_title}.*"):
                    if file_path.suffix in ['.m4a', '.mp3', '.webm']:
                        return str(file_path)
                
        except Exception as e:
            print(f"❌ Error downloading video: {e}")
        
        return None
    
    async def generate_transcript(self, audio_file_path: str, video_title: str) -> Optional[Dict]:
        """Generate timed transcript using OpenAI Whisper."""
        print(f"🎤 Generating transcript for: {video_title}")
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                transcript = await openai.Audio.atranscribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            # Process the transcript
            processed_transcript = {
                'title': video_title,
                'segments': transcript.get('segments', []),
                'words': transcript.get('words', []),
                'language': transcript.get('language', 'en'),
                'duration': transcript.get('duration', 0),
                'text': transcript.get('text', ''),
                'generated_at': datetime.now().isoformat()
            }
            
            print(f"✅ Transcript generated successfully")
            return processed_transcript
            
        except Exception as e:
            print(f"❌ Error generating transcript: {e}")
            return None
    
    def save_transcript(self, transcript: Dict, creator_name: str, video_title: str):
        """Save transcript to JSON file."""
        # Clean filename
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        # Create transcripts directory
        transcripts_dir = self.output_dir / creator_name / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        
        # Save transcript
        transcript_file = transcripts_dir / f"{safe_title}_transcript.json"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Transcript saved to: {transcript_file}")
        return str(transcript_file)
    
    def create_readable_transcript(self, transcript: Dict, creator_name: str, video_title: str):
        """Create a human-readable transcript file."""
        # Clean filename
        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        # Create transcripts directory
        transcripts_dir = self.output_dir / creator_name / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)
        
        # Create readable transcript
        readable_file = transcripts_dir / f"{safe_title}_readable.txt"
        
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(f"Transcript for: {video_title}\n")
            f.write(f"Creator: {creator_name}\n")
            f.write(f"Generated: {transcript['generated_at']}\n")
            f.write(f"Duration: {transcript['duration']:.2f} seconds\n")
            f.write("=" * 80 + "\n\n")
            
            # Write segments with timestamps
            for segment in transcript['segments']:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                if text:
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {text}\n\n")
        
        print(f"📝 Readable transcript saved to: {readable_file}")
        return str(readable_file)
    
    async def process_creator(self, creator_name: str, creator_info: Dict) -> List[Dict]:
        """Process all videos for a single creator."""
        print(f"\n🎬 Processing creator: {creator_name}")
        print(f"📺 Channel: {creator_info['url']}")
        print(f"📝 Description: {creator_info['description']}")
        print("-" * 60)
        
        # Get recent videos
        videos = self.get_channel_videos(creator_info['url'], creator_name)
        
        if not videos:
            print(f"⚠️  No videos found for {creator_name}")
            return []
        
        processed_videos = []
        
        for i, video in enumerate(videos[:self.max_videos], 1):
            print(f"\n[{i}/{len(videos[:self.max_videos])}] Processing: {video['title']}")
            
            # Download video
            audio_file = self.download_video(video['url'], creator_name)
            
            if not audio_file:
                print(f"❌ Failed to download video: {video['title']}")
                continue
            
            # Generate transcript
            transcript = await self.generate_transcript(audio_file, video['title'])
            
            if transcript:
                # Save transcript files
                json_file = self.save_transcript(transcript, creator_name, video['title'])
                readable_file = self.create_readable_transcript(transcript, creator_name, video['title'])
                
                # Add file paths to video info
                video['audio_file'] = audio_file
                video['transcript_json'] = json_file
                video['transcript_readable'] = readable_file
                video['transcript_data'] = transcript
                
                processed_videos.append(video)
                
                # Clean up audio file to save space
                try:
                    os.remove(audio_file)
                    print(f"🗑️  Cleaned up audio file: {audio_file}")
                except:
                    pass
            else:
                print(f"❌ Failed to generate transcript for: {video['title']}")
        
        return processed_videos
    
    async def process_all_creators(self) -> Dict[str, List[Dict]]:
        """Process all creators and their videos."""
        print("🚀 Starting YouTube Creator Video Download and Transcript Generation")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎥 Max videos per creator: {self.max_videos}")
        print("=" * 80)
        
        all_results = {}
        
        for creator_name, creator_info in CREATORS.items():
            try:
                videos = await self.process_creator(creator_name, creator_info)
                all_results[creator_name] = videos
                
                print(f"\n✅ Completed {creator_name}: {len(videos)} videos processed")
                
            except Exception as e:
                print(f"❌ Error processing {creator_name}: {e}")
                all_results[creator_name] = []
        
        return all_results
    
    def save_summary_report(self, results: Dict[str, List[Dict]]):
        """Save a summary report of all processed videos."""
        summary_file = self.output_dir / "processing_summary.json"
        
        summary = {
            'processed_at': datetime.now().isoformat(),
            'total_creators': len(results),
            'total_videos': sum(len(videos) for videos in results.values()),
            'creators': {}
        }
        
        for creator_name, videos in results.items():
            summary['creators'][creator_name] = {
                'videos_processed': len(videos),
                'videos': [
                    {
                        'title': video['title'],
                        'url': video['url'],
                        'upload_date': video['upload_date'],
                        'duration': video['duration'],
                        'transcript_json': video.get('transcript_json', ''),
                        'transcript_readable': video.get('transcript_readable', '')
                    }
                    for video in videos
                ]
            }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Summary report saved to: {summary_file}")
        return str(summary_file)


async def main():
    parser = argparse.ArgumentParser(
        description="Download videos from YouTube creators and generate transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all creators (default 5 videos each)
  python youtube_transcript_downloader.py
  
  # Process with custom video limit
  python youtube_transcript_downloader.py --max-videos 3
  
  # Custom output directory
  python youtube_transcript_downloader.py --output-dir "my_videos"
        """
    )
    
    parser.add_argument('--max-videos', type=int, default=5,
                       help='Maximum number of videos to process per creator (default: 5)')
    parser.add_argument('--output-dir', default='creator_videos',
                       help='Output directory for downloads and transcripts (default: creator_videos)')
    parser.add_argument('--creator', choices=list(CREATORS.keys()),
                       help='Process only a specific creator')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = YouTubeTranscriptDownloader(
        output_dir=args.output_dir,
        max_videos=args.max_videos
    )
    
    try:
        if args.creator:
            # Process single creator
            creator_info = CREATORS[args.creator]
            results = {args.creator: await downloader.process_creator(args.creator, creator_info)}
        else:
            # Process all creators
            results = await downloader.process_all_creators()
        
        # Save summary report
        downloader.save_summary_report(results)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("🎉 PROCESSING COMPLETE!")
        print("=" * 80)
        
        total_videos = sum(len(videos) for videos in results.values())
        print(f"📊 Total videos processed: {total_videos}")
        
        for creator_name, videos in results.items():
            print(f"📺 {creator_name}: {len(videos)} videos")
        
        print(f"\n📁 Check the '{args.output_dir}' directory for:")
        print("   • Downloaded video metadata")
        print("   • JSON transcripts with timestamps")
        print("   • Human-readable transcript files")
        print("   • Processing summary report")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 