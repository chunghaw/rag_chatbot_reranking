#!/usr/bin/env python3
"""
Test Script for YouTube Transcript System

Tests the YouTube transcript downloader with a single video to verify everything works
before running the full batch processing.
"""

import os
import sys
import asyncio
from pathlib import Path
from youtube_transcript_downloader import YouTubeTranscriptDownloader

# Test configuration
TEST_CREATOR = "Eczachly_"
TEST_VIDEOS = 1
TEST_OUTPUT_DIR = "test_videos"


async def test_single_creator():
    """Test the transcript system with a single creator and one video."""
    print("🧪 Testing YouTube Transcript System")
    print("=" * 50)
    
    # Initialize downloader with test settings
    downloader = YouTubeTranscriptDownloader(
        output_dir=TEST_OUTPUT_DIR,
        max_videos=TEST_VIDEOS
    )
    
    try:
        # Test with a single creator
        from youtube_transcript_downloader import CREATORS
        
        if TEST_CREATOR not in CREATORS:
            print(f"❌ Test creator '{TEST_CREATOR}' not found in CREATORS dictionary")
            return False
        
        creator_info = CREATORS[TEST_CREATOR]
        print(f"🎬 Testing with creator: {TEST_CREATOR}")
        print(f"📺 Channel: {creator_info['url']}")
        print(f"🎥 Max videos: {TEST_VIDEOS}")
        print(f"📁 Output directory: {TEST_OUTPUT_DIR}")
        print("-" * 50)
        
        # Process the test creator
        videos = await downloader.process_creator(TEST_CREATOR, creator_info)
        
        if videos:
            print(f"\n✅ Test successful! Processed {len(videos)} video(s)")
            
            # Print details of processed video
            for video in videos:
                print(f"\n📹 Video: {video['title']}")
                print(f"⏱️  Duration: {video.get('duration', 0)} seconds")
                print(f"📅 Upload date: {video.get('upload_date', 'Unknown')}")
                print(f"📝 Transcript JSON: {video.get('transcript_json', 'N/A')}")
                print(f"📖 Readable transcript: {video.get('transcript_readable', 'N/A')}")
            
            return True
        else:
            print("❌ No videos were processed successfully")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_rag_integration():
    """Test RAG integration with the generated transcript."""
    print("\n🔗 Testing RAG Integration")
    print("=" * 50)
    
    try:
        from add_transcripts_to_rag import TranscriptRAGIntegrator
        
        integrator = TranscriptRAGIntegrator(
            transcripts_dir=TEST_OUTPUT_DIR,
            api_url="http://localhost:8000"
        )
        
        # Load transcript files
        transcripts = integrator.load_transcript_files()
        
        if not transcripts:
            print("❌ No transcript files found for RAG integration test")
            return False
        
        print(f"📁 Found {len(transcripts)} transcript(s) to test")
        
        # Test with first transcript
        test_transcript = transcripts[0]
        print(f"🎬 Testing with: {test_transcript['title']}")
        
        # Create segments
        segments = integrator.create_segments_from_transcript(test_transcript)
        print(f"📝 Created {len(segments)} segments")
        
        if segments:
            # Test adding first segment to RAG
            print("🔗 Testing RAG API connection...")
            success = await integrator.add_segment_to_rag(segments[0])
            
            if success:
                print("✅ RAG integration test successful!")
                return True
            else:
                print("❌ RAG integration test failed")
                return False
        else:
            print("❌ No segments created from transcript")
            return False
            
    except Exception as e:
        print(f"❌ RAG integration test failed: {e}")
        return False


async def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    test_dir = Path(TEST_OUTPUT_DIR)
    if test_dir.exists():
        import shutil
        try:
            shutil.rmtree(test_dir)
            print(f"✅ Cleaned up {TEST_OUTPUT_DIR}")
        except Exception as e:
            print(f"⚠️  Could not clean up test files: {e}")


async def main():
    """Run the complete test suite."""
    print("🚀 YouTube Transcript System Test Suite")
    print("=" * 60)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("💡 Add it to your .env file: OPENAI_API_KEY=your_key_here")
        return False
    
    print("✅ OpenAI API key found")
    
    # Check if RAG system is running
    try:
        import requests
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ RAG system is running")
        else:
            print("⚠️  RAG system may not be running properly")
    except:
        print("⚠️  RAG system not accessible (make sure it's running on localhost:8000)")
    
    print("\n" + "=" * 60)
    
    # Run tests
    test_results = []
    
    # Test 1: Video download and transcription
    print("🧪 Test 1: Video Download & Transcription")
    result1 = await test_single_creator()
    test_results.append(("Video Download & Transcription", result1))
    
    # Test 2: RAG integration
    print("\n🧪 Test 2: RAG Integration")
    result2 = await test_rag_integration()
    test_results.append(("RAG Integration", result2))
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your YouTube transcript system is ready to use")
        print("\n💡 Next steps:")
        print("   1. Run: python youtube_transcript_downloader.py")
        print("   2. Run: python add_transcripts_to_rag.py")
        print("   3. Query your RAG system with transcript content")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please check the errors above and fix them before proceeding")
    
    # Ask if user wants to clean up test files
    print(f"\n🗑️  Test files are in: {TEST_OUTPUT_DIR}")
    print("💡 You can keep them for inspection or clean them up")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1) 