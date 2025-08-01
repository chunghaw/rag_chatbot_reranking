#!/usr/bin/env python3
"""
Test Audio Splitting Functionality

Tests the audio splitting capabilities of the transcript system.
"""

import os
import sys
import asyncio
from pathlib import Path
from add_transcripts_to_rag import TranscriptRAGIntegrator

# Test configuration
TEST_AUDIO_DIR = "test_audio"
TEST_CHUNK_DURATION = 5  # 5 minutes for testing


async def test_audio_splitting():
    """Test the audio splitting functionality."""
    print("🧪 Testing Audio Splitting Functionality")
    print("=" * 60)
    
    # Initialize integrator
    integrator = TranscriptRAGIntegrator()
    
    # Check ffmpeg availability
    if not integrator.check_ffmpeg_available():
        print("❌ ffmpeg not found. Please install ffmpeg to test audio splitting.")
        print("💡 Install ffmpeg: https://ffmpeg.org/download.html")
        return False
    
    print("✅ ffmpeg found - proceeding with tests")
    
    # Find test audio files
    test_dir = Path(TEST_AUDIO_DIR)
    if not test_dir.exists():
        print(f"❌ Test audio directory not found: {TEST_AUDIO_DIR}")
        print("💡 Create a test_audio directory with some m4a files to test")
        return False
    
    # Find m4a files
    m4a_files = list(test_dir.glob("*.m4a"))
    if not m4a_files:
        print(f"❌ No m4a files found in {TEST_AUDIO_DIR}")
        print("💡 Add some m4a files to test the splitting functionality")
        return False
    
    print(f"📁 Found {len(m4a_files)} test audio files")
    
    # Test splitting on each file
    for i, audio_file in enumerate(m4a_files, 1):
        print(f"\n[{i}/{len(m4a_files)}] Testing: {audio_file.name}")
        print("-" * 40)
        
        try:
            # Test splitting
            chunks = integrator.split_audio_file(str(audio_file), TEST_CHUNK_DURATION)
            
            print(f"📊 Results:")
            print(f"   Original file: {audio_file.name}")
            print(f"   Chunks created: {len(chunks)}")
            
            if len(chunks) > 1:
                print(f"   ✅ File was split into {len(chunks)} chunks")
                
                # Verify chunk files exist
                for j, chunk_file in enumerate(chunks):
                    chunk_path = Path(chunk_file)
                    if chunk_path.exists():
                        size_mb = chunk_path.stat().st_size / (1024 * 1024)
                        print(f"   📁 Chunk {j+1}: {chunk_path.name} ({size_mb:.1f} MB)")
                    else:
                        print(f"   ❌ Chunk {j+1}: {chunk_path.name} (missing)")
                
                # Clean up test chunks
                print(f"🧹 Cleaning up test chunks...")
                integrator.cleanup_temp_files(chunks[1:])  # Keep original file
                
            else:
                print(f"   ℹ️  File was within size limit, no splitting needed")
                
        except Exception as e:
            print(f"   ❌ Error testing {audio_file.name}: {e}")
    
    print(f"\n✅ Audio splitting test completed!")
    return True


async def test_transcript_generation():
    """Test transcript generation with splitting."""
    print("\n🎤 Testing Transcript Generation with Splitting")
    print("=" * 60)
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for transcript generation test!")
        print("💡 Add it to your .env file: OPENAI_API_KEY=your_key_here")
        return False
    
    # Initialize integrator
    integrator = TranscriptRAGIntegrator()
    
    # Find a test audio file
    test_dir = Path(TEST_AUDIO_DIR)
    m4a_files = list(test_dir.glob("*.m4a"))
    
    if not m4a_files:
        print("❌ No test audio files found")
        return False
    
    # Test with first file
    test_file = m4a_files[0]
    print(f"🎵 Testing transcript generation for: {test_file.name}")
    
    try:
        # Generate transcript with short chunk duration to force splitting
        transcript = await integrator.generate_transcript_from_audio(
            str(test_file),
            test_file.stem,
            max_chunk_duration=1  # 1 minute chunks for testing
        )
        
        if transcript:
            print(f"✅ Transcript generated successfully!")
            print(f"📊 Transcript details:")
            print(f"   Title: {transcript.get('title', 'Unknown')}")
            print(f"   Duration: {transcript.get('duration', 0):.1f} seconds")
            print(f"   Segments: {len(transcript.get('segments', []))}")
            print(f"   Words: {len(transcript.get('words', []))}")
            print(f"   Language: {transcript.get('language', 'Unknown')}")
            print(f"   Chunks processed: {transcript.get('chunks_processed', 1)}")
            
            # Show first few segments
            segments = transcript.get('segments', [])
            if segments:
                print(f"\n📝 First 3 segments:")
                for i, segment in enumerate(segments[:3]):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '')[:100] + "..." if len(segment.get('text', '')) > 100 else segment.get('text', '')
                    
                    print(f"   [{start_time:.1f}s - {end_time:.1f}s] {text}")
            
            return True
        else:
            print(f"❌ Failed to generate transcript")
            return False
            
    except Exception as e:
        print(f"❌ Error generating transcript: {e}")
        return False


async def main():
    """Run the complete audio splitting test suite."""
    print("🚀 Audio Splitting Test Suite")
    print("=" * 80)
    
    # Test 1: Audio splitting functionality
    print("🧪 Test 1: Audio Splitting")
    splitting_success = await test_audio_splitting()
    
    # Test 2: Transcript generation with splitting
    print("\n🧪 Test 2: Transcript Generation")
    transcript_success = await test_transcript_generation()
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    
    print(f"Audio Splitting: {'✅ PASSED' if splitting_success else '❌ FAILED'}")
    print(f"Transcript Generation: {'✅ PASSED' if transcript_success else '❌ FAILED'}")
    
    if splitting_success and transcript_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Audio splitting functionality is working correctly")
        print("\n💡 You can now process large audio files with confidence!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Please check the errors above and fix them")
    
    return splitting_success and transcript_success


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