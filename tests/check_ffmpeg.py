#!/usr/bin/env python3
"""
Check ffmpeg Installation

Simple script to verify that ffmpeg and ffprobe are properly installed.
"""

import subprocess
import sys


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        version_line = result.stdout.split('\n')[0]
        print(f"✅ ffmpeg found: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found")
        return False


def check_ffprobe():
    """Check if ffprobe is available."""
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, check=True)
        version_line = result.stdout.split('\n')[0]
        print(f"✅ ffprobe found: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffprobe not found")
        return False


def main():
    """Check ffmpeg installation."""
    print("🔍 Checking ffmpeg installation...")
    print("=" * 50)
    
    ffmpeg_ok = check_ffmpeg()
    ffprobe_ok = check_ffprobe()
    
    print("\n" + "=" * 50)
    if ffmpeg_ok and ffprobe_ok:
        print("🎉 ffmpeg installation is complete!")
        print("✅ Audio splitting functionality is available")
        return True
    else:
        print("❌ ffmpeg installation is incomplete")
        print("\n💡 Installation instructions:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   Or visit: https://ffmpeg.org/download.html")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 