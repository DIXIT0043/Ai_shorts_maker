# Viral Shorts Extractor

This project analyzes videos (particularly podcasts) to identify and extract segments that have high viral potential for social media platforms. It uses AI to analyze video transcripts and automatically create short-form content.

## Features

- Video download from URLs
- Automatic video transcription using Whisper
- AI-powered analysis of viral potential using Groq
- Automatic extraction of viral-worthy segments (20-30 seconds)
- Support for multiple viral segments per video

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Groq API key:
```bash
export GROQ_API_KEY='your-api-key-here'
```

## Usage

```python
from app import ShortsExtractor

# Initialize the extractor
extractor = ShortsExtractor(groq_api_key="your-api-key-here")

# Process a video
video_url = "https://example.com/video-url"
output_dir = "output_shorts"

# Extract viral shorts
shorts = extractor.process_video(video_url, output_dir)

# Print generated shorts
for short in shorts:
    print(f"Generated short: {short}")
```

## How it Works

1. The video is downloaded from the provided URL
2. The video is transcribed using OpenAI's Whisper model
3. The transcript is analyzed by Groq's LLM to identify viral-worthy segments
4. The identified segments are extracted and saved as separate video files
5. The original video is cleaned up automatically

## Requirements

- Python 3.8+
- FFmpeg (for video processing)
- Groq API key

## Notes

- The system is optimized for podcast-style content
- Each extracted short is 20-30 seconds long
- Multiple viral segments can be extracted from a single video
- The output videos are saved in MP4 format with H.264 video codec and AAC audio codec 