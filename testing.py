from pytubefix import YouTube
import subprocess
import openai
import numpy as np
import json
import math
import pdb
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from moviepy.editor import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
from youtube_transcript_api._errors import TranscriptsDisabled

groq_api_key = os.getenv("GROQ_API_KEY")

"""Cell 3: Download YouTube Video function"""

def download_video(url, filename):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension='mp4').first()
    video.download(filename=filename)

#Segment Video function
def segment_video(response):
    try:
        # Load the video file
        video = VideoFileClip("input_video.mp4")
        
        for i, segment in enumerate(response):
            start_time = float(segment.get("start_time", 0))
            end_time = float(segment.get("end_time", 0))
            
            # Extract the segment
            video_segment = video.subclip(start_time, end_time)
            
            # Save the segment
            output_file = f"output{str(i).zfill(3)}.mp4"
            video_segment.write_videofile(output_file, codec='libx264', audio_codec='aac')
            
            # Close the segment to free up resources
            video_segment.close()
            
            print(f"Successfully created segment {i+1}")
        
        # Close the main video
        video.close()
        return True
        
    except Exception as e:
        print(f"Error in segment_video: {str(e)}")
        return False

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file using moviepy"""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

def transcribe_with_whisper(audio_path):
    """Transcribe audio using OpenAI's Whisper"""
    try:
        # Load the Whisper model
        model = whisper.load_model("base")
        
        # Transcribe the audio
        result = model.transcribe(audio_path)
        
        # Format the transcript similar to YouTube transcript format
        formatted_transcript = ''
        for segment in result["segments"]:
            start_time = "{:.2f}".format(segment['start'])
            end_time = "{:.2f}".format(segment['end'])
            text = segment['text']
            formatted_transcript += f"{start_time} --> {end_time} : {text}\n"
            
        return formatted_transcript
    except Exception as e:
        print(f"Error in Whisper transcription: {str(e)}")
        return None

def get_transcript(video_id):
    try:
        # First try to get YouTube transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format the transcript
        formatted_transcript = ''
        for entry in transcript:
            start_time = "{:.2f}".format(entry['start'])
            end_time = "{:.2f}".format(entry['start'] + entry['duration'])
            text = entry['text']
            formatted_transcript += f"{start_time} --> {end_time} : {text}\n"
            
        return formatted_transcript
        
    except TranscriptsDisabled:
        print("YouTube transcript not available. Falling back to Whisper transcription...")
        
        # Extract audio from the downloaded video
        audio_path = "temp_audio.wav"
        if not extract_audio_from_video("input_video.mp4", audio_path):
            raise Exception("Failed to extract audio from video")
            
        # Transcribe using Whisper
        transcript = transcribe_with_whisper(audio_path)
        if transcript:
            # Clean up temporary audio file
            try:
                os.remove(audio_path)
            except:
                pass
            return transcript
        else:
            raise Exception("Failed to transcribe audio with Whisper")
            
    except Exception as e:
        print(f"Error getting transcript: {str(e)}")
        return None

#Analyze transcript with GPT-3 function
def analyze_transcript(transcript):
    # Create the chat model with correct initialization
    chat = ChatGroq(
        api_key=groq_api_key,
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.7,
        max_tokens=512
    )
    
    # Create the prompt template with more explicit instructions and example
    template = """You are a ViralGPT helpful assistant. You are master at reading youtube transcripts and identifying the most Interesting and Viral Content.
    
    This is a transcript of a video. Please identify the 3 most viral sections from the whole, make sure they are more than 30 seconds in duration.
    
    IMPORTANT: Your response MUST be a valid JSON array with exactly this format:
    [
        {{
            "start_time": 12.84,
            "end_time": 45.56,
            "description": "Discussion about unrealistic aspects of Indian spy movies",
            "duration": 32.72
        }},
        {{
            "start_time": 127.20,
            "end_time": 170.88,
            "description": "Explanation of real spy movie techniques and dead letter boxes",
            "duration": 43.68
        }},
        {{
            "start_time": 259.70,
            "end_time": 298.94,
            "description": "Discussion about Cambridge Five and recommendation of Tinker Tailor Soldier Spy",
            "duration": 39.24
        }}
    ]
    
    Rules:
    1. Each segment must be at least 30 seconds long
    2. Use the exact timestamps from the transcript
    3. Include a brief description of what happens in each segment
    4. Calculate duration as end_time - start_time
    5. Return exactly 3 segments
    6. DO NOT include any thinking or explanation, just return the JSON array
    
    Here is the Transcription:
    {transcript}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format the messages
    messages = prompt.format_messages(
        transcript=transcript
    )
    
    # Get the response
    response = chat.invoke(messages)
    
    # Parse the response content
    try:
        # Clean the response content to ensure it's valid JSON
        content = response.content.strip()
        # Remove any markdown code block indicators if present
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Remove any thinking or explanation text
        if '<think>' in content:
            content = content[content.find('['):content.rfind(']')+1]
        
        parsed_response = json.loads(content)
        
        # Validate the response format
        if not isinstance(parsed_response, list):
            raise ValueError("Response is not a list")
        
        # Ensure each segment has the required fields
        for segment in parsed_response:
            required_fields = ["start_time", "end_time", "description", "duration"]
            if not all(field in segment for field in required_fields):
                raise ValueError(f"Segment missing required fields: {segment}")
            
            # Validate duration
            calculated_duration = segment["end_time"] - segment["start_time"]
            if abs(calculated_duration - segment["duration"]) > 0.1:  # Allow small floating point differences
                segment["duration"] = calculated_duration
            
            # Ensure duration is at least 30 seconds
            if segment["duration"] < 30:
                raise ValueError(f"Segment duration is less than 30 seconds: {segment}")
        
        return parsed_response
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {str(e)}")
        print("Raw response:", response.content)
        return []

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    elif 'youtube.com' in url:
        if 'v=' in url:
            return url.split('v=')[1].split('&')[0]
    return None

def main():
    # Get video URL from user input
    url = input("Enter YouTube video URL: ")
    video_id = extract_video_id(url)
    
    if not video_id:
        print("Invalid YouTube URL. Please provide a valid YouTube video URL.")
        return
        
    filename = 'input_video.mp4'
    download_video(url, filename)
    
    transcript = get_transcript(video_id)
    print("Transcript:", transcript)
    interesting_segments = analyze_transcript(transcript)
    print("Interesting segments:", interesting_segments)
    
    if interesting_segments:
        if segment_video(interesting_segments):
            print("Successfully created video segments")
        else:
            print("Failed to create video segments")
    else:
        print("No valid segments found to process")

# Run the main function
if __name__ == "__main__":
    main()