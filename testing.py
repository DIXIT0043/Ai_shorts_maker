from pytubefix import YouTube
import subprocess
import openai
import numpy as np
import json
import math
import pdb
# from groq import Groq
# from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from moviepy.editor import VideoFileClip
import speech_recognition as sr 
from pydub import AudioSegment
import tempfile
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from youtube_transcript_api import YouTubeTranscriptApi
# groq_api_key = os.getenv("GROQ_API_KEY")

import google.generativeai as genai  # Add this import
from urllib.error import HTTPError
import time
import random


def download_video(url, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            yt = YouTube(url)
            # Get the highest quality stream available
            video = yt.streams.get_highest_resolution()
            if not video:
                # Fallback to any available stream
                video = yt.streams.first()
            
            if not video:
                raise Exception("No suitable video stream found")
                
            # Add a small delay before downloading to avoid rate limiting
            time.sleep(random.uniform(1, 3))
            
            video.download(filename=filename)
            print(f"Successfully downloaded video to {filename}")
            return
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                # Add exponential backoff between retries
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Download failed.")
                raise Exception(f"Failed to download video after {max_retries} attempts: {str(e)}")


#Segment Video function
def segment_video(response):
    try:
        # Load the video file
        video = VideoFileClip("input_video.mp4")
        
        for i, segment in enumerate(response):
            try:
                start_time = float(segment.get("start_time", 0))
                end_time = float(segment.get("end_time", 0))
                
                # Basic validation only
                if start_time >= end_time:
                    print(f"Invalid time range for segment {i+1}: start_time ({start_time}) >= end_time ({end_time})")
                    continue

                # Extract the segment exactly as specified by LLM
                video_segment = video.subclip(start_time, end_time)
                
                # Save the segment with high quality settings
                output_file = f"output{str(i).zfill(3)}.mp4"
                try:
                    video_segment.write_videofile(
                        output_file,
                        codec='libx264',
                        audio_codec='aac',
                        bitrate='10000k',  # High bitrate for better quality
                        preset='slow',    # Better compression
                        threads=4,        # Use multiple threads
                        fps=30,          # Maintain original frame rate
                        ffmpeg_params=[
                            '-crf', '10',  # High quality (lower is better, range 0-51)
                            '-profile:v', 'high',
                            '-level', '4.0',
                            '-pix_fmt', 'yuv420p'
                        ],
                        logger=None,
                        verbose=False
                    )
                    print(f"Successfully created segment {i+1} (duration: {end_time - start_time:.2f}s)")
                except Exception as e:
                    print(f"Error writing segment {i+1}: {str(e)}")
                    continue
                finally:
                    try:
                        video_segment.close()
                    except:
                        pass
                
            except Exception as e:
                print(f"Error processing segment {i+1}: {str(e)}")
                continue
        
        # Close the main video
        try:
            video.close()
        except:
            pass
            
        return True
        
    except Exception as e:
        print(f"Error in segment_video: {str(e)}")
        try:
            video.close()
        except:
            pass
        return False


def get_transcript(video_id):
    try:
        # First try YouTubeTranscriptApi with Hindi
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
        except:
            # Then try English
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except:
                # Then try any available language
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                except:
                    # Fallback to pytube captions
                    print("Trying to get captions using pytube...")
                    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                    captions = yt.captions
                    
                    # Try to get Hindi captions first
                    caption = captions.get_by_language_code('hi')
                    if not caption:
                        # Then try English
                        caption = captions.get_by_language_code('en')
                    if not caption:
                        # If no Hindi or English captions, get the first available caption
                        caption = list(captions.values())[0]
                    
                    if caption:
                        # Convert caption to transcript format
                        transcript = []
                        for caption_item in caption.generate_srt_captions().split('\n\n'):
                            if caption_item.strip():
                                lines = caption_item.strip().split('\n')
                                if len(lines) >= 3:
                                    time_line = lines[1]
                                    text = ' '.join(lines[2:])
                                    
                                    # Parse time
                                    start_time = time_line.split(' --> ')[0]
                                    end_time = time_line.split(' --> ')[1]
                                    
                                    # Convert time to seconds
                                    def time_to_seconds(time_str):
                                        h, m, s = time_str.replace(',', '.').split(':')
                                        return float(h) * 3600 + float(m) * 60 + float(s)
                                    
                                    transcript.append({
                                        'start': time_to_seconds(start_time),
                                        'duration': time_to_seconds(end_time) - time_to_seconds(start_time),
                                        'text': text
                                    })
                    else:
                        raise Exception("No captions found")
    except Exception as e:
        print(f"Could not get transcript from YouTube: {str(e)}")
        print("Attempting to generate transcript from audio...")
        return generate_transcript_from_audio()
    
    # Format the transcript
    formatted_transcript = ''
    for entry in transcript:
        start_time = "{:.2f}".format(entry['start'])
        end_time = "{:.2f}".format(entry['start'] + entry['duration'])
        text = entry['text']
        formatted_transcript += f"{start_time} --> {end_time} : {text}\n"
    
    return formatted_transcript

def generate_transcript_from_audio():
    try:
        # Load the video file
        video = VideoFileClip("input_video.mp4")
        
        # Extract audio
        audio = video.audio
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Save audio to temporary file
        audio.write_audiofile(temp_audio_path)
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Load audio file
        audio_segment = AudioSegment.from_wav(temp_audio_path)
        
        # Split audio into 30-second chunks
        chunk_length = 30 * 1000  # 30 seconds in milliseconds
        chunks = [audio_segment[i:i + chunk_length] for i in range(0, len(audio_segment), chunk_length)]
        
        formatted_transcript = ""
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Export chunk to temporary file
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            
            # Recognize speech in chunk
            with sr.AudioFile(chunk_path) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    start_time = i * 30
                    end_time = (i + 1) * 30
                    formatted_transcript += f"{start_time:.2f} --> {end_time:.2f} : {text}\n"
                except sr.UnknownValueError:
                    print(f"Could not understand audio in chunk {i}")
                except sr.RequestError as e:
                    print(f"Error with the speech recognition service: {e}")
            
            # Clean up chunk file
            os.remove(chunk_path)
        
        # Clean up temporary files
        os.remove(temp_audio_path)
        video.close()
        
        return formatted_transcript
        
    except Exception as e:
        print(f"Error generating transcript from audio: {str(e)}")
        return ""


#Analyze transcript with GPT-3 function
def analyze_transcript(transcript):
    template = """
    Here is the transcript to analyze: {transcript}
    You are a viral content curator specialized in identifying highly engaging short-form video segments. 
    Your task is to identify 3-7 most viral viral-worthy segments from the transcript that would perform well on platforms like YouTube Shorts, Instagram Reels.
    STRICTLY THE GENERATED SEGMENTS SHOULD BE SELF-CONTAINED AND MAKE SENSE(THEY SHOULD MAKE SENSE EVEN IF THEY ARE TAKEN OUT OF CONTEXT) ON THEIR OWN.
    KEY REQUIREMENTS:
    1. Segment Length: MUST be 25-70(STRICTLY) seconds long (optimal for short-form platforms)
    2. Content Type: Must be engaging, shareable, and self-contained
    3. Content Categories to Look For:
       - Funny moments or jokes
       - Surprising revelations or plot twists
       - Emotional or inspiring moments
       - Useful tips or life hacks
       - Interesting facts or educational bits
       - Dramatic or intense scenes
       - Relatable situations
       - Impressive skills or talents

    OUTPUT FORMAT:
    Return a JSON array with ONLY the segments that meet ALL requirements:
    [
        {{
            "start_time": 12.84,
            "end_time": 45.56,
            "description": "Brief description of the segment and why it's viral-worthy",
            "duration": 32.72,
            "category": "One of: Humor, Education, Inspiration, Drama, Surprise, Tutorial, Talent"
        }}
    ]

    IMPORTANT RULES:
    - REJECT any segment longer than 70 seconds
    - REJECT any segment shorter than 25 seconds
    - REJECT any segment without a clear hook
    - REJECT any segment with technical issues
    - Each segment must be self-contained and make sense on its own
    - Prioritize segments with strong emotional impact or value
    - Look for moments that would make viewers want to share or rewatch
"""
    
    # Google Gemini API setup
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)
    
    # Use gemini-2.0-flash model
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Format prompt for Gemini
    prompt = template.format(transcript=transcript)

    # Get the response from Gemini
    response = model.generate_content(prompt)
    content = response.text.strip()
    
    # Parse the response content
    try:
        # Clean the response content to ensure it's valid JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # Remove any thinking or explanation text
        if '<think>' in content:
            content = content[content.find('['):content.rfind(']')+1]
        
        # If content is empty or doesn't start with '[', return empty list
        if not content or not content.startswith('['):
            print("Invalid response format from LLM")
            return []
            
        parsed_response = json.loads(content)
        
        # Validate the response format
        if not isinstance(parsed_response, list):
            print("Response is not a list")
            return []
        
        # Ensure each segment has the required fields
        for segment in parsed_response:
            required_fields = ["start_time", "end_time", "description", "duration", "category"]
            if not all(field in segment for field in required_fields):
                print(f"Segment missing required fields: {segment}")
                return []
            
            # Update duration in the segment
            segment["duration"] = segment["end_time"] - segment["start_time"]
        
        return parsed_response
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {str(e)}")
        print("Raw response:", response)
        return []

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    import re
    
    # Regular expression patterns for different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?]+)',
        r'youtube\.com\/shorts\/([^&\n?]+)',
        r'youtube\.com\/v\/([^&\n?]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def main():
    # Get video URL from user input
    url = input("Enter YouTube video URL: ")
    
    # Extract video ID
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