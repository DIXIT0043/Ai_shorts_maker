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
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from youtube_transcript_api import YouTubeTranscriptApi
groq_api_key = os.getenv("GROQ_API_KEY")

"""Cell 3: Download YouTube Video function"""

def download_video(url, filename):
    yt = YouTube(url)
    # Get the highest resolution stream that's 1080p or lower
    video = yt.streams.filter(progressive=True, file_extension='mp4', resolution='1080p').first()
    if not video:
        # If 1080p is not available, get the highest available resolution
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
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
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="deepseek-r1-distill-llama-70b",
        )
    # Initialize Ollama
    # llm = Ollama(
    #     model="mistral",
    #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    #     temperature=0.7,
    # )
    
    # Create the prompt template with more explicit instructions and example
    template = """You are a ViralGPT helpful assistant. You are master at reading youtube transcripts and identifying the most Interesting and Viral Content.
    
    This is a transcript of a video. Please identify 3-7 segments (20-30 seconds each) that have the highest potential to go viral on social media. Consider factors like:
    - Engaging storytelling
    - Emotional moments 
    - Surprising revelations
    - Humorous content
    - Valuable insights
    
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
    1. Each segment should be 20-30 seconds in duration
    2. Use the exact timestamps from the transcript
    3. Include a brief description of what happens in each segment
    4. Calculate duration as end_time - start_time
    6. DO NOT include any thinking or explanation, just return the JSON array
    7. The transcript is in Hindi, but provide descriptions in English
    8. Make sure to select segments with high viral potential based on the factors above
    
    Here is the Transcription:
    {transcript}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Format the messages
    messages = prompt.format_messages(
        transcript=transcript
    )
    
    # Get the response
    response = llm.invoke(messages[0].content)
    
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
            required_fields = ["start_time", "end_time", "description", "duration"]
            if not all(field in segment for field in required_fields):
                print(f"Segment missing required fields: {segment}")
                return []
            
            # Validate duration
            calculated_duration = segment["end_time"] - segment["start_time"]
            if abs(calculated_duration - segment["duration"]) > 0.1:
                segment["duration"] = calculated_duration
        
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