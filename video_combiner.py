from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
import os
import random

def combine_videos(top_video_path, bottom_video_path, output_path):
    # Load both videos
    top_clip = VideoFileClip(top_video_path)
    bottom_clip = VideoFileClip(bottom_video_path)
    
    # Mute the bottom video
    bottom_clip = bottom_clip.without_audio()
    
    # Get the duration of the top video
    target_duration = top_clip.duration
    
    # Randomly select a start time for the bottom video
    max_start_time = max(0, bottom_clip.duration - target_duration)
    start_time = random.uniform(0, max_start_time)
    
    # Trim the bottom video to match the top video's duration
    bottom_clip = bottom_clip.subclip(start_time, start_time + target_duration)
    
    # Calculate dimensions for 9:16 ratio
    # We'll use the width of the top video as reference
    target_width = top_clip.w
    target_height = int(target_width * 16 / 9)  # This gives us the full 9:16 height
    
    # Calculate individual heights for top and bottom
    half_height = target_height // 2
    
    # Resize both videos to fit the target width while maintaining aspect ratio
    top_clip = top_clip.resize(width=target_width)
    bottom_clip = bottom_clip.resize(width=target_width)
    
    # Create a black background with the correct 9:16 dimensions
    background = ColorClip(size=(target_width, target_height), color=(0, 0, 0), duration=target_duration)
    
    # Position the clips
    top_clip = top_clip.set_position(("center", 0))
    bottom_clip = bottom_clip.set_position(("center", half_height))
    
    # Combine all clips
    final_clip = CompositeVideoClip([background, top_clip, bottom_clip])
    
    # Write the output file with high quality settings
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        bitrate="8000k",  # High bitrate for better quality
        preset="slow",    # Better compression
        threads=4,        # Use multiple threads for faster processing
        fps=30           # Maintain original frame rate
    )

    top_clip.close()
    bottom_clip.close()
    final_clip.close()

if __name__ == "__main__":
    folder_1 = "./out_7"  # Folder containing shorts
    folder_2 = "./folder_3"  # Folder containing the single video
    output_folder = "./combined_videos"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the single video from folder_2
    bottom_video = None
    for filename in os.listdir(folder_2):
        if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            bottom_video = os.path.join(folder_2, filename)
            break
    
    if not bottom_video:
        print("No video found in folder_2")
        exit(1)
    
    # Process each video in folder_1
    for filename in os.listdir(folder_1):
        if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            top_video = os.path.join(folder_1, filename)
            output_video = os.path.join(output_folder, f"combined_{filename}")
            print(f"Processing {top_video} with {bottom_video} -> {output_video}")
            combine_videos(top_video, bottom_video, output_video) 