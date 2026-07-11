import os
import requests
import cv2
import tempfile

def download_video(url, save_path):
    """Downloads the video file from the provided URL."""
    print(f"Downloading video from {url}...")
    try:
        # Stream the download so we don't overload memory with large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() # Check for HTTP errors
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download video: {e}")
        return False
def resize_frame(frame,target_width=1024):
    '''
    Resizes a frame down to target_width while preserving aspect ratio
    '''
    height, width=frame.shape[:2]
    if width<target_width:
        return frame #no upscaling for small frames
    
    scale=target_width/width
    new_ht=int(height*scale)
    return cv2.resize(frame, (target_width,new_ht),interpolation=cv2.INTER_AREA)
def extract_frames(video_path, max_frames=10):
    """
    Extracts evenly spaced frames from the video.
    Returns a list of file paths to the extracted images.
    """
    print(f"Extracting up to {max_frames} frames...")
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video file.")
        return []

    # Get total frames to calculate interval
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames<=0:
        print("WARNING: frame count not available, fall back to sequence read")
        interval=1

    else:
        interval=max(1, total_frames//max_frames)

    # Calculate how many frames to skip to get 'max_frames' evenly spaced


    extracted_frame_paths = []
    temp_dir = tempfile.mkdtemp() # Create a temporary folder for images

    current_frame = 0
    saved_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break # End of video
            
        if current_frame % interval == 0 and saved_count < max_frames:
            # Save the frame as a JPEG
            frame_filename = os.path.join(temp_dir, f"frame_{saved_count:03d}.jpg")
            #Downscaling the image from large resolutions like 4K but preserving an acceptable quality for faster API calls
            frame=resize_frame(frame, target_width=1024)
            cv2.imwrite(frame_filename, frame)
            extracted_frame_paths.append(frame_filename)
            saved_count += 1
            
        current_frame += 1

    video.release()
    print(f"Extracted {len(extracted_frame_paths)} frames.")
    return extracted_frame_paths