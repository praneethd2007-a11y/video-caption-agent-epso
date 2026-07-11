"""
Video Captioning Agent - Track 2 Submission
This script reads tasks from a JSON input, processes video URLs, 
and generates captions in requested styles using AI inference.
"""

import json
import os
import sys
import tempfile
import video_utils  
import ai_inference

#FILE_PATH = "/input/tasks.json"
FILE_PATH = os.environ.get("TASKS_PATH", "tasks.json")
OUTPUT_FILE = os.environ.get("RESULTS_PATH", "results.json")

def read_tasks(inputpath):
    """
    Processing input
    """
    if not os.path.exists(inputpath):
        print(f"Error: Input file not found at {inputpath}")
        sys.exit(1)

    with open(inputpath, 'r', encoding="utf-8") as file:
        try:
            tasks = json.load(file)
            return tasks
        except json.JSONDecodeError:
            print("Error: Malformed JSON in input file")
            sys.exit(1)

def main():
    """
    Main entry point for the hackathon video captioning agent.
    Reads tasks, extracts frames, and pings the Fireworks AI API.
    """
    print("Initializing Video Captioning Agent...")
    
    tasks=read_tasks(FILE_PATH)
        
    print(f"Successfully loaded {len(tasks)} tasks.\n")

    # This list will hold all our final answers for the judges
    all_results = []

    for task in tasks:
        task_id = task.get("task_id")
        video_url = task.get("video_url")
        styles = task.get("styles", [])
        
        print(f"--- Processing Task ID: {task_id} ---")
        print(f"Target Video URL: {video_url}")
        
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        
        # Dictionary to hold the captions for THIS specific task
        task_captions = {}
        
        if video_utils.download_video(video_url, temp_video_path):
            frame_paths = video_utils.extract_frames(temp_video_path, max_frames=10)
            
            # --- STEP 4: Hand frames to the AI ---
            if frame_paths:
                for style in styles:
                    # Call Fireworks for each requested style!
                    caption = ai_inference.generate_caption(frame_paths, style)
                    task_captions[style] = caption
            else:
                print("No frames extracted. Marking all styles as failed for this task.")
                for style in styles:
                    task_captions[style] = f"[CAPTION_FAILED: '{style}' - no frames extracted from video]"

            # Clean up the temporary files so we don't run out of memory
            for path in frame_paths:
                if os.path.exists(path):
                    os.remove(path)
            
        else:
            print("Download failed. Marking all styles as failed for this task.")
            for style in styles:
                task_captions[style] = f"[CAPTION_FAILED: '{style}' - video download failed]"
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        # Add this task's completed captions to our final results list
        all_results.append({
            "task_id": task_id,
            "captions": task_captions
        })
        print(f"Task {task_id} complete!\n")

    # --- STEP 5: Save everything to results.json ---
    print("Writing final results to JSON...")
    with open(OUTPUT_FILE, 'w', encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Success! Check the {OUTPUT_FILE} file in your folder.")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)
