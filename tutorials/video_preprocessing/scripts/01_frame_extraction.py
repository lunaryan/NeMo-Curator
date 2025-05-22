import os
import subprocess
import argparse
import cv2

def extract_frames_ffmpeg(input_video, output_dir, fps, frame_pattern):
    """Extracts frames from a video using FFmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    output_path_pattern = os.path.join(output_dir, frame_pattern)
    
    # Construct FFmpeg command with quotes for paths
    command = [
        'ffmpeg', '-i', input_video,
        '-vf', f'fps={fps}',
        output_path_pattern
    ]
    
    try:
        # Execute FFmpeg command
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"FFmpeg output:\n{process.stdout}")
        print(f"Frames extracted successfully to {output_dir} using FFmpeg.")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution: {e}")
        print(f"FFmpeg stderr:\n{e.stderr}")
    except FileNotFoundError:
        print("Error: FFmpeg command not found. Please ensure FFmpeg is installed and in your PATH.")

def extract_frames_opencv(input_video, output_dir, target_fps, frame_pattern):
    """Extracts frames from a video using OpenCV."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("Warning: Video FPS is 0. Cannot calculate frame skip interval. Extracting all frames.")
        frame_skip_interval = 1
    elif target_fps <= 0:
        print("Warning: Target FPS is 0 or less. Extracting all frames.")
        frame_skip_interval = 1
    else:
        frame_skip_interval = max(1, round(original_fps / target_fps))

    frame_counter = 0
    save_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_counter % frame_skip_interval == 0:
            frame_filename = os.path.join(output_dir, frame_pattern % (save_counter + 1)) # Use save_counter + 1 for 1-based indexing
            cv2.imwrite(frame_filename, frame)
            save_counter += 1
            
        frame_counter += 1
            
    cap.release()
    print(f"{save_counter} frames extracted successfully to {output_dir} using OpenCV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--input_video", required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted frames.")
    parser.add_argument("--method", default="ffmpeg", choices=["ffmpeg", "opencv"], help="Frame extraction method (ffmpeg or opencv).")
    parser.add_argument("--fps", type=float, default=1.0, help="Target frames per second to extract.")
    parser.add_argument("--frame_pattern", default="frame_%06d.png", help="Naming pattern for output frames (e.g., frame_%%06d.png).")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
    else:
        if args.method == "ffmpeg":
            extract_frames_ffmpeg(args.input_video, args.output_dir, args.fps, args.frame_pattern)
        elif args.method == "opencv":
            extract_frames_opencv(args.input_video, args.output_dir, args.fps, args.frame_pattern)
        else:
            print("Error: Invalid method specified. Choose 'ffmpeg' or 'opencv'.")
