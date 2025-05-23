import os
import subprocess
import shutil # For checking ffmpeg path and rmtree

def extract_all_frames(video_filepath: str, 
                           output_dir: str, 
                           ffmpeg_path: str = "ffmpeg",
                           frame_filename_pattern: str = "frame_%07d.jpg",
                           target_fps: float = -1.0) -> tuple[list[str], int]:
    """
    Extracts all frames from a video file using FFmpeg.

    Args:
        video_filepath (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        ffmpeg_path (str): Path to the FFmpeg executable.
        frame_filename_pattern (str): Output filename pattern for frames 
                                      (e.g., "frame_%07d.jpg").
        target_fps (float): If > 0, specifies the target FPS for extraction. 
                            If <=0, extracts all frames at original video FPS.

    Returns:
        tuple[list[str], int]: A tuple containing:
            - A list of file paths to the extracted frames (if successful).
            - The total number of frames extracted.
        Returns ([], 0) if extraction fails or no frames are extracted.
    """
    if not os.path.exists(video_filepath):
        print(f"Error: Video file not found: {video_filepath}")
        return [], 0

    if not shutil.which(ffmpeg_path):
        print(f"Error: FFmpeg not found at path: {ffmpeg_path}. Please install FFmpeg "
              "or provide the correct path.")
        return [], 0

    os.makedirs(output_dir, exist_ok=True)
    
    # Clean the output directory before extraction to ensure fresh frames
    # and accurate count from listing directory later.
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Warning: Failed to delete {item_path}. Reason: {e}")


    output_pattern_for_ffmpeg = os.path.join(output_dir, frame_filename_pattern)
    
    cmd = [
        ffmpeg_path,
        "-i", video_filepath,
        "-hide_banner", # Suppress version and build info
        "-loglevel", "warning", # Print warnings and errors
    ]

    if target_fps > 0:
        cmd.extend(["-vf", f"fps={target_fps}"])
    
    # Standard arguments for image sequence output
    cmd.extend(["-start_number", "0"]) # Ensure numbering starts from 0 if pattern supports it
    cmd.append(output_pattern_for_ffmpeg)

    try:
        # print(f"Executing FFmpeg: {' '.join(cmd)}") # Debugging
        result = subprocess.run(cmd, capture_output=True, text=True, check=False) # Use capture_output
        
        if result.returncode != 0:
            print(f"Error extracting frames from {video_filepath} (FFmpeg return code: {result.returncode}):")
            # print(f"FFmpeg stdout: {result.stdout}") # Often empty on error
            print(f"FFmpeg stderr: {result.stderr}")
            return [], 0

        # List files based on the part of the pattern before sequence numbers
        # and the extension. This is more robust than just listing all files.
        prefix = frame_filename_pattern.split('%')[0]
        suffix = "." + frame_filename_pattern.split('.')[-1]
        
        extracted_frames = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir) 
            if f.startswith(prefix) and f.endswith(suffix)
        ])
        
        num_extracted = len(extracted_frames)
        if num_extracted == 0 and result.returncode == 0 :
             # FFmpeg might succeed but produce no output if video is corrupt or very short
             print(f"Warning: FFmpeg ran successfully but no frames matching pattern were found in {output_dir} for {video_filepath}.")
             print(f"FFmpeg stderr for clues: {result.stderr}")


        # print(f"Successfully extracted {num_extracted} frames from {video_filepath} to {output_dir}") # Debugging
        return extracted_frames, num_extracted

    except Exception as e:
        print(f"An exception occurred while running FFmpeg for {video_filepath}: {e}")
        return [], 0

if __name__ == '__main__':
    # Example Usage (for testing the function directly)
    # Create a dummy small video for testing
    sample_video_filename = "test_sample_video.mp4" 
    test_data_dir = "test_data_frame_extractor" # To hold the sample video
    test_output_dir = os.path.join(test_data_dir, "test_extracted_frames")

    os.makedirs(test_data_dir, exist_ok=True)
    sample_video_path = os.path.join(test_data_dir, sample_video_filename)

    if not os.path.exists(sample_video_path):
        try:
            print(f"Creating dummy video file: {sample_video_path}")
            ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"
            subprocess.run([
                ffmpeg_exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=128x128:d=1:r=10", 
                "-c:v", "libx264", "-pix_fmt", "yuv420p", sample_video_path
            ], check=True, capture_output=True, text=True)
            print(f"Dummy video created at {sample_video_path}")
        except Exception as e:
            print(f"Could not create dummy video for testing: {e}")
            if hasattr(e, 'stderr'): print(f"FFmpeg stderr: {e.stderr}")

    if os.path.exists(sample_video_path):
        print(f"Testing frame_extractor with video: {sample_video_path}")
        
        # Test 1: Extract all frames (original FPS)
        print("\n--- Test 1: Extract all frames ---")
        frames1, count1 = extract_all_frames(sample_video_path, test_output_dir)
        if count1 > 0:
            print(f"Test 1 successful: {count1} frames extracted to {test_output_dir}")
            # print(f"First few frames: {frames1[:2]}")
        else:
            print(f"Test 1 failed or produced no frames.")
        shutil.rmtree(test_output_dir, ignore_errors=True) # Clean up for next test

        # Test 2: Extract at target FPS (e.g., 1 FPS)
        print("\n--- Test 2: Extract at 1 FPS ---")
        frames2, count2 = extract_all_frames(sample_video_path, test_output_dir, target_fps=1)
        if count2 > 0:
            print(f"Test 2 successful: {count2} frames extracted to {test_output_dir}")
            # print(f"First few frames: {frames2[:2]}")
        else:
            print(f"Test 2 failed or produced no frames.")
        shutil.rmtree(test_output_dir, ignore_errors=True)

        # Test 3: Non-existent video
        print("\n--- Test 3: Non-existent video ---")
        frames3, count3 = extract_all_frames("non_existent_video.mp4", test_output_dir)
        if count3 == 0 and not frames3:
             print("Test 3 successful: Correctly handled non-existent video.")
        else:
             print(f"Test 3 failed: count={count3}, frames={frames3}")
        shutil.rmtree(test_output_dir, ignore_errors=True)
        
        # Clean up the whole test_data_dir
        # shutil.rmtree(test_data_dir)
        print(f"\nTesting complete. Please manually delete test directory: {test_data_dir}")

    else:
        print(f"Skipping tests, sample video '{sample_video_path}' not found and could not be created.")
