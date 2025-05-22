import argparse
import json
import os
import subprocess
import shutil

def check_ffmpeg(ffmpeg_path):
    """Checks if FFmpeg is available and executable."""
    if shutil.which(ffmpeg_path):
        try:
            process = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
            print(f"FFmpeg found at '{ffmpeg_path}'. Version info (first line): {process.stdout.splitlines()[0]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error checking FFmpeg version with path '{ffmpeg_path}': {e}")
            return False
    else:
        print(f"Error: FFmpeg executable not found at '{ffmpeg_path}' or in PATH.")
        return False

def main(args):
    # 1. Setup
    if not check_ffmpeg(args.ffmpeg_path):
        print("Exiting due to FFmpeg not being available.")
        return

    os.makedirs(args.output_frames_dir, exist_ok=True)
    print(f"Output frames will be saved in subdirectories under: {args.output_frames_dir}")

    processed_videos = 0
    skipped_videos_missing_fields = 0
    skipped_videos_duration = 0
    skipped_videos_file_not_found = 0
    total_frames_extracted = 0
    total_ffmpeg_errors = 0

    # 2. Process Manifest
    try:
        with open(args.input_manifest, 'r', encoding='utf-8') as f_manifest:
            for line_number, line in enumerate(f_manifest, 1):
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_number} due to JSON decode error: {e}")
                    continue

                try:
                    video_filepath = record[args.text_field_for_filepath]
                    video_id = str(record[args.text_field_for_video_id]) # Ensure video_id is string for path creation
                    duration_str = record.get(args.text_field_for_duration) # Use .get for duration as it might be missing
                except KeyError as e:
                    print(f"Warning: Skipping record (approx line {line_number}) due to missing key: {e}. Record: {record}")
                    skipped_videos_missing_fields += 1
                    continue
                
                if duration_str is None:
                    print(f"Warning: Skipping video ID '{video_id}' (approx line {line_number}) due to missing duration field ('{args.text_field_for_duration}').")
                    skipped_videos_missing_fields +=1 # Count as missing field
                    continue

                try:
                    duration = float(duration_str)
                except ValueError:
                    print(f"Warning: Skipping video ID '{video_id}' (approx line {line_number}) due to invalid duration value: '{duration_str}'.")
                    skipped_videos_duration += 1
                    continue

                if duration <= 0 or duration < args.min_duration_for_extraction:
                    # print(f"Info: Skipping video ID '{video_id}' (duration: {duration}s) as it's too short or invalid for extraction (min_duration: {args.min_duration_for_extraction}s).")
                    skipped_videos_duration += 1
                    continue
                
                processed_videos += 1

                # 3. Extract Frames for Eligible Videos
                video_specific_output_dir = os.path.join(args.output_frames_dir, video_id)
                os.makedirs(video_specific_output_dir, exist_ok=True)

                if not os.path.exists(video_filepath):
                    print(f"Warning: Video file for ID '{video_id}' not found at '{video_filepath}'. Skipping frame extraction for this video.")
                    skipped_videos_file_not_found += 1
                    continue

                print(f"\nProcessing video ID '{video_id}': {video_filepath} (Duration: {duration:.2f}s)")

                N = args.frames_per_video
                frames_extracted_for_this_video = 0
                for k in range(N):
                    # Calculate timestamp - spread reasonably, avoid very start/end for robustness.
                    # Using (k+1) / (N+1) ensures points are within (0, duration).
                    timestamp_ratio = (k + 1) / (N + 1)
                    timestamp = timestamp_ratio * duration
                    
                    # Ensure timestamp is slightly offset if it's too close to 0, especially for N=1
                    if N == 1 and timestamp < 0.1: # For single frame, try to avoid very first frame
                        timestamp = max(0.1, duration * 0.1) # e.g. 10% into video or 0.1s, whichever is larger
                    elif timestamp < 0.05: # General small offset from absolute start
                        timestamp = 0.05

                    output_frame_filename = f"frame_{k+1:03d}.jpg"
                    output_frame_path = os.path.join(video_specific_output_dir, output_frame_filename)

                    ffmpeg_command = [
                        args.ffmpeg_path,
                        "-ss", str(timestamp),
                        "-i", video_filepath,
                        "-vframes", "1",      # Extract a single frame
                        "-q:v", "2",          # Reasonably good quality for JPG
                        "-y",                 # Overwrite output files without asking
                        output_frame_path
                    ]
                    
                    print(f"  Extracting frame {k+1}/{N} at {timestamp:.2f}s to {output_frame_path}...")
                    try:
                        process = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False)
                        if process.returncode == 0:
                            total_frames_extracted += 1
                            frames_extracted_for_this_video += 1
                        else:
                            total_ffmpeg_errors += 1
                            print(f"    Warning: FFmpeg failed for video ID '{video_id}', frame {k+1}. Timestamp: {timestamp:.2f}s.")
                            print(f"    FFmpeg stderr: {process.stderr.strip()}")
                    except Exception as e:
                        total_ffmpeg_errors += 1
                        print(f"    Error executing FFmpeg for video ID '{video_id}', frame {k+1}: {e}")
                
                if frames_extracted_for_this_video > 0 :
                     print(f"  Successfully extracted {frames_extracted_for_this_video} frames for video ID '{video_id}'.")
                else:
                     print(f"  No frames extracted for video ID '{video_id}' (possibly all FFmpeg attempts failed).")


    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{args.input_manifest}'.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while processing the manifest: {e}")
        return
    
    # 4. Reporting
    print("\n--- Extraction Summary ---")
    print(f"Total manifest entries processed (attempted): {line_number if 'line_number' in locals() else 0}")
    print(f"Videos eligible and attempted for frame extraction: {processed_videos}")
    print(f"Total sample frames extracted successfully: {total_frames_extracted}")
    print(f"Videos skipped due to missing fields: {skipped_videos_missing_fields}")
    print(f"Videos skipped due to short/invalid duration: {skipped_videos_duration}")
    print(f"Videos skipped due to source file not found: {skipped_videos_file_not_found}")
    print(f"Total FFmpeg errors encountered during frame extraction: {total_ffmpeg_errors}")
    print("--- End of Summary ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts sample frames from videos listed in a JSONL manifest.")
    parser.add_argument("--input_manifest", required=True, 
                        help="Path to the input JSONL manifest file.")
    parser.add_argument("--output_frames_dir", required=True, 
                        help="Base directory to save sample frames. Subdirectories will be created per video_id.")
    parser.add_argument("--text_field_for_filepath", default="filepath", 
                        help="Name of the field in the manifest holding the video file path (default: filepath).")
    parser.add_argument("--text_field_for_video_id", default="video_id", 
                        help="Name of the field for the unique video identifier (default: video_id).")
    parser.add_argument("--text_field_for_duration", default="duration", 
                        help="Name of the field for video duration in seconds (default: duration).")
    parser.add_argument("--frames_per_video", type=int, default=3, 
                        help="Number of sample frames to extract per video (default: 3).")
    parser.add_argument("--min_duration_for_extraction", type=float, default=1.0, 
                        help="Minimum video duration (seconds) required for frame extraction (default: 1.0).")
    parser.add_argument("--ffmpeg_path", default="ffmpeg", 
                        help="Path to the FFmpeg executable (default: ffmpeg).")
    
    args = parser.parse_args()
    main(args)
