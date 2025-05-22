import argparse
import json
import os
import subprocess
import sys # For sys.executable
import pandas as pd
import dask.dataframe as dd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client

def run_extraction_for_row(row, script_path, base_out_dir, id_col, path_col, fps, frame_format_ext):
    """
    Runs the external frame extraction script for a single row (Pandas Series).
    Returns a dictionary with the status of the extraction.
    """
    video_id = str(row[id_col])
    video_filepath = row[path_col]

    if not os.path.exists(video_filepath):
        return {
            'video_id': video_id, 
            'status': 'error', 
            'message': f'File not found at {video_filepath}', 
            'output_path': None
        }

    video_frames_output_dir = os.path.join(base_out_dir, video_id)
    try:
        os.makedirs(video_frames_output_dir, exist_ok=True)
    except OSError as e:
        return {
            'video_id': video_id,
            'status': 'error',
            'message': f'Could not create output directory {video_frames_output_dir}: {e}',
            'output_path': None
        }

    # Construct command for the external script
    # Assuming the external script (e.g., 01_frame_extraction.py) supports these args
    cmd = [
        sys.executable,  # Path to current Python interpreter
        script_path,
        "--input_video", video_filepath,
        "--output_dir", video_frames_output_dir,
        "--fps", str(fps),
        "--method", "ffmpeg",  # Assuming robust method from external script
        "--frame_pattern", f"frame_%06d.{frame_format_ext.lower()}" 
    ]

    try:
        # print(f"Running command for video_id {video_id}: {' '.join(cmd)}") # For debugging
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            # Check if frames were actually created (optional, but good for robustness)
            # For now, assume success on returncode 0
            return {
                'video_id': video_id, 
                'status': 'success', 
                'message': 'Frames extracted.', 
                'output_path': video_frames_output_dir
            }
        else:
            # Capture first 500 chars of stderr for brevity in logs
            error_message = result.stderr[:500].strip() if result.stderr else "No stderr output from script."
            if not error_message and result.stdout: # If stderr is empty but stdout has info
                error_message = f"Script failed with no stderr, stdout (first 500 chars): {result.stdout[:500].strip()}"

            print(f"Failure for video_id {video_id}. Stderr: {error_message}")
            return {
                'video_id': video_id, 
                'status': 'failure', 
                'message': error_message,
                'output_path': video_frames_output_dir # Output path still relevant for inspection
            }
    except Exception as e:
        print(f"Exception during subprocess run for video_id {video_id}: {e}")
        return {
            'video_id': video_id, 
            'status': 'error', 
            'message': f'Exception during script execution: {e}', 
            'output_path': video_frames_output_dir
        }


def main(args):
    # 1. Setup
    print("--- Frame Extraction Orchestration Script Start ---")
    
    # Create output directories
    log_output_dir = os.path.dirname(args.output_orchestration_log)
    if log_output_dir and not os.path.exists(log_output_dir):
        os.makedirs(log_output_dir, exist_ok=True)
    
    if not os.path.exists(args.base_output_dir_for_frames):
        os.makedirs(args.base_output_dir_for_frames, exist_ok=True)
        print(f"Created base output directory for frames: {args.base_output_dir_for_frames}")

    # Check frame extraction script
    if not os.path.isfile(args.frame_extraction_script_path):
        print(f"Error: Frame extraction script not found or is not a file at '{args.frame_extraction_script_path}'.")
        return
    print(f"Using frame extraction script: {args.frame_extraction_script_path}")

    # Initialize Dask client
    if args.num_workers:
        client = get_client(num_workers=args.num_workers, create_cluster_if_needed=True)
        print(f"Dask client configured: {client}")
    else:
        client = get_client(create_cluster_if_needed=True)
        print(f"Dask client configured (default settings): {client}")

    # 2. Load Manifest
    print(f"Loading manifest from: {args.input_manifest}")
    try:
        dataset = DocumentDataset.read_json(args.input_manifest, backend='pandas')
        if dataset.df is None or dataset.df.empty:
            print("Input manifest is empty or failed to load. No frames to extract.")
            # Create an empty log file
            with open(args.output_orchestration_log, 'w') as f: pass
            print(f"Empty orchestration log saved to {args.output_orchestration_log}")
            return
        
        num_rows_initial = len(dataset.df)
        npartitions = args.num_workers if args.num_workers and args.num_workers > 0 else (os.cpu_count() or 2)
        npartitions = min(npartitions, num_rows_initial) if num_rows_initial > 0 else 1
        
        dask_df = dd.from_pandas(dataset.df, npartitions=npartitions)
        print(f"Manifest loaded into Dask DataFrame with {dask_df.npartitions} partitions from {num_rows_initial} records.")

    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{args.input_manifest}'.")
        return
    except Exception as e:
        print(f"Error loading input manifest '{args.input_manifest}': {e}")
        return

    # Check for required columns
    required_cols = [args.text_field_for_video_id, args.text_field_for_filepath]
    missing_cols = [col for col in required_cols if col not in dask_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the manifest: {', '.join(missing_cols)}. Available: {list(dask_df.columns)}")
        return

    # 3. Apply extraction function to Dask DataFrame
    print(f"Starting frame extraction orchestration for {len(dask_df)} videos...")
    meta = {
        'video_id': 'object', 
        'status': 'object', 
        'message': 'object', 
        'output_path': 'object'
    }
    
    log_results_ddf = dask_df.apply(
        run_extraction_for_row,
        args=(
            args.frame_extraction_script_path, 
            args.base_output_dir_for_frames,
            args.text_field_for_video_id,
            args.text_field_for_filepath,
            args.frames_per_second,
            args.frame_format
        ),
        axis=1,
        meta=meta
    )

    # 4. Compute and Save Log
    print("Computing results and saving orchestration log...")
    log_results_df_pd = log_results_ddf.compute()
    
    try:
        with open(args.output_orchestration_log, 'w', encoding='utf-8') as f:
            for _, row in log_results_df_pd.iterrows():
                f.write(json.dumps(row.to_dict()) + '\n')
        print(f"Orchestration log saved to: {args.output_orchestration_log}")
    except IOError as e:
        print(f"Error writing orchestration log file: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while saving the log: {e}")
        return

    # 5. Logging Summary
    total_processed = len(log_results_df_pd)
    success_count = log_results_df_pd[log_results_df_pd['status'] == 'success'].shape[0]
    failure_count = log_results_df_pd[log_results_df_pd['status'] == 'failure'].shape[0]
    error_count = log_results_df_pd[log_results_df_pd['status'] == 'error'].shape[0] # Includes file not found, dir creation errors

    print("\n--- Orchestration Summary ---")
    print(f"Total videos processed: {total_processed}")
    print(f"  Successfully extracted frames for: {success_count} videos")
    print(f"  Failed frame extraction (script failure): {failure_count} videos")
    print(f"  Errors (file not found, setup issues): {error_count} videos")
    
    if failure_count > 0 or error_count > 0:
        print("  Check the log file for details on failures/errors.")
    print("--- End of Summary ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrates frame extraction for videos in a manifest.")
    parser.add_argument("--input_manifest", required=True, help="Path to the input JSONL manifest.")
    parser.add_argument("--output_orchestration_log", required=True, help="Path to save the JSONL log file for extraction status.")
    parser.add_argument("--text_field_for_filepath", default="filepath", help="Field name for video file path (default: filepath).")
    parser.add_argument("--text_field_for_video_id", default="video_id", help="Field name for unique video ID (default: video_id).")
    parser.add_argument("--base_output_dir_for_frames", required=True, help="Root directory where frame subdirectories (per video_id) will be created.")
    parser.add_argument("--frame_extraction_script_path", required=True, help="Path to the external Python frame extraction script (e.g., 01_frame_extraction.py).")
    parser.add_argument("--frames_per_second", type=int, default=1, help="Target FPS for frame extraction (default: 1).")
    parser.add_argument("--frame_format", default="jpg", help="Output format for frames (e.g., jpg, png) (default: jpg).")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of Dask workers (default: None, Dask decides).")
    
    args = parser.parse_args()
    main(args)
