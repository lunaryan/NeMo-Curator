import argparse
import json
import os
import subprocess
import pandas as pd
import dask.dataframe as dd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client

def get_default_error_properties(error_msg):
    """Returns a dictionary with default/error values for video properties."""
    return {
        'width': -1,
        'height': -1,
        'duration': 0.0,
        'codec_name': 'error',
        'bit_rate': 'error',
        'error_message': str(error_msg)
    }

def extract_properties_for_row(row, ffprobe_path, filepath_col):
    """
    Extracts video properties for a single row (Pandas Series) using ffprobe.
    """
    video_path = row[filepath_col]

    if not os.path.exists(video_path):
        error_message = f"File not found at '{video_path}'"
        print(f"Warning (row-level): {error_message} for video_id: {row.get('video_id', 'N/A')}")
        return pd.Series(get_default_error_properties(error_message))

    command = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0", # Select the first video stream
        video_path
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)

        if process.returncode != 0 or not process.stdout:
            error_detail = process.stderr.strip() if process.stderr else "No stderr output."
            if not process.stdout and process.returncode == 0 : 
                 error_message = f"ffprobe produced no video stream output for {video_path}. Detail: {error_detail}"
            else: 
                 error_message = f"ffprobe error for {video_path}. Return code: {process.returncode}. Detail: {error_detail}"
            print(f"Warning (row-level): {error_message} for video_id: {row.get('video_id', 'N/A')}")
            return pd.Series(get_default_error_properties(error_message))

        ffprobe_output = json.loads(process.stdout)

        if not ffprobe_output.get("streams") or len(ffprobe_output["streams"]) == 0:
            error_message = f"No video streams found by ffprobe for {video_path}."
            print(f"Warning (row-level): {error_message} for video_id: {row.get('video_id', 'N/A')}")
            return pd.Series(get_default_error_properties(error_message))

        stream_info = ffprobe_output["streams"][0]
        
        properties = {
            'width': int(stream_info.get('width', -1)),
            'height': int(stream_info.get('height', -1)),
            'duration': float(stream_info.get('duration', 0.0)),
            'codec_name': str(stream_info.get('codec_name', 'unknown')),
            'bit_rate': str(stream_info.get('bit_rate', 'N/A')), 
            'error_message': None 
        }
        return pd.Series(properties)

    except json.JSONDecodeError as e:
        error_message = f"Error decoding ffprobe JSON output for {video_path}: {e}. Output: {process.stdout[:500]}"
        print(f"Warning (row-level): {error_message} for video_id: {row.get('video_id', 'N/A')}")
        return pd.Series(get_default_error_properties(error_message))
    except FileNotFoundError: 
        error_message = f"Error: {ffprobe_path} not found during row processing. This should have been caught by pre-flight check."
        print(f"CRITICAL (row-level): {error_message}")
        return pd.Series(get_default_error_properties(error_message))
    except Exception as e:
        error_message = f"An unexpected error occurred while processing {video_path}: {e}"
        print(f"Warning (row-level): {error_message} for video_id: {row.get('video_id', 'N/A')}")
        return pd.Series(get_default_error_properties(error_message))


def main(args):
    # 1. Setup
    output_dir = os.path.dirname(args.output_manifest)
    if output_dir and not os.path.exists(output_dir): # Ensure output_dir is not an empty string
        os.makedirs(output_dir, exist_ok=True)

    if args.num_workers:
        client = get_client(num_workers=args.num_workers, create_cluster_if_needed=True)
        print(f"Dask client configured: {client}")
    
    # Pre-flight check for ffprobe
    try:
        ffprobe_check = subprocess.run([args.ffprobe_path, "-version"], capture_output=True, text=True, check=True)
        print(f"ffprobe found at '{args.ffprobe_path}' and is executable. Version output (first 100 chars): \n{ffprobe_check.stdout[:100]}...")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: ffprobe executable not found at '{args.ffprobe_path}'. Please install ffprobe or correct the path. Cannot proceed.")
        return
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL ERROR: ffprobe at '{args.ffprobe_path}' is not executable or failed version check: {e.stderr}. Cannot proceed.")
        return

    # 2. Load Manifest
    print(f"Loading manifest from: {args.input_manifest}")
    try:
        # Requirement: Use DocumentDataset.read_json with pandas backend
        dataset = DocumentDataset.read_json(args.input_manifest, backend='pandas')
        
        if dataset.df is None or dataset.df.empty:
            print("Input manifest is empty or failed to load. Saving an empty manifest.")
            with open(args.output_manifest, 'w') as f: # Create an empty file
                pass 
            print(f"Empty output manifest saved to {args.output_manifest}")
            return
        
        # Convert pandas DataFrame from DocumentDataset to Dask DataFrame
        num_rows = len(dataset.df)
        if args.num_workers and args.num_workers > 0:
            npartitions = min(args.num_workers, num_rows) if num_rows > 0 else 1
        else:
            # Default to number of CPU cores, but not more than number of rows
            npartitions = min(os.cpu_count() or 2, num_rows) if num_rows > 0 else 1
        
        dask_df = dd.from_pandas(dataset.df, npartitions=npartitions)
        print(f"Manifest loaded into Dask DataFrame with {dask_df.npartitions} partitions from {num_rows} records.")

    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{args.input_manifest}'.")
        return
    except Exception as e:
        print(f"Error loading input manifest '{args.input_manifest}': {e}")
        return

    # Check if the filepath column exists in the Dask DataFrame
    if args.text_field_for_filepath not in dask_df.columns:
        print(f"CRITICAL ERROR: The specified filepath column '{args.text_field_for_filepath}' does not exist in the input manifest.")
        print(f"Available columns: {list(dask_df.columns)}")
        return
        
    # 3. Define meta for the new columns
    meta = {
        'width': 'i4',          
        'height': 'i4',         
        'duration': 'f8',       
        'codec_name': 'object', 
        'bit_rate': 'object',   
        'error_message': 'object' 
    }

    # 4. Apply to Dask DataFrame
    print(f"Starting ffprobe processing for video properties using column '{args.text_field_for_filepath}'...")
    
    # Persist the original dask_df to prevent re-reading if underlying data source is complex
    # For from_pandas, this might not be strictly necessary but is good practice in general Dask workflows.
    dask_df = dask_df.persist()

    new_properties_ddf = dask_df.apply(
        extract_properties_for_row,
        args=(args.ffprobe_path, args.text_field_for_filepath),
        axis=1,
        meta=meta
    ).persist() # Persist the results of the expensive computation

    # Concatenate the original Dask DataFrame with the new properties Dask DataFrame
    enriched_ddf = dd.concat([dask_df, new_properties_ddf], axis=1)
    
    # Trigger computation and get count
    num_total_records = len(enriched_ddf) 
    print(f"Video properties extraction applied. Total records in enriched manifest: {num_total_records}.")
    
    # Report on errors
    # Compute only the error_message column and the filepath for error reporting
    error_reporting_df = enriched_ddf[[args.text_field_for_filepath, 'error_message']].copy()
    error_messages_computed = error_reporting_df[error_reporting_df['error_message'].notnull()].compute()
    num_errors = len(error_messages_computed)

    if num_errors > 0:
        print(f"Encountered {num_errors} errors during ffprobe processing. Error examples:")
        for _, row_err in error_messages_computed.head(min(num_errors, 5)).iterrows():
            print(f"  File: {row_err[args.text_field_for_filepath]}, Error: {row_err['error_message']}")
    else:
        print("No errors encountered during ffprobe processing.")

    # Create a new DocumentDataset from the enriched Dask DataFrame
    enriched_dataset = DocumentDataset(df=enriched_ddf)

    # 5. Save Enriched Manifest
    print(f"Saving enriched manifest to: {args.output_manifest}")
    try:
        enriched_dataset.to_json(args.output_manifest)
        print(f"Enriched manifest saved successfully to: {args.output_manifest}")
    except Exception as e:
        print(f"Error saving enriched manifest using DocumentDataset.to_json: {e}")
        print("Attempting fallback save with Dask's to_json (will create parts if multiple partitions).")
        try:
            # Define output path pattern for Dask's to_json
            # If more than one partition, save to a directory with part files
            # Otherwise, try to save as a single file.
            output_path_for_dask = args.output_manifest
            if enriched_ddf.npartitions > 1:
                # Create a directory for parts
                parts_dir = args.output_manifest + "_parts"
                os.makedirs(parts_dir, exist_ok=True)
                output_path_for_dask = os.path.join(parts_dir, "part-*.jsonl")
                print(f"Saving {enriched_ddf.npartitions} partitions to directory: {parts_dir}")
            else: # Single partition
                print(f"Saving single partition to: {output_path_for_dask}")

            enriched_ddf.to_json(output_path_for_dask, orient="records", lines=True, compute=True)
            
            if enriched_ddf.npartitions > 1:
                 print(f"Fallback Dask save successful. Parts saved in: {parts_dir}")
            else:
                 print(f"Fallback Dask save successful to: {output_path_for_dask}")

        except Exception as e2:
            print(f"Fallback Dask save also failed: {e2}")

    print("Processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enriches a video manifest with technical properties using ffprobe.")
    parser.add_argument("--input_manifest", required=True, help="Path to the input JSONL manifest.")
    parser.add_argument("--output_manifest", required=True, help="Path to save the enriched JSONL manifest.")
    parser.add_argument("--text_field_for_filepath", default="filepath", help="The field in the JSONL that contains the video file path.")
    parser.add_argument("--ffprobe_path", default="ffprobe", help="Path to the ffprobe executable.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of Dask workers. If None, Dask decides (uses local scheduler).")
    
    args = parser.parse_args()
    main(args)
