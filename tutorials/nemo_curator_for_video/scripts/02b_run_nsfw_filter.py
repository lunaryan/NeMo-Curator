import argparse
import json
import os
import glob
import pandas as pd
import dask.dataframe as dd
import random # For mock NSFW function
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client

# --- Mock NSFW Model Function ---
def mock_nsfw_classify_image(image_path):
    """
    Mock NSFW classification function.
    Returns a random float between 0.0 and 1.0.
    In a real scenario, this would be replaced by an actual model inference.
    """
    # print(f"Mock NSFW: Classifying {os.path.basename(image_path)}...") # Can be too verbose
    return random.random()

# --- Real NSFW Model (Conceptual - commented out) ---
# try:
#     from nemo_curator.image.classifiers import NSFWClassifier
#     # One would typically initialize the NSFWClassifier once globally if possible,
#     # or pass the model path/config to be loaded by workers.
#     # For example:
#     # nsfw_classifier_instance = NSFWClassifier(model_path_or_config)
#     # def real_nsfw_classify_image(image_path):
#     #     return nsfw_classifier_instance.compute_score(image_path) # Assuming such a method
#     print("Note: Real NSFWClassifier found but mock function will be used for this script.")
#     # Set this to True to attempt using the real classifier if available and configured.
#     USE_REAL_NSFW_CLASSIFIER = False 
# except ImportError:
#     print("Info: NeMo Curator's NSFWClassifier not found or not installed. Using mock function.")
#     USE_REAL_NSFW_CLASSIFIER = False
# except Exception as e:
#     print(f"Info: Error importing or initializing NSFWClassifier: {e}. Using mock function.")
#     USE_REAL_NSFW_CLASSIFIER = False

# --- End of NSFW Model Stubs ---

def get_video_nsfw_score(row, sample_frames_base_dir, video_id_col, nsfw_model_func, nsfw_threshold_arg):
    """
    Calculates max NSFW score for a video based on its sample frames.
    """
    video_id = str(row[video_id_col])
    video_frames_path = os.path.join(sample_frames_base_dir, video_id)

    default_return = pd.Series({'max_nsfw_score': -1.0, 'is_nsfw_video': False, 'frames_analyzed': 0})

    if not os.path.isdir(video_frames_path):
        # print(f"Info: Frame directory not found for video ID '{video_id}' at '{video_frames_path}'.")
        return default_return

    frame_files = glob.glob(os.path.join(video_frames_path, "*.jpg")) # Assuming JPG frames
    if not frame_files:
        # print(f"Info: No JPG frames found in '{video_frames_path}' for video ID '{video_id}'.")
        return default_return

    max_score = 0.0
    frames_analyzed_count = 0

    for frame_path in frame_files:
        try:
            score = nsfw_model_func(frame_path)
            if score > max_score:
                max_score = score
            frames_analyzed_count += 1
        except Exception as e:
            print(f"Warning: Error during NSFW classification for frame '{frame_path}' (video ID '{video_id}'): {e}")
            # Optionally, count this as an error or handle differently
            continue # Skip this frame

    if frames_analyzed_count == 0:
        # This case might occur if all frames failed processing inside the loop
        # print(f"Info: Zero frames successfully analyzed for video ID '{video_id}' in '{video_frames_path}'.")
        return default_return
    
    is_nsfw = True if max_score >= nsfw_threshold_arg else False
    
    return pd.Series({
        'max_nsfw_score': float(max_score), 
        'is_nsfw_video': bool(is_nsfw), 
        'frames_analyzed': int(frames_analyzed_count)
    })


def main(args):
    # 1. Setup
    print("--- NSFW Filter Script Start ---")
    print(f"Using MOCK NSFW classification function. Scores will be random.")
    output_dir = os.path.dirname(args.output_manifest)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.num_workers:
        client = get_client(num_workers=args.num_workers, create_cluster_if_needed=True)
        print(f"Dask client configured: {client}")

    # 2. Load Manifest
    print(f"Loading manifest from: {args.input_manifest}")
    try:
        dataset = DocumentDataset.read_json(args.input_manifest, backend='pandas')
        if dataset.df is None or dataset.df.empty:
            print("Input manifest is empty or failed to load. Saving an empty manifest.")
            with open(args.output_manifest, 'w') as f: pass
            print(f"Empty output manifest saved to {args.output_manifest}")
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
    if args.text_field_for_video_id not in dask_df.columns:
        print(f"CRITICAL ERROR: Video ID column '{args.text_field_for_video_id}' not found in manifest.")
        print(f"Available columns: {list(dask_df.columns)}")
        return

    # 3. Define meta for new columns
    meta = {
        'max_nsfw_score': 'f8', 
        'is_nsfw_video': 'bool', 
        'frames_analyzed': 'i4'
    }
    
    # Select the NSFW model function to use
    # For this script, mock_nsfw_classify_image is used directly as per prioritization.
    # If USE_REAL_NSFW_CLASSIFIER were true, one might select `real_nsfw_classify_image` here.
    current_nsfw_model_func = mock_nsfw_classify_image
    
    # 4. Apply to Dask DataFrame
    print(f"Starting NSFW score calculation for videos using sample frames from: {args.sample_frames_base_dir}")
    nsfw_cols_ddf = dask_df.apply(
        get_video_nsfw_score, 
        args=(args.sample_frames_base_dir, args.text_field_for_video_id, current_nsfw_model_func, args.nsfw_threshold),
        axis=1, 
        meta=meta
    ).persist()

    enriched_ddf = dd.concat([dask_df.persist(), nsfw_cols_ddf], axis=1) # Persist dask_df too
    
    # Trigger computation for reporting
    total_videos_processed = len(enriched_ddf) # This computes
    print(f"NSFW scoring applied. Total videos processed: {total_videos_processed}")

    # Report on videos with no frames found or analyzed
    no_frames_analyzed_df = enriched_ddf[enriched_ddf['frames_analyzed'] <= 0].compute()
    if not no_frames_analyzed_df.empty:
        print(f"Found {len(no_frames_analyzed_df)} videos with no frames found/analyzed or errors during frame processing.")
        # for vid_id_err in no_frames_analyzed_df[args.text_field_for_video_id].head(3).tolist():
        #     print(f"  - Video ID (no frames analyzed): {vid_id_err}")


    # 5. Filtering (if not keeping NSFW)
    dataset_to_save = DocumentDataset(df=enriched_ddf) # Wrap the Dask DF
    
    num_flagged_as_nsfw = dataset_to_save.df[dataset_to_save.df['is_nsfw_video'] == True].shape[0].compute()
    print(f"Number of videos flagged as NSFW (max_score >= {args.nsfw_threshold}): {num_flagged_as_nsfw}")

    if not args.keep_nsfw_videos:
        print(f"Filtering out NSFW videos. Keeping only videos with is_nsfw_video == False.")
        dataset_to_save.df = dataset_to_save.df[dataset_to_save.df['is_nsfw_video'] == False]
        num_after_filtering = dataset_to_save.df.shape[0].compute() # Re-compute count after filter
        print(f"Number of videos remaining after filtering: {num_after_filtering}")
    else:
        print("Keeping all videos (NSFW and non-NSFW) in the output manifest.")
        num_after_filtering = total_videos_processed # No change in count

    # 6. Save Manifest
    print(f"Saving final manifest to: {args.output_manifest}")
    try:
        dataset_to_save.to_json(args.output_manifest)
        print(f"Final manifest saved successfully to: {args.output_manifest}")
    except Exception as e:
        print(f"Error saving final manifest using DocumentDataset.to_json: {e}")
        # Fallback save (optional, can be complex if DocumentDataset itself failed on its Dask DF)
        print("Attempting fallback save with Dask's to_json directly (may create parts).")
        try:
            output_path_for_dask = args.output_manifest
            if dataset_to_save.df.npartitions > 1:
                parts_dir = args.output_manifest + "_parts_fallback"
                os.makedirs(parts_dir, exist_ok=True)
                output_path_for_dask = os.path.join(parts_dir, "part-*.jsonl")
            dataset_to_save.df.to_json(output_path_for_dask, orient="records", lines=True, compute=True)
            print(f"Fallback Dask save successful to: {output_path_for_dask}")
        except Exception as e2:
            print(f"Fallback Dask save also failed: {e2}")
            
    print(f"--- NSFW Filter Script End. Processed {num_rows_initial} initial manifest entries. Output {num_after_filtering} entries. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Applies a mock NSFW filter to video sample frames and updates a manifest.")
    parser.add_argument("--input_manifest", required=True, help="Path to an enriched JSONL manifest.")
    parser.add_argument("--sample_frames_base_dir", required=True, help="Base directory of sample frames (e.g., output from 02a_extract_sample_frames.py).")
    parser.add_argument("--output_manifest", required=True, help="Path to save the NSFW-filtered/annotated manifest.")
    parser.add_argument("--text_field_for_video_id", default="video_id", help="Field name for video ID (default: video_id).")
    parser.add_argument("--nsfw_threshold", type=float, default=0.8, help="Threshold for NSFW classification (default: 0.8).")
    parser.add_argument("--keep_nsfw_videos", action="store_true", help="If set, keeps NSFW videos in the manifest; otherwise, filters them out.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of Dask workers (default: None, Dask decides).")
    
    args = parser.parse_args()
    main(args)
