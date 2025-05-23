import argparse
import os
import shutil
import json
import pandas as pd
import dask.dataframe as dd
import torch # For checking cuda availability

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, load_object_on_worker
from nemo_curator.utils.script_utils import ArgumentHelper

# --- Local module imports (adjust paths if needed based on final project structure) ---
try:
    from frame_extractor import extract_all_frames
    import frame_deduplicator # Explicitly import for its functions
    import dali_augmenter # Explicitly import for its functions
    from nemo_curator_adapters.path_based_frame_embedder import PathBasedFrameEmbedder
    print("Successfully imported local modules: frame_extractor, frame_deduplicator, dali_augmenter, PathBasedFrameEmbedder.")
except ImportError as e:
    print(f"Error importing local modules: {e}. Ensure they are in the Python path or correct relative locations.")
    # Depending on strictness, might want to exit here or let it fail later.
    # For now, let it proceed so the script structure can be reviewed.
    PathBasedFrameEmbedder = None # Placeholder if import fails, to allow script structure.

# --- NeMo Curator Classifier imports ---
try:
    from nemo_curator.image.classifiers import AestheticClassifier, NSFWClassifier
    print("Successfully imported NeMo Curator classifiers: AestheticClassifier, NSFWClassifier.")
except ImportError as e:
    print(f"Error importing NeMo Curator classifiers: {e}. Ensure NeMo Curator is correctly installed.")
    AestheticClassifier, NSFWClassifier = None, None # Placeholders

# Global dictionary to hold loaded models per worker (conceptual for optimization)
# This is a more advanced pattern; for the simplified approach, models are loaded in process_video_row.
# WORKER_MODELS = {}

def process_video_row(row: pd.Series, args_ns: argparse.Namespace) -> pd.Series:
    """
    Processes a single video row from the manifest through the frame-level pipeline.
    """
    video_id = str(row[args_ns.video_id_col])
    raw_video_filepath = row[args_ns.video_filepath_col]

    # Initialize result dictionary with default values for all expected output fields
    res = {
        'video_id': video_id,
        'status': 'unprocessed',
        'error_message': None,
        'original_frame_count': 0,
        'frames_after_embedding': 0, # Number of frames successfully embedded
        'frames_after_aesthetic_filter': 0,
        'frames_after_nsfw_filter': 0,
        'frames_after_dedup': 0,
        'augmented_frames_count': 0,
        'path_to_extracted_frames_dir': None, # Temp dir for raw frames
        'path_to_augmented_frames_output_dir': None,
        'avg_aesthetic_score_selected': None, # Example of an aggregated metric
        'max_nsfw_score_selected': None      # Example
    }
    
    # 1. Setup per video
    video_temp_dir = os.path.join(args_ns.base_temp_frame_extraction_dir, video_id)
    res['path_to_extracted_frames_dir'] = video_temp_dir
    try:
        os.makedirs(video_temp_dir, exist_ok=True)
    except Exception as e:
        res['status'] = 'error_setup'
        res['error_message'] = f"Failed to create temp directory {video_temp_dir}: {e}"
        return pd.Series(res)

    try:
        # A. Frame Extraction
        print(f"[Video: {video_id}] Starting frame extraction from {raw_video_filepath}...")
        raw_frame_paths, original_frame_count_val = extract_all_frames(
            video_filepath=raw_video_filepath,
            output_dir=video_temp_dir,
            ffmpeg_path=args_ns.ffmpeg_path,
            target_fps=args_ns.raw_frame_fps
        )
        res['original_frame_count'] = original_frame_count_val
        if not raw_frame_paths or original_frame_count_val == 0:
            res['status'] = 'error_frame_extraction_no_frames'
            res['error_message'] = "No frames extracted or frame_extractor returned empty list."
            if args_ns.cleanup_temp_frames and os.path.exists(video_temp_dir): shutil.rmtree(video_temp_dir)
            return pd.Series(res)
        print(f"[Video: {video_id}] Extracted {res['original_frame_count']} raw frames to {video_temp_dir}")

        # B. Create Frame DataFrame
        frame_df_pd = pd.DataFrame({
            'frame_path': raw_frame_paths,
            'frame_id': [os.path.splitext(os.path.basename(p))[0] for p in raw_frame_paths],
            'video_id': video_id
        })
        if frame_df_pd.empty:
            res['status'] = 'error_frame_df_empty'
            res['error_message'] = "Frame DataFrame is empty after extraction (unexpected)."
            if args_ns.cleanup_temp_frames and os.path.exists(video_temp_dir): shutil.rmtree(video_temp_dir)
            return pd.Series(res)

        # C. Embeddings & Classification (Simplified Approach)
        print(f"[Video: {video_id}] Starting embedding and classification for {len(frame_df_pd)} frames...")
        if PathBasedFrameEmbedder is None or AestheticClassifier is None or NSFWClassifier is None:
            raise ImportError("PathBasedFrameEmbedder or NeMo Curator Classifiers not available on worker.")

        # Initialize classifiers (loaded per video in this simplified model)
        # Note: For production, these should be loaded once per worker.
        aesthetic_classifier = AestheticClassifier(model_name_or_path="sac_public_2022_06_29_vit_l_14_linear.pth") # Example model
        nsfw_classifier = NSFWClassifier(model_name_or_path="simonduq.nsfw_image_detection") # Example from HF

        embedder = PathBasedFrameEmbedder(
            model_name=args_ns.embedding_model_name,
            frame_paths_col="frame_path", # PathBasedFrameEmbedder uses this to find paths
            batch_size=args_ns.embedding_batch_size,
            classifiers=[aesthetic_classifier, nsfw_classifier] # Pass instances
        )
        
        # Convert Pandas DF to cuDF if on GPU for PathBasedFrameEmbedder (if it expects cuDF)
        # PathBasedFrameEmbedder's _run_inference uses cudf.DataFrame for partition_df.
        # Here, we are processing a single video's frames, so we make a "partition" out of them.
        try:
            import cudf # Try to import cudf
            frame_cudf = cudf.from_pandas(frame_df_pd)
            frame_doc_dataset_input = DocumentDataset(df=frame_cudf)
            print(f"[Video: {video_id}] Converted frame_df to cuDF for embedder.")
        except ImportError:
            print(f"[Video: {video_id}] cuDF not available, using Pandas DataFrame for embedder (may be slower).")
            frame_doc_dataset_input = DocumentDataset(df=frame_df_pd) # PathBasedFrameEmbedder should ideally handle Pandas too

        # PathBasedFrameEmbedder returns a DocumentDataset with a Dask DataFrame.
        # For single video processing here, we compute it immediately.
        processed_frames_dataset = embedder(frame_doc_dataset_input) 
        processed_frames_df = processed_frames_dataset.df.compute() # Results for this video's frames
        res['frames_after_embedding'] = len(processed_frames_df)
        
        if 'image_embedding' not in processed_frames_df.columns:
            res['status'] = 'error_embedding_no_output_col'
            res['error_message'] = "Embedder did not produce 'image_embedding' column."
            if args_ns.cleanup_temp_frames and os.path.exists(video_temp_dir): shutil.rmtree(video_temp_dir)
            return pd.Series(res)
            
        print(f"[Video: {video_id}] Finished embedding. Got {res['frames_after_embedding']} results.")
        # Expected columns from classifiers: 'aesthetic_score', 'nsfw_score' (or as defined by classifier.pred_column)

        # D. Filter Frames
        current_frames_df = processed_frames_df
        if 'aesthetic_score' in current_frames_df.columns:
            current_frames_df = current_frames_df[current_frames_df['aesthetic_score'] >= args_ns.aesthetic_threshold]
            res['frames_after_aesthetic_filter'] = len(current_frames_df)
            print(f"[Video: {video_id}] Frames after aesthetic filter: {res['frames_after_aesthetic_filter']}")
        else:
            print(f"[Video: {video_id}] Warning: 'aesthetic_score' column not found. Skipping aesthetic filter.")
            res['frames_after_aesthetic_filter'] = -1 # Indicate not run

        if 'nsfw_score' in current_frames_df.columns:
            current_frames_df = current_frames_df[current_frames_df['nsfw_score'] < args_ns.nsfw_threshold]
            res['frames_after_nsfw_filter'] = len(current_frames_df)
            print(f"[Video: {video_id}] Frames after NSFW filter: {res['frames_after_nsfw_filter']}")
        else:
            print(f"[Video: {video_id}] Warning: 'nsfw_score' column not found. Skipping NSFW filter.")
            res['frames_after_nsfw_filter'] = -1 # Indicate not run

        if current_frames_df.empty:
            res['status'] = 'success_no_frames_after_filters'
            res['error_message'] = "No frames remained after content filtering."
            if args_ns.cleanup_temp_frames and os.path.exists(video_temp_dir): shutil.rmtree(video_temp_dir)
            return pd.Series(res)

        # E. Intra-Video Semantic Deduplication
        # frame_deduplicator expects Pandas DataFrame
        # Convert current_frames_df (which is Pandas after .compute())
        dedup_input_df = current_frames_df.copy() # Ensure it's Pandas
        
        print(f"[Video: {video_id}] Starting intra-video deduplication for {len(dedup_input_df)} frames...")
        dedup_frames_df = frame_deduplicator.deduplicate_video_frames_sequential(
            frame_df=dedup_input_df,
            embedding_col='image_embedding', # From PathBasedFrameEmbedder
            similarity_threshold=args_ns.dedup_similarity_threshold,
            min_sequence_break=args_ns.dedup_min_sequence_break
        )
        final_selected_frames_df = dedup_frames_df[~dedup_frames_df['is_intra_video_duplicate']]
        res['frames_after_dedup'] = len(final_selected_frames_df)
        print(f"[Video: {video_id}] Frames after deduplication: {res['frames_after_dedup']}")

        if final_selected_frames_df.empty:
            res['status'] = 'success_no_frames_after_dedup'
            res['error_message'] = "No frames remained after deduplication."
            if args_ns.cleanup_temp_frames and os.path.exists(video_temp_dir): shutil.rmtree(video_temp_dir)
            return pd.Series(res)

        # Aggregate scores for selected frames
        if 'aesthetic_score' in final_selected_frames_df.columns and not final_selected_frames_df['aesthetic_score'].empty:
            res['avg_aesthetic_score_selected'] = final_selected_frames_df['aesthetic_score'].mean()
        if 'nsfw_score' in final_selected_frames_df.columns and not final_selected_frames_df['nsfw_score'].empty:
            res['max_nsfw_score_selected'] = final_selected_frames_df['nsfw_score'].max()

        # F. DALI Augmentation
        selected_frame_paths_for_aug = final_selected_frames_df['frame_path'].tolist()
        video_augmented_output_dir = os.path.join(args_ns.base_augmented_frames_output_dir, video_id)
        res['path_to_augmented_frames_output_dir'] = video_augmented_output_dir
        
        print(f"[Video: {video_id}] Starting DALI augmentation for {len(selected_frame_paths_for_aug)} selected frames...")
        _, aug_count = dali_augmenter.augment_frames_dali(
            selected_frame_paths=selected_frame_paths_for_aug,
            video_id=video_id,
            base_augmented_output_dir=args_ns.base_augmented_frames_output_dir, # DALI augmenter creates video_id subdir
            dali_batch_size=args_ns.dali_batch_size,
            dali_device=args_ns.dali_device,
            num_threads=args_ns.dali_num_threads,
            num_augmentations_per_frame=args_ns.num_augmentations_per_frame,
            target_size_wh=(args_ns.dali_target_width, args_ns.dali_target_height)
        )
        res['augmented_frames_count'] = aug_count
        print(f"[Video: {video_id}] Generated {res['augmented_frames_count']} augmented frames.")
        
        res['status'] = 'success'

    except Exception as e:
        import traceback
        print(f"Error processing video {video_id}: {e}\n{traceback.format_exc()}")
        res['status'] = 'error_processing_video'
        res['error_message'] = str(e)
    
    finally:
        # G. Cleanup
        if args_ns.cleanup_temp_frames and os.path.exists(video_temp_dir):
            try:
                shutil.rmtree(video_temp_dir)
                print(f"[Video: {video_id}] Cleaned up temporary frame directory: {video_temp_dir}")
            except Exception as e:
                print(f"[Video: {video_id}] Error cleaning up temp directory {video_temp_dir}: {e}")
                if res['status'] == 'success': # If main processing was success, but cleanup failed
                    res['status'] = 'success_cleanup_failed' 
                    res['error_message'] = f"Cleanup failed: {e}"

    return pd.Series(res)


def main(args):
    print("--- Advanced Video Pipeline Main Script Start ---")
    # 1. Setup Dask client
    client = get_client(num_workers=args.num_workers, create_cluster_if_needed=True)
    print(f"Dask client configured: {client}")

    # Create base output directories if they don't exist
    os.makedirs(args.base_temp_frame_extraction_dir, exist_ok=True)
    os.makedirs(args.base_augmented_frames_output_dir, exist_ok=True)
    output_manifest_dir = os.path.dirname(args.output_manifest)
    if output_manifest_dir: os.makedirs(output_manifest_dir, exist_ok=True)

    # 2. Load initial video manifest
    print(f"Loading input manifest: {args.input_manifest}")
    try:
        # Load as pandas first, then convert to Dask. DocumentDataset is a wrapper.
        # NeMo Curator modules often expect DocumentDataset as input.
        # However, for .apply, we need a Dask DataFrame.
        initial_pd_df = pd.read_json(args.input_manifest, lines=True)
        if initial_pd_df.empty:
            print("Input manifest is empty. Exiting.")
            with open(args.output_manifest, 'w') as f: pass # Create empty output
            return
        
        if args.max_videos is not None and args.max_videos > 0:
            initial_pd_df = initial_pd_df.head(args.max_videos)
            print(f"Processing a maximum of {args.max_videos} videos.")

        # Determine number of partitions for Dask
        num_rows = len(initial_pd_df)
        if args.num_workers and args.num_workers > 0:
            npartitions = min(args.num_workers * 2, num_rows) if num_rows > 0 else 1 # Example: 2 partitions per worker
        else:
            npartitions = min( (os.cpu_count() or 2) * 2, num_rows) if num_rows > 0 else 1
        
        # Use partition_size if specified
        if args.partition_size:
             video_ddf = dd.from_pandas(initial_pd_df, npartitions=None, chunksize=args.partition_size)
             print(f"Created Dask DataFrame with partition_size = {args.partition_size}. Partitions: {video_ddf.npartitions}")
        else:
             video_ddf = dd.from_pandas(initial_pd_df, npartitions=npartitions)
             print(f"Created Dask DataFrame with npartitions = {video_ddf.npartitions}")


    except Exception as e:
        print(f"Failed to load or partition input manifest: {e}")
        return
    
    # Define meta for the output of process_video_row
    meta = {
        'video_id': 'object', 'status': 'object', 'error_message': 'object',
        'original_frame_count': 'i8', 'frames_after_embedding': 'i8',
        'frames_after_aesthetic_filter': 'i8', 'frames_after_nsfw_filter': 'i8',
        'frames_after_dedup': 'i8', 'augmented_frames_count': 'i8',
        'path_to_extracted_frames_dir': 'object',
        'path_to_augmented_frames_output_dir': 'object',
        'avg_aesthetic_score_selected': 'f8',
        'max_nsfw_score_selected': 'f8'
    }

    # Apply processing function
    print("Starting Dask computation for video processing...")
    results_ddf = video_ddf.apply(
        process_video_row, 
        args=(args,), # Pass the full args namespace
        axis=1, 
        meta=meta
    )
    
    computed_results_df = results_ddf.compute()
    print(f"Dask computation finished. Processed {len(computed_results_df)} videos.")

    # Save results
    print(f"Saving updated manifest to: {args.output_manifest}")
    try:
        computed_results_df.to_json(args.output_manifest, orient='records', lines=True)
        print("Manifest saved successfully.")
    except Exception as e:
        print(f"Error saving output manifest: {e}")

    # Final summary
    print("\n--- Pipeline Summary ---")
    status_counts = computed_results_df['status'].value_counts()
    print("Status counts:")
    for status_val, count in status_counts.items():
        print(f"  {status_val}: {count}")
    
    successful_videos = computed_results_df[computed_results_df['status'] == 'success']
    if not successful_videos.empty:
        print(f"\nMetrics for successfully processed videos (average over {len(successful_videos)} videos):")
        print(f"  Avg. Original Frames: {successful_videos['original_frame_count'].mean():.2f}")
        print(f"  Avg. Frames after Deduplication: {successful_videos['frames_after_dedup'].mean():.2f}")
        print(f"  Avg. Augmented Frames Generated: {successful_videos['augmented_frames_count'].mean():.2f}")
        if 'avg_aesthetic_score_selected' in successful_videos.columns:
            print(f"  Avg. Aesthetic Score (selected frames): {successful_videos['avg_aesthetic_score_selected'].mean():.2f}")

    print("--- Advanced Video Pipeline Main Script End ---")


if __name__ == '__main__':
    helper = ArgumentHelper(description="Main orchestration script for advanced frame-level video processing.")

    # Input/Output
    helper.add_arg_input_file(required=True, help_msg="Path to video-level JSONL manifest.")
    helper.add_arg_output_file(required=True, help_msg="Path to save updated video-level JSONL (processing results).")
    helper.parser.add_argument("--base_temp_frame_extraction_dir", required=True, help="Base temporary directory for storing raw extracted frames.")
    helper.parser.add_argument("--base_augmented_frames_output_dir", required=True, help="Base output directory for storing augmented frames.")

    # Column Names
    helper.add_arg_id_field(default="video_id", help_msg="Column name for video ID in manifest.") # --id_field
    helper.parser.add_argument("--video_filepath_col", default="filepath", help="Column name for raw video file path.")

    # Dask args
    helper.add_distributed_args() # --num_workers, --scheduler_file, etc.
    helper.add_arg_partitions() # --num_partitions, --partition_size

    # Frame Extractor args
    helper.parser.add_argument("--ffmpeg_path", default="ffmpeg", help="Path to FFmpeg executable.")
    helper.parser.add_argument("--raw_frame_fps", type=float, default=-1.0, help="Target FPS for raw frame extraction (-1.0 for original FPS).")

    # Embedding/Classifier args
    helper.parser.add_argument("--embedding_model_name", required=True, help="Name of TIMM model for embeddings (e.g., vit_large_patch14_clip_224.openai).")
    helper.parser.add_argument("--embedding_batch_size", type=int, default=32, help="Batch size for frame embedding.")
    helper.parser.add_argument("--aesthetic_threshold", type=float, default=4.5, help="Aesthetic score threshold for filtering frames.")
    helper.parser.add_argument("--nsfw_threshold", type=float, default=0.8, help="NSFW score threshold (frames >= threshold are considered NSFW and potentially filtered).")

    # Frame Deduplicator args
    helper.parser.add_argument("--dedup_similarity_threshold", type=float, default=0.98, help="Cosine similarity threshold for sequential frame deduplication.")
    helper.parser.add_argument("--dedup_min_sequence_break", type=int, default=1, help="Min dissimilar frames to break a duplicate sequence.")
    
    # DALI Augmenter args
    helper.parser.add_argument("--dali_batch_size", type=int, default=32, help="Batch size for DALI augmentation pipeline.")
    helper.parser.add_argument("--dali_device", choices=['cpu', 'gpu'], default='cpu', help="DALI device for augmentation.")
    helper.parser.add_argument("--dali_num_threads", type=int, default=2, help="Number of CPU threads for DALI pipeline.")
    helper.parser.add_argument("--num_augmentations_per_frame", type=int, default=1, help="Number of augmented versions to create per selected frame.")
    helper.parser.add_argument("--dali_target_width", type=int, default=256, help="Target width for DALI augmented frames.")
    helper.parser.add_argument("--dali_target_height", type=int, default=256, help="Target height for DALI augmented frames.")

    # Control args
    helper.parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process from the input manifest (for testing).")
    helper.parser.add_argument("--cleanup_temp_frames", action="store_true", help="If set, removes the temporary raw frame extraction directory for each video after processing.")
    
    args = helper.parse_args()
    main(args)
