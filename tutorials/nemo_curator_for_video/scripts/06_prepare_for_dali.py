import argparse
import os
import pandas as pd

# Attempt to import DALI for initial check, but allow script to load if not present
DALI_AVAILABLE = False
try:
    import nvidia.dali.pipeline as dali_pipeline # Use alias to avoid conflict with local 'pipeline'
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
    print("NVIDIA DALI modules found.")
except ImportError:
    print("Warning: NVIDIA DALI modules not found. DALI pipeline execution will fail.")
    # Define dummy classes/functions if needed for type hinting or basic structure,
    # though for this script, direct use in functions should be fine with runtime checks.
    pass 

class VideoPipe(dali_pipeline.Pipeline if DALI_AVAILABLE else object): # Inherit from object if DALI not found
    def __init__(self, batch_size, num_threads, device_str, device_id_int, file_list, labels_list, sequence_length, seed=12345):
        if not DALI_AVAILABLE:
            raise RuntimeError("DALI is not available. Cannot initialize VideoPipe.")
        
        # device_id_int is used by Pipeline constructor
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id_int, seed=seed)
        
        # self.device is used by fn.resize, fn.crop_mirror_normalize
        self.device = device_str # 'cpu' or 'gpu' 
        
        # For fn.readers.video, device can be 'cpu' or 'gpu'.
        # We pass the string directly.
        
        self.reader = fn.readers.video(
            device=device_str,  # 'cpu' or 'gpu'
            file_list=file_list,
            labels=labels_list if labels_list else None, # DALI handles None correctly
            sequence_length=sequence_length,
            normalized=False, # Output pixel values in [0, 255]
            image_type=types.RGB,
            dtype=types.UINT8,
            name="VideoReader",
            random_shuffle=True if not labels_list else False,
            initial_fill=16 if not labels_list else None
        )
        self.has_labels = bool(labels_list)

    def define_graph(self):
        video_data_tuple = self.reader() # Returns a tuple (frames, labels) or just (frames)
        
        frames = video_data_tuple[0] if self.has_labels else video_data_tuple
        
        # Example processing steps
        # Resize to a common size
        processed_frames = fn.resize(frames, size=[224, 224], device=self.device)
        
        # Normalize and change layout (example for typical VLM/CNN input)
        # FCHW: Frames, Channels, Height, Width
        processed_frames = fn.crop_mirror_normalize(
            processed_frames,
            dtype=types.FLOAT,
            output_layout="FCHW", # Frames, Channels, Height, Width
            crop=(224, 224),      # Crop size (can be same as resize if no random crop)
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], # ImageNet mean for RGB
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]   # ImageNet std for RGB
        )

        if self.has_labels:
            labels = video_data_tuple[1]
            return processed_frames, labels
        else:
            return processed_frames

def main(args):
    # 1. Setup
    print("--- Prepare for DALI Script Start ---")
    if not DALI_AVAILABLE and args.dali_device == "gpu": # Or even for CPU if we intend to run
         print("Error: DALI is not installed, but DALI execution is requested. Please install NVIDIA DALI to run the pipeline.")
         return
    elif not DALI_AVAILABLE:
         print("Warning: DALI is not installed. Script will perform data preparation but cannot run the DALI pipeline.")
         # Allow to proceed to show data prep, but pipeline execution part will be skipped.

    # 2. Data Preparation
    print(f"Loading manifest from: {args.input_manifest}")
    try:
        df = pd.read_json(args.input_manifest, lines=True)
        if df.empty:
            print("Input manifest is empty. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{args.input_manifest}'.")
        return
    except ValueError as e: # Handles JSON decoding errors in read_json
        print(f"Error decoding JSON from manifest file '{args.input_manifest}': {e}")
        return
    except Exception as e:
        print(f"Error loading manifest: {e}")
        return

    if args.text_field_for_filepath not in df.columns:
        print(f"Error: Filepath field '{args.text_field_for_filepath}' not found in manifest. Available columns: {list(df.columns)}")
        return
    
    file_list = df[args.text_field_for_filepath].tolist()
    original_count = len(file_list)
    
    labels_list = None
    if args.text_field_for_label:
        if args.text_field_for_label not in df.columns:
            print(f"Warning: Label field '{args.text_field_for_label}' not found in manifest. Proceeding without labels.")
        else:
            try:
                labels_list = df[args.text_field_for_label].astype(int).tolist()
                print(f"Loaded labels from field '{args.text_field_for_label}'.")
            except ValueError:
                print(f"Warning: Could not convert label field '{args.text_field_for_label}' to integer. Proceeding without labels.")
                labels_list = None # Reset if conversion failed
            except Exception as e:
                print(f"Warning: Error processing label field '{args.text_field_for_label}': {e}. Proceeding without labels.")
                labels_list = None


    # Filter out non-existent files
    existent_files = []
    corresponding_labels = [] if labels_list else None

    for i, f_path in enumerate(file_list):
        if os.path.exists(f_path):
            existent_files.append(f_path)
            if labels_list:
                corresponding_labels.append(labels_list[i])
        else:
            print(f"Warning: File not found: '{f_path}'. It will be excluded from DALI pipeline.")
            
    if not existent_files:
        print("Error: No valid (existing) video files found after checking paths from manifest. Exiting.")
        return
        
    print(f"Filtered file list: {len(existent_files)} videos remaining out of {original_count} initial entries.")
    file_list = existent_files
    if labels_list:
        labels_list = corresponding_labels


    # 3. DALI Pipeline Execution (guarded by DALI_AVAILABLE)
    if not DALI_AVAILABLE:
        print("\nNVIDIA DALI is not available. Skipping pipeline creation and execution.")
        print("Data preparation completed. File list and labels (if any) are ready.")
        return

    print(f"\nInitializing DALI VideoPipe: batch_size={args.batch_size}, sequence_length={args.sequence_length}, device='{args.dali_device}'")
    
    # DALI device_id is an integer; 0 for CPU if device is 'cpu', typically 0 for first GPU if 'gpu'.
    # For simplicity, using 0. For multi-GPU, this would need more sophisticated handling.
    dali_device_id_int = 0 

    try:
        pipe = VideoPipe(
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            device_str=args.dali_device, # 'cpu' or 'gpu' string for fn.readers.video and internal logic
            device_id_int=dali_device_id_int, # Integer for Pipeline superclass
            file_list=file_list,
            labels_list=labels_list,
            sequence_length=args.sequence_length
        )
        pipe.build()
        print("DALI pipeline built successfully.")
    except Exception as e:
        print(f"Error building DALI pipeline: {e}")
        print("This might be due to DALI setup issues, or problems with the video files/formats.")
        return

    print(f"\nRunning DALI pipeline for {args.max_iterations} iterations...")
    for i in range(args.max_iterations):
        try:
            pipe_out = pipe.run()
            
            # pipe_out is a tuple of DALI Tensors (or list of Tensors)
            # If labels exist, it's (videos_batch, labels_batch)
            # Otherwise, it's (videos_batch,)
            
            videos_batch = pipe_out[0] if isinstance(pipe_out, tuple) and len(pipe_out)>0 else pipe_out
            
            # DALI tensors are on the device they were produced. For CPU device, can convert to numpy.
            # For GPU, need to copy to host first if you want to interact with numpy.
            # Example: videos_batch_cpu = videos_batch.as_cpu().as_array() (if on GPU)
            # For just printing shapes/types, DALI tensor methods are fine.
            
            print(f"Iteration {i+1}/{args.max_iterations}:")
            print(f"  Video batch shape: {videos_batch.shape()}, dtype: {videos_batch.dtype()}, device: {'GPU' if videos_batch.is_gpu() else 'CPU'}")

            if pipe.has_labels and isinstance(pipe_out, tuple) and len(pipe_out) > 1:
                labels_batch = pipe_out[1]
                print(f"  Labels batch shape: {labels_batch.shape()}, dtype: {labels_batch.dtype()}, device: {'GPU' if labels_batch.is_gpu() else 'CPU'}")
                # print(f"  Sample labels: {labels_batch.as_cpu().as_array()[:min(args.batch_size, 4)]}") # Example: print first few labels

        except RuntimeError as e:
            print(f"DALI runtime error at iteration {i+1}: {e}")
            print("This could be due to a problematic video file or DALI configuration issues.")
            break # Stop on runtime error
        except Exception as e:
            print(f"An unexpected error occurred during DALI pipeline run at iteration {i+1}: {e}")
            break

    print("\n--- Prepare for DALI Script End ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares data from a manifest and demonstrates a DALI video pipeline.")
    parser.add_argument("--input_manifest", required=True, help="Path to JSONL manifest (must contain filepaths).")
    parser.add_argument("--text_field_for_filepath", default="filepath", help="Field name for video file paths (default: filepath).")
    parser.add_argument("--text_field_for_label", default=None, help="Field name for labels (e.g., integer class for each video) (default: None).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DALI pipeline (default: 4).")
    parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames per video sequence in a batch (default: 16).")
    parser.add_argument("--dali_device", choices=['cpu', 'gpu'], default='cpu', help="DALI device to run on ('cpu' or 'gpu') (default: cpu).")
    parser.add_argument("--num_threads", type=int, default=2, help="Number of CPU threads for DALI pipeline (default: 2).")
    parser.add_argument("--max_iterations", type=int, default=5, help="Number of DALI pipeline iterations to run and print info for (default: 5).")
    
    args = parser.parse_args()
    main(args)
