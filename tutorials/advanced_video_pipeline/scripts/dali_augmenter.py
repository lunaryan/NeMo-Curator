import os
import shutil
import numpy as np # For image array manipulation if needed by save_dali_batch

# Conditional DALI import
try:
    from nvidia.dali import pipeline_def, fn, types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    # Define dummy types for class definition and type hints if DALI not available
    class types: 
        RGB = None
        UINT8 = None
        FLOAT = None
        DALIDataType = None
        IMAGE_TYPE = None # Placeholder for actual DALI image type if used
    def pipeline_def(*args, **kwargs): 
        def decorator(f):
            return f
        return decorator
    class fn_dummy:
        def __getattr__(self, name):
            def dummy_fn(*args, **kwargs):
                raise ImportError(f"NVIDIA DALI not installed, cannot use fn.{name}")
            return dummy_fn
    fn = fn_dummy()


def save_dali_batch_to_files(output_dir, batch_data_list, batch_start_index, original_filenames_for_naming_inspiration=None):
    """ Helper to save a batch of DALI output images (list of NumPy arrays). """
    saved_paths = []
    try:
        import cv2 # OpenCV for saving
    except ImportError:
        print("Error: OpenCV is required by save_dali_batch_to_files to save images.")
        return saved_paths

    for i in range(len(batch_data_list)):
        img_array = batch_data_list[i] # Expects a list of NumPy HWC arrays
        
        if img_array.dtype != np.uint8:
            if img_array.dtype == np.float32: # Assuming 0-1 range from augmentations
                img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
            else: 
                img_array = np.clip(img_array, 0, 255).astype(np.uint8) # General case, clip and cast
        
        original_name_part = ""
        if original_filenames_for_naming_inspiration and i < len(original_filenames_for_naming_inspiration):
            base, _ = os.path.splitext(os.path.basename(original_filenames_for_naming_inspiration[i]))
            original_name_part = f"{base}_aug_"

        out_filename = f"{original_name_part}aug_frame_{batch_start_index + i:07d}.jpg"
        out_filepath = os.path.join(output_dir, out_filename)
        
        try:
            # DALI usually outputs RGB, OpenCV imwrite expects BGR
            cv2.imwrite(out_filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            saved_paths.append(out_filepath)
        except Exception as e:
            print(f"Error saving augmented frame {out_filepath}: {e}")
    return saved_paths


def augment_frames_dali(
    selected_frame_paths: list[str], 
    video_id: str,
    base_augmented_output_dir: str,
    dali_batch_size: int = 4, 
    dali_device: str = "cpu",
    num_threads: int = 2,
    num_augmentations_per_frame: int = 1,
    target_size_wh: tuple = (256, 256)
) -> tuple[list[str], int]:
    """
    Uses DALI to augment a list of selected frames and saves them.
    """
    if not DALI_AVAILABLE:
        print("Error: NVIDIA DALI is not installed. Cannot perform DALI augmentation.")
        return [], 0
    
    if not selected_frame_paths:
        print(f"No frames provided for DALI augmentation for video {video_id}.")
        return [], 0

    video_specific_aug_dir = os.path.join(base_augmented_output_dir, str(video_id), "augmented_frames")
    os.makedirs(video_specific_aug_dir, exist_ok=True)

    all_input_paths_for_dali = []
    for p in selected_frame_paths:
        all_input_paths_for_dali.extend([p] * num_augmentations_per_frame)
    
    if not all_input_paths_for_dali:
         return [], 0

    @pipeline_def(batch_size=dali_batch_size, num_threads=num_threads, device_id=0 if dali_device=="gpu" else None, prefetch_queue_depth=1)
    def dali_augmentation_pipeline_def(frame_files_input):
        encoded_images, _ = fn.readers.file(
            files=frame_files_input, 
            name="FrameFileInputReader",
            shuffle_after_epoch=True if num_augmentations_per_frame > 1 else False, 
            pad_last_batch=True 
        )
        
        images = fn.decoders.image(encoded_images, device="mixed" if dali_device=="gpu" else "cpu", output_type=types.RGB)
        images = fn.resize(images, size=[target_size_wh[1], target_size_wh[0]], device=dali_device) # H, W for DALI resize
        
        images = fn.brightness_contrast(images, 
                                        brightness=fn.random.uniform(range=(0.7, 1.3)),
                                        contrast=fn.random.uniform(range=(0.7, 1.3)),
                                        device=dali_device)
        images = fn.hsv(images, hue=fn.random.uniform(range=(-20, 20)), device=dali_device)
        images = fn.flip(images, horizontal=fn.random.coin_flip(), device=dali_device)
        images = fn.rotate(images, angle=fn.random.uniform(range=(-15, 15)), fill_value=0, device=dali_device)
        
        return images

    pipe = dali_augmentation_pipeline_def(frame_files_input=all_input_paths_for_dali)
    pipe.build()

    saved_augmented_frame_paths = []
    total_augmented_frames_saved = 0
    num_total_inputs_for_dali = len(all_input_paths_for_dali)
    
    # Calculate iterations carefully based on DALI's behavior with pad_last_batch
    # Each pipe.run() gives one batch.
    num_batches = (num_total_inputs_for_dali + dali_batch_size - 1) // dali_batch_size

    for i in range(num_batches):
        try:
            pipe_out = pipe.run()
            output_batch_dali_tensorlist = pipe_out[0] # This is a DALI TensorList (CPU or GPU)
            
            # Convert DALI TensorList to list of NumPy arrays on CPU
            output_batch_cpu_list = []
            if dali_device == "gpu":
                for j in range(output_batch_dali_tensorlist.num_samples()):
                    output_batch_cpu_list.append(output_batch_dali_tensorlist.at(j).as_cpu().as_array())
            else: # Already on CPU
                for j in range(output_batch_dali_tensorlist.num_samples()):
                     output_batch_cpu_list.append(output_batch_dali_tensorlist.at(j).as_array())
            
            # Determine original filenames for naming inspiration for this batch
            # This is simplified; if shuffle_after_epoch is True, direct mapping is lost.
            naming_inspiration_list = None
            if num_augmentations_per_frame == 1 and not (True if num_augmentations_per_frame > 1 else False): # No shuffle
                start_idx_original_paths = i * dali_batch_size
                end_idx_original_paths = start_idx_original_paths + len(output_batch_cpu_list)
                naming_inspiration_list = selected_frame_paths[start_idx_original_paths:end_idx_original_paths]

            saved_in_batch = save_dali_batch_to_files(
                video_specific_aug_dir, 
                output_batch_cpu_list, 
                batch_start_index=total_augmented_frames_saved,
                original_filenames_for_naming_inspiration=naming_inspiration_list
            )
            saved_augmented_frame_paths.extend(saved_in_batch)
            total_augmented_frames_saved += len(saved_in_batch)

        except StopIteration: # Should not happen with pre-calculated num_batches if pad_last_batch=True
            break 
        except Exception as e:
            print(f"Error during DALI augmentation pipeline for video {video_id} (iteration {i}): {e}")
            break 

    print(f"Video {video_id}: Saved {total_augmented_frames_saved} augmented frames to {video_specific_aug_dir}")
    return saved_augmented_frame_paths, total_augmented_frames_saved

if __name__ == '__main__':
    if not DALI_AVAILABLE:
        print("DALI is not available, cannot run DALI augmenter example.")
    else:
        print("DALI is available. Running conceptual example for dali_augmenter.py.")
        # Setup a dummy environment for testing
        test_video_id = "dali_test_vid_01"
        base_test_output_dir = "temp_dali_augmenter_output"
        
        # Create some dummy input frame files (e.g., using OpenCV)
        sample_input_frames_dir = os.path.join(base_test_output_dir, "sample_input_frames", test_video_id)
        os.makedirs(sample_input_frames_dir, exist_ok=True)
        
        dummy_frame_paths = []
        try:
            import cv2
            for i in range(5): # Create 5 dummy frames
                frame_path = os.path.join(sample_input_frames_dir, f"frame_{i:03d}.jpg")
                # Create a small black image with some text
                img = np.zeros((64, 64, 3), dtype=np.uint8) 
                cv2.putText(img, f"F{i}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imwrite(frame_path, img)
                dummy_frame_paths.append(os.path.abspath(frame_path))
            print(f"Created {len(dummy_frame_paths)} dummy frames for testing in {sample_input_frames_dir}")

            if dummy_frame_paths:
                saved_paths, count = augment_frames_dali(
                    selected_frame_paths=dummy_frame_paths,
                    video_id=test_video_id,
                    base_augmented_output_dir=os.path.join(base_test_output_dir, "augmented_output"),
                    dali_batch_size=2,
                    dali_device="cpu", # Change to "gpu" if you have a GPU and DALI GPU build
                    num_threads=2,
                    num_augmentations_per_frame=2, # Create 2 augmented versions of each input frame
                    target_size_wh=(50, 50)
                )
                print(f"DALI example finished. Saved {count} augmented frames.")
                if count > 0: print(f"Example augmented frame path: {saved_paths[0]}")
            else:
                print("No dummy frames created, skipping DALI augmentation example.")

        except ImportError:
            print("OpenCV is required to create dummy frames for the DALI example. Skipping.")
        except Exception as e:
            print(f"An error occurred during the DALI example: {e}")
        finally:
            # shutil.rmtree(base_test_output_dir, ignore_errors=True) # Clean up test data
            print(f"If test ran, please manually delete: {base_test_output_dir}")
```
