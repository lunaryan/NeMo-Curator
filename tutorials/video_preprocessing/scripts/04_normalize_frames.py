import os
import argparse
import cv2
import numpy as np
import glob

def get_np_dtype(dtype_str):
    """Maps a dtype string to a np.dtype object."""
    try:
        return np.dtype(dtype_str)
    except TypeError:
        return None

def normalize_frames(input_dir, output_dir, method, range_min, range_max, user_mean, user_std, output_dtype_str):
    """Normalizes frames from input_dir and saves them to output_dir."""

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    output_dtype_np = get_np_dtype(output_dtype_str)
    if output_dtype_np is None:
        print(f"Error: Invalid output_dtype '{output_dtype_str}'. Please use a valid NumPy dtype string (e.g., 'float32', 'uint8').")
        return

    image_patterns = [
        '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif',
        '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF', '*.GIF'
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not image_paths:
        print(f"No images found in '{input_dir}' with supported extensions.")
        return

    total_frames = len(image_paths)
    print(f"Found {total_frames} frames to normalize.")

    for i, image_path in enumerate(image_paths):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: Could not read image '{image_path}'. Skipping.")
                continue

            img_float = image.astype(np.float32)
            norm_img = None

            if method == 'minmax':
                current_min = np.min(img_float)
                current_max = np.max(img_float)
                if current_max == current_min:
                    norm_img = np.full(img_float.shape, range_min, dtype=np.float32)
                else:
                    norm_img = (img_float - current_min) / (current_max - current_min)
                norm_img = norm_img * (range_max - range_min) + range_min
            
            elif method == 'meanstd':
                num_channels = img_float.shape[2] if img_float.ndim == 3 else 1
                
                calc_mean = None
                if user_mean:
                    calc_mean = np.array(user_mean, dtype=np.float32)
                    if calc_mean.size == 1 and num_channels > 1: # Single mean value for multi-channel image
                         calc_mean = np.full(num_channels, calc_mean[0])
                    if calc_mean.shape[0] != num_channels:
                        print(f"Warning: Provided mean has {calc_mean.shape[0]} values, but image '{os.path.basename(image_path)}' has {num_channels} channels. Using image-specific mean.")
                        calc_mean = None 
                if calc_mean is None: # If not provided or incompatible
                    calc_mean = np.mean(img_float, axis=(0, 1))

                calc_std = None
                if user_std:
                    calc_std = np.array(user_std, dtype=np.float32)
                    if calc_std.size == 1 and num_channels > 1: # Single std value for multi-channel image
                        calc_std = np.full(num_channels, calc_std[0])
                    if calc_std.shape[0] != num_channels:
                        print(f"Warning: Provided std has {calc_std.shape[0]} values, but image '{os.path.basename(image_path)}' has {num_channels} channels. Using image-specific std.")
                        calc_std = None
                if calc_std is None: # If not provided or incompatible
                    calc_std = np.std(img_float, axis=(0, 1))
                
                # Ensure calc_std is not too small to avoid division by zero or instability
                calc_std = np.where(calc_std < 1e-7, 1e-7, calc_std)
                
                norm_img = (img_float - calc_mean) / calc_std

            # Convert to output dtype and save
            save_img = None
            if output_dtype_np == np.uint8:
                if method == 'minmax':
                    # norm_img is already in range_min to range_max. Clip to 0-255.
                    # This assumes range_min, range_max are somewhat compatible with 0-255.
                    # E.g. if range_max is 1.0, it scales to 255. If range_max is 255, it's direct.
                    if range_max <= 1.0 and range_min >=0: # Typical 0-1 float range
                         save_img = np.clip(norm_img * 255.0, 0, 255).astype(np.uint8)
                    else: # Assume norm_img is already in a 0-255 like range
                         save_img = np.clip(norm_img, 0, 255).astype(np.uint8)
                elif method == 'meanstd':
                    print(f"Warning: Saving mean/std normalized image '{os.path.basename(image_path)}' as uint8. This involves re-scaling and potential data loss. Consider 'float32' for output_dtype.")
                    # Re-normalize to 0-1 range first
                    norm_img_min = np.min(norm_img)
                    norm_img_max = np.max(norm_img)
                    if norm_img_max == norm_img_min:
                        rescaled_img = np.zeros(norm_img.shape, dtype=np.float32)
                    else:
                        rescaled_img = (norm_img - norm_img_min) / (norm_img_max - norm_img_min)
                    save_img = (np.clip(rescaled_img, 0, 1) * 255).astype(np.uint8)
            elif np.issubdtype(output_dtype_np, np.floating):
                save_img = norm_img.astype(output_dtype_np)
            elif np.issubdtype(output_dtype_np, np.integer):
                print(f"Warning: Saving to integer type '{output_dtype_str}' other than uint8. This may involve clipping/scaling not explicitly defined. Output may not be as expected.")
                # For other int types, we might need specific scaling logic.
                # A common approach for uint16 is to scale 0-1 float to 0-65535
                if output_dtype_np == np.uint16:
                    if method == 'minmax' and range_min >= 0 and range_max <=1:
                         save_img = np.clip(norm_img * 65535.0, 0, 65535).astype(np.uint16)
                    else: # General case, rescale to 0-1 then to 0-65535
                         norm_img_min = np.min(norm_img)
                         norm_img_max = np.max(norm_img)
                         if norm_img_max == norm_img_min:
                             rescaled_img = np.full(norm_img.shape, 0, dtype=np.float32)
                         else:
                             rescaled_img = (norm_img - norm_img_min) / (norm_img_max - norm_img_min)
                         save_img = (np.clip(rescaled_img, 0, 1) * 65535).astype(np.uint16)
                else: # For other integer types, just cast, might not be ideal
                    save_img = norm_img.astype(output_dtype_np)
            else:
                print(f"Error: Unsupported output_dtype '{output_dtype_str}'. Cannot save image.")
                continue

            original_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, original_filename)
            
            cv2.imwrite(output_path, save_img)
            
            print(f"Normalized ({method}) {i+1} of {total_frames}: {original_filename} -> {output_path} (dtype: {output_dtype_str})")

        except Exception as e:
            print(f"Error processing image '{image_path}': {e}")
            
    print(f"Successfully normalized {total_frames} frames and saved them to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize image frames.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the frames to normalize.")
    parser.add_argument("--output_dir", required=True, help="Directory to save normalized frames.")
    parser.add_argument("--method", default="minmax", choices=["minmax", "meanstd"], help="Normalization method.")
    parser.add_argument("--range_min", type=float, default=0.0, help="Target minimum for minmax normalization.")
    parser.add_argument("--range_max", type=float, default=1.0, help="Target maximum for minmax normalization.")
    parser.add_argument("--mean", nargs='+', type=float, default=None, help="Mean for meanstd normalization (list of floats, e.g., 0.5 0.5 0.5). Calculated per-image if not provided.")
    parser.add_argument("--std", nargs='+', type=float, default=None, help="Standard deviation for meanstd normalization (list of floats, e.g., 0.5 0.5 0.5). Calculated per-image if not provided.")
    parser.add_argument("--output_dtype", default="float32", help="NumPy dtype for the output array before saving (e.g., float32, uint8, float64, uint16).")
    
    args = parser.parse_args()

    # Validate mean/std length if provided (basic check, more detailed in function)
    if args.mean is not None and args.std is None:
        parser.error("--mean requires --std to be specified as well.")
    if args.std is not None and args.mean is None:
        parser.error("--std requires --mean to be specified as well.")
    if args.mean is not None and args.std is not None and len(args.mean) != len(args.std) and not (len(args.mean)==1 or len(args.std)==1) :
         # Allow single value mean/std to be broadcasted later, but if multiple values are given, they must match.
         # This check is simplified; the core logic inside the function handles channel matching more robustly.
         pass


    normalize_frames(args.input_dir, args.output_dir, args.method, 
                     args.range_min, args.range_max, 
                     args.mean, args.std, args.output_dtype)
