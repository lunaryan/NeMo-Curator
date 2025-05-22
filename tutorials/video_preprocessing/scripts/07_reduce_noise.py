import os
import argparse
import cv2
import numpy as np
import glob

def reduce_noise(args):
    """Reduces noise in images in input_dir and saves them to output_dir."""

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate kernel sizes
    if args.method in ['gaussian', 'median']:
        if args.kernel_size <= 0:
            print(f"Error: Kernel size for {args.method} must be positive. Got {args.kernel_size}.")
            return
        if args.kernel_size % 2 == 0:
            print(f"Error: Kernel size for {args.method} must be odd. Got {args.kernel_size}.")
            return
    elif args.method == 'bilateral':
        if args.kernel_size <= 0: # d parameter for bilateralFilter
            print(f"Error: Kernel size (diameter d) for Bilateral filter must be positive. Got {args.kernel_size}.")
            return
    elif args.method == 'nlm':
        if args.nlm_template_window_size <= 0 or args.nlm_template_window_size % 2 == 0:
            print(f"Error: NLM template window size must be positive and odd. Got {args.nlm_template_window_size}.")
            return
        if args.nlm_search_window_size <= 0 or args.nlm_search_window_size % 2 == 0:
            print(f"Error: NLM search window size must be positive and odd. Got {args.nlm_search_window_size}.")
            return

    image_patterns = [
        '*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif',
        '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF', '*.GIF'
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, pattern)))
    
    if not image_paths:
        print(f"No images found in '{args.input_dir}' with supported extensions.")
        return

    total_frames = len(image_paths)
    print(f"Found {total_frames} frames to denoise.")

    for i, image_path in enumerate(image_paths):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image '{image_path}'. Skipping.")
                continue

            denoised_img = None
            if args.method == 'gaussian':
                denoised_img = cv2.GaussianBlur(img, (args.kernel_size, args.kernel_size), 0)
            elif args.method == 'median':
                denoised_img = cv2.medianBlur(img, args.kernel_size)
            elif args.method == 'bilateral':
                # For bilateralFilter, d (kernel_size) is the diameter.
                denoised_img = cv2.bilateralFilter(img, int(args.kernel_size), args.sigma_color, args.sigma_space)
            elif args.method == 'nlm':
                # Determine if grayscale based on shape
                is_grayscale_proper = len(img.shape) == 2 
                is_grayscale_3channel = len(img.shape) == 3 and img.shape[2] == 1
                
                img_for_nlm = img.copy() # Work on a copy for potential dtype conversion

                if is_grayscale_proper or is_grayscale_3channel:
                    if is_grayscale_3channel: # HxWx1
                        img_for_nlm = img_for_nlm[:,:,0] # Convert to HxW for NLM
                    
                    # cv2.fastNlMeansDenoising expects uint8 input
                    if img_for_nlm.dtype != np.uint8:
                        print(f"Warning: NLM for grayscale expects uint8 input, got {img_for_nlm.dtype}. Converting image '{os.path.basename(image_path)}' to uint8 by scaling.")
                        if np.max(img_for_nlm) > 1.001 and np.max(img_for_nlm) <= 255: # Already in 0-255 range but not uint8
                             img_for_nlm = np.clip(img_for_nlm, 0, 255).astype(np.uint8)
                        elif np.max(img_for_nlm) > 1.001 : # Likely larger range, e.g. uint16
                             img_for_nlm = (np.clip(img_for_nlm / np.max(img_for_nlm),0,1) * 255).astype(np.uint8)
                        else: # Assuming range like 0-1 for floats
                            img_for_nlm = (np.clip(img_for_nlm, 0, 1) * 255).astype(np.uint8)

                    denoised_channel = cv2.fastNlMeansDenoising(
                        img_for_nlm, None, h=args.nlm_h, 
                        templateWindowSize=args.nlm_template_window_size, 
                        searchWindowSize=args.nlm_search_window_size
                    )
                    
                    if is_grayscale_3channel: 
                        denoised_img = cv2.cvtColor(denoised_channel, cv2.COLOR_GRAY2BGR) 
                    else: 
                        denoised_img = denoised_channel
                else: # Color image
                    # cv2.fastNlMeansDenoisingColored expects uint8 input
                    if img_for_nlm.dtype != np.uint8:
                        print(f"Warning: NLM for color expects uint8 input, got {img_for_nlm.dtype}. Converting image '{os.path.basename(image_path)}' to uint8 by scaling.")
                        if np.max(img_for_nlm) > 1.001 and np.max(img_for_nlm) <= 255:
                             img_for_nlm = np.clip(img_for_nlm, 0, 255).astype(np.uint8)
                        elif np.max(img_for_nlm) > 1.001:
                             img_for_nlm = (np.clip(img_for_nlm / np.max(img_for_nlm),0,1) * 255).astype(np.uint8)
                        else:
                             img_for_nlm = (np.clip(img_for_nlm, 0, 1) * 255).astype(np.uint8)
                    
                    denoised_img = cv2.fastNlMeansDenoisingColored(
                        img_for_nlm, None, h=args.nlm_h, hColor=args.nlm_h_color, 
                        templateWindowSize=args.nlm_template_window_size, 
                        searchWindowSize=args.nlm_search_window_size
                    )
            else:
                print(f"Error: Unknown method '{args.method}'. Skipping image '{image_path}'.")
                continue
            
            original_filename = os.path.basename(image_path)
            output_path = os.path.join(args.output_dir, original_filename)
            
            cv2.imwrite(output_path, denoised_img)
            
            print(f"Denoised ({args.method}) {i+1}/{total_frames}: {original_filename} -> {output_path}")

        except Exception as e:
            print(f"Error processing image '{image_path}': {e}")
            
    print(f"Successfully denoised {total_frames} frames and saved them to '{args.output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce noise in image frames.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the frames.")
    parser.add_argument("--output_dir", required=True, help="Directory to save denoised frames.")
    parser.add_argument("--method", default="gaussian", choices=["gaussian", "median", "bilateral", "nlm"], 
                        help="Denoising method (default: gaussian).")
    parser.add_argument("--kernel_size", type=int, default=5, 
                        help="Kernel size for Gaussian/Median (odd, positive). Diameter 'd' for Bilateral (positive). (default: 5).")
    parser.add_argument("--sigma_color", type=float, default=75, 
                        help="Sigma color for Bilateral filter (default: 75).")
    parser.add_argument("--sigma_space", type=float, default=75, 
                        help="Sigma space for Bilateral filter (default: 75).")
    parser.add_argument("--nlm_h", type=float, default=10, 
                        help="Filter strength for NLM (luminance/grayscale) (default: 10).")
    parser.add_argument("--nlm_h_color", type=float, default=10, 
                        help="Filter strength for NLM (color components) (default: 10).")
    parser.add_argument("--nlm_template_window_size", type=int, default=7, 
                        help="Template window size for NLM (odd, positive) (default: 7).")
    parser.add_argument("--nlm_search_window_size", type=int, default=21, 
                        help="Search window size for NLM (odd, positive) (default: 21).")
    
    parsed_args = parser.parse_args()
    reduce_noise(parsed_args)
