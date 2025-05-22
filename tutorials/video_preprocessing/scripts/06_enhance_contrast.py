import os
import argparse
import cv2
import numpy as np
import glob

def enhance_contrast(args):
    """Enhances contrast of images in input_dir and saves them to output_dir."""

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

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
    print(f"Found {total_frames} frames to enhance contrast.")

    for i, image_path in enumerate(image_paths):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image '{image_path}'. Skipping.")
                continue

            is_grayscale = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
            
            target_channel = None
            a_channel = None # For LAB
            b_channel = None # For LAB

            if is_grayscale:
                if len(img.shape) == 3 and img.shape[2] == 1: # Grayscale but 3-channel
                    target_channel = img[:,:,0]
                else: # Truly 2D grayscale
                    target_channel = img.copy()
            else: # Color image
                lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                target_channel, a_channel, b_channel = cv2.split(lab_img)

            enhanced_channel = None
            if args.method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=args.clip_limit, tileGridSize=tuple(args.tile_grid_size))
                enhanced_channel = clahe.apply(target_channel)
            elif args.method == 'hist_eq':
                enhanced_channel = cv2.equalizeHist(target_channel)
            else:
                print(f"Error: Unknown method '{args.method}'. Skipping image '{image_path}'.")
                continue

            final_image = None
            if is_grayscale:
                # If original was 3-channel grayscale, make enhanced also 3-channel for consistency
                if len(img.shape) == 3 and img.shape[2] == 1:
                    final_image = cv2.cvtColor(enhanced_channel, cv2.COLOR_GRAY2BGR)
                else:
                    final_image = enhanced_channel
            else: # Color image
                merged_lab = cv2.merge((enhanced_channel, a_channel, b_channel))
                final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
            
            original_filename = os.path.basename(image_path)
            output_path = os.path.join(args.output_dir, original_filename)
            
            cv2.imwrite(output_path, final_image)
            
            print(f"Enhanced contrast ({args.method}) {i+1}/{total_frames}: {original_filename} -> {output_path}")

        except Exception as e:
            print(f"Error processing image '{image_path}': {e}")
            
    print(f"Successfully enhanced contrast for {total_frames} frames and saved them to '{args.output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance contrast of image frames.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the frames.")
    parser.add_argument("--output_dir", required=True, help="Directory to save enhanced frames.")
    parser.add_argument("--method", default="clahe", choices=["clahe", "hist_eq"], 
                        help="Contrast enhancement method (default: clahe).")
    parser.add_argument("--clip_limit", type=float, default=2.0, 
                        help="Clip limit for CLAHE (default: 2.0).")
    parser.add_argument("--tile_grid_size", type=int, nargs=2, default=[8, 8], 
                        metavar=('ROWS', 'COLS'), help="Tile grid size for CLAHE (e.g., 8 8) (default: [8, 8]).")
    
    parsed_args = parser.parse_args()
    enhance_contrast(parsed_args)
