import os
import argparse
import cv2
import glob

def get_cv2_interpolation_flag(interpolation_str):
    """Maps an interpolation string to a cv2.INTER_* flag."""
    mapping = {
        "INTER_NEAREST": cv2.INTER_NEAREST,
        "INTER_LINEAR": cv2.INTER_LINEAR,
        "INTER_AREA": cv2.INTER_AREA,
        "INTER_CUBIC": cv2.INTER_CUBIC,
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
    }
    return mapping.get(interpolation_str.upper())

def resize_frames(input_dir, output_dir, width, height, interpolation_str):
    """Resizes frames from input_dir and saves them to output_dir."""
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    cv2_interpolation_flag = get_cv2_interpolation_flag(interpolation_str)
    if cv2_interpolation_flag is None:
        print(f"Error: Invalid interpolation method '{interpolation_str}'. Valid methods are: INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4.")
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
    print(f"Found {total_frames} frames to resize.")

    for i, image_path in enumerate(image_paths):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image '{image_path}'. Skipping.")
                continue

            resized_image = cv2.resize(image, (width, height), interpolation=cv2_interpolation_flag)
            
            original_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, original_filename)
            
            cv2.imwrite(output_path, resized_image)
            
            print(f"Resized {i+1} of {total_frames}: {original_filename} -> {output_path}")

        except Exception as e:
            print(f"Error processing image '{image_path}': {e}")
            
    print(f"Successfully resized {total_frames} frames and saved them to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize image frames.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the frames to resize.")
    parser.add_argument("--output_dir", required=True, help="Directory to save resized frames.")
    parser.add_argument("--width", required=True, type=int, help="Target width for resizing.")
    parser.add_argument("--height", required=True, type=int, help="Target height for resizing.")
    parser.add_argument("--interpolation", default="INTER_LINEAR", 
                        help="OpenCV interpolation method (e.g., INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_NEAREST, INTER_LANCZOS4).")
    
    args = parser.parse_args()
    
    resize_frames(args.input_dir, args.output_dir, args.width, args.height, args.interpolation)
