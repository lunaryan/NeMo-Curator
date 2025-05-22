import os
import argparse
import cv2
import glob

def convert_frames(input_dir, output_dir, output_format, grayscale, jpg_quality):
    """Converts frames from input_dir and saves them to output_dir with specified modifications."""

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if output_format and output_format.startswith('.'):
        output_format = output_format[1:] # Remove leading dot if present

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
    print(f"Found {total_frames} frames to convert.")

    for i, image_path in enumerate(image_paths):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image '{image_path}'. Skipping.")
                continue

            image_to_save = image
            if grayscale:
                image_to_save = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            current_output_format = output_format
            if not current_output_format:
                current_output_format = os.path.splitext(image_path)[1][1:] # Get original ext without dot
            
            if not current_output_format: # Handle case where original file has no extension
                print(f"Warning: Image '{image_path}' has no extension. Skipping conversion for this file.")
                continue

            output_filename = f"{base_name}.{current_output_format.lower()}"
            output_path = os.path.join(output_dir, output_filename)
            
            imwrite_params = []
            if current_output_format.lower() in ['jpg', 'jpeg']:
                if not (0 <= jpg_quality <= 100):
                    print(f"Warning: JPG quality ({jpg_quality}) is outside the 0-100 range. Clamping to this range.")
                    clamped_jpg_quality = max(0, min(100, jpg_quality))
                    imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, clamped_jpg_quality]
                else:
                    imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
            
            cv2.imwrite(output_path, image_to_save, imwrite_params)
            
            print(f"Converted {i+1} of {total_frames}: {os.path.basename(image_path)} -> {output_filename}")

        except Exception as e:
            print(f"Error processing image '{image_path}': {e}")
            
    print(f"Successfully converted {total_frames} frames and saved them to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image frames format and properties.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the frames to convert.")
    parser.add_argument("--output_dir", required=True, help="Directory to save converted frames.")
    parser.add_argument("--output_format", default=None, help="Target image format extension (e.g., png, jpg). If None, uses original extension.")
    parser.add_argument("--grayscale", action="store_true", help="If set, convert frames to grayscale.")
    parser.add_argument("--jpg_quality", type=int, default=95, help="Quality for JPG output (0-100).")
    
    args = parser.parse_args()

    # Basic validation for jpg_quality if argparse doesn't handle it fully for custom messages
    # However, type=int and a default usually suffice, and we handle clamping in the function.
    # For this specific requirement, clamping is done inside the function.

    convert_frames(args.input_dir, args.output_dir, args.output_format, args.grayscale, args.jpg_quality)
