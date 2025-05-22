import os
import argparse
import pydicom
import numpy as np
import cv2

def dicom_to_frames(args):
    """Converts a DICOM file to image frames."""

    if not os.path.isfile(args.input_dicom):
        print(f"Error: Input DICOM file '{args.input_dicom}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        ds = pydicom.dcmread(args.input_dicom)
    except Exception as e:
        print(f"Error reading DICOM file '{args.input_dicom}': {e}")
        return

    pixel_array = ds.pixel_array
    number_of_frames = getattr(ds, 'NumberOfFrames', 1)

    frames_data = []
    if number_of_frames > 1:
        frames_data = [pixel_array[i] for i in range(number_of_frames)]
    else:
        frames_data = [pixel_array] # Wrap single frame in a list

    print(f"Found {number_of_frames} frame(s) in the DICOM file.")

    # DICOM tags
    rescale_slope = float(getattr(ds, 'RescaleSlope', 1))
    rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0))
    photometric_interpretation = getattr(ds, 'PhotometricInterpretation', "MONOCHROME2")
    dicom_bits_stored = int(getattr(ds, 'BitsStored', 16)) # Default to 16 for safety if not present

    for i, frame_data in enumerate(frames_data):
        current_frame_pixels = frame_data.copy().astype(np.float32)

        # Apply Rescale Slope/Intercept
        current_frame_pixels = current_frame_pixels * rescale_slope + rescale_intercept

        # Determine WindowCenter (WC) and WindowWidth (WW)
        window_center = args.wc
        window_width = args.ww

        if window_center is None and args.apply_windowing:
            wc_tag = ds.get('WindowCenter')
            if wc_tag:
                # Handle multi-valued WC (use first)
                window_center = float(wc_tag[0]) if isinstance(wc_tag.value, pydicom.multival.MultiValue) else float(wc_tag.value)
        
        if window_width is None and args.apply_windowing:
            ww_tag = ds.get('WindowWidth')
            if ww_tag:
                # Handle multi-valued WW (use first)
                window_width = float(ww_tag[0]) if isinstance(ww_tag.value, pydicom.multival.MultiValue) else float(ww_tag.value)
        
        # Apply Windowing if WC/WW are determined
        if window_center is not None and window_width is not None:
            min_val = window_center - window_width / 2.0
            max_val = window_center + window_width / 2.0
            current_frame_pixels = np.clip(current_frame_pixels, min_val, max_val)
            print(f"Frame {i+1}: Applied windowing WC={window_center}, WW={window_width}. Range: [{min_val}, {max_val}]")
        elif args.wc is not None or args.ww is not None or args.apply_windowing:
            print(f"Frame {i+1}: Windowing requested but WC/WW parameters incomplete or not found in DICOM. WC={window_center}, WW={window_width}")


        # Handle Photometric Interpretation MONOCHROME1 (inversion)
        # This is done *before* normalization to [0,1] to ensure min/max are correct for MONOCHROME1
        if photometric_interpretation == "MONOCHROME1":
            current_frame_pixels = np.max(current_frame_pixels) - current_frame_pixels

        # Pixel Data Conversion for Saving
        # Normalize current_frame_pixels from its current range to [0,1]
        # Add a small epsilon to prevent division by zero if max == min
        min_pixel_val = np.min(current_frame_pixels)
        max_pixel_val = np.max(current_frame_pixels)
        
        if max_pixel_val == min_pixel_val:
            norm_pixels = np.zeros(current_frame_pixels.shape, dtype=np.float32)
        else:
            norm_pixels = (current_frame_pixels - min_pixel_val) / (max_pixel_val - min_pixel_val + 1e-7)

        is_output_png = args.output_format.lower() == 'png'
        # JPG/BMP always become 8-bit due to cv2.imwrite limitations for these formats or common practice
        force_8bit = args.rescale_to_8bit or not is_output_png or args.output_format.lower() in ['jpg', 'jpeg', 'bmp']

        final_pixels = None
        output_bit_depth_msg = ""

        if force_8bit:
            final_pixels = (norm_pixels * 255).astype(np.uint8)
            output_bit_depth_msg = "8-bit"
        else: # Implicitly is_output_png and not args.rescale_to_8bit
            if dicom_bits_stored > 8:
                final_pixels = (norm_pixels * 65535).astype(np.uint16)
                output_bit_depth_msg = "16-bit"
            else:
                final_pixels = (norm_pixels * 255).astype(np.uint8)
                output_bit_depth_msg = "8-bit"
        
        # Construct filename
        frame_filename_base = args.frame_pattern % (i + 1) # 1-based indexing for frames
        output_filename = f"{frame_filename_base}.{args.output_format.lower()}"
        output_path = os.path.join(args.output_dir, output_filename)

        try:
            cv2.imwrite(output_path, final_pixels)
            print(f"Saved frame {i+1}/{number_of_frames}: {output_filename} ({output_bit_depth_msg})")
        except Exception as e:
            print(f"Error saving frame {i+1} ('{output_filename}'): {e}")
            
    print(f"Successfully processed {number_of_frames} frame(s) and saved to '{args.output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DICOM file to image frames.")
    parser.add_argument("--input_dicom", required=True, help="Path to the input DICOM file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted frames.")
    parser.add_argument("--output_format", default="png", choices=["png", "jpg", "jpeg", "bmp", "tiff"], 
                        help="Output image format (default: png).")
    parser.add_argument("--frame_pattern", default="frame_%06d", 
                        help="Naming pattern for output frames (e.g., frame_%%06d). Extension added automatically.")
    parser.add_argument("--apply_windowing", action="store_true", 
                        help="Apply WindowCenter/WindowWidth from DICOM tags if custom WC/WW not provided.")
    parser.add_argument("--wc", type=float, default=None, help="Custom WindowCenter value.")
    parser.add_argument("--ww", type=float, default=None, help="Custom WindowWidth value.")
    parser.add_argument("--rescale_to_8bit", action="store_true", 
                        help="Force rescale to 8-bit (0-255). If false, PNG might be saved as 16-bit if DICOM BitsStored > 8. JPG/BMP are always 8-bit.")
    
    parsed_args = parser.parse_args()
    
    # Normalize output_format (e.g. if user types JPEG or JPG)
    parsed_args.output_format = parsed_args.output_format.lower()
    if parsed_args.output_format == "jpeg":
        parsed_args.output_format = "jpg"

    dicom_to_frames(parsed_args)
