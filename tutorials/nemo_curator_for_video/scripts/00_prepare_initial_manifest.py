import argparse
import json
import os

def prepare_initial_manifest(output_file, num_samples):
    """Generates a sample JSONL file containing initial video metadata."""

    output_dir = os.path.dirname(output_file)
    if output_dir: # Ensure output_dir is not an empty string (e.g. if output_file is just a filename)
        os.makedirs(output_dir, exist_ok=True)

    predefined_samples = [
        {
            "video_id": "vid001",
            "filepath": "/path/to/sample_video_1.mp4",
            "source_description": "Routine check-up footage.",
            "tags": ["medical", "checkup", "clinical"]
        },
        {
            "video_id": "vid002",
            "filepath": "/path/to/surgery_archive/archive_05/proc_dr_eva.avi",
            "source_description": "Surgical procedure extract by Dr. Eva Rostova.",
            "tags": ["surgery", "Dr. Eva Rostova", "medical_staff_present"]
        },
        {
            "video_id": "vid003",
            "filepath": "/mnt/research_data/microscopy/exp_alpha/cells_on_slide.mkv",
            "source_description": "Microscopy video of cells, experiment Alpha.",
            "tags": ["research", "microscopy", "lab_data"]
        },
        {
            "video_id": "vid004",
            "filepath": "/path/to/conference_recordings/keynote_main_hall.mp4",
            "source_description": "Conference recording - keynote speech from main hall.",
            "tags": ["conference", "presentation", "public_event"]
        },
        {
            "video_id": "vid005",
            "filepath": "/secure_storage/lobby_cam/2023-10-26_14h00m.mov",
            "source_description": "Security camera footage - lobby area, includes identifiable faces.",
            "tags": ["security", "PII_potential", "surveillance"]
        }
    ]

    generated_entries = []
    num_predefined = len(predefined_samples)

    for i in range(num_samples):
        sample_template = predefined_samples[i % num_predefined]
        entry = sample_template.copy() # Start with a copy of the template

        if i >= num_predefined:
            # Modify video_id and filepath to ensure uniqueness for cycled samples
            original_id_base = sample_template["video_id"].split('_cycle_')[0] # Get base if already cycled
            entry["video_id"] = f"{original_id_base}_cycle_{i // num_predefined}_{i % num_predefined + 1}"
            
            original_filepath = sample_template["filepath"]
            path_part, ext_part = os.path.splitext(original_filepath)
            # Remove previous cycle numbers from path_part if any
            path_part_base = path_part.split('_cycle_')[0]
            entry["filepath"] = f"{path_part_base}_cycle_{i // num_predefined}_{i % num_predefined + 1}{ext_part}"
        else:
            # For the first cycle, use predefined values directly
            entry["video_id"] = sample_template["video_id"]
            entry["filepath"] = sample_template["filepath"]
            
        # Ensure tags is a new list object for each entry if it's copied by reference
        entry["tags"] = list(sample_template["tags"]) 
        generated_entries.append(entry)

    try:
        with open(output_file, 'w') as f:
            for entry in generated_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"Successfully generated {num_samples} sample entries and saved to '{output_file}'.")
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a sample JSONL file with initial video metadata.")
    parser.add_argument("--output_file", required=True, 
                        help="Full path for the output JSONL file (e.g., data/initial_video_manifest.jsonl).")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of sample video metadata entries to generate (default: 5).")
    
    args = parser.parse_args()
    
    if args.num_samples <= 0:
        print("Error: --num_samples must be a positive integer.")
    else:
        prepare_initial_manifest(args.output_file, args.num_samples)
