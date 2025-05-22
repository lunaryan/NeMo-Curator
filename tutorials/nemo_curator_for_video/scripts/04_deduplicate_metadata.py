import argparse # Keep for direct custom args if any, but ArgumentHelper handles most
import os
import pandas as pd # For saving duplicate IDs if needed
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates, FuzzyDuplicates, FuzzyDuplicatesConfig
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper

def main(args):
    # 1. Setup
    print("--- Deduplicate Metadata Script Start ---")
    
    output_manifest_dir = os.path.dirname(args.output_manifest)
    if output_manifest_dir and not os.path.exists(output_manifest_dir):
        os.makedirs(output_manifest_dir, exist_ok=True)

    if args.dedup_method == 'fuzzy':
        if args.cache_dir and not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir, exist_ok=True)
            print(f"Created cache directory for fuzzy deduplication: {args.cache_dir}")

    # Initialize Dask client using num_workers from ArgumentHelper
    client = get_client(num_workers=args.num_workers, create_cluster_if_needed=True)
    print(f"Dask client configured: {client}")


    # 2. Load Manifest
    print(f"Loading manifest from: {args.input_manifest}")
    try:
        dataset = DocumentDataset.read_json(args.input_manifest, backend='pandas')
        
        if dataset.df is None or dataset.df.empty:
            print("Input manifest is empty or failed to load. Saving an empty manifest.")
            with open(args.output_manifest, 'w') as f: pass
            print(f"Empty output manifest saved to {args.output_manifest}")
            if args.output_duplicate_ids_file:
                 with open(args.output_duplicate_ids_file, 'w') as f: pass
                 print(f"Empty duplicate IDs file saved to {args.output_duplicate_ids_file}")
            return
        
        initial_count = len(dataset.df)
        print(f"Manifest loaded. Initial number of records: {initial_count}")

    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{args.input_manifest}'.")
        return
    except Exception as e:
        print(f"Error loading input manifest '{args.input_manifest}': {e}")
        return

    # Check for required fields (id_field is from ArgumentHelper, dedup_field is custom)
    if args.id_field not in dataset.df.columns:
        print(f"Error: ID field '{args.id_field}' not found in manifest columns: {list(dataset.df.columns)}")
        return
    if args.dedup_field not in dataset.df.columns:
        print(f"Error: Deduplication field '{args.dedup_field}' not found in manifest columns: {list(dataset.df.columns)}")
        return

    # 3. Instantiate Deduplication Module
    dedup_module = None
    print(f"Using deduplication method: {args.dedup_method}")
    print(f"Deduplicating on field: '{args.dedup_field}' using ID field: '{args.id_field}'")
    print(f"Perform removal of duplicates: {args.perform_removal}")

    if args.dedup_method == 'exact':
        dedup_module = ExactDuplicates(
            id_field=args.id_field, 
            text_field=args.dedup_field, 
            perform_removal=args.perform_removal
        )
    elif args.dedup_method == 'fuzzy':
        # Map LSH argument names from ArgumentHelper to FuzzyDuplicatesConfig
        # ArgumentHelper.add_lsh_args provides: args.num_output_bands, args.num_output_rows, args.ngram_size
        fuzzy_config = FuzzyDuplicatesConfig(
            ngram_size=args.ngram_size,
            bands=args.num_output_bands,  # Map from num_output_bands
            rows=args.num_output_rows     # Map from num_output_rows
        )
        dedup_module = FuzzyDuplicates(
            config=fuzzy_config, 
            id_field=args.id_field, 
            text_field=args.dedup_field, 
            cache_dir=args.cache_dir,
            perform_removal=args.perform_removal
        )
        print(f"Fuzzy deduplication config: ngram_size={args.ngram_size}, bands={args.num_output_bands}, rows={args.num_output_rows}, cache_dir='{args.cache_dir}'")
    else:
        print(f"Error: Unknown deduplication method '{args.dedup_method}'. Choose 'exact' or 'fuzzy'.")
        return

    # 4. Apply Deduplication
    print("Applying deduplication...")
    output_dataset = dedup_module(dataset) 
    
    # Handling output_duplicate_ids_file
    if args.output_duplicate_ids_file:
        if not args.perform_removal:
            # output_dataset.df contains the duplicate items if perform_removal=False
            # For ExactDuplicates, this contains all items part of a duplicate group except one original.
            # For FuzzyDuplicates, this contains items clustered as duplicates.
            duplicate_ids_df = output_dataset.df.compute() 
            if not duplicate_ids_df.empty:
                duplicate_ids = duplicate_ids_df[args.id_field].unique().tolist()
                try:
                    with open(args.output_duplicate_ids_file, 'w', encoding='utf-8') as f:
                        for item_id in duplicate_ids:
                            f.write(str(item_id) + '\n')
                    print(f"Saved {len(duplicate_ids)} unique IDs of records identified in duplicate sets to: {args.output_duplicate_ids_file}")
                except IOError as e:
                    print(f"Error writing duplicate IDs file: {e}")
            else:
                print("No duplicates found to save to duplicate IDs file (or output_dataset was empty after non-removal dedup step).")
                with open(args.output_duplicate_ids_file, 'w') as f: pass
        else: # args.perform_removal is True
            print(f"Info: `output_duplicate_ids_file` is specified, but `perform_removal` is True.")
            print(f"  In this mode, this script does not save duplicate IDs separately as they are removed directly.")
            print(f"  To get duplicate IDs, run with --perform_removal unset (or False).")
            with open(args.output_duplicate_ids_file, 'w') as f: pass

    # 5. Save Output Manifest
    print(f"Saving deduplicated manifest to: {args.output_manifest}")
    try:
        output_dataset.to_json(args.output_manifest) 
        # Compute final count after saving (or after dedup_module call if perform_removal=True)
        final_count_df = output_dataset.df.compute() if hasattr(output_dataset.df, 'compute') else output_dataset.df
        final_count = len(final_count_df)
        print(f"Deduplicated manifest saved successfully. Final number of records: {final_count}")
    except Exception as e:
        print(f"Error saving deduplicated manifest: {e}")
        final_count = "N/A (Save failed or error during count)"
        if hasattr(output_dataset, 'df') and isinstance(output_dataset.df, dd.DataFrame):
            print("Attempting fallback save with Dask's to_json directly.")
            try:
                output_path_for_dask = args.output_manifest
                if output_dataset.df.npartitions > 1:
                    parts_dir = args.output_manifest + "_parts_dedup_fallback"
                    os.makedirs(parts_dir, exist_ok=True)
                    output_path_for_dask = os.path.join(parts_dir, "part-*.jsonl")
                output_dataset.df.to_json(output_path_for_dask, orient="records", lines=True, compute=True)
                print(f"Fallback Dask save successful to: {output_path_for_dask}")
            except Exception as e2:
                print(f"Fallback Dask save also failed: {e2}")

    print(f"--- Deduplicate Metadata Script End. Initial: {initial_count}, Final: {final_count} records. ---")

if __name__ == "__main__":
    helper = ArgumentHelper()
    # Required script-specific arguments
    helper.add_arg_input_file(required=True, help_msg="Path to input JSONL manifest.")
    helper.add_arg_output_file(required=True, help_msg="Path to save deduplicated JSONL manifest.")
    
    # Custom arguments for this script
    # For argparse.ArgumentParser, one would use parser.add_argument(...)
    # For ArgumentHelper, it's helper.parser.add_argument(...)
    helper.parser.add_argument("--dedup_field", required=True, help="Metadata field for duplicate checking (e.g., 'title', 'source_description').")
    helper.parser.add_argument("--dedup_method", required=True, choices=['exact', 'fuzzy'], help="Deduplication method.")
    helper.parser.add_argument("--perform_removal", action="store_true", help="If set, removes duplicates from the output manifest.")
    helper.parser.add_argument("--output_duplicate_ids_file", default=None, help="Path to save IDs of videos identified as duplicates (primarily for when perform_removal is False).")

    # Standard arguments from ArgumentHelper
    helper.add_arg_id_field(default="video_id") # Provides --id_field
    helper.add_cache_dir_arg(default="./dedup_cache") # Provides --cache_dir
    
    # LSH args for fuzzy deduplication. ArgumentHelper provides:
    # --ngram_size, --num_output_bands, --num_output_rows
    helper.add_lsh_args(ngram_size_default=5, num_output_bands_default=40, num_output_rows_default=2)
    
    helper.add_distributed_args() # Provides --num_workers

    args = helper.parse_args()
    main(args)
