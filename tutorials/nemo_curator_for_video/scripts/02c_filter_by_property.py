import argparse
import json
import os
import pandas as pd
import dask.dataframe as dd
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
import re # For regex_contains if we want to compile regex for efficiency

def load_filters(filters_json_arg):
    """Loads filter criteria from a JSON string or a JSON file path."""
    if os.path.exists(filters_json_arg):
        try:
            with open(filters_json_arg, 'r', encoding='utf-8') as f:
                filters = json.load(f)
            print(f"Loaded filters from file: {filters_json_arg}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in filter file '{filters_json_arg}': {e}")
            return None
        except IOError as e:
            print(f"Error reading filter file '{filters_json_arg}': {e}")
            return None
    else:
        try:
            filters = json.loads(filters_json_arg)
            print("Loaded filters from JSON string.")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON string for filters_json argument: {e}")
            return None

    if not isinstance(filters, list):
        print("Error: Filters must be a list of dictionaries.")
        return None
    
    # Basic validation of filter structure (can be expanded)
    for i, f_rule in enumerate(filters):
        if not isinstance(f_rule, dict) or "field" not in f_rule:
            print(f"Error: Filter rule at index {i} is malformed (must be a dict with a 'field' key): {f_rule}")
            return None
        # Check that at least one valid filter condition is present
        valid_conditions = {"min", "max", "equals", "contains", "regex_contains"}
        if not any(key in f_rule for key in valid_conditions):
            print(f"Error: Filter rule for field '{f_rule['field']}' (index {i}) has no valid condition ({', '.join(valid_conditions)}): {f_rule}")
            return None
            
    return filters

def main(args):
    # 1. Setup
    print("--- Filter by Property Script Start ---")
    output_dir = os.path.dirname(args.output_manifest)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if args.num_workers:
        client = get_client(num_workers=args.num_workers, create_cluster_if_needed=True)
        print(f"Dask client configured: {client}")

    # 2. Load Filters JSON
    filters = load_filters(args.filters_json)
    if filters is None:
        print("Exiting due to errors in loading or validating filters.")
        return
    if not filters: # Empty list of filters
        print("Warning: No filters provided. The output manifest will be a copy of the input.")
    
    # 3. Load Manifest
    print(f"Loading manifest from: {args.input_manifest}")
    try:
        dataset = DocumentDataset.read_json(args.input_manifest, backend='pandas')
        if dataset.df is None or dataset.df.empty:
            print("Input manifest is empty or failed to load. Saving an empty manifest.")
            with open(args.output_manifest, 'w') as f: pass
            print(f"Empty output manifest saved to {args.output_manifest}")
            return

        num_rows_initial = len(dataset.df)
        npartitions = args.num_workers if args.num_workers and args.num_workers > 0 else (os.cpu_count() or 2)
        npartitions = min(npartitions, num_rows_initial) if num_rows_initial > 0 else 1
        
        ddf = dd.from_pandas(dataset.df, npartitions=npartitions)
        print(f"Manifest loaded into Dask DataFrame with {ddf.npartitions} partitions from {num_rows_initial} initial records.")

    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{args.input_manifest}'.")
        return
    except Exception as e:
        print(f"Error loading input manifest '{args.input_manifest}': {e}")
        return

    initial_count = len(ddf) # This triggers computation if ddf is not already computed
    print(f"Initial number of records: {initial_count}")
    
    # 4. Apply Filters Sequentially
    filtered_ddf = ddf
    
    for i, filter_rule in enumerate(filters):
        field_name = filter_rule["field"]
        print(f"\nApplying filter {i+1}/{len(filters)} on field '{field_name}': {filter_rule}")

        if field_name not in filtered_ddf.columns:
            print(f"  Warning: Field '{field_name}' not found in manifest columns. Skipping this filter.")
            continue

        current_filter_conditions = []

        # Min/Max filter
        min_val = filter_rule.get("min")
        max_val = filter_rule.get("max")
        if min_val is not None:
            # Ensure the column is numeric before comparison, or handle potential errors
            # Dask will try to infer, but explicit conversion might be needed for mixed types
            # For now, assume column type is appropriate or Dask handles it.
            current_filter_conditions.append(filtered_ddf[field_name] >= min_val)
            print(f"    Condition: {field_name} >= {min_val}")
        if max_val is not None:
            current_filter_conditions.append(filtered_ddf[field_name] <= max_val)
            print(f"    Condition: {field_name} <= {max_val}")

        # Equals filter
        if "equals" in filter_rule:
            equals_val = filter_rule["equals"]
            current_filter_conditions.append(filtered_ddf[field_name] == equals_val)
            print(f"    Condition: {field_name} == {equals_val}")
            
        # Contains filter (for list-like columns)
        if "contains" in filter_rule:
            contains_val = filter_rule["contains"]
            # This apply might be slow on Dask if not optimized.
            # It's important that the meta is correct.
            # The lambda checks for list type before attempting 'in'.
            # In Dask, if a column has mixed types (e.g. some rows are lists, some are not),
            # .apply is flexible but can be less performant than vectorized ops.
            # Assuming 'object' dtype for columns that might contain lists.
            condition = filtered_ddf[field_name].apply(
                lambda x: isinstance(x, list) and contains_val in x, 
                meta=(field_name, 'bool')
            )
            current_filter_conditions.append(condition)
            print(f"    Condition: {field_name} contains '{contains_val}' (for list elements)")

        # Regex Contains filter (for string columns)
        if "regex_contains" in filter_rule:
            regex_val = filter_rule["regex_contains"]
            # .astype(str) is important for columns that might have mixed types or NaNs
            # na=False means NaNs will not match the regex (evaluates to False)
            condition = filtered_ddf[field_name].astype(str).str.contains(regex_val, na=False, regex=True)
            current_filter_conditions.append(condition)
            print(f"    Condition: {field_name} matches regex '{regex_val}'")
        
        if not current_filter_conditions:
            print(f"  Warning: No valid filter operations found for rule on field '{field_name}'. Skipping this filter rule.")
            continue

        # Combine all conditions for the current filter rule using logical AND
        combined_rule_condition = current_filter_conditions[0]
        for cond in current_filter_conditions[1:]:
            combined_rule_condition = combined_rule_condition & cond
            
        filtered_ddf = filtered_ddf[combined_rule_condition]
        
        # Persist intermediate results if many filters or large data
        if (i + 1) % 5 == 0 and len(filters) > 5 : # Persist every 5 filters for longer filter chains
             filtered_ddf = filtered_ddf.persist()
             print(f"  Persisted intermediate DataFrame after filter {i+1}.")
        
        # Optionally, print count after each filter (can be expensive)
        # current_count_after_filter = len(filtered_ddf.compute()) # compute() is expensive
        # print(f"  Number of records after this filter: {current_count_after_filter} (Note: This count is computed)")


    # Compute final count
    final_count = len(filtered_ddf) # This will trigger computation for all filters
    print(f"\nTotal records after applying all filters: {final_count}")
    
    # 5. Save Filtered Manifest
    print(f"Saving filtered manifest to: {args.output_manifest}")
    final_dataset = DocumentDataset(df=filtered_ddf)
    try:
        final_dataset.to_json(args.output_manifest)
        print(f"Filtered manifest saved successfully to: {args.output_manifest}")
    except Exception as e:
        print(f"Error saving filtered manifest using DocumentDataset.to_json: {e}")
        # Fallback
        print("Attempting fallback save with Dask's to_json directly.")
        try:
            output_path_for_dask = args.output_manifest
            if filtered_ddf.npartitions > 1:
                parts_dir = args.output_manifest + "_parts_fallback"
                os.makedirs(parts_dir, exist_ok=True)
                output_path_for_dask = os.path.join(parts_dir, "part-*.jsonl")
            filtered_ddf.to_json(output_path_for_dask, orient="records", lines=True, compute=True) # Ensure compute=True
            print(f"Fallback Dask save successful to: {output_path_for_dask}")
        except Exception e2:
            print(f"Fallback Dask save also failed: {e2}")

    print(f"--- Filter by Property Script End. Initial: {initial_count}, Final: {final_count} records. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filters a video manifest based on specified property criteria.")
    parser.add_argument("--input_manifest", required=True, help="Path to an enriched JSONL manifest.")
    parser.add_argument("--output_manifest", required=True, help="Path to save the filtered JSONL manifest.")
    parser.add_argument("--filters_json", required=True, help="JSON string or path to a JSON file defining filters.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of Dask workers (default: None, Dask decides).")
    
    args = parser.parse_args()
    main(args)
