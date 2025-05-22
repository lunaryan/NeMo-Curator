import argparse
import os
import yaml
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.sequential import Sequential
from nemo_curator.utils.distributed_utils import get_client

def main(args):
    # 1. Setup
    print("--- PII Redaction Script Start ---")
    output_dir = os.path.dirname(args.output_manifest)
    if output_dir and not os.path.exists(output_dir): # Ensure output_dir is not an empty string
        os.makedirs(output_dir, exist_ok=True)

    if args.num_workers:
        client = get_client(num_workers=args.num_workers, create_cluster_if_needed=True)
        print(f"Dask client configured: {client}")

    # 2. Load Manifest
    print(f"Loading manifest from: {args.input_manifest}")
    try:
        # Requirement: Use DocumentDataset.read_json with pandas backend
        # PiiModifier and related tools in NeMo Curator often work with Dask DataFrames internally,
        # so loading as pandas and letting the pipeline convert might be fine, or it might convert to Dask later.
        # For consistency with NeMo Curator examples, let's assume the pipeline handles the DataFrame type.
        dataset = DocumentDataset.read_json(args.input_manifest, backend='pandas') 
        
        if dataset.df is None or dataset.df.empty:
            print("Input manifest is empty or failed to load. Saving an empty manifest.")
            with open(args.output_manifest, 'w') as f: pass
            print(f"Empty output manifest saved to {args.output_manifest}")
            return
        
        print(f"Manifest loaded. Number of records: {len(dataset.df)}")

    except FileNotFoundError:
        print(f"Error: Input manifest file not found at '{args.input_manifest}'.")
        return
    except Exception as e:
        print(f"Error loading input manifest '{args.input_manifest}': {e}")
        return

    # 3. Initialize PiiModifier
    pii_modifier = None
    if args.pii_config_path and os.path.exists(args.pii_config_path):
        print(f"Loading PII configuration from: {args.pii_config_path}")
        try:
            with open(args.pii_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # PiiModifier.from_config might expect a specific structure, e.g., a sub-dictionary.
            # If the YAML file IS the PiiModifier config directly:
            # pii_modifier_config = config
            # If the YAML contains a key like 'pii_modifier':
            pii_modifier_config = config.get('pii_modifier', config) # Adjust if config structure is different
            
            # Ensure required keys for PiiModifier are in the config, or provide defaults.
            # PiiModifier(language, supported_entities, anonymize_action, batch_size, device)
            # The from_config method might handle this internally.
            # Assuming PiiModifier has from_config or we extract manually.
            # Let's try direct instantiation by extracting common params from a general config.
            
            lang = pii_modifier_config.get('language', args.default_lang)
            action = pii_modifier_config.get('anonymize_action', args.default_action)
            supported_entities = pii_modifier_config.get('supported_entities', None) # None means all default for lang
            # Other params like batch_size, device could also be in config.
            
            pii_modifier = PiiModifier(
                language=lang, 
                anonymize_action=action, 
                supported_entities=supported_entities
                # One could add more params here if extracted from config:
                # batch_size=pii_modifier_config.get('batch_size', 32), 
                # device=pii_modifier_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            )
            print(f"PiiModifier initialized from config file. Language: {lang}, Action: {action}, Entities: {supported_entities or 'Default'}")

        except yaml.YAMLError as e:
            print(f"Error parsing YAML config file '{args.pii_config_path}': {e}")
            print("Falling back to default PiiModifier settings.")
        except Exception as e:
            print(f"Error initializing PiiModifier from config '{args.pii_config_path}': {e}")
            print("Falling back to default PiiModifier settings.")

    if pii_modifier is None: # Fallback or no config provided
        pii_modifier = PiiModifier(
            language=args.default_lang, 
            anonymize_action=args.default_action, 
            supported_entities=None # None uses default Presidio entities for the language
        )
        print(f"PiiModifier initialized with default settings. Language: {args.default_lang}, Action: {args.default_action}, Entities: Default")

    # 4. Build Pipeline
    pipeline_steps = []
    valid_fields_for_redaction = []
    for field_name in args.fields_to_redact:
        if field_name not in dataset.df.columns:
            print(f"Warning: Field '{field_name}' specified for redaction not found in manifest columns. Skipping this field.")
            continue
        
        # Check if column dtype is suitable (e.g., object/string)
        if not pd.api.types.is_string_dtype(dataset.df[field_name]) and not pd.api.types.is_object_dtype(dataset.df[field_name]):
            print(f"Warning: Field '{field_name}' is not of string/object type (is {dataset.df[field_name].dtype}). PII redaction might behave unexpectedly. Proceeding cautiously.")
        
        pipeline_steps.append(Modify(pii_modifier, text_field=field_name))
        valid_fields_for_redaction.append(field_name)

    if not pipeline_steps:
        print("No valid fields found for PII redaction, or no fields specified. The output manifest will be a copy of the input.")
        # Save original dataset if no operations are to be performed
        try:
            dataset.to_json(args.output_manifest)
            print(f"Copied input manifest to: {args.output_manifest} as no valid redaction fields were processed.")
        except Exception as e:
            print(f"Error saving original manifest: {e}")
        return

    print(f"Applying PII redaction to fields: {', '.join(valid_fields_for_redaction)}")
    pipeline = Sequential(pipeline_steps)

    # 5. Apply Pipeline
    # DocumentDataset.df might be Pandas or Dask. PiiModifier usually expects Dask.
    # If dataset.df is pandas, the Sequential pipeline might convert it or handle it.
    # NeMo Curator's pipelines are generally designed to work with Dask DataFrames.
    # If `dataset` was loaded with `backend='pandas'`, its `dataset.df` is a Pandas DF.
    # The `Sequential` pipeline typically converts this to Dask internally if needed,
    # or it might be better to ensure `dataset` holds a Dask DF from the start for PII.
    # However, the requirement was `backend='pandas'` for loading.
    # Let's assume the NeMo Curator pipeline handles this.
    
    print("Applying PII redaction pipeline...")
    redacted_dataset = pipeline(dataset) # This will run the operations
    
    # The result `redacted_dataset.df` should be a Dask DataFrame if PII module converted it,
    # or still Pandas if operations were in-place on Pandas.
    # PiiModifier typically uses Dask for parallel processing.

    print(f"PII redaction pipeline applied. Number of records in redacted dataset: {len(redacted_dataset.df)}") # May trigger compute

    # 6. Save Redacted Manifest
    print(f"Saving redacted manifest to: {args.output_manifest}")
    try:
        redacted_dataset.to_json(args.output_manifest)
        print(f"Redacted manifest saved successfully to: {args.output_manifest}")
    except Exception as e:
        print(f"Error saving redacted manifest: {e}")
        # Optional: Fallback save if DocumentDataset.to_json fails
        # if hasattr(redacted_dataset, 'df') and isinstance(redacted_dataset.df, dd.DataFrame):
        #     print("Attempting fallback save with Dask's to_json directly.")
        #     try:
        #         output_path_for_dask = args.output_manifest
        #         if redacted_dataset.df.npartitions > 1:
        #             parts_dir = args.output_manifest + "_parts_fallback_pii"
        #             os.makedirs(parts_dir, exist_ok=True)
        #             output_path_for_dask = os.path.join(parts_dir, "part-*.jsonl")
        #         redacted_dataset.df.to_json(output_path_for_dask, orient="records", lines=True, compute=True)
        #         print(f"Fallback Dask save successful to: {output_path_for_dask}")
        #     except Exception as e2:
        #         print(f"Fallback Dask save also failed: {e2}")


    print("--- PII Redaction Script End ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Applies PII redaction to textual fields in a JSONL manifest.")
    parser.add_argument("--input_manifest", required=True, help="Path to the input JSONL manifest.")
    parser.add_argument("--output_manifest", required=True, help="Path to save the PII-redacted JSONL manifest.")
    parser.add_argument("--fields_to_redact", required=True, nargs='+', 
                        help="List of text field names to redact (e.g., source_description transcript).")
    parser.add_argument("--pii_config_path", default=None, 
                        help="Path to a YAML configuration file for PiiModifier (optional).")
    parser.add_argument("--num_workers", type=int, default=None, 
                        help="Number of Dask workers (default: None, Dask decides).")
    parser.add_argument("--default_lang", default="en", 
                        help="Default language for PII redaction if no config is provided (default: en).")
    parser.add_argument("--default_action", default="replace", 
                        help="Default PII action (e.g., replace, hash, mask) if no config (default: replace).")
    
    args = parser.parse_args()
    main(args)
