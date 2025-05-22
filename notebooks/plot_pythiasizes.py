# %% 
try:
    # Only attempt to use autoreload in an interactive environment
    import IPython
    if IPython.get_ipython() is not None:
        IPython.get_ipython().run_line_magic('load_ext', 'autoreload')
        IPython.get_ipython().run_line_magic('autoreload', '2')
except (ImportError, AttributeError):
    # Not running in IPython/Jupyter, continue without autoreload
    pass
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import glob
from matplotlib.lines import Line2D # For custom legend entry
import re # Import the regular expression module

from das.plotting import plot_results, set_plt_settings, PLOTS_DIR, get_color_palette
# %%

# Define model configurations and parent directory for results
MODEL_CONFIGURATIONS_FOR_PLOT = {
    "31m": {"folder_suffix": "EleutherAI_pythia-31m", "target_config_str": "RevNetB8H64D1"},
    "70m": {"folder_suffix": "EleutherAI_pythia-70m", "target_config_str": "RevNetB8H64D1"},
    "160m": {"folder_suffix": "EleutherAI_pythia-160m", "target_config_str": "RevNetB8H64D1"},
    "410m": {"folder_suffix": "EleutherAI_pythia-410m", "target_config_str": "RevNetB8H64D1"},
}
RESULTS_PARENT_DIR = "../results/"
REVISION_TYPES = ["main", "step0"] # Revisions to load
MODEL_TYPES_TO_SEARCH = ["RevNet", "Rotation"] # For searching different model architectures

def get_base_display_name_from_config_key(config_key_part):
    """Derives a base display name from a configuration key string."""
    match = re.match(r"RevNetB(\d+)H\d+D\d+", config_key_part)
    if match:
        return f"non-lin."
    elif "Rotation" in config_key_part.lower():
        return f"lin."
    else:
        return config_key_part # Fallback

def load_and_process_results(model_configs, parent_results_dir, revision_types_to_load):
    """
    Loads results for specified Pythia models, targeting specified revisions and model configurations (RevNet, Rotation).
    The layer is detected from the filename.
    Index of DataFrames: "model_short_name (LX)" e.g. "pythia-70m (L3)"
    Columns of DataFrames: "Base Config Name (revision_type)" e.g., "$K=8$ (Full)", "Rotation (Init.)"
    """
    iia_data_dict = {}
    accuracy_data_dict = {}

    for model_short_name, config_details in model_configs.items():
        model_folder_name = config_details["folder_suffix"]
        # target_model_config_key_part is specific to RevNet, used for filtering RevNet files.
        revnet_target_config_key_part = config_details.get("target_config_str") # Expect this to be like "RevNetB8H64D1"

        for revision_type in revision_types_to_load:
            current_model_base_dir = os.path.join(parent_results_dir, model_folder_name)

            for model_search_key in MODEL_TYPES_TO_SEARCH: # Iterate RevNet, Rotation
                base_model_display_name_for_df_col = ""
                search_pattern_glob = ""

                if model_search_key == "RevNet":
                    if not revnet_target_config_key_part:
                        print(f"Warning: 'target_config_str' not defined for {model_short_name} when searching for RevNet. Skipping RevNet for this model configuration.")
                        continue
                    search_pattern_glob = os.path.join(os.path.dirname(__file__), current_model_base_dir,
                                                       f"{revision_type}_m0_l*_{revnet_target_config_key_part}*_results.json")
                    base_model_display_name_for_df_col = get_base_display_name_from_config_key(revnet_target_config_key_part)
                elif model_search_key == "Rotation":
                    search_pattern_glob = os.path.join(os.path.dirname(__file__), current_model_base_dir,
                                                       f"{revision_type}_m0_l*_Rotation*_results.json")
                    base_model_display_name_for_df_col = "lin." # Model type is 'Rotation'
                else:
                    # This case should not be reached with the current MODEL_TYPES_TO_SEARCH
                    print(f"Warning: Unknown model_search_key '{model_search_key}'. Skipping.")
                    continue

                json_files = glob.glob(search_pattern_glob)

                if not json_files:
                    # This is an informational message, as not all combinations might exist.
                    print(f"Info: No JSON files matching pattern '{search_pattern_glob}' found for {model_short_name}, model type '{model_search_key}', revision '{revision_type}'. This might be expected.")
                    continue

                # Process ALL files found for this model_search_key and revision_type.
                # Each file typically corresponds to a different layer 'l*'.
                for file_path in json_files:
                    filename = os.path.basename(file_path)
                    core_name = filename[:-len("_results.json")]
                    name_parts = core_name.split('_')

                    if len(name_parts) < 4:
                        print(f"Warning: Filename {filename} for {model_short_name} (rev: {revision_type}, type: {model_search_key}) does not have enough parts ({len(name_parts)}) to parse. Skipping.")
                        continue

                    file_revision_part = name_parts[0]
                    if file_revision_part != revision_type:
                        print(f"Warning: File {filename} (revision part '{file_revision_part}') for {model_short_name} does not match expected revision '{revision_type}'. Skipping.")
                        continue

                    layer_spec_part = name_parts[2] # e.g., 'l3'
                    model_key_original_from_filename = name_parts[3] # e.g., 'RevNetB8H64D1' or 'Rotation'

                    # Additional check for model key consistency with the search type
                    if model_search_key == "RevNet":
                        if revnet_target_config_key_part not in model_key_original_from_filename:
                            print(f"Warning: File {filename} (searched as RevNet) model key '{model_key_original_from_filename}' does not contain target '{revnet_target_config_key_part}'. Skipping.")
                            continue
                    elif model_search_key == "Rotation":
                        if not model_key_original_from_filename.startswith("Rotation"):
                            print(f"Warning: File {filename} (searched as Rotation) model key '{model_key_original_from_filename}' does not appear to be a Rotation model. Skipping.")
                            continue
                    
                    revision_suffix_for_col = revision_type.replace('step', 'Step ').replace('Step 0', 'Init.').replace('main', 'Full')
                    df_column_label = f"{base_model_display_name_for_df_col} ({revision_suffix_for_col})"

                    try:
                        with open(file_path, 'r') as f:
                            loaded_content = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from {filename} for {model_short_name} ({df_column_label}). Skipping.")
                        continue
                    except FileNotFoundError:
                        print(f"Warning: File {filename} for {model_short_name} ({df_column_label}) not found. Skipping.")
                        continue

                    actual_data = loaded_content
                    if isinstance(loaded_content, list):
                        if len(loaded_content) > 0 and isinstance(loaded_content[0], dict):
                            actual_data = loaded_content[0]
                        else:
                            print(f"Warning: File {filename} ({df_column_label}) is a list but not in expected format. Skipping.")
                            continue
                    
                    if not isinstance(actual_data, dict):
                        print(f"Warning: Parsed data from {filename} ({df_column_label}) is not a dictionary. Skipping.")
                        continue
                    
                    stage_accuracy_val = actual_data.get("accuracy")
                    if not isinstance(stage_accuracy_val, (int, float)):
                        print(f"Warning: 'accuracy' for {model_short_name} ({df_column_label}) in {filename} is not a number or not found: {stage_accuracy_val}. Using NaN.")
                        stage_accuracy_val = np.nan

                    iia_val = np.nan
                    current_layer_num_str = None
                    if layer_spec_part.startswith('l') and layer_spec_part[1:].isdigit():
                        current_layer_num_str = layer_spec_part[1:]
                        target_layer_name_in_json = f"Layer{current_layer_num_str}"
                        layer_content_from_json = actual_data.get(target_layer_name_in_json)

                        if isinstance(layer_content_from_json, dict):
                            extracted_iia = layer_content_from_json.get("[[0], [1]]")
                            if extracted_iia is None and len(layer_content_from_json) == 1:
                                potential_iia = list(layer_content_from_json.values())[0]
                                if isinstance(potential_iia, (int, float)):
                                     extracted_iia = potential_iia
                                else:
                                    print(f"Warning: Single value in {target_layer_name_in_json} for {filename} ({df_column_label}) is not a number: {potential_iia}. IIA set to NaN.")
                            elif extracted_iia is None and len(layer_content_from_json) > 1:
                                 print(f"Warning: Ambiguous IIA in {target_layer_name_in_json} for {filename} ({df_column_label}). Key '[[0], [1]]' not found and multiple entries exist. IIA set to NaN.")
                            
                            if isinstance(extracted_iia, (int, float)):
                                iia_val = extracted_iia
                            elif extracted_iia is not None: # Was found but not a number
                                print(f"Warning: Extracted IIA from {target_layer_name_in_json} for {filename} ({df_column_label}) is not a number: {extracted_iia}. Using NaN.")
                        else:
                            print(f"Warning: Content of {target_layer_name_in_json} in {filename} ({df_column_label}) is not a dictionary or layer not found. Cannot extract IIA.")
                    else:
                        print(f"Warning: Layer specification '{layer_spec_part}' in {filename} ({df_column_label}) is not valid. Cannot determine layer for IIA.")

                    if current_layer_num_str is None:
                        # This message clarifies which model type/revision failed for layer determination
                        print(f"Error: Could not determine layer number for {model_short_name} (Type: {model_search_key}, Rev: {revision_type}) from {filename}. Skipping data for this entry.")
                        continue
                    
                    data_label_for_index = f"$\\text{{{model_short_name}}} \\text{{\\large (L{current_layer_num_str})}}$" # Row index for DF
                    
                    if data_label_for_index not in iia_data_dict:
                        iia_data_dict[data_label_for_index] = {}
                    iia_data_dict[data_label_for_index][df_column_label] = iia_val

                    if data_label_for_index not in accuracy_data_dict:
                        accuracy_data_dict[data_label_for_index] = {}
                    # Store accuracy with the same detailed column label, linking it to this specific file's model type and revision
                    accuracy_data_dict[data_label_for_index][df_column_label] = stage_accuracy_val
    
    iia_df = pd.DataFrame.from_dict(iia_data_dict, orient='index')
    accuracy_df = pd.DataFrame.from_dict(accuracy_data_dict, orient='index')
        
    return iia_df, accuracy_df

# %%
if __name__ == '__main__':
    print(f"Attempting to load data for multiple Pythia models from parent directory: {os.path.abspath(os.path.join(os.path.dirname(__file__), RESULTS_PARENT_DIR))}")

    iia_results_df, accuracy_results_df = load_and_process_results(
        MODEL_CONFIGURATIONS_FOR_PLOT,
        RESULTS_PARENT_DIR,
        REVISION_TYPES
    )
    
    processed_iia_df = iia_results_df.copy()
    processed_accuracy_df = accuracy_results_df.copy()
    


    model_size_sort_order = {
        "pythia-31m": 1,
        "pythia-70m": 2,
        "pythia-160m": 3,
        "pythia-410m": 4,
    }
    
    def get_sort_key_for_model_label(label):
        model_name_part = label.split(" ")[0] # Extracts "pythia-31m" from "pythia-31m (L2)"
        return model_size_sort_order.get(model_name_part, float('inf')) # Sort unknown/new models last

    final_ordered_index = []
    # Determine the order from available data (prefer IIA's index if available)
    if not processed_iia_df.empty:
        final_ordered_index = sorted(processed_iia_df.index.unique(), key=get_sort_key_for_model_label)
    elif not processed_accuracy_df.empty: # Fallback if IIA df is empty
        final_ordered_index = sorted(processed_accuracy_df.index.unique(), key=get_sort_key_for_model_label)
    
    if not final_ordered_index and (not processed_iia_df.empty or not processed_accuracy_df.empty):
        # This case means sorting failed to produce an order, but there's data.
        # Fallback to alphabetical sort of whatever index exists.
        print(f"Warning: Custom x-axis ordering based on model size resulted in an empty order, but data exists. Using alphabetical sort for x-axis.")
        combined_indices = processed_iia_df.index.union(processed_accuracy_df.index).unique()
        final_ordered_index = sorted(combined_indices.tolist())
    elif not final_ordered_index:
        print("No data loaded or unable to determine x-axis order. Plot might be empty.")

    # Create the x-coordinates map
    x_coordinates_map = {}
    # Define the extra gap you want before "Full" compared to standard step spacing - NOT USED IN THIS VERSION
    # gap_before_full = 1.0 

    if final_ordered_index: # Ensure final_ordered_index is not empty
        for i, label in enumerate(final_ordered_index):
            x_coordinates_map[label] = i # Simple linear spacing
        print(f"X-coordinates map for plotting: {x_coordinates_map}")

    def sort_columns(df):
        if df.empty:
            return df

        def custom_sort_key(col_name):
            # col_name examples: "$K=8$ (Init.)", "$K=8$ (Full)", "Rotation (Init.)", "Rotation (Full)"
            revision_prio = 2 # Default, for sorting stability if no suffix
            if "(Init.)" in col_name:
                revision_prio = 0 # Init comes before Full
            elif "(Full)" in col_name:
                revision_prio = 1

            # Check for $K=...$ type (RevNet)
            match_k_value = re.search(r"\\$K=(\\d+)\\$", col_name) # Regex for $K=...$
            if match_k_value:
                k_value = int(match_k_value.group(1))
                # Primary sort by Model Type (1 for RevNet), then K-value, then revision_prio
                return (1, k_value, revision_prio, col_name)

            # Check for Rotation type
            if "Rotation" in col_name:
                # Sort Rotation after RevNet types (2 for Rotation), then by revision_prio
                # Using 0 as a placeholder for k_value for Rotation as it's not applicable
                return (2, 0, revision_prio, col_name) 

            # Fallback for any other column types (e.g., if accuracy columns are named differently by mistake)
            # This helps sort them predictably, though ideally all columns fit the above patterns.
            print(f"Warning: Column '{col_name}' in sort_columns did not match expected RevNet or Rotation patterns.")
            return (3, 0, revision_prio, col_name)

        sorted_cols = sorted(df.columns, key=custom_sort_key)
        return df[sorted_cols]

    if not processed_iia_df.empty:
        # Reindex to ensure sorted order and consistent columns.
        # Columns will be like "$K=8$(main)", "$K=8$(step0)"
        processed_iia_df = processed_iia_df.reindex(index=final_ordered_index if final_ordered_index else processed_iia_df.index) 
        processed_iia_df = sort_columns(processed_iia_df) # sort_columns should handle the new names
        print(f"IIA DataFrame final X-axis: {processed_iia_df.index.tolist()}")
        print(f"IIA DataFrame final columns: {processed_iia_df.columns.tolist()}")


    if not processed_accuracy_df.empty:
        processed_accuracy_df = processed_accuracy_df.reindex(index=final_ordered_index if final_ordered_index else processed_accuracy_df.index)
        processed_accuracy_df = sort_columns(processed_accuracy_df)
        print(f"Accuracy DataFrame final X-axis: {processed_accuracy_df.index.tolist()}")
        print(f"Accuracy DataFrame final columns: {processed_accuracy_df.columns.tolist()}")
        
    # --- End of Pre-processing ---


    print("\nFinal Processed IIA DataFrame for Plotting:")
    print(processed_iia_df)
    print("\nFinal Processed Accuracy DataFrame for Plotting:")
    print(processed_accuracy_df)
    
    # Rename columns to only include the elements in brackets and remove duplicates
    def rename_and_deduplicate_columns(df):
        if df.empty:
            return df
        
        # Create a mapping of old column names to new column names (bracket content only)
        rename_map = {}
        for col in df.columns:
            # Extract content within brackets using regex
            match = re.search(r'\((.*?)\)', col)
            if match:
                # Use only the content within brackets as the new column name
                new_name = match.group(1)
                rename_map[col] = new_name
            else:
                # If no brackets found, keep the original name
                rename_map[col] = col
        
        # Rename columns
        df = df.rename(columns=rename_map)
        
        # Remove duplicate columns (keep the first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]

        # Inverse the naming
        inverse_rename_map = {v: k for k, v in rename_map.items()}
        df = df.rename(columns=inverse_rename_map)
        return df
    
    processed_accuracy_df = rename_and_deduplicate_columns(processed_accuracy_df)
    
    # Define custom color and marker maps based on user requirements
    iia_color_map_custom = {
        "non-lin.": get_color_palette(2)[-1],  # Or any other color you prefer for non-linear
        "lin.": (0, 204/255, 102/255) # Specific green for linear
    }
    iia_marker_map_custom = {
        "non-lin.": "o",  # Circle for non-linear IIA lines
        "lin.": "s"      # Square for linear IIA lines
    }
    acc_revision_marker_map_custom = {
        "Full": "D",    # Diamond for Full accuracy
        "Init.": "^"   # Triangle_up for Init. accuracy
    }


    # Sort columns for both DataFrames to ensure consistent ordering
    def sort_columns_by_model_type_and_revision(df):
        if df.empty:
            return df
            
        def get_sort_key(col_name):
            # Extract model type and revision information
            model_type = "unknown"
            revision = "unknown"
            
            # Check for model type patterns
            if "non-lin." in col_name:
                model_type = "non-lin."
                priority = 1
            elif "lin." in col_name:
                model_type = "lin."
                priority = 2
            else:
                priority = 3
                
            # Extract revision information
            if "(Full)" in col_name:
                revision = "Full"
                rev_priority = 1
            elif "(Init.)" in col_name:
                revision = "Init."
                rev_priority = 2
            else:
                rev_priority = 3
                
            # Return tuple for sorting: (model_type_priority, revision_priority, original_name)
            # This ensures stable sorting even with unexpected column names
            return (priority, rev_priority, col_name)
        

        # Sort columns based on the custom key function
        sorted_cols = sorted(df.columns, key=get_sort_key)
        return df[sorted_cols]

    processed_iia_df = sort_columns_by_model_type_and_revision(processed_iia_df)
    # Apply column sorting to both DataFrames
    if not processed_iia_df.empty:
        processed_iia_df = sort_columns_by_model_type_and_revision(processed_iia_df)
        print("\nSorted IIA DataFrame columns:")
        print(processed_iia_df.columns.tolist())
        
    if not processed_accuracy_df.empty:
        processed_accuracy_df = sort_columns_by_model_type_and_revision(processed_accuracy_df)
        print("\nSorted Accuracy DataFrame columns:")
        print(processed_accuracy_df.columns.tolist())
    fig, ax = plot_results(
        processed_iia_df, 
        processed_accuracy_df, 
        x_coordinates_map,
        revision_aware=True, 
        acc_marker_color="red",  # Ensure actual accuracy markers are red
        acc_legend_marker_color_override="red", # Ensure accuracy legend also uses red
        x_label="DNN Size",
        legend_bbox_final=(1.01, -0.01),
        legend_font_size=18, 
        acc_marker_label="DNN",
        x_axis_label_pad=10,
        xlabel_fontsize=24,
        ylabel_fontsize=24,
        acc_marker_size=13, # Global acc marker size, can be overridden if needed per revision by enhancing plotting.py more
        xtick_label_fontsize=24,
        ytick_label_fontsize=24,
        xtick_label_rotation=0,
        figsize=(8, 4.2),
        legend_background_alpha=0.9,
        legend_label_spacing=0.1,
        # Pass the custom maps
        revision_pattern_str=r"^(.*?) ?\(([^)]+)\)$", # Base name (revision_tag)
        iia_base_name_color_map=iia_color_map_custom,
        iia_base_name_marker_map=iia_marker_map_custom,
        acc_revision_marker_map=acc_revision_marker_map_custom,
        legend_ncol=3,
        legend_columnspacing=0.6,
        legend_handlelength=1.5,
        legend_handletextpad=0.3,
    )

    # Set LaTeX font for high-quality publication-ready plots
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    
    ax.set_xticklabels(ax.get_xticklabels(), ha='center')

    # Ensure all text elements use the LaTeX renderer
    for text in ax.texts:
        text.set_usetex(True)

    
    ax.set_ylim(0.21, 1.01)

    # Set fontsize for axis ticks
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.show()
    fig.savefig(PLOTS_DIR / f"pythia_sizes.pdf", dpi=300)

# %%
