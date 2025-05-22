# %% 
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

from das.plotting import plot_results, set_plt_settings, PLOTS_DIR
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

def get_base_display_name_from_config_key(config_key_part):
    """Derives a base display name from a configuration key string."""
    match = re.match(r"RevNetB(\d+)H\d+D\d+", config_key_part)
    if match:
        return f"$K={match.group(1)}$"
    return config_key_part # Fallback

def load_and_process_results(model_configs, parent_results_dir, revision_types_to_load):
    """
    Loads results for specified Pythia models, targeting specified revisions and a specific model configuration.
    The layer is detected from the filename.
    Index of DataFrames: "model_short_name (LX)" e.g. "pythia-70m (L3)"
    Columns of DataFrames: "Base Config Name (revision_type)" e.g., "$K=8$(main)"
    """
    iia_data_dict = {} 
    accuracy_data_dict = {} 
    
    for model_short_name, config_details in model_configs.items():
        model_folder_name = config_details["folder_suffix"]
        target_model_config_key_part = config_details["target_config_str"] # e.g., "RevNetB8H64D1"

        # Determine the base display name for DataFrame columns ONCE per target_model_config_key_part
        # Since target_config_str is the same for all models here, this will be consistent.
        base_display_name_for_df_cols = get_base_display_name_from_config_key(target_model_config_key_part)

        for revision_type in revision_types_to_load:
            current_model_base_dir = os.path.join(parent_results_dir, model_folder_name)
            # Construct search pattern: {revision_type}_m0_l*_RevNetB8H64D1*_results.json
            search_pattern_glob = os.path.join(os.path.dirname(__file__), current_model_base_dir, 
                                               f"{revision_type}_m0_l*_{target_model_config_key_part}*_results.json")
            
            json_files = glob.glob(search_pattern_glob)
            
            if not json_files:
                print(f"Warning: No JSON files matching pattern '{search_pattern_glob}' found for {model_short_name} (revision: {revision_type}). Skipping this entry.")
                continue
            
            if len(json_files) > 1:
                print(f"Warning: Multiple JSON files found for {model_short_name}, config {target_model_config_key_part}, revision {revision_type}. Using first one: {json_files[0]}")
            
            file_path = json_files[0]
            filename = os.path.basename(file_path)
            core_name = filename[:-len("_results.json")]
            name_parts = core_name.split('_')

            if len(name_parts) < 4: 
                print(f"Warning: Filename {filename} for {model_short_name} (rev: {revision_type}) does not have enough parts ({len(name_parts)}) to parse. Skipping.")
                continue

            # Validate parts: revision type must match, layer spec 'lX', model key
            if name_parts[0] != revision_type:
                print(f"Warning: File {filename} for {model_short_name} is not for '{revision_type}' revision (found '{name_parts[0]}'). Skipping.")
                continue
            
            layer_spec_part = name_parts[2] # e.g., 'l3'
            model_key_original_from_filename = name_parts[3] # e.g., 'RevNetB8H64D1'

            if target_model_config_key_part not in model_key_original_from_filename:
                print(f"Warning: File {filename} model key '{model_key_original_from_filename}' does not contain target '{target_model_config_key_part}'. This might indicate a glob pattern issue or incorrect file. Skipping.")
                continue
            
            # Construct the final DataFrame column name with revision
            df_column_label_with_revision = f"{revision_type.replace('step', 'Step ').replace('main', 'Full')} – {base_display_name_for_df_cols}"

            try:
                with open(file_path, 'r') as f:
                    loaded_content = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename} for {model_short_name} (rev: {revision_type}). Skipping.")
                continue
            except FileNotFoundError: 
                print(f"Warning: File {filename} for {model_short_name} (rev: {revision_type}) not found. Skipping.")
                continue

            actual_data = loaded_content
            if isinstance(loaded_content, list):
                if len(loaded_content) > 0 and isinstance(loaded_content[0], dict):
                    actual_data = loaded_content[0]
                else:
                    print(f"Warning: File {filename} ({model_short_name}, rev: {revision_type}) is a list but not in expected format. Skipping.")
                    continue
            
            if not isinstance(actual_data, dict):
                print(f"Warning: Parsed data from {filename} ({model_short_name}, rev: {revision_type}) is not a dictionary. Skipping.")
                continue
            
            stage_accuracy_val = actual_data.get("accuracy")
            if not isinstance(stage_accuracy_val, (int, float)):
                print(f"Warning: 'accuracy' for {model_short_name} ({df_column_label_with_revision}) in {filename} is not a number or not found: {stage_accuracy_val}. Using NaN.")
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
                            print(f"Warning: Single value in {target_layer_name_in_json} for {filename} ({df_column_label_with_revision}) is not a number: {potential_iia}. IIA set to NaN.")
                    elif extracted_iia is None and len(layer_content_from_json) > 1:
                         print(f"Warning: Ambiguous IIA in {target_layer_name_in_json} for {filename} ({df_column_label_with_revision}). Key '[[0], [1]]' not found and multiple entries exist. IIA set to NaN.")
                    
                    if isinstance(extracted_iia, (int, float)):
                        iia_val = extracted_iia
                    elif extracted_iia is not None: # Was found but not a number
                        print(f"Warning: Extracted IIA from {target_layer_name_in_json} for {filename} ({df_column_label_with_revision}) is not a number: {extracted_iia}. Using NaN.")
                else:
                    print(f"Warning: Content of {target_layer_name_in_json} in {filename} ({df_column_label_with_revision}) is not a dictionary or layer not found. Cannot extract IIA.")
            else:
                print(f"Warning: Layer specification '{layer_spec_part}' in {filename} ({df_column_label_with_revision}) is not valid. Cannot determine layer for IIA.")

            if current_layer_num_str is None:
                print(f"Error: Could not determine layer number for {model_short_name} (rev: {revision_type}) from {filename}. Skipping data for this entry.")
                continue
            
            data_label_for_index = f"$\\text{{{model_short_name}}} \\text{{\\large (L{current_layer_num_str})}}$" # Row index for DF
            
            if data_label_for_index not in iia_data_dict:
                iia_data_dict[data_label_for_index] = {}
            iia_data_dict[data_label_for_index][df_column_label_with_revision] = iia_val

            if data_label_for_index not in accuracy_data_dict:
                accuracy_data_dict[data_label_for_index] = {}
            accuracy_data_dict[data_label_for_index][df_column_label_with_revision] = stage_accuracy_val
    
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

    # --- Pre-processing before plotting ---
    # No model column filtering needed as we load specific data.

    # 1. Custom sort for x-axis (index) for both DataFrames
    # Index will be like "pythia-31m (L2)", "pythia-70m (L3)"
    # We want to sort them by model size primarily.
    
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

    def sort_columns(df): # This function is from the original script, can be kept
        if df.empty:
            return df
        
        # Custom sorting function for columns
        def custom_sort_key(col_name):
            if col_name == "Rotation": # Unlikely to be present now
                return (0, 0) 
            elif col_name and col_name[0].isdigit(): # Handles "$K=8$"
                try:
                    # Extracts numeric part for sorting, e.g., "8" from "$K=8$"
                    step_num = int(col_name.split(" ")[0]) 
                    return (1, step_num)  
                except ValueError: # If parsing fails (e.g. not "X YYY ZZZ" format)
                    return (2, col_name)  # Fallback sort alphabetically after numeric ones
            else: # Other non-numeric columns (if any) come last
                return (3, col_name) 
        
        # Sort columns using the custom sorting function
        # If only one column (e.g. "$K=8$"), this will just return it.
        sorted_cols = sorted(df.columns, key=custom_sort_key)
        df = df[sorted_cols]
        return df

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
    
    fig, ax = plot_results(
        processed_iia_df, 
        processed_accuracy_df, 
        x_coordinates_map,
        revision_aware=True, # Enable new revision handling
        acc_legend_marker_color_override="grey", # Make legend diamond neutral color
        x_label="Model Size",
        legend_bbox_final=(1.0, 0.0),
        legend_font_size=18, 
        acc_marker_label="Model Acc.",
        x_axis_label_pad=10,
        xlabel_fontsize=24,
        ylabel_fontsize=24,
        xtick_label_fontsize=24,
        ytick_label_fontsize=24,
        xtick_label_rotation=0,
        figsize=(8, 5),
        legend_background_alpha=0.9,
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
