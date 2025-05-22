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
from collections import defaultdict # Import defaultdict

from das.plotting import set_plt_settings, PLOTS_DIR
# %%

# Define model configurations and parent directory for results
TARGET_MODEL_BASE_NAME = "pythia-410m"
TARGET_MODEL_CONFIG_KEY = "RevNetB8H64D1" # The specific model configuration string from filenames
SEEDS_TO_PLOT = list(range(1, 6)) + [None] # Seeds 1 to 5
RESULTS_PARENT_DIR = "../results/"
REVISION_TYPES = ["step0", "step143000"] # Explicit order for x-axis

# Helper function for the new plot type
def plot_grouped_bars_mean_std(
    data_sources: dict, # {'Metric Name': pd.DataFrame (index=revisions, columns=seeds)}
    revision_types: list, 
    revision_type_labels: list,
    title: str,
    bar_labels: list = None, # Optional specific labels for bar groups
):
    """
    Plots grouped bars for different metrics, showing mean across seeds with std dev error bars.
    Also plots individual data points (seeds) as small dots.
    """
    if not data_sources:
        print("No data sources provided to plot_grouped_bars_mean_std.")
        return

    if bar_labels is None:
        bar_labels = list(data_sources.keys())
    
    n_groups = len(revision_types)
    n_bars_per_group = len(bar_labels)
    
    means = {}
    std_devs = {}
    raw_values = {} # To store raw values for scatter plot

    for label in bar_labels:
        df = data_sources.get(label)
        if df is not None and not df.empty:
            df_reindexed = df.reindex(revision_types)
            means[label] = df_reindexed.mean(axis=1) 
            std_devs[label] = df_reindexed.std(axis=1)
            raw_values[label] = df_reindexed # Store the reindexed DataFrame
        else:
            means[label] = pd.Series([np.nan] * n_groups, index=revision_types)
            std_devs[label] = pd.Series([np.nan] * n_groups, index=revision_types)
            raw_values[label] = pd.DataFrame(index=revision_types, columns=df.columns if df is not None else [])


    fig, ax = plt.subplots(figsize=(12, 7)) # Slightly wider for better dot visibility
    
    index = np.arange(n_groups) 
    bar_width = 0.25 
    
    colors = ['red', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    dot_color = 'black' # Or a contrasting color like 'gray'
    dot_size = 30
    dot_alpha = 0.6

    for i, metric_label in enumerate(bar_labels):
        bar_positions = index + (i - (n_bars_per_group -1 ) / 2) * bar_width
        ax.bar(bar_positions, means[metric_label].fillna(0), bar_width, 
               yerr=std_devs[metric_label].fillna(0), label=metric_label, capsize=5, color=colors[i % len(colors)],
               alpha=0.7) # Make bars slightly transparent if dots are overlaid

        # Add individual data points (dots)
        metric_df = raw_values.get(metric_label)
        if metric_df is not None and not metric_df.empty:
            for revision_idx, revision_key in enumerate(revision_types):
                if revision_key in metric_df.index:
                    seed_values = metric_df.loc[revision_key].values
                    # X-coordinates for the dots: same as the bar for this metric and revision
                    x_coords_for_dots = np.full(len(seed_values), bar_positions[revision_idx])
                    ax.scatter(x_coords_for_dots, seed_values, 
                               color=dot_color, s=dot_size, zorder=3, alpha=dot_alpha,
                               edgecolor='white', linewidth=0.5) # Use same color as the corresponding metric bar

    ax.set_xlabel("DNN Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(index)
    ax.set_xticklabels(revision_type_labels)
    ax.legend()
    
    fig.tight_layout()
    return fig, ax

def load_and_process_results(target_model_name, model_config_key, seeds_to_load, parent_results_dir, revision_types_to_load):
    """
    Loads results for a specific Pythia model variant across multiple seeds and revisions.
    Organizes IIA data per layer and accuracy data overall.
    IIA DataFrames (per layer): Index="revision_type", Columns="Seed X"
    Accuracy DataFrame: Index="revision_type", Columns="Seed X"
    """
    # iia_data_temp: {layer_str: {revision_type: {seed_label: iia_value}}}
    iia_data_temp = defaultdict(lambda: defaultdict(dict))
    # accuracy_data_temp: {revision_type: {seed_label: acc_value}}
    accuracy_data_temp = defaultdict(dict)
    
    script_dir = os.path.dirname(__file__)

    for seed_num in seeds_to_load:
        if seed_num is None:
            model_folder_name = f"EleutherAI_{target_model_name}"
            seed_label = "Seed -1"
        else:
            model_folder_name = f"EleutherAI_{target_model_name}-seed{seed_num}"
            seed_label = f"Seed {seed_num}"
   
        for revision_type in revision_types_to_load:
            # Construct the base path for glob search carefully
            glob_search_base_path = os.path.join(script_dir, parent_results_dir, model_folder_name)
            search_pattern_glob = os.path.join(glob_search_base_path, 
                                               f"{revision_type}_m0_l*_{model_config_key}*_results.json")
            
            json_files_for_layers = glob.glob(search_pattern_glob)
            
            if not json_files_for_layers:
                print(f"Warning: No JSON files matching pattern '{search_pattern_glob}' found for {target_model_name}, seed {seed_num}, revision {revision_type}. Skipping this entry.")
                continue
            
            first_file_processed_for_acc_in_rev = False # Reset for each seed-revision

            for file_path in json_files_for_layers:
                filename = os.path.basename(file_path)
                core_name = filename[:-len("_results.json")]
                name_parts = core_name.split('_')

                if len(name_parts) < 4: 
                    print(f"Warning: Filename {filename} for {target_model_name} seed {seed_num} (rev: {revision_type}) does not have enough parts ({len(name_parts)}) to parse. Skipping.")
                    continue

                # Validate parts
                if name_parts[0] != revision_type:
                    print(f"Warning: File {filename} for {target_model_name} seed {seed_num} is not for '{revision_type}' revision (found '{name_parts[0]}'). Skipping.")
                    continue
                
                layer_spec_part = name_parts[2] # e.g., 'l3'
                model_key_original_from_filename = name_parts[3] # e.g., 'RevNetB8H64D1'

                if model_config_key not in model_key_original_from_filename:
                    print(f"Warning: File {filename} model key '{model_key_original_from_filename}' does not contain target '{model_config_key}'. This might indicate a glob pattern issue or incorrect file. Skipping.")
                    continue
                
                try:
                    with open(file_path, 'r') as f:
                        loaded_content = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {filename} for {target_model_name} seed {seed_num} (rev: {revision_type}). Skipping.")
                    continue
                except FileNotFoundError: 
                    print(f"Warning: File {file_path} for {target_model_name} seed {seed_num} (rev: {revision_type}) not found. Skipping.")
                    continue

                actual_data = loaded_content
                if isinstance(loaded_content, list):
                    if len(loaded_content) > 0 and isinstance(loaded_content[0], dict):
                        actual_data = loaded_content[0]
                    else:
                        print(f"Warning: File {filename} ({target_model_name} seed {seed_num}, rev: {revision_type}) is a list but not in expected format. Skipping.")
                        continue
                
                if not isinstance(actual_data, dict):
                    print(f"Warning: Parsed data from {filename} ({target_model_name} seed {seed_num}, rev: {revision_type}) is not a dictionary. Skipping.")
                    continue
                
                # Accuracy: Assumed to be the same for all layer files of the same (seed, revision)
                if not first_file_processed_for_acc_in_rev:
                    stage_accuracy_val = actual_data.get("accuracy")
                    if isinstance(stage_accuracy_val, (int, float)):
                        accuracy_data_temp[revision_type][seed_label] = stage_accuracy_val
                    else:
                        print(f"Warning: 'accuracy' for {seed_label} (rev: {revision_type}) in {filename} is not a number or not found: {stage_accuracy_val}. Using NaN.")
                        accuracy_data_temp[revision_type][seed_label] = np.nan
                    first_file_processed_for_acc_in_rev = True

                # IIA (per layer)
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
                                print(f"Warning: Single value in {target_layer_name_in_json} for {filename} ({seed_label}, rev: {revision_type}) is not a number: {potential_iia}. IIA set to NaN.")
                        elif extracted_iia is None and len(layer_content_from_json) > 1:
                             print(f"Warning: Ambiguous IIA in {target_layer_name_in_json} for {filename} ({seed_label}, rev: {revision_type}). Key '[[0], [1]]' not found and multiple entries exist. IIA set to NaN.")
                        
                        if isinstance(extracted_iia, (int, float)):
                            iia_val = extracted_iia
                        elif extracted_iia is not None: 
                            print(f"Warning: Extracted IIA from {target_layer_name_in_json} for {filename} ({seed_label}, rev: {revision_type}) is not a number: {extracted_iia}. Using NaN.")
                    else:
                        print(f"Warning: Content of {target_layer_name_in_json} in {filename} ({seed_label}, rev: {revision_type}) is not a dictionary or layer not found. Cannot extract IIA.")
                else:
                    print(f"Warning: Layer specification '{layer_spec_part}' in {filename} ({seed_label}, rev: {revision_type}) is not valid. Cannot determine layer for IIA.")

                if current_layer_num_str is None:
                    print(f"Error: Could not determine layer number for {target_model_name} seed {seed_num} (rev: {revision_type}) from {filename}. Skipping IIA data for this entry.")
                    continue
                
                layer_dict_key = f"L{current_layer_num_str}" # Key for our iia_data_temp dictionary
                iia_data_temp[layer_dict_key][revision_type][seed_label] = iia_val
    
    # Convert collected data to DataFrames
    dict_of_iia_dfs = {}
    for layer_str, data_for_layer in iia_data_temp.items():
        # data_for_layer is {revision_type: {seed_label: iia_value}}
        # orient='index' means revision_types become rows, seed_labels become columns
        df = pd.DataFrame.from_dict(data_for_layer, orient='index')
        dict_of_iia_dfs[layer_str] = df

    accuracy_df_overall = pd.DataFrame.from_dict(accuracy_data_temp, orient='index')
        
    return dict_of_iia_dfs, accuracy_df_overall

# %%
if __name__ == '__main__':
    print(f"Attempting to load data for {TARGET_MODEL_BASE_NAME} seeds from parent directory: {os.path.abspath(os.path.join(os.path.dirname(__file__), RESULTS_PARENT_DIR))}")
    # Load RevNet IIA and Accuracy data
    revnet_iia_dfs_by_layer, accuracy_results_df_overall = load_and_process_results(
        TARGET_MODEL_BASE_NAME,
        TARGET_MODEL_CONFIG_KEY, # This is for "RevNet"
        SEEDS_TO_PLOT,
        RESULTS_PARENT_DIR,
        REVISION_TYPES
    )
    
    if not revnet_iia_dfs_by_layer and accuracy_results_df_overall.empty:
        print("No RevNet IIA or Accuracy data loaded. Exiting.")
        sys.exit()

    ROTATION_TARGET_MODEL_CONFIG_KEY = "Rotation"
    rotation_iia_dfs_by_layer, rotation_accuracy_results_df_overall = load_and_process_results(
        TARGET_MODEL_BASE_NAME,
        ROTATION_TARGET_MODEL_CONFIG_KEY,
        SEEDS_TO_PLOT,
        RESULTS_PARENT_DIR,
        REVISION_TYPES
    )

    sorted_seed_column_labels =  sorted([f"Seed {s}" for s in SEEDS_TO_PLOT if s is not None], key=lambda x: int(re.search(r'\d+', x).group())) + ["Seed -1"]

    all_layer_keys = set(revnet_iia_dfs_by_layer.keys())
    if rotation_iia_dfs_by_layer: 
        all_layer_keys.update(rotation_iia_dfs_by_layer.keys())
    
    if not all_layer_keys:
        print("No layer data found to plot. Exiting.")
        sys.exit()
        
    sorted_layer_keys = sorted(list(all_layer_keys), key=lambda x: int(x[1:]))

    # --- Plotting Loop (per layer using the new bar plot function) ---
    for layer_key in sorted_layer_keys:
        print(f"\n--- Generating Bar Plot for Layer: {layer_key} ---")

        current_layer_data_sources = {}

        # 1. Accuracy Data
        if not accuracy_results_df_overall.empty:
            processed_accuracy_df = accuracy_results_df_overall.reindex(
                index=REVISION_TYPES, columns=sorted_seed_column_labels
            )
            current_layer_data_sources['DNN'] = processed_accuracy_df
        else:
            print(f"Warning: No accuracy data available for plot of layer {layer_key}.")
            current_layer_data_sources['DNN'] = pd.DataFrame(index=REVISION_TYPES, columns=sorted_seed_column_labels)

        # 2. RevNet IIA Data
        revnet_df_layer = revnet_iia_dfs_by_layer.get(layer_key)
        if revnet_df_layer is not None and not revnet_df_layer.empty:
            processed_revnet_iia_df = revnet_df_layer.reindex(
                index=REVISION_TYPES, columns=sorted_seed_column_labels
            )
            current_layer_data_sources['non-linear'] = processed_revnet_iia_df
        else:
            print(f"Warning: No RevNet IIA data for layer {layer_key}.")
            current_layer_data_sources['non-linear'] = pd.DataFrame(index=REVISION_TYPES, columns=sorted_seed_column_labels)

        # 3. Rotation IIA Data
        rotation_df_layer = rotation_iia_dfs_by_layer.get(layer_key) 
        if rotation_df_layer is not None and not rotation_df_layer.empty:
            processed_rotation_iia_df = rotation_df_layer.reindex(
                index=REVISION_TYPES, columns=sorted_seed_column_labels
            )
            current_layer_data_sources['linear'] = processed_rotation_iia_df
        else:
            print(f"Warning: No Rotation IIA data for layer {layer_key} (using placeholder or actual).")
            current_layer_data_sources['linear'] = pd.DataFrame(index=REVISION_TYPES, columns=sorted_seed_column_labels)
        
        bar_group_labels = ['DNN', 'non-linear', 'linear']

        # Ensure the new plotting function is called here
        fig, ax = plot_grouped_bars_mean_std(
            current_layer_data_sources,
            REVISION_TYPES,
            ["Init.", "Full"],
            "",
            bar_labels=bar_group_labels,
        )
        set_plt_settings(font_size=30, legend_font_size=25)

        # Add horizontal gridlines for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Set figure size for better proportions and readability
        fig.set_size_inches(8, 6)
        # Adjust layout to ensure all elements fit properly
        fig.tight_layout(pad=0.5)
        # Set y-axis limit to 1.0 for consistent scale across plots
        ax.set_ylim(0, 1.0)
        # Set legend position to lower left
        ax.legend(loc='lower right')
        
        save_dir = PLOTS_DIR / f"{TARGET_MODEL_BASE_NAME}_layer_{layer_key}"
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"mean_scores_by_seed.pdf", dpi=300)
    

# %%
