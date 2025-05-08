# %% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import glob
from matplotlib.lines import Line2D # For custom legend entry
import re # Import the regular expression module

# %%
def load_and_process_results(base_dir):
    """
    Loads results from JSON files, processes them, and returns two DataFrames:
    one for IIA values and one for model accuracies at each stage.
    """
    iia_data_dict = {}
    accuracy_data_dict = {} # For stage-specific accuracies
    search_path = os.path.join(os.path.dirname(__file__), base_dir, "*_results.json")
    
    json_files = glob.glob(search_path)
    if not json_files:
        print(f"Warning: No JSON files found in {search_path}")
        return pd.DataFrame(), pd.DataFrame()

    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        if not filename.endswith("_results.json"):
            continue
        
        core_name = filename[:-len("_results.json")]
        name_parts = core_name.split('_')

        if len(name_parts) < 4:
            print(f"Warning: Filename {filename} does not have enough parts to parse. Skipping.")
            continue

        original_first_key = name_parts[0]
        current_first_key = "Full" if original_first_key == "main" else original_first_key
        
        layer_spec_part = name_parts[2] 
        model_key_original = name_parts[3]
        model_key_to_use = model_key_original

        match = re.match(r"RevNetB(\d+)H\d+D\d+", model_key_original)
        if match:
            number_of_blocks = match.group(1)
            model_key_to_use = f"{number_of_blocks} RevNet Blocks"
            if model_key_original != model_key_to_use:
                 print(f"Dynamically renamed model '{model_key_original}' to '{model_key_to_use}' from file {filename}")
       
        try:
            with open(file_path, 'r') as f:
                loaded_content = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}. Skipping.")
            continue
        except FileNotFoundError:
            print(f"Warning: File {filename} not found. Skipping.")
            continue

        actual_data = loaded_content
        if isinstance(loaded_content, list):
            if len(loaded_content) > 0 and isinstance(loaded_content[0], dict):
                actual_data = loaded_content[0]
            else:
                print(f"Warning: File {filename} is a list but not in expected format. Skipping.")
                continue
        
        if not isinstance(actual_data, dict):
            print(f"Warning: Parsed data from {filename} is not a dictionary. Skipping.")
            continue
        
        # Extract stage-specific accuracy from every file
        stage_accuracy = actual_data.get("accuracy")
        if isinstance(stage_accuracy, (int, float)):
            if current_first_key not in accuracy_data_dict:
                accuracy_data_dict[current_first_key] = {}
            accuracy_data_dict[current_first_key][model_key_to_use] = stage_accuracy
        else:
            print(f"Warning: 'accuracy' for {model_key_to_use} (original: {model_key_original}) in {filename} (stage: {current_first_key}) is not a number or not found: {stage_accuracy}.")

        iia_value = None
        target_layer_name = None
        if layer_spec_part.startswith('l') and layer_spec_part[1:].isdigit():
            layer_num = layer_spec_part[1:]
            target_layer_name = f"Layer{layer_num}"
            layer_content = actual_data.get(target_layer_name)

            if isinstance(layer_content, dict):
                iia_value = layer_content.get("[[0], [1]]")
                if iia_value is None and len(layer_content) == 1:
                    potential_iia = list(layer_content.values())[0]
                    if isinstance(potential_iia, (int, float)):
                         iia_value = potential_iia
                    else:
                        print(f"Warning: Single value in {target_layer_name} for {filename} is not a number: {potential_iia}. Skipping IIA.")
                elif iia_value is None and len(layer_content) > 1:
                     print(f"Warning: Ambiguous IIA in {target_layer_name} for {filename}. Key '[[0], [1]]' not found and multiple entries exist. Skipping IIA.")
                elif iia_value is None and len(layer_content) == 0:
                    print(f"Warning: No data in {target_layer_name} for {filename}. Skipping IIA.")
            else:
                print(f"Warning: Content of {target_layer_name} in {filename} is not a dictionary or layer not found. Skipping IIA.")
        else:
            print(f"Warning: Layer specification '{layer_spec_part}' in {filename} is not valid. Skipping IIA.")

        if iia_value is not None:
            if current_first_key not in iia_data_dict:
                iia_data_dict[current_first_key] = {}
            iia_data_dict[current_first_key][model_key_to_use] = iia_value
        elif target_layer_name :
             print(f"Could not extract IIA for {filename} (model: {model_key_to_use}) from {target_layer_name}")
    
    iia_df = pd.DataFrame.from_dict(iia_data_dict, orient='index')
    accuracy_df = pd.DataFrame.from_dict(accuracy_data_dict, orient='index')

    if not iia_df.empty:
        iia_df = iia_df[sorted(iia_df.columns)]
    if not accuracy_df.empty:
        accuracy_df = accuracy_df[sorted(accuracy_df.columns)]
        
    return iia_df, accuracy_df

def plot_results(iia_df, acc_df, x_coords_map):
    """
    Plots IIA and accuracies using a map for x-coordinate positions.
    Assumes DFs are filtered and sorted.
    """

    if iia_df.empty and acc_df.empty:
        print("Both IIA and Accuracy DataFrames are empty. No plot will be generated.")
        return
    if iia_df.empty:
        print("IIA DataFrame is empty. Plotting accuracies only if available.")
    if acc_df.empty:
        print("Accuracy DataFrame is empty. Plotting IIA only if available.")


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    plotted_any_diamonds = False 

    # Plot diamonds first (zorder=1 so they are behind lines)
    if not acc_df.empty:
        for model_name in acc_df.columns: 
            for x_label_str in acc_df.index:  
                if x_label_str in x_coords_map and pd.notna(acc_df.loc[x_label_str, model_name]):
                    numeric_x_coord = x_coords_map[x_label_str]
                    accuracy_val = acc_df.loc[x_label_str, model_name]
                    
                    # Condition to plot diamond: IIA data exists for this point OR iia_df is empty
                    plot_this_diamond = False
                    if iia_df.empty:
                        plot_this_diamond = True
                    elif model_name in iia_df.columns and x_label_str in iia_df.index and pd.notna(iia_df.loc[x_label_str, model_name]):
                        plot_this_diamond = True
                    
                    if plot_this_diamond:
                        ax.plot(numeric_x_coord, accuracy_val, 'D', color='red', markersize=10, label='_nolegend_', zorder=1)
                        plotted_any_diamonds = True

    # Plot IIA lines (zorder=2 so they are on top)
    if not iia_df.empty:
        num_lines = len(iia_df.columns)
        colormap = plt.cm.viridis 
        colors = [colormap(i) for i in np.linspace(0, 1, num_lines)] if num_lines > 0 else []

        for i, column_name in enumerate(iia_df.columns):
            # Prepare x and y data for plotting using the x_coords_map
            plot_x_values = []
            plot_y_values = []
            for x_label_str in iia_df.index: # iia_df.index is already sorted
                if x_label_str in x_coords_map: # Ensure label is in map
                    numeric_x = x_coords_map[x_label_str]
                    y_val = iia_df.loc[x_label_str, column_name]
                    # Only add points that have a valid y-value
                    if pd.notna(y_val):
                        plot_x_values.append(numeric_x)
                        plot_y_values.append(y_val)
            
            if plot_x_values: # Only plot if there's data for this line
                ax.plot(plot_x_values, plot_y_values, marker='o', linestyle='-', label=column_name, color=colors[i], linewidth=2, markersize=8, zorder=2)
        
    ax.set_xlabel("Model Training Step", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14) # More general Y-axis label
    ax.set_title("Pythia 410M: DAS for IOI", fontsize=16, fontweight='bold') # Updated title
    
    handles, labels = ax.get_legend_handles_labels()
    
    if plotted_any_diamonds:
        diamond_legend_entry = Line2D([0], [0], marker='D', color='red', label='Model Accuracy',
                                      linestyle='None', markersize=10)
        if "Model Accuracy" not in labels:
            handles.append(diamond_legend_entry)
            labels.append('Model Accuracy')

    if handles: # Only create legend if there's something to show
        legend = ax.legend(handles=handles, labels=labels, title="Intervention / Metric", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
        plt.setp(legend.get_title(),fontsize='12')

    # Set custom x-ticks based on the x_coords_map
    # Ensure the labels correspond to the sorted tick positions
    if x_coords_map:
        # Sort tick positions numerically to ensure correct order for labels
        sorted_tick_positions = sorted(x_coords_map.values())
        # Create a reverse map from position to label
        pos_to_label_map = {v: k for k, v in x_coords_map.items()}
        # Get labels in the order of sorted tick positions
        sorted_tick_labels = [pos_to_label_map[pos].replace("step", "Step ") for pos in sorted_tick_positions]
        
        ax.set_xticks(sorted_tick_positions)
        ax.set_xticklabels(sorted_tick_labels, rotation=45, ha="right", fontsize=12)
    else: # Fallback if x_coords_map is empty
        plt.xticks(rotation=45, ha="right", fontsize=12)

    plt.yticks(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Original tight_layout call, happens before final legend placement
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    # Move and restyle the legend to the lower right position
    if handles:  # Only adjust legend if it exists
        # Check if the first legend object (legend_upper_left) exists and is part of an axes
        if 'legend' in locals() and legend is not None and legend.axes is not None:
            legend.remove()  # Remove the previous legend (the one at upper left)

        # Create the new legend with desired properties directly
        final_legend = ax.legend(handles=handles, labels=labels, title="Intervention Model", 
                                bbox_to_anchor=(0.98, 0.02), # x, y position in axes coords
                                loc='lower right', 
                                borderaxespad=0.5, # Padding inside the legend box
                                fontsize=10,
                                facecolor='white',  # Set facecolor directly
                                edgecolor='black',  # Set edgecolor directly
                                frameon=True        # Ensure frame is drawn
                               )
        
        # Further customize the frame for opacity and linewidth
        if final_legend:
            frame = final_legend.get_frame()
            frame.set_linewidth(1.0)  # Set border thickness
            frame.set_alpha(1.0)      # Ensure the frame (and its facecolor) is fully opaque
            plt.setp(final_legend.get_title(), fontsize='12')
    
    # Increase the space between the x-axis label and the x-axis for better readability.
    ax.xaxis.labelpad = 25  # Increase labelpad (default is usually 4)
    plt.show()

# %%
if __name__ == '__main__':
    data_folder = "../results/EleutherAI_pythia-410m/" 
    print(f"Attempting to load data from: {os.path.abspath(os.path.join(os.path.dirname(__file__), data_folder))}")

    iia_results_df, accuracy_results_df = load_and_process_results(data_folder)
    
    processed_iia_df = iia_results_df.copy()
    processed_accuracy_df = accuracy_results_df.copy()

    # # --- Pre-processing before plotting ---
    # # Determine the actual column name to filter after potential renaming
    original_model_to_remove = "RevNetB16H64D1"
    filter_target_name = original_model_to_remove
    match_filter = re.match(r"RevNetB(\d+)H\d+D\d+", original_model_to_remove)
    if match_filter:
        filter_target_name = f"{match_filter.group(1)} RevNet Blocks"
        print(f"Filter target '{original_model_to_remove}' maps to potential renamed column '{filter_target_name}'.")
    else:
        print(f"Filter target '{original_model_to_remove}' will be used as is (does not match RevNet renaming pattern).")


    # 1. Filter out the specified model
    if not processed_iia_df.empty and filter_target_name in processed_iia_df.columns:
        processed_iia_df = processed_iia_df.drop(columns=[filter_target_name])
        print(f"Removed model column: {filter_target_name} from IIA DataFrame.")
    elif not processed_iia_df.empty:
        print(f"Model column: {filter_target_name} (for removal) not found in IIA DataFrame columns: {processed_iia_df.columns.tolist()}")


    if not processed_accuracy_df.empty and filter_target_name in processed_accuracy_df.columns:
        processed_accuracy_df = processed_accuracy_df.drop(columns=[filter_target_name])
        print(f"Removed model column: {filter_target_name} from Accuracy DataFrame.")
    elif not processed_accuracy_df.empty :
        print(f"Model column: {filter_target_name} (for removal) not found in Accuracy DataFrame columns: {processed_accuracy_df.columns.tolist()}")

    # 2. Custom sort for x-axis (index) for both DataFrames
    # Determine combined unique sorted index from both dataframes
    combined_index = pd.Index([])
    if not processed_iia_df.empty:
        combined_index = combined_index.union(processed_iia_df.index)
    if not processed_accuracy_df.empty:
        combined_index = combined_index.union(processed_accuracy_df.index)
    
    step_keys = sorted(
        [idx for idx in combined_index if idx.startswith('step') and idx[4:].isdigit()],
        key=lambda x: int(x[4:])
    )
    ordered_index_list = step_keys
    if 'Full' in combined_index:
        ordered_index_list.append('Full')
    
    # Keep only existing keys in the desired order that are present in the combined set
    final_ordered_index = [idx for idx in ordered_index_list if idx in combined_index]
    
    if not final_ordered_index and combined_index.tolist():
        print(f"Warning: Custom x-axis ordering resulted in an empty order. Using original unique sorted index for x-axis.")
        final_ordered_index = sorted(combined_index.tolist())

    # Create the x-coordinates map
    x_coordinates_map = {}
    current_numeric_pos = 0
    # Define the extra gap you want before "Full" compared to standard step spacing
    gap_before_full = 1.0  # e.g., standard step is 1 unit, "Full" will be 1+gap_before_full units after last step

    if final_ordered_index: # Ensure final_ordered_index is not empty
        for i, label in enumerate(final_ordered_index):
            x_coordinates_map[label] = current_numeric_pos
            if label.startswith("step") and (i + 1 < len(final_ordered_index)) and final_ordered_index[i+1] == "Full":
                current_numeric_pos += (1 + gap_before_full) # Apply standard spacing + extra gap
            else:
                current_numeric_pos += 1 # Standard spacing
        print(f"X-coordinates map for plotting: {x_coordinates_map}")

    def sort_columns(df):
        if df.empty:
            return df
        
        # Custom sorting function for columns
        def custom_sort_key(col_name):
            if col_name == "Rotation":
                return (0, 0)  # Place "Rotation" first
            elif col_name[0].isdigit():
                try:
                    step_num = int(col_name.split(" ")[0])
                    return (1, step_num)  
                except ValueError:
                    return (2, col_name)  # If parsing fails, place after numeric steps
            else:
                return (3, col_name)  # Other columns come last
        
        # Sort columns using the custom sorting function
        sorted_columns = sorted(df.columns, key=custom_sort_key)
        df = df[sorted_columns]
        return df
        # Ren
    if not processed_iia_df.empty:
        # Reindex, keeping only columns that actually exist after filtering
        # Ensure index is exactly final_ordered_index if it's not empty
        processed_iia_df = processed_iia_df.reindex(index=final_ordered_index if final_ordered_index else processed_iia_df.index, columns=processed_iia_df.columns) 
        processed_iia_df = sort_columns(processed_iia_df)
        print(f"IIA DataFrame final X-axis: {processed_iia_df.index.tolist()}")
        print(f"IIA DataFrame final columns: {processed_iia_df.columns.tolist()}")


    if not processed_accuracy_df.empty:
        processed_accuracy_df = processed_accuracy_df.reindex(index=final_ordered_index if final_ordered_index else processed_accuracy_df.index, columns=processed_accuracy_df.columns)
        processed_accuracy_df = sort_columns(processed_accuracy_df)
        print(f"Accuracy DataFrame final X-axis: {processed_accuracy_df.index.tolist()}")
        print(f"Accuracy DataFrame final columns: {processed_accuracy_df.columns.tolist()}")
        
    # --- End of Pre-processing ---

    print("\nFinal Processed IIA DataFrame for Plotting:")
    print(processed_iia_df)
    print("\nFinal Processed Accuracy DataFrame for Plotting:")
    print(processed_accuracy_df)
    
    plot_results(processed_iia_df, processed_accuracy_df, x_coordinates_map)

 
# %%
