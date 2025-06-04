# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

from pathlib import Path

RESULTS_DIR = Path("../Results/Standard_DAS")

def calculate_mean_std(dict_list, greedy=False):
    print(greedy)
    if not dict_list:
        return {}

    def recursive_stats(keys, data, greedy=False):
        if isinstance(data[0], dict):
            return {
                key: recursive_stats(keys + [key], [d[key] for d in data], greedy)
                for key in data[0]
            }
        else:
            if isinstance(data[0], list) and greedy:
                print("Detected Greedy")

                out = defaultdict(list)
                for seed in data:
                    for el in seed:
                        print(el)
                        key, value = el
                        out[len(key[0])].append(value)

                print(out)
                return {
                    key: recursive_stats(keys + [key], out[key], greedy)
                    for key in out
                }
            values = np.array(data)
            return (
                float(round(values.mean(), 2)),
                float(values.std()),
                float(round(max(values), 2)),
            )

    return recursive_stats([], dict_list, greedy)


def load_file(intervention, algorithm, greedy=False):
    print(intervention, algorithm, greedy)
    with open(
        RESULTS_DIR / intervention / "FullyTrained" / algorithm / "results.json"
    ) as f:
        return calculate_mean_std(json.load(f), greedy)


def load(
    interventions=["Rotation", "RevNet", "Greedy_Neuronset"],
    algorithms=[
        "Both_Equality_Relations",
        "Left_Equality_Relation",
        "Identity_of_First_Argument",
    ],
):
    results = {}
    for intervention in interventions:
        results[intervention] = {}
        for algorithm in algorithms:
            results[intervention][algorithm] = load_file(intervention, algorithm, greedy="greedy"in intervention.lower())
    return results


def autolabel(rects_list, ax_to_annotate):
    "Attach a text label above each bar, displaying its height (main value)."
    for bar_container in rects_list:
        for rect_patch in bar_container.patches:
            height = rect_patch.get_height()
            ax_to_annotate.annotate(
                f"{height:.2f}",  # Format height to 2 decimal places
                xy=(rect_patch.get_x() + rect_patch.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=5,
            )  # Adjusted fontsize for clarity


def plot_data(
    full_data_to_plot,
    ordered_algorithms = ["Both_Equality_Relations", "Left_Equality_Relation", "Identity_of_First_Argument"],
    font_size=24,
    axes_labelsize=24,
    axes_titlesize=24,
    xtick_labelsize=24,
    ytick_labelsize=24,
    legend_fontsize=24,
    legend_title_fontsize=24,
    legend_loc="upper right",
    task_name="Hierarchical Equality"
):
    # LaTeX font rendering and global font size settings
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": font_size,  # General font size
            "axes.labelsize": axes_labelsize,
            "axes.titlesize": axes_titlesize,
            "xtick.labelsize": xtick_labelsize,  # Smaller for potentially long intervention labels
            "ytick.labelsize": ytick_labelsize,
            "legend.fontsize": legend_fontsize,
            "legend.title_fontsize": legend_title_fontsize,
        }
    )

    model_names = list(full_data_to_plot.keys())
    num_models = len(model_names)

    first_model_data = full_data_to_plot[model_names[0]]

    # Check if all ordered algorithms are present in the data
    available_data_algorithms = list(first_model_data.keys())
    algorithms_to_plot = []
    for algo in ordered_algorithms:
        if algo in available_data_algorithms:
            algorithms_to_plot.append(algo)
        else:
            print(f"Warning: Algorithm '{algo}' specified in ordered_algorithms not found in data. It will be skipped.")
    
    if not algorithms_to_plot:
        print("Error: None of the specified algorithms for subplots are available in the data.")
        return
    
    num_subplots = len(algorithms_to_plot)

    # Assuming layer names are consistent across models and algorithms (after filtering)
    # Use the first algorithm in our plotting list to determine layers
    potential_layer_keys = first_model_data[algorithms_to_plot[0]].keys()
    layer_names_unsorted = [
        key for key in potential_layer_keys if key.startswith("Layer")
    ]

    # Sort layer names numerically if possible (e.g., Layer1, Layer2, Layer10)
    try:
        layer_names = sorted(
            layer_names_unsorted, key=lambda x: int(x.replace("Layer", ""))
        )
    except ValueError:
        layer_names = sorted(layer_names_unsorted)  # Fallback to alphanumeric sort

    num_layers = len(layer_names)

    bar_width = 0.22 # Adjusted for three bars
    model_pair_gap = 0.05 
    model_group_width = num_models * bar_width + (num_models -1) * model_pair_gap # Renamed for clarity

    intervention_group_padding = 0.25 
    layer_group_padding = 0.8 

    errorbar_color = "black"
    errorbar_capsize = 3
    errorbar_elinewidth = 1
    errorbar_markersize = 4
    errorbar_zorder = 3

    # Define the mapping for intervention string/integer labels to concise display numbers
    intervention_display_map = {
        '[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]': 8,
        '[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]': 12,
        '[[0, 1], [2, 3]]': 2,
        '[[0], [1]]': 1,
        '[[0, 1, 2, 3, 4, 5, 6, 7]]': 8,
        '[[0, 1]]': 2,
        '[[0]]': 1,
        '1': 1, '2': 2, '8': 8, '12': 12,
        # Integer keys (expected from Greedy_Neuronset data)
        1: 1, 2: 2, 8: 8, 12: 12
    }

    # Define hatch patterns for colorblind accessibility
    hatch_patterns = ['o', '\\', '+', 'x', 'o', 'O', '.', '*']

    plot_colors = ["skyblue", "salmon", "lightgreen"] # Colors for up to 3 models

    img_lambda = None # Initialize
    try:
        img_lambda = plt.imread('intervensionsize_label.png')
    except FileNotFoundError:
        print("Warning: notebooks/intervensionsize_label.png not found. Lambda image label will be skipped.")

    # Create a figure with num_subplots subplots
    # Adjust figsize: a bit more width per subplot than original single '5' or '6'
    # Let's aim for roughly 6 units width per subplot if it has y-axis, 5 if not.
    # Total width for 3 plots: 6 (first) + 5 (middle) + 5 (last, but legend needs space) -> ~16-18
    fig_width_total = len(algorithms_to_plot) * 6.5 + 1.5 
    fig, axes = plt.subplots(1, num_subplots, figsize=(fig_width_total, 5), sharey=True)
    
    if num_subplots == 1: # If only one algorithm ends up being plotted
        axes = [axes] # Make it iterable for the loop

    for idx, algo_t in enumerate(algorithms_to_plot):
        ax = axes[idx] # Current subplot
        
        # Set subplot title
        ax.set_title(algo_t.replace("_", " "))

        # Determine intervention_tick_labels for the current algorithm
        # Assuming they are consistent across layers and models for this algo_t
        if not layer_names:
            print(f"Warning: No layers found for algorithm {algo_t}. Skipping plot.")
            plt.close(fig)  # Close if fig was already created
            continue

        # Safely get intervention_tick_labels (original keys from data)
        try:
            # These keys can be strings (e.g., '[[0],[1]]') or integers (e.g., 1, 2, 8 from Greedy)
            current_intervention_keys_unsorted = list(full_data_to_plot[model_names[0]][algo_t][layer_names[0]].keys())
        except KeyError:
            print(
                f"Warning: Could not retrieve intervention keys for {model_names[0]}/{algo_t}/{layer_names[0]}. Skipping plot for {algo_t}."
            )
            plt.close(fig) # Close the empty figure
            continue
    
        
        # Filter and map intervention keys to display labels [1, 2, 8]
        processed_interventions = []
        for original_key in current_intervention_keys_unsorted:
            display_label = intervention_display_map.get(original_key)
            if display_label in [1, 2, 8, 12]: # Only consider if it maps to our target display values
                processed_interventions.append({'original_key': original_key, 'display_label': display_label})
        
        # Sort by the 'display_label' (1, 2, 8)
        processed_interventions.sort(key=lambda x: x['display_label'])
            
        intervention_tick_labels_sorted_keys = [item['original_key'] for item in processed_interventions]
        x_ticks_for_display                  = [item['display_label'] for item in processed_interventions]

            
        num_interventions = len(intervention_tick_labels_sorted_keys)

        if num_interventions == 0:
            print(
                f"Warning: No interventions found for algorithm {algo_t}, layer {layer_names[0]} that map to sizes 1, 2, or 8. Skipping plot for this algorithm."
            )
            plt.close(fig)
            continue

        current_x_offset = 0
        x_ticks_intervention_positions = []
        layer_label_info = []
        all_bar_containers_for_labeling = []
        legend_added_flags = [False] * num_models
        
        actual_xticklabels_for_plot = [] # Initialize list for repeated tick labels

        for layer_idx, layer_n in enumerate(layer_names):
            layer_block_start_x = current_x_offset
            # The loop for plotting bars should iterate based on the filtered and sorted keys
            # for int_idx, current_intervention_key_str in enumerate(intervention_tick_labels_sorted_keys):
            # Change this to iterate through the conceptual display labels
            for int_idx, target_display_label in enumerate(x_ticks_for_display):
                bar_positions = [
                    current_x_offset + i * (bar_width + model_pair_gap)
                    for i in range(num_models)
                ]

                for model_idx, model_name in enumerate(model_names):
                    mean_val, std_val, max_val = 0, 0, 0 # Default to zeros
                    found_data_for_model = False

                    # Search for the actual data key for the current model that maps to target_display_label
                    available_keys_for_current_model = list(full_data_to_plot[model_name][algo_t][layer_n].keys())
                    actual_key_for_this_model = None
                    for key_from_model in available_keys_for_current_model:
                        if intervention_display_map.get(key_from_model) == target_display_label:
                            actual_key_for_this_model = key_from_model
                            found_data_for_model = True
                            break
                    
                    if found_data_for_model and actual_key_for_this_model is not None:
                        try:
                            mean_val, std_val, max_val = full_data_to_plot[model_name][algo_t][layer_n][actual_key_for_this_model]
                        except KeyError: # Should ideally not happen if key was from .keys()
                            print(f"ERROR: Key '{actual_key_for_this_model}' not found for {model_name}/{algo_t}/{layer_n} despite being in its keys. Plotting as zero.")
                            # mean_val, std_val, max_val remain 0,0,0
                    else:
                        print(f"Warning: No data key found for {model_name}/{algo_t}/{layer_n} that maps to display label {target_display_label}. Plotting as zero.")
                        # mean_val, std_val, max_val remain 0,0,0

                    # Cycle through colors if more models than defined colors (for future flexibility)
                    color = plot_colors[model_idx % len(plot_colors)]
                    # Cycle through hatch patterns
                    hatch = hatch_patterns[model_idx % len(hatch_patterns)]

                    label_for_bar = None # Initialize label for this specific bar
                    if not legend_added_flags[model_idx]:
                        # This is the first time this model type is plotted in this subplot,
                        # so assign a label for the legend.
                        if model_name == "Rotation":
                            label_for_bar = "linear"
                        elif model_name == "Greedy_Neuronset":
                            label_for_bar = "identity"
                        elif model_name == "RevNet":
                            label_for_bar = "non-linear"
                        else:
                            label_for_bar = model_name
                        legend_added_flags[model_idx] = True # Mark as labeled for this subplot

                    rect_container = ax.bar(
                        bar_positions[model_idx],
                        max_val,  # Bar height is now max_val
                        bar_width,
                        color=color,
                        label=label_for_bar, # Use the determined label
                        hatch=hatch, # Add hatch pattern
                        edgecolor='white', # Set hatch pattern color to white
                        zorder=2,
                    )

                    all_bar_containers_for_labeling.append(rect_container)

                    bar_patch = rect_container.patches[0]
                    ax.errorbar(
                        bar_patch.get_x() + bar_patch.get_width() / 2,
                        mean_val,
                        yerr=std_val,
                        fmt="o",
                        color=errorbar_color,
                        capsize=errorbar_capsize,
                        elinewidth=errorbar_elinewidth,
                        markeredgewidth=errorbar_elinewidth,
                        markersize=errorbar_markersize,
                        zorder=errorbar_zorder,
                    )

                intervention_tick_pos = (
                    current_x_offset + model_group_width / 2.0
                )  # Centered under the group of bars
                x_ticks_intervention_positions.append(intervention_tick_pos)
                
                # Append the correct display label for this tick position
                actual_xticklabels_for_plot.append(x_ticks_for_display[int_idx]) 

                current_x_offset += model_group_width
                if int_idx < num_interventions - 1:
                    current_x_offset += intervention_group_padding

            layer_block_end_x = current_x_offset
            layer_label_pos = (
                layer_block_start_x
                + (
                    layer_block_end_x
                    - layer_block_start_x
                    - (intervention_group_padding if num_interventions > 1 else 0)
                )
                / 2.0
            )
            layer_label_info.append(
                (layer_label_pos, layer_n.replace("Layer", "L"))
            )  # Shorten Layer names for display

            if layer_idx < num_layers - 1:
                current_x_offset += layer_group_padding

        # autolabel(all_bar_containers_for_labeling, ax) # autolabel remains for potential future use

        ax.set_xticks(x_ticks_intervention_positions)
        # Use the mapped display labels and adjust rotation/alignment
        ax.set_xticklabels(actual_xticklabels_for_plot, rotation=0, ha="center") 

        # Adjust y-position for layer labels dynamically based on current xtick label properties
        # This is an estimate; might need fine-tuning
        y_pos_level1_labels = -0.15
        # Check if any label is a long string (e.g. unmapped)
        if any(isinstance(lbl, str) and len(lbl) > 10 for lbl in actual_xticklabels_for_plot):
            y_pos_level1_labels = (
                -0.20
            )  # Make more space if long string labels are present
            # If xtick labels are rotated (e.g. ax.get_xticklabels()[0].get_rotation() != 0),
            # this might need further adjustment, but current default is rotation=0.

        for pos, label in layer_label_info:
            ax.text(
                pos,
                y_pos_level1_labels,
                label,
                ha="center",
                va="top",
                fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )

        # Y-axis label handling
        if idx == 0: # Only for the first (leftmost) subplot
            ax.set_ylabel("Accuracy")
            if img_lambda is not None:
                imagebox = OffsetImage(img_lambda, zoom=0.2) # Adjust zoom as needed
                # Position using axes fraction. xy is the point, box_alignment aligns the imagebox to this point.
                # (1, 0.5) means right-center of imagebox is at xy.
                ab = AnnotationBbox(imagebox, 
                                   xy=(-0.03, -0.07), 
                                   xycoords='axes fraction', 
                                   box_alignment=(1, 0.5), # Align right-center of image to xy
                                   frameon=False,
                                   pad=0) # No padding around the image
                ax.add_artist(ab)
            else:
                # Fallback to text if image not loaded
                ax.text(-0.05, -0.05, r"$\lambda$", 
                        transform=ax.transAxes, 
                        ha='right', 
                        va='center', 
                        fontsize=axes_labelsize)
        else:
            ax.yaxis.set_ticklabels([]) # Hide tick labels for shared y-axis - Reinstated
            ax.set_ylabel("") # Hide y-axis label

        # Legend handling - only for the last (rightmost) subplot
        if idx == num_subplots - 1: # Check if it's the last subplot being plotted
            # The legend_added_flags ensure labels are only added once per model type
            # The actual legend creation happens here for the last plot
            handles, labels = ax.get_legend_handles_labels()
            if handles: # Only show legend if there are items to show
                 # Place legend in top right corner
                 ax.legend(handles, labels, loc=legend_loc, ncol=len(labels)-1, frameon=True, framealpha=0.9, facecolor='white') # Use collected handles/labels with horizontal layout
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
        
    # Common settings for all subplots (applied once due to sharey=True for ylim)
    axes[0].set_ylim(0, 1) 

    # Define custom y-axis tick formatter
    def custom_y_formatter(x, pos):
        if x == 0:  # Check if x is a whole number (e.g., 0.0, 1.0)
            return f""  # Format as integer (e.g., "0", "1")
        else:
            return f"{x:.2f}"  # Format with one decimal place (e.g., "0.2")

    axes[0].yaxis.set_major_formatter(FuncFormatter(custom_y_formatter))

    fig.tight_layout() # Adjust subplot params for a tight layout

    
    output_filename = f"plots/{task_name}_combined.pdf" # Single filename for the combined plot
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved as {output_filename}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    plt.show()
    plt.close(fig)
# %%

## Hierarchical Equality
# %%
results = load()
# %%
plot_data(results, task_name="hierarchical_equality") # Single call to the revised function
# %%

## Distributed Law
# %%
algorithms = ["AndOr", "AndOrAnd"]
results = load(algorithms=algorithms)
# %%
plot_data(results, ordered_algorithms=algorithms, legend_loc="lower right", task_name="distributed_law") # Single call to the revised function

# %%
