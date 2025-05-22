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
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import re
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

from das.plotting import plot_results, set_plt_settings

import matplotlib.image as mpimg
from matplotlib.offsetbox import HPacker, TextArea, OffsetImage, AnchoredOffsetbox


RESULTS_DIR = Path("../Results_Cleaned")
PLOTS_DIR = Path("plots/mlp")



def aggregate_stats(
    data: List[List[Dict[str, Dict[str, float]]]]
) -> List[Dict[str, Dict[str, Tuple[float, float, float]]]]:
    if not data:
        return []

    num_items = len(data[0])
    result = []
    accuracy_data = []
    for i in range(num_items):
        accumulator: Dict[str, Dict[str, List[float]]] = {}
        _accuracy_data = []
        for seed in data:
            entry = seed[i]
            for key1, subdict in entry.items():
                if key1 == "accuracy":
                    _accuracy_data.append(subdict)
                    continue
                if key1 not in accumulator:
                    accumulator[key1] = {}
                for key2, value in subdict.items():
                    accumulator[key1].setdefault(key2, []).append(value)

        accuracy_data.append(float(np.mean(_accuracy_data)))
        # Compute (max, mean, std) for each float list
        stats_item: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        for key1, subdict in accumulator.items():
            stats_item[key1] = {}
            for key2, values in subdict.items():
                arr = np.array(values)
                max_val = float(np.max(arr))
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                stats_item[key1][key2] = (max_val, mean_val, std_val)

        result.append(stats_item)

    return result, accuracy_data

models=["Rotation","L1HS16-24","L5HS16-24","L10HS16-24","L10HS128"]
layers = ["Layer1", "Layer2", "Layer3"]
epoch_names=["Random Init",
            "EP1-B131072",
            "EP1-B262144",
            "EP1-B393216",
            "EP1-B524288",
            "EP1-B655360",
            "EP1-B786432",
            "EP1-B917504",
            "EP1-B1048576",
            "EP2-B131072",
            "EP2-B262144",
            "EP2-B393216",
            "EP2-B524288",
            "EP2-B655360",
            "EP2-B786432",
            "EP2-B917504",
            "EP2-B1048576",]

def prepare_data(algorithm_name,layers_label,intervention_sizes,intervention_label):
    residict={}
    accuracy=None
    for ac_model_pos,ac_Model in enumerate(models):
        with open(RESULTS_DIR / 'TrainingProgression' / ac_Model / algorithm_name
     / 'results.json') as f:
            results= json.load(f)
        results, accuracy_data = aggregate_stats(results)
        if accuracy is None:
            # acc is the same for all models
            accuracy = accuracy_data

        for ac_train_step,ac_train_list in enumerate(results):
            if epoch_names[ac_train_step] not in residict:
                residict[epoch_names[ac_train_step]]={}
            for aclayer_pos,aclayer in enumerate(layers):
                if layers_label[aclayer_pos] not in residict[epoch_names[ac_train_step]]:
                    residict[epoch_names[ac_train_step]][layers_label[aclayer_pos]]={}
                for acint_pos, acint in enumerate(intervention_sizes):
                    if intervention_label[acint_pos] not in residict[epoch_names[ac_train_step]][layers_label[aclayer_pos]]:
                        residict[epoch_names[ac_train_step]][layers_label[aclayer_pos]][intervention_label[acint_pos]]=[[],[],[]]

                    residict[epoch_names[ac_train_step]][layers_label[aclayer_pos]][intervention_label[acint_pos]][0].append(results[ac_train_step][aclayer][acint][0])
                    residict[epoch_names[ac_train_step]][layers_label[aclayer_pos]][intervention_label[acint_pos]][1].append(results[ac_train_step][aclayer][acint][1])
                    residict[epoch_names[ac_train_step]][layers_label[aclayer_pos]][intervention_label[acint_pos]][2].append(results[ac_train_step][aclayer][acint][2])
    return residict, accuracy

def extract_training_data(data, layer, intervention, models):
    """Extracts means, stds, and max values for a specific metric across training epochs."""
    epochs_raw = list(data.keys())
    num_models = len(models)
    # Sort epochs: 'Random Init' first, then EPx-Byyy sorted numerically
    def sort_key(epoch_name):
        if epoch_name == 'Random Init':
            return (0, 0, 0) # Ensure it's first
        match = re.match(r'EP(\d+)-B(\d+)', epoch_name)
        if match:
            ep_num = int(match.group(1))
            batch_num = int(match.group(2))
            return (1, ep_num, batch_num) # Sort by type, then epoch, then batch
        print(f"Warning: Unrecognized epoch format: {epoch_name}")
        return (2, 0, 0) # Place unrecognized formats last

    sorted_epochs = sorted(epochs_raw, key=sort_key)

    plot_data_mean = {}
    plot_data_std = {}
    plot_data_max = {}

    for epoch in sorted_epochs:
        try:
            # Access the mean list (index 0), std list (index 1), and max list (index 2)
            data_point = data[epoch][layer][intervention]
            max_list, means_list, stds_list = data_point[0], data_point[1], data_point[2]

            if len(means_list) == num_models and len(stds_list) == num_models and len(max_list) == num_models:
                 plot_data_mean[epoch] = means_list
                 plot_data_std[epoch] = stds_list
                 plot_data_max[epoch] = max_list
            else:
                # Handle potential length mismatches if data is inconsistent
                print(f"Warning: Data length mismatch in {epoch}/{layer}/{intervention}. Expected {num_models}, got {len(means_list)}. Padding with NaN.")
                plot_data_mean[epoch] = (means_list + [np.nan]*num_models)[:num_models]
                plot_data_std[epoch] = (stds_list + [np.nan]*num_models)[:num_models]
                plot_data_max[epoch] = (max_list + [np.nan]*num_models)[:num_models]

        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Could not extract data for {epoch}/{layer}/{intervention}. Error: {e}. Filling with NaN.")
            plot_data_mean[epoch] = [np.nan] * num_models
            plot_data_std[epoch] = [np.nan] * num_models
            plot_data_max[epoch] = [np.nan] * num_models

    # Create DataFrames
    mean_df = pd.DataFrame.from_dict(plot_data_mean, orient='index', columns=models)
    std_df = pd.DataFrame.from_dict(plot_data_std, orient='index', columns=models)
    max_df = pd.DataFrame.from_dict(plot_data_max, orient='index', columns=models)

    # Drop rows where all values are NaN (might happen if an epoch had missing data)
    mean_df.dropna(axis=0, how='all', inplace=True)
    std_df = std_df.reindex(mean_df.index) # Ensure std_df matches filtered mean_df index
    max_df = max_df.reindex(mean_df.index) # Ensure max_df matches filtered mean_df index

    return max_df, mean_df, std_df


def plot_layer_intervention(results, accuracy_data, target_layer, target_intervention, out_file_name, legend=True, collapse_strategy="max"):
    # Determine number of models/seeds from data structure
    try:
        first_epoch_key = next(iter(results))
        num_models = len(results[first_epoch_key][target_layer][target_intervention][0])
        print(f"Detected {num_models} models/seeds per data point.")
    except (StopIteration, KeyError, IndexError, TypeError):
        print("Warning: Could not determine number of models automatically. Defaulting to 1.")
        num_models = 1
    # Extract the data for the chosen layer/intervention
    metric_max_df, metric_mean_df, metric_std_df = extract_training_data(results, target_layer, target_intervention, model_names)
    x_coords_map = {epoch: i for i, epoch in enumerate(metric_mean_df.index)}
    if collapse_strategy == "max":
        df = metric_max_df
    elif collapse_strategy == "mean":
        df = metric_mean_df
    elif collapse_strategy == "std":
        df = metric_std_df
    else:
        raise ValueError(f"Invalid collapse strategy: {collapse_strategy}")

    accuracy_df = pd.DataFrame(accuracy_data, index = epoch_names, columns = ["Accuracy"])

    # Define a simple label replacement for epochs if desired (optional)
    def format_epoch_label(s):
        if s == 'Random Init':
            return 'Step 0'
        s = s.replace('-B', '/') # Shorten label
        # Potentially convert large batch numbers to scientific notation or k/M units
        parts = s.split('/')
        if len(parts) == 2:
            try:
                batch_num = int(parts[1])
                if batch_num >= 1_000_000:
                        s = f"{batch_num/1_000_000:.0f}M"
                elif batch_num >= 1000:
                        s = f"{batch_num/1000:.0f}k"
            except ValueError:
                pass # Keep original if batch number isn't integer
        return s
    
    fig, ax = plot_results(
        iia_df=df,
        acc_df=accuracy_df,      # No separate accuracy markers needed here
        iia_std_df=pd.DataFrame(),   # Standard deviations for error bands
        x_coords_map=x_coords_map,
        # Use a descriptive Y label based on what the metric actually is
        y_label=f"Accuracy",
        error_bar_alpha=0.15,        # Adjust transparency of error bands
        error_bar_multiplier=1.0,    # Show +/- 1 standard deviation
        xtick_label_rotation=0,
        xtick_label_fontsize=24,
        ytick_label_fontsize=24,
        xlabel_fontsize=24,
        ylabel_fontsize=24,
        x_label="DNN Training Steps",
        legend_font_size=18,
        legend_loc_final='lower right', # Adjust legend position
        legend_bbox_final=(1.0, 0.0),   # Adjust legend position
        figsize=(8, 4.2),              # Adjust figure size
        inject_color_at_start=(0, 204/255, 102/255),
        acc_marker_color="red",
        acc_marker_label="DNN",
        x_axis_label_pad=10
        # xtick_label_replace=format_epoch_label, # Apply custom label formatting
    )

    # --- X-axis Tick Customization (as requested) ---
    # Only show labels for the first and last training steps plotted
    if len(metric_mean_df.index) > 1:
        start_epoch_label = metric_mean_df.index[0]
        end_epoch_label = metric_mean_df.index[-1]
        start_pos = x_coords_map[start_epoch_label]
        end_pos = x_coords_map[end_epoch_label]

        # Get formatted labels if replacement function is used
        formatted_start = "0"
        formatted_end = "Full"

        labels = ["" for _ in range(len(metric_mean_df.index))]
        labels[0] = formatted_start
        labels[len(labels)//2] = format_epoch_label(metric_mean_df.index[len(labels)//2])
        labels[-1] = formatted_end
        ax.set_xticklabels(labels, rotation=0, ha="center")
        print(f"Customized x-axis ticks to show only: {formatted_start}, {formatted_end}")
    elif len(metric_mean_df.index) == 1:
            start_epoch_label = metric_mean_df.index[0]
            start_pos = x_coords_map[start_epoch_label]
            formatted_start = format_epoch_label(start_epoch_label)
            ax.set_xticks([start_pos])
            ax.set_xticklabels([formatted_start], rotation=45, ha="right")

    if not legend:
        ax.get_legend().remove()

    # Optional: Adjust Y-axis limits if needed
    ax.set_ylim(0.45, 1.01)
    set_plt_settings()

    fig.tight_layout() # Adjust layout to prevent overlap
    fig.savefig(out_file_name, bbox_inches='tight')
    plt.show()


def plot_single_layer_intervention_ax(ax, results, accuracy_data, target_layer, target_intervention, model_names, collapse_strategy="max", show_legend=False, show_xlabel=True, show_ylabel=True):
    """Plots a single layer/intervention combination on a given matplotlib Axes object."""
    try:
        first_epoch_key = next(iter(results))
        num_models = len(results[first_epoch_key][target_layer][target_intervention][0])
    except (StopIteration, KeyError, IndexError, TypeError):
        print(f"Warning: Could not determine number of models for {target_layer}/{target_intervention}. Defaulting to 1.")
        num_models = 1

    metric_max_df, metric_mean_df, metric_std_df = extract_training_data(results, target_layer, target_intervention, model_names)
    x_coords_map = {epoch: i for i, epoch in enumerate(metric_mean_df.index)}

    if collapse_strategy == "max":
        df_to_plot = metric_max_df
    elif collapse_strategy == "mean":
        df_to_plot = metric_mean_df
    elif collapse_strategy == "std":
        df_to_plot = metric_std_df
    else:
        raise ValueError(f"Invalid collapse strategy: {collapse_strategy}")

    accuracy_df = pd.DataFrame(accuracy_data, index=epoch_names, columns=["Accuracy"])

    def format_epoch_label(s):
        if s == 'Random Init':
            return 'Step 0'
        s = s.replace('-B', '/')
        parts = s.split('/')
        if len(parts) == 2:
            try:
                batch_num = int(parts[1])
                if batch_num >= 1_000_000:
                    s = f"{batch_num/1_000_000:.0f}M"
                elif batch_num >= 1000:
                    s = f"{batch_num/1000:.0f}k"
            except ValueError:
                pass
        return s

    # Use the existing plot_results, but pass the ax
    plot_results(
        ax=ax, # Pass the subplot axis
        iia_df=df_to_plot,
        acc_df=accuracy_df,
        iia_std_df=pd.DataFrame(), # Assuming std is handled by collapse_strategy or not shown here
        x_coords_map=x_coords_map,
        y_label="Accuracy" if show_ylabel else "",
        error_bar_alpha=0.15,
        error_bar_multiplier=1.0,
        xtick_label_rotation=0,
        xtick_label_fontsize=18, # Adjusted for subplots
        ytick_label_fontsize=18, # Adjusted for subplots
        xlabel_fontsize=20 if show_xlabel else 0, # Adjusted for subplots
        ylabel_fontsize=20 if show_ylabel else 0, # Adjusted for subplots
        x_label="DNN Training Steps" if show_xlabel else "",
        legend_font_size=14, # Adjusted for subplots
        legend_loc_final='lower right',
        legend_bbox_final=(1.0, 0.0),
        # figsize is controlled by the parent grid function
        inject_color_at_start=(0, 204/255, 102/255),
        acc_marker_color="red",
        acc_marker_label="DNN",
        x_axis_label_pad=10,
        show_legend=show_legend # Pass legend visibility
    )

    if len(metric_mean_df.index) > 1:
        start_epoch_label = metric_mean_df.index[0]
        end_epoch_label = metric_mean_df.index[-1]

        formatted_start = "0"
        formatted_end = "Full"
        
        middle_idx = len(metric_mean_df.index) // 2
        formatted_middle = format_epoch_label(metric_mean_df.index[middle_idx])


        labels = ["" for _ in range(len(metric_mean_df.index))]
        if show_xlabel: # Only set labels if xlabel is shown (typically bottom row)
            labels[0] = formatted_start
            labels[middle_idx] = formatted_middle
            labels[-1] = formatted_end
            ax.set_xticklabels(labels, rotation=0, ha="center")
        else:
            ax.set_xticklabels([])


    elif len(metric_mean_df.index) == 1 and show_xlabel:
        start_epoch_label = metric_mean_df.index[0]
        start_pos = x_coords_map[start_epoch_label]
        formatted_start = format_epoch_label(start_epoch_label)
        ax.set_xticks([start_pos])
        ax.set_xticklabels([formatted_start], rotation=45, ha="right")
    
    if not show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_ylim(0.45, 1.01)
    # set_plt_settings() # Applied globally for the figure by the grid function

def plot_layer_intervention_grid(results, accuracy_data, layers_label, intervention_label, model_names, algorithm_name, out_file_name, collapse_strategy="max"):
    num_rows = len(layers_label)
    num_cols = len(intervention_label)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharex=False, sharey=True)
    # If only one row or col, axes might not be a 2D array
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)


    for i, layer in enumerate(layers_label):
        for j, intervention in enumerate(intervention_label):
            ax = axes[i, j]
            show_legend_for_subplot = (i == num_rows - 1 and j == num_cols -1) # Legend only on top-right
            show_xlabel_for_subplot = (i == num_rows - 1) # X-labels only on bottom row
            show_ylabel_for_subplot = (j == 0) # Y-labels only on left-most column

            plot_single_layer_intervention_ax(
                ax=ax,
                results=results,
                accuracy_data=accuracy_data,
                target_layer=layer,
                target_intervention=intervention,
                model_names=model_names,
                collapse_strategy=collapse_strategy,
                show_legend=show_legend_for_subplot,
                show_xlabel=show_xlabel_for_subplot,
                show_ylabel=show_ylabel_for_subplot
            )

            # Row titles (Layer names)
            if j == 0:
                # Use a proper line break in the label for LaTeX rendering in matplotlib
                current_ylabel = ax.get_ylabel()
                if current_ylabel:
                    ax.set_ylabel(f"{layer}\n{current_ylabel}", fontsize=20, labelpad=20)
                else:
                    ax.set_ylabel(f"{layer}", fontsize=20)
            
            # Column titles (Intervention sizes)
            if i == 0:
                current_intervention_display_label = intervention_label[j] # e.g., "Intervention Size 8"
                image_path = Path('intervensionsize_label.png')
                
                # Extract the numeric part of the label, e.g., "1", "2", "8"
                try:
                    numeric_part_for_label = current_intervention_display_label.split(" ")[-1]
                    # You might want to add validation here if the format isn't guaranteed
                except IndexError:
                    numeric_part_for_label = "?" # Fallback if split fails

                use_image_title = False
                if image_path.exists():
                    try:
                        img_data = mpimg.imread(image_path)
                        image_artist = OffsetImage(img_data, zoom=0.25) # User's preferred zoom
                        
                        text_props = dict(fontsize=20)
                        text_after = TextArea(f" = {numeric_part_for_label}", textprops=text_props)

                        packer = HPacker(children=[image_artist, text_after],
                                         sep=5, # Separation in points
                                         align="center",
                                         pad=0)

                        anchored_box = AnchoredOffsetbox(loc='lower center',
                                                         child=packer,
                                                         frameon=False,
                                                         bbox_to_anchor=(0.5, 1.05), 
                                                         bbox_transform=ax.transAxes,
                                                         borderpad=0.)
                        ax.add_artist(anchored_box)
                        use_image_title = True
                    except Exception as e:
                        print(f"Error processing image for column title '{current_intervention_display_label}': {e}. Using text fallback.")
                        # Will fall through to text title logic
                
                if not use_image_title:
                    # Text fallback (if image doesn't exist or failed to load)
                    base_name_parts = current_intervention_display_label.split(" ")[:-1]
                    # Join parts like "Intervention", "Size" to form "Intervention Size"
                    base_name = " ".join(base_name_parts) if base_name_parts else "Label" # Fallback base name

                    status_suffix = ""
                    if not image_path.exists():
                        status_suffix = " (img not found)"
                    else: # Image exists but failed to load (implies an error occurred)
                        status_suffix = " (img error)"
                    
                    ax_title_text = f"{base_name} = {numeric_part_for_label}{status_suffix}"
                    
                    ax_top = ax.twiny()
                    ax_top.set_xticks([])
                    ax_top.set_xlabel(ax_title_text, fontsize=20, labelpad=10)
                    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    for spine_name in ['top', 'right', 'left', 'bottom']:
                        ax_top.spines[spine_name].set_visible(False)

    set_plt_settings() # Apply global settings
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent overlap and make space for suptitle and legend
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure plots directory exists
    fig.savefig(PLOTS_DIR / out_file_name, bbox_inches='tight')
    plt.show()


# %%
intervention_sizes_double_intervention =   ['[[0], [1]]',
                '[[0, 1], [2, 3]]',
                '[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]'
                ]
intervention_sizes_single_intervention =   ['[[0]]',
                '[[0, 1]]',
                '[[0, 1, 2, 3, 4, 5, 6, 7]]'
                ]
layers_label=["Layer 3","Layer 2","Layer 1"]
intervention_label=["Intervention Size 1","Intervention Size 2","Intervention Size 8"]
model_names=["linear","$L_{{\\mathrm{{rn}}}}=1,d_\\mathrm{rn}=2^4$","$L_{{\\mathrm{{rn}}}}=5,d_\\mathrm{rn}=2^4$","$L_{{\\mathrm{{rn}}}}=10,d_\\mathrm{rn}=2^4$","$L_{{\\mathrm{{rn}}}}=10,d_\\mathrm{rn}=2^7$"]

def generate_plots(algorithm_name, intervention_sizes=intervention_sizes_double_intervention):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    results, accuracy_data = prepare_data(algorithm_name
,layers_label,intervention_sizes,intervention_label)
    legend_idx = ("Layer 1", "Intervention Size 8")
    for layer in layers_label:
        for intervention in intervention_label:
            legend = layer == legend_idx[0] and intervention == legend_idx[1]
            plot_layer_intervention(results, accuracy_data, layer, intervention, PLOTS_DIR / f"training_progression_{algorithm_name}_{layer}_{intervention}.pdf".replace(" ", "_"), legend=legend)
    plot_layer_intervention_grid(
        results=results,
        accuracy_data=accuracy_data,
        layers_label=layers_label, # Your list of layer labels
        intervention_label=intervention_label, # Your list of intervention labels
        model_names=model_names,
        algorithm_name=algorithm_name
    ,
        out_file_name=f"max_training_progression_{algorithm_name}.pdf",
        collapse_strategy="max" # or "mean"
    )
    plot_layer_intervention_grid(
        results=results,
        accuracy_data=accuracy_data,
        layers_label=layers_label, # Your list of layer labels
        intervention_label=intervention_label, # Your list of intervention labels
        model_names=model_names,
        algorithm_name=algorithm_name,
        out_file_name=f"mean_training_progression_{algorithm_name}.pdf",
        collapse_strategy="mean" # or "mean"
    )
# %%
generate_plots("Both_Equality_Relations")

# %%
generate_plots("Left_Equality_Relation")

# %%
generate_plots("Identity_of_First_Argument", intervention_sizes=intervention_sizes_single_intervention)

# %%
generate_plots("Identity_of_Second_Argument", intervention_sizes=intervention_sizes_single_intervention)

# %%

## Distributive Law

intervention_sizes =   ['[[0], [1]]',
                        '[[0, 1], [2, 3]]',
                        '[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]'
                        ]
intervention_label=["Intervention Size 1","Intervention Size 2","Intervention Size 12"]


# %%
generate_plots("AndOr", intervention_sizes=intervention_sizes)

# %%
generate_plots("AndOrAnd", intervention_sizes=intervention_sizes)
# %%
