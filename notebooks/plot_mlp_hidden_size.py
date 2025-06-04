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
from matplotlib.offsetbox import HPacker, TextArea, OffsetImage, AnchoredOffsetbox, DrawingArea

# Added imports for custom legend handler
from matplotlib.legend_handler import HandlerBase

RESULTS_DIR = Path("../Results")
PLOTS_DIR = Path("plots/mlp")
IMAGE_LEGEND_PATH = Path("intervensionsize_label_small.png")

# Define custom legend handler to draw a pre-made artist (our HPacker)
class HandlerToDrawArtist(HandlerBase):
    def __init__(self, artist_to_draw, **kwargs):
        self.artist_to_draw = artist_to_draw
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, 
                       xdescent, ydescent, width, height, fontsize, trans):
        # self.artist_to_draw is our HPacker. It should be returned as is.
        # The legend's drawing mechanism will handle its transform and placement.
        return [self.artist_to_draw]

def put_together(data1,data2):
    for i in range(len(data1)):
        for a in data2[i].keys():
            if a!="accuracy":
                data1[i][a]=data2[i][a]
    return data1

def calculate_mean_std(dict_list):
    if not dict_list:
        return {}

    def recursive_stats(keys, data):
        if isinstance(data[0], dict):
            return {key: recursive_stats(keys + [key], [d[key] for d in data]) for key in data[0]}
        else:
            values = np.array(data)
            return (values.mean(), values.std())

    return recursive_stats([], dict_list)

layers = ["Layer1", "Layer2", "Layer3"]
intervention_sizes =   ['[[0], [1]]',
                        '[[0, 1], [2, 3]]',
                        '[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]'
                        ]
layers_label=["Layer 1","Layer 2","Layer 3"]
intervention_label=["Intervetion Size 1","Intervetion Size 2","Intervetion Size 8"]
hiddensize_range=list(range(1,17))


def prepare_data(results,layers,intervention_sizes,layers_label,intervention_label,hiddensize_range):
    residict={}
    for achiddensize in hiddensize_range:
        achiddensize=str(achiddensize)
        for aclayer_pos,aclayer in enumerate(layers):
            if layers_label[aclayer_pos] not in residict:
                residict[layers_label[aclayer_pos]]={}
            for acint_pos, acint in enumerate(intervention_sizes):
                if intervention_label[acint_pos] not in residict[layers_label[aclayer_pos]]:
                    residict[layers_label[aclayer_pos]][intervention_label[acint_pos]]=[[],[]]
                residict[layers_label[aclayer_pos]][intervention_label[acint_pos]][0].append(results[achiddensize][aclayer][acint][0])
                residict[layers_label[aclayer_pos]][intervention_label[acint_pos]][1].append(results[achiddensize][aclayer][acint][1])
    return residict


def make_plot_single_layer(results, mean_data, target_layer, intervention_sizes, out_file_name):
    layer_data = results.get(target_layer)
    if layer_data:
        num_points = len(layer_data[next(iter(layer_data.keys()))][0])
        print(num_points)
        hidden_sizes_index = [i+1 for i in range(num_points)]
        layer_data = results.get(target_layer)
        mean_data_for_df = {}
        std_data_for_df = {}
        intervention_sizes = [] # Keep track of column order

        for intervention_key, (means, stds) in layer_data.items():
            intervention_sizes.append(intervention_key)
            if len(means) == len(hidden_sizes_index) and len(stds) == len(hidden_sizes_index):
                mean_data_for_df[intervention_key] = means
                std_data_for_df[intervention_key] = stds
            else:
                print(f"Warning: Data length mismatch for {intervention_key}. Expected {len(hidden_sizes_index)}, got {len(means)}. Skipping.")

        # Create DataFrames with the defined index
        iia_mean_df = pd.DataFrame(mean_data_for_df, index=hidden_sizes_index)
        iia_std_df = pd.DataFrame(std_data_for_df, index=hidden_sizes_index)

        # Ensure columns are in a consistent order (optional but good practice)
        iia_mean_df = iia_mean_df[intervention_sizes]
        iia_std_df = iia_std_df[intervention_sizes]

        # Create the mapping for x-coordinates (using simple numeric positions)
        x_coords_map = {label: i for i, label in enumerate(hidden_sizes_index)}

        # --- Plotting ---
        # Use the modified plot_results function

        from matplotlib import rc
        rc('text.latex', preamble=r'\usepackage{color}')


        fig, ax = plot_results(
            iia_df=iia_mean_df,
            acc_df=pd.DataFrame(), # No accuracy data in this specific example format
            iia_std_df=iia_std_df, # Pass the standard deviation data
            x_coords_map=x_coords_map,
            x_label="$d_{\\mathrm{rn}}$", # Customize axis label
            y_label="Accuracy", # Customize axis label
            title=f"Results for {target_layer}", # Customize title
            error_bar_alpha=0.25, # Customize alpha
            error_bar_multiplier=1.0, # Plot +/- 1 std dev
            # Add any other plot_results customization parameters as needed
            legend_font_size=18,
            xtick_label_rotation=0,
            figsize=(8, 4.2),
            x_axis_label_pad=10,
            legend_handletextpad=2
        )
        
        # Add a single legend entry for the reference lines
        ax.plot([], [], color='black', linestyle='--', linewidth=2, label='Accuracy')
        
        ax.tick_params(axis='both', which='major', labelsize=24)
        # Set axis label sizes
        ax.set_xlabel(ax.get_xlabel(), fontsize=24)
        ax.set_ylabel(ax.get_ylabel(), fontsize=24)
        
        # Set title size if needed
        # Set x-axis ticks to show only powers of 2 and 1
        power_of_2_ticks = [1, 2, 4, 8, 16]
        ax.set_xticks([el-1 for el in power_of_2_ticks])
                # Add custom legend with image markers
       


        # Adjust horizontal alignment of tick labels to center them properly
        for label in ax.get_xticklabels():
            label.set_ha('center')

        set_plt_settings()
        
        plt.tight_layout()
        plt.draw()
        
        fig.savefig(out_file_name, bbox_inches='tight')
    
        return fig, ax

    else:
        print(f"Error: Data for layer '{target_layer}' not found in results.")


def load_data(algorithm_name, intervention_sizes, hiddensize_range):

    with open(RESULTS_DIR / 'HiddenSizeProgression' / algorithm_name / 'results_1.json') as f:
        results1 = json.load(f)
    with open(RESULTS_DIR / 'HiddenSizeProgression' / algorithm_name / 'results_2.json') as f:
        results2 = json.load(f)
    with open(RESULTS_DIR / 'Standard_DAS/Rotation/FullyTrained' / algorithm_name / 'results.json') as f:
        mean_data = json.load(f)
    mean_data = calculate_mean_std(mean_data)
    results = put_together(results1, results2)
    results = calculate_mean_std(results)
    results = prepare_data(results, layers, intervention_sizes, layers_label, intervention_label, hiddensize_range)
    return results, mean_data



def plot_single_layer_hidden_size_ax(ax, layer_plot_data, plot_config, show_legend_on_ax, show_xlabel_on_ax, show_ylabel_on_ax, title_text, image_for_legend_path: Path):
    """
    Plots data for a single layer onto a given matplotlib Axes object.
    Includes a custom legend with images if show_legend_on_ax is True.
    """
    if not layer_plot_data:
        ax.text(0.5, 0.5, f"No data for\\n{title_text}", ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        if not show_ylabel_on_ax: # if it's not the first plot in a row/col that needs y-label
             ax.set_yticklabels([])
        return

    num_points = len(layer_plot_data[next(iter(layer_plot_data.keys()))][0])
    hidden_sizes_index = [i + 1 for i in range(num_points)]
    
    mean_data_for_df = {}
    std_data_for_df = {}
    intervention_keys_ordered = []

    for intervention_key, (means, stds) in layer_plot_data.items():
        intervention_keys_ordered.append(intervention_key)
        if len(means) == len(hidden_sizes_index) and len(stds) == len(hidden_sizes_index):
            mean_data_for_df[intervention_key] = means
            std_data_for_df[intervention_key] = stds
        else:
            print(f"Warning: Data length mismatch for {intervention_key} in {title_text}. Expected {len(hidden_sizes_index)}, got {len(means)}. Skipping.")
            # Fill with NaNs to maintain DataFrame shape if necessary, or handle error
            mean_data_for_df[intervention_key] = [np.nan] * len(hidden_sizes_index)
            std_data_for_df[intervention_key] = [np.nan] * len(hidden_sizes_index)


    iia_mean_df = pd.DataFrame(mean_data_for_df, index=hidden_sizes_index)
    iia_std_df = pd.DataFrame(std_data_for_df, index=hidden_sizes_index)

    if not intervention_keys_ordered: # No data was valid
        ax.text(0.5, 0.5, f"No valid data series for\\n{title_text}", ha='center', va='center', transform=ax.transAxes)
        return

    iia_mean_df = iia_mean_df[intervention_keys_ordered]
    iia_std_df = iia_std_df[intervention_keys_ordered]

    x_coords_map = {label: i for i, label in enumerate(hidden_sizes_index)}

    # Call plot_results, instructing it to draw on our ax and not to create its own legend
    # We will handle the legend manually to inject images.
    _fig, _ax = plot_results(
        ax=ax,
        iia_df=iia_mean_df,
        acc_df=pd.DataFrame(), # No separate accuracy data series for this plot type
        iia_std_df=iia_std_df,
        x_coords_map=x_coords_map,
        x_label="", # Labels will be set conditionally later
        y_label="", # Labels will be set conditionally later
        show_legend=show_legend_on_ax, # Suppress default legend from plot_results
        error_bar_alpha=plot_config.get("error_bar_alpha", 0.25),
        error_bar_multiplier=plot_config.get("error_bar_multiplier", 1.0),
        legend_font_size=plot_config.get("legend_font_size", 18), # Still passed for other text elements if used
        xtick_label_rotation=plot_config.get("xtick_label_rotation", 0),
        x_axis_label_pad=plot_config.get("x_axis_label_pad", 10),
        legend_handletextpad=plot_config.get("legend_handletextpad", 2), # For internal logic if any
        iia_linewidth=2.5,
    )

    if show_legend_on_ax:
        # This section manually constructs the legend, potentially including an image,
        # if show_legend_on_ax is True. This aligns with the comment indicating manual legend handling.

        # First, if plot_results (or any prior call) created a legend on this ax, remove it.
        # This ensures we don't stack legends or have conflicts.
        current_ax_legend = ax.get_legend()
        # Get the handles and labels from the artists plotted on the axes (e.g., lines from plot_results).
        handles, labels = ax.get_legend_handles_labels()

        # If there are no handles (e.g., no data lines were plotted), then no legend is typically needed.
        if not handles:
            print("No handles found")
            pass
        else:
            # Attempt to load the image for the legend.
            try:
                # Very hacky i'm sorry
                img_arr = mpimg.imread(IMAGE_LEGEND_PATH)
                image_zoom = plot_config.get("legend_image_zoom", 0.2)  # Default zoom, adjust as needed.
                ab = OffsetImage(img_arr, zoom=image_zoom)
                annotation_box = AnchoredOffsetbox(
                    loc='upper right',
                    child=ab,
                    pad=0.0,
                    frameon=False,
                    bbox_to_anchor=(0.88, 0.3),
                    bbox_transform=ax.transAxes,  
                    borderpad=0.0,
                )
                annotation_box.set_zorder(10)
                ax.add_artist(annotation_box)

                # Add three images at the same y but below the main image
                x_base = 0.88  # y position for the row of images (adjust as needed)
                y_positions = [0.15, 0.225, 0.3]  # x positions for the three images

                for i, y in enumerate(y_positions):
                    ab_row = OffsetImage(img_arr, zoom=image_zoom)
                    annotation_box_row = AnchoredOffsetbox(
                        loc='upper right',
                        child=ab_row,
                        pad=0.0,    
                        frameon=False,
                        bbox_to_anchor=(x_base, y),
                        bbox_transform=ax.transAxes,
                        borderpad=0.0,
                    )
                    annotation_box_row.set_zorder(10)
                    ax.add_artist(annotation_box_row)

            except FileNotFoundError:
                print(f"Warning: Legend image {IMAGE_LEGEND_PATH} not found. Drawing legend without image.")
                # Fallback: Draw legend with original handles and labels if image is not found.
                # (Assumes legend properties are defined as above for consistency)
                loc = plot_config.get("legend_loc", "best")
                legend_fontsize = plot_config.get("legend_font_size", 18)
                ncol = plot_config.get("legend_ncol", 1)
                frameon = plot_config.get("legend_frameon", True)
                title = plot_config.get("legend_title", None)
                if handles: # Only draw if there are actual items to show
                    ax.legend(handles=handles, labels=labels, loc=loc, fontsize=legend_fontsize, ncol=ncol, frameon=frameon, title=title)
            except Exception as e:
                print(f"Warning: Could not create custom legend with image due to: {e}. Drawing default legend.")
                # Fallback for other errors during custom legend creation.
                loc = plot_config.get("legend_loc", "best")
                legend_fontsize = plot_config.get("legend_font_size", 18)
                ncol = plot_config.get("legend_ncol", 1)
                frameon = plot_config.get("legend_frameon", True)
                title = plot_config.get("legend_title", None)
                if handles: # Only draw if there are actual items to show
                    ax.legend(handles=handles, labels=labels, loc=loc, fontsize=legend_fontsize, ncol=ncol, frameon=frameon, title=title)
    if show_xlabel_on_ax:
        ax.set_xlabel(plot_config.get("x_label_text", "$d_{\\mathrm{rn}}$"), fontsize=plot_config.get("axis_labelsize", 24))
    else:
        ax.set_xlabel("")

    if show_ylabel_on_ax:
        ax.set_ylabel(plot_config.get("y_label_text", "Accuracy"), fontsize=plot_config.get("axis_labelsize", 24))
    else:
        ax.set_ylabel("")


    ax.set_title(title_text, fontsize=plot_config.get("title_fontsize", 22))
    
    power_of_2_ticks = [1, 2, 4, 8, 16]
    # x_coords_map maps 1-16 to 0-15. Ticks should be at these 0-indexed positions.
    ax.set_xticks([x_coords_map[val] for val in power_of_2_ticks if val in x_coords_map])
    ax.set_xticklabels([str(val) for val in power_of_2_ticks if val in x_coords_map])


    for label in ax.get_xticklabels():
        label.set_ha('center')


def plot_hidden_size_grid(results_data, layer_names_to_plot, out_file_name_suffix, image_path_for_legend: Path):
    """
    Creates a 1xN grid of plots, where N is the number of layers.
    Each subplot shows hidden size progression for a layer, with custom image legends.
    """
    num_rows = 1
    num_cols = len(layer_names_to_plot)
    
    # Adjust figsize: make it wider for 3 plots. (width per plot, height per plot)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4.5 * num_rows), sharey=True)
    
    if num_cols == 1: # Ensure axes is always a list-like structure
        axes = [axes]

    plot_config = {
        "x_label_text": "$d_{\\mathrm{rn}}$",
        "y_label_text": "Accuracy",
        "error_bar_alpha": 0.25,
        "error_bar_multiplier": 1.0,
        "legend_font_size": 15,
        "xtick_label_rotation": 0,
        "x_axis_label_pad": 10,
        "legend_custom_handletextpad": 0.2, # For our custom legend
        "legend_line_width_pts": 15, # New: Nominal width for the line segment in legend item
        "legend_line_height_pts": 10, # New: Nominal height for the line segment area
        "legend_image_zoom": 0.5,
        "legend_image_sep": 1, # pixels between line, image, text in legend item
        "legend_handletextpad": 2,
        "legend_borderpad": 0.3,
        "legend_loc": 'best', # 'upper center'
        # "legend_title": "Intervention Size",
        # "legend_title_fontsize": 13,
        "tick_labelsize": 20,
        "axis_labelsize": 22,
        "title_fontsize": 20,
    }

    for col_idx, layer_name in enumerate(layer_names_to_plot):
        ax = axes[col_idx]
        layer_data = results_data.get(layer_name)

        plot_single_layer_hidden_size_ax(
            ax=ax,
            layer_plot_data=layer_data,
            plot_config=plot_config,
            show_legend_on_ax=(col_idx == num_cols - 1), # Legend only on the last plot
            show_xlabel_on_ax=True, # X-labels on all plots in the bottom row
            show_ylabel_on_ax=(col_idx == 0), # Y-labels only on the first plot of a row
            title_text=layer_name,
            image_for_legend_path=image_path_for_legend
        )

    set_plt_settings() 
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout; rect might need tuning top for suptitle
    # fig.suptitle(f"MLP Hidden Size Progression ({out_file_name_suffix.split('_')[0]})", fontsize=24, y=0.99)


    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    full_out_file_name = PLOTS_DIR / f"mlp_hidden_size_grid_{out_file_name_suffix}.pdf"
    fig.savefig(full_out_file_name, bbox_inches='tight')
    print(f"Saved grid plot to {full_out_file_name}")
    plt.show()

def plot(algorithm_name, intervention_size, hidden_size_range):
    results, mean_data=load_data(algorithm_name, intervention_size, hidden_size_range)
    make_plot_single_layer(results,mean_data,"Layer 1",intervention_sizes,f"plots/mlp/hidden_size_{algorithm_name}.pdf")
    plot_hidden_size_grid(
        results_data=results,
        layer_names_to_plot=layers_label, # ["Layer 1", "Layer 2", "Layer 3"]
        out_file_name_suffix=algorithm_name, # Suffix for the output PDF
        image_path_for_legend=IMAGE_LEGEND_PATH
    )

# %%
layers = ["Layer1", "Layer2", "Layer3"]
intervention_sizes_double_intervention =   ['[[0], [1]]',
                '[[0, 1], [2, 3]]',
                '[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]'
                ]
intervention_sizes_single_intervention =   ['[[0]]',
                '[[0, 1]]',
                '[[0, 1, 2, 3, 4, 5, 6, 7]]'
                ]        
layers_label=["Layer 1","Layer 2","Layer 3"]
intervention_label=["\qquad $=1$","\qquad $=2$","\qquad $=8$"]
hiddensize_range=list(range(1,17))

# %%
plot("Both_Equality_Relations", intervention_sizes_double_intervention, hiddensize_range)
#%%
plot("Identity_of_First_Argument", intervention_sizes_single_intervention, hiddensize_range)
#%%
plot("Left_Equality_Relation", intervention_sizes_double_intervention, hiddensize_range)
#make_plot_single_layer(results,mean_data,"Layer 1",intervention_sizes,"plots/mlp/Identity_of_Second_Argument_Layer1.pdf")
# %%
intervention_sizes =   ['[[0], [1]]',
                        '[[0, 1], [2, 3]]',
                        '[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]'
                        ]
hiddensize_range=list(range(1,25))
# %%
plot("AndOr", intervention_sizes, hiddensize_range)
# %%
plot("AndOrAnd", intervention_sizes, hiddensize_range)
# %%
