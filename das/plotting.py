from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re # Import re for revision parsing
from pathlib import Path

PLOTS_DIR = Path("plots")

def get_color_palette(num_colors, inject_at_start=None):
    colors=[]
    if inject_at_start is not None:
        colors.append(inject_at_start)
    start_color=(52, 225, 235)
    end_color=(5, 7, 66)
    #start_color=(0, 255, 0)
    #end_color=(255, 0, 0)
    training_steps=num_colors
    for i in range(training_steps):
        fac=i/(training_steps-1)
        R=start_color[0]+fac*(end_color[0]-start_color[0])
        G=start_color[1]+fac*(end_color[1]-start_color[1])
        B=start_color[2]+fac*(end_color[2]-start_color[2])
        colors.append((R/255,G/255,B/255))
    return colors

def plot_results(
    iia_df,
    acc_df,
    x_coords_map,
    *,
    iia_std_df=None,  # New: DataFrame for IIA standard deviations
    error_bar_alpha=0.2, # New: Alpha for error shading
    error_bar_multiplier=1.0, # New: Multiplier for std dev range
    x_label="Model Training Step",
    y_label="Accuracy",
    title="Results Plot",
    iia_line_label="Intervention Model",
    acc_marker_label="Model Accuracy",
    acc_marker_color=None,
    acc_marker_shape="D",
    acc_marker_size=10,
    iia_marker_shape="o",
    iia_marker_size=8,
    legend_font_size=20,
    iia_linewidth=2,
    legend_loc_initial="upper left",
    legend_bbox_initial=(1.05, 1),
    legend_loc_final="lower right",
    legend_bbox_final=(0.98, 0.02),
    legend_facecolor="white",
    legend_edgecolor="black",
    legend_frameon=True,
    xtick_label_replace=None,  # e.g., lambda s: s.replace("step", "Step ")
    xtick_label_rotation=45,
    xtick_label_fontsize=12,
    ytick_label_fontsize=12,
    xlabel_fontsize=14,
    ylabel_fontsize=14,
    grid_linestyle="--",
    grid_alpha=0.7,
    figsize=(14, 8),
    style="seaborn-v0_8-whitegrid",
    revision_aware=False,
    revision_pattern_str=r"^(.*?) ?\(([^)]+)\)$", # Base name (revision_tag)
    revision_linestyles=None, # Default: ['-', '--', ':', '-.'] if revision_aware
    acc_legend_marker_color_override=None, # e.g., "grey" for neutral legend diamond
    inject_color_at_start=None,
    legend_background_alpha=0.8,
    x_axis_label_pad=None,
    y_axis_label_pad=None,
):
    """
    Plots IIA and accuracy results with extensive customization via parameters.

    Parameters
    ----------
    iia_df : pd.DataFrame
        DataFrame of IIA (or other metric) values.
    acc_df : pd.DataFrame
        DataFrame of accuracy (or other metric) values.
    x_coords_map : dict
        Mapping from index labels to numeric x-axis positions.
    iia_std_df : pd.DataFrame, optional
        DataFrame of standard deviations corresponding to iia_df. If provided,
        shaded error bars (mean +/- multiplier * std) will be plotted for IIA lines.
    error_bar_alpha : float, optional
        Alpha transparency for the shaded error region (default is 0.2).
    error_bar_multiplier : float, optional
        Multiplier for the standard deviation to determine the error band width
        (default is 1.0, i.e., mean +/- 1*std).
    All other parameters are for plot customization.
    """
    if iia_df.empty and acc_df.empty:
        print("Both IIA and Accuracy DataFrames are empty. No plot will be generated.")
        return
    if iia_df.empty:
        print("IIA DataFrame is empty. Plotting accuracies only if available.")
    if acc_df.empty:
        print("Accuracy DataFrame is empty. Plotting IIA only if available.")

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)
    plotted_any_diamonds = False

    # --- Revision Handling Setup ---
    _effective_revision_linestyles = revision_linestyles
    if revision_aware and _effective_revision_linestyles is None:
        _effective_revision_linestyles = ['-', '--', ':', '-.']
    
    parsed_column_details = {} # {original_col_name: (base_name, revision_tag)}
    
    all_potential_columns = set()
    if not iia_df.empty:
        all_potential_columns.update(iia_df.columns)
    if not acc_df.empty:
        all_potential_columns.update(acc_df.columns)
    sorted_all_columns = sorted(list(all_potential_columns))

    for col_name in sorted_all_columns:
        base_name_for_col = col_name
        revision_tag_for_col = None
        if revision_aware:
            match = re.match(revision_pattern_str, col_name)
            if match:
                base_name_for_col = match.group(1).strip()
                revision_tag_for_col = match.group(2)
        parsed_column_details[col_name] = (base_name_for_col, revision_tag_for_col)

    # Color and linestyle assignment for IIA lines
    iia_line_attributes = {} # {original_iia_col_name: {'color': color, 'linestyle': linestyle}}
    
    if not iia_df.empty:
        iia_columns_to_plot_ordered = list(iia_df.columns) # Use df's column order for legend consistency
        num_iia_lines = len(iia_columns_to_plot_ordered)
        iia_color_palette = get_color_palette(num_iia_lines, inject_at_start=inject_color_at_start)

        if revision_aware:
            # Group IIA columns by base name to assign linestyles systematically for revisions of the same base
            iia_base_to_revs_map = {}
            for iia_col_name in iia_columns_to_plot_ordered: # Process in original order
                base_name, rev_tag = parsed_column_details.get(iia_col_name, (iia_col_name, None))
                if base_name not in iia_base_to_revs_map:
                    iia_base_to_revs_map[base_name] = []
                iia_base_to_revs_map[base_name].append({'name': iia_col_name, 'tag': rev_tag})
            
            # Sort revisions within each base for consistent linestyle assignment (e.g., by tag name)
            for base_name_key in iia_base_to_revs_map:
                iia_base_to_revs_map[base_name_key].sort(key=lambda x: x['tag'] if x['tag'] is not None else '')

            # Assign unique color to each IIA line, and cycled linestyle within a base group
            current_color_index = 0
            # Iterate through original iia_df.columns to maintain order for color assignment
            # but use the grouped map to find the rev_idx for linestyle.
            
            # Create a lookup for rev_idx within a base_name
            rev_idx_lookup = {}
            for base_name_key, revs_list in iia_base_to_revs_map.items():
                for idx, rev_item in enumerate(revs_list):
                    rev_idx_lookup[rev_item['name']] = idx


            for iia_col_name in iia_columns_to_plot_ordered:
                color_for_line = iia_color_palette[current_color_index % len(iia_color_palette)] if iia_color_palette else 'black'
                
                # Determine linestyle
                # base_name_of_col, _ = parsed_column_details.get(iia_col_name, (iia_col_name, None))
                rev_idx_for_linestyle = rev_idx_lookup.get(iia_col_name, 0)
                linestyle_for_line = _effective_revision_linestyles[rev_idx_for_linestyle % len(_effective_revision_linestyles)]
                
                iia_line_attributes[iia_col_name] = {
                    'color': color_for_line,
                    'linestyle': linestyle_for_line
                }
                current_color_index += 1
        else: # Not revision_aware: each IIA line gets a unique color and solid linestyle
            for i, iia_col_name in enumerate(iia_columns_to_plot_ordered):
                color_val = iia_color_palette[i % len(iia_color_palette)] if iia_color_palette else 'black'
                iia_line_attributes[iia_col_name] = {
                    'color': color_val,
                    'linestyle': '-' 
                }
    # --- End Revision Handling Setup ---

    # Plot accuracy markers (diamonds or other shape)
    if not acc_df.empty:
        for model_name_col_acc in acc_df.columns: 

            current_marker_color = None
            if acc_marker_color is None and model_name_col_acc in iia_line_attributes:
                    current_marker_color = iia_line_attributes[model_name_col_acc]['color']
            else:
                current_marker_color = acc_marker_color

            for x_label_str in acc_df.index:
                if x_label_str in x_coords_map and pd.notna(acc_df.loc[x_label_str, model_name_col_acc]):
                    numeric_x_coord = x_coords_map[x_label_str]
                    accuracy_val = acc_df.loc[x_label_str, model_name_col_acc]

                    plot_this_marker = False
                    if iia_df.empty:
                        plot_this_marker = True
                    # Check if the *exact* column (potentially with revision) exists in IIA for this point
                    elif model_name_col_acc in iia_df.columns and \
                         x_label_str in iia_df.index and \
                         pd.notna(iia_df.loc[x_label_str, model_name_col_acc]):
                        plot_this_marker = True
                    # Fallback for non-revision aware mode, or if IIA only has base name
                    elif not revision_aware and base_name_for_col in iia_df.columns and \
                         x_label_str in iia_df.index and \
                         pd.notna(iia_df.loc[x_label_str, base_name_for_col]):
                         plot_this_marker = True


                    if plot_this_marker:
                        ax.plot(
                            numeric_x_coord,
                            accuracy_val,
                            acc_marker_shape,
                            color=current_marker_color, # Use determined color
                            markersize=acc_marker_size,
                            label='_nolegend_',
                            zorder=1,
                        )
                        plotted_any_diamonds = True

    # Plot IIA lines
    if not iia_df.empty:
        # Iterate through columns in their original order for consistent legend
        for original_iia_col_name in iia_df.columns:
            if original_iia_col_name not in iia_line_attributes:
                # This case should ideally not be hit if setup is correct
                print(f"Warning: IIA column '{original_iia_col_name}' not found in attribute map. Skipping.")
                continue

            attrs = iia_line_attributes[original_iia_col_name]
            line_color = attrs['color']
            linestyle_for_line = attrs['linestyle']

            # Prepare lists to store plot data points for this line
            plot_x_coords = []
            plot_y_means = []
            plot_y_stds = [] # Store std devs corresponding to means

            for x_label_str_iia in iia_df.index:
                if x_label_str_iia in x_coords_map:
                    numeric_x_iia = x_coords_map[x_label_str_iia]
                    y_val_iia = iia_df.loc[x_label_str_iia, original_iia_col_name]

                    # Only add points if the mean value is valid
                    if pd.notna(y_val_iia):
                        plot_x_coords.append(numeric_x_iia)
                        plot_y_means.append(y_val_iia)

                        # Check for corresponding std dev if requested
                        std_val = np.nan # Default to NaN if no std data
                        if iia_std_df is not None and \
                           original_iia_col_name in iia_std_df.columns and \
                           x_label_str_iia in iia_std_df.index:
                            std_val_candidate = iia_std_df.loc[x_label_str_iia, original_iia_col_name]
                            if pd.notna(std_val_candidate):
                                std_val = std_val_candidate
                        plot_y_stds.append(std_val)


            if plot_x_coords: # Proceed only if there are valid points to plot
                plot_x_np = np.array(plot_x_coords)
                plot_y_means_np = np.array(plot_y_means)

                # Plot shaded error bars if std data is available and valid
                if iia_std_df is not None:
                    plot_y_stds_np = np.array(plot_y_stds)
                    valid_std_mask = ~np.isnan(plot_y_stds_np)

                    if np.any(valid_std_mask): # Check if there's any valid std data
                        upper_bound = plot_y_means_np + error_bar_multiplier * plot_y_stds_np
                        lower_bound = plot_y_means_np - error_bar_multiplier * plot_y_stds_np

                        # Plot shaded region *first* so line is on top
                        # fill_between handles NaNs by creating gaps in the shading
                        ax.fill_between(
                            plot_x_np, lower_bound, upper_bound,
                            color=line_color,
                            alpha=error_bar_alpha,
                            linewidth=0, # No lines for the boundary of the fill
                            zorder=1.5 # Below line markers but above grid
                        )
                    # Optionally print a warning if std_df was provided but no valid std values were found for this line
                    # elif error_bar_multiplier > 0: # Only warn if user intended to plot errors
                    #    print(f"Warning: No valid standard deviation data found for column '{original_iia_col_name}' corresponding to its mean values. Skipping error bars for this line.")


                # Plot the main line (mean values)
                ax.plot(
                    plot_x_np,
                    plot_y_means_np,
                    marker=iia_marker_shape,
                    linestyle=linestyle_for_line, # Use determined linestyle
                    label=original_iia_col_name,
                    color=line_color, # Use determined color
                    linewidth=iia_linewidth,
                    markersize=iia_marker_size,
                    zorder=2, # Ensure line is plotted above fill_between and markers above diamonds
                )

    ax.set_xlabel(x_label, fontsize=xlabel_fontsize)
    ax.set_ylabel(y_label, fontsize=ylabel_fontsize)

    handles, labels = ax.get_legend_handles_labels()
    # Add a legend entry for accuracy markers (diamonds) if any were plotted
    if plotted_any_diamonds:
        # Determine the color for the legend entry
        legend_diamond_color = acc_legend_marker_color_override if acc_legend_marker_color_override is not None else acc_marker_color
        
        # Create a custom Line2D object for the legend
        diamond_legend_entry = Line2D(
            [0], [0],
            marker=acc_marker_shape,
            color=legend_diamond_color,
            label=acc_marker_label,
            linestyle='None',
            markersize=acc_marker_size,
        )
        
        # Only add the legend entry if it's not already present
        if acc_marker_label not in labels:
            print(f"Adding diamond legend entry for {acc_marker_label}")
            # Prepend to handles and labels to ensure it appears first in the legend
            handles.insert(0, diamond_legend_entry)
            labels.insert(0, acc_marker_label)

    # Initial legend (for layout)
    if handles:
        legend = ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=legend_bbox_initial,
            loc=legend_loc_initial,
            borderaxespad=0.,
            fontsize=legend_font_size,
        )
    else:
        legend = None

    # Set custom x-ticks
    if x_coords_map:
        sorted_tick_positions = sorted(x_coords_map.values())
        pos_to_label_map = {v: k for k, v in x_coords_map.items()}
        if xtick_label_replace is not None:
            sorted_tick_labels = [xtick_label_replace(pos_to_label_map[pos]) for pos in sorted_tick_positions]
        else:
            sorted_tick_labels = [pos_to_label_map[pos] for pos in sorted_tick_positions]
        ax.set_xticks(sorted_tick_positions)
        ax.set_xticklabels(sorted_tick_labels, rotation=xtick_label_rotation, ha="right", fontsize=xtick_label_fontsize)
    else:
        plt.xticks(rotation=xtick_label_rotation, ha="right", fontsize=xtick_label_fontsize)

    plt.yticks(fontsize=ytick_label_fontsize)
    ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

    # Move and restyle the legend to the final position
    if handles:
        if 'legend' in locals() and legend is not None and legend.axes is not None:
            legend.remove()
        
        final_legend = ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=legend_bbox_final,
            loc=legend_loc_final,
            borderaxespad=0.5,
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
            frameon=legend_frameon,
            fontsize=legend_font_size,
        )
        if final_legend:
            frame = final_legend.get_frame()
            frame.set_linewidth(1.0)
            frame.set_alpha(1.0)

    if x_axis_label_pad is not None:
        ax.xaxis.labelpad = x_axis_label_pad
    if y_axis_label_pad is not None:
        ax.yaxis.labelpad = y_axis_label_pad
    ax.get_legend().get_frame().set_alpha(legend_background_alpha)
    return fig, ax


def set_plt_settings(font_size=None, legend_font_size=None):
    # Set up matplotlib with LaTeX fonts and larger font size
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    if legend_font_size is not None:
        plt.rcParams.update({
            "legend.fontsize": legend_font_size,
        })
    if font_size is not None:
        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
        })
