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
    ax: plt.Axes = None, # New: For plotting on existing axes
    show_legend: bool = True, # New: To control legend visibility
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
    legend_label_spacing=0.2, 
    legend_ncol=1,
    legend_columnspacing=1.0,
    legend_handlelength=2.0,
    legend_handletextpad=0.8,
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
    iia_base_name_color_map: dict | None = None,
    iia_base_name_marker_map: dict | None = None,
    acc_revision_marker_map: dict | None = None,
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

    _ax_passed_in = ax is not None
    if not _ax_passed_in:
        plt.style.use(style) # Apply style only when creating a new figure
        fig, current_ax = plt.subplots(figsize=figsize)
    else:
        current_ax = ax
        fig = current_ax.get_figure()

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
    columns_to_parse = list(all_potential_columns)

    for col_name in columns_to_parse:
        base_name_for_col = col_name
        revision_tag_for_col = None
        if revision_aware:
            match = re.match(revision_pattern_str, col_name)
            if match:
                base_name_for_col = match.group(1).strip()
                revision_tag_for_col = match.group(2)
        parsed_column_details[col_name] = (base_name_for_col, revision_tag_for_col)

    # Color, linestyle and marker assignment for IIA lines
    iia_line_attributes = {} # {original_iia_col_name: {'color': color, 'linestyle': linestyle, 'marker': marker}}
    
    if not iia_df.empty:
        iia_columns_to_plot_ordered = list(iia_df.columns) # Use df's column order for legend consistency
        
        unique_iia_base_names = []
        seen_bases = set()
        for col in iia_columns_to_plot_ordered:
            base, _ = parsed_column_details.get(col, (col,None))
            if base not in seen_bases:
                unique_iia_base_names.append(base)
                seen_bases.add(base)

        _iia_color_assignments = {}
        _iia_marker_assignments = {}
        
        # Assign colors to base names
        unmapped_base_names_for_color = [bn for bn in unique_iia_base_names if not (iia_base_name_color_map and bn in iia_base_name_color_map)]
        fallback_iia_color_palette = []
        if unmapped_base_names_for_color:
            fallback_iia_color_palette = get_color_palette(len(unmapped_base_names_for_color), inject_at_start=inject_color_at_start)
        
        fallback_color_idx = 0
        for bn in unique_iia_base_names:
            if iia_base_name_color_map and bn in iia_base_name_color_map:
                _iia_color_assignments[bn] = iia_base_name_color_map[bn]
            elif fallback_iia_color_palette and bn in unmapped_base_names_for_color : # check bn in unmapped.. ensures correct indexing
                _iia_color_assignments[bn] = fallback_iia_color_palette[fallback_color_idx % len(fallback_iia_color_palette)]
                fallback_color_idx += 1
            else:
                _iia_color_assignments[bn] = 'black' # Ultimate fallback

        # Assign markers to base names
        for bn in unique_iia_base_names:
            if iia_base_name_marker_map and bn in iia_base_name_marker_map:
                _iia_marker_assignments[bn] = iia_base_name_marker_map[bn]
            else:
                _iia_marker_assignments[bn] = iia_marker_shape # Global default

        if revision_aware:
            iia_base_to_revs_map = {}
            for iia_col_name in iia_columns_to_plot_ordered:
                base_name, rev_tag = parsed_column_details.get(iia_col_name, (iia_col_name, None))
                if base_name not in iia_base_to_revs_map:
                    iia_base_to_revs_map[base_name] = []
                iia_base_to_revs_map[base_name].append({'name': iia_col_name, 'tag': rev_tag})
            
            for base_name_key in iia_base_to_revs_map:
                iia_base_to_revs_map[base_name_key].sort(key=lambda x: (str(x['tag'] is None), x['tag'] if x['tag'] is not None else ''))

            rev_idx_lookup = {}
            for base_name_key, revs_list in iia_base_to_revs_map.items():
                for idx, rev_item in enumerate(revs_list):
                    rev_idx_lookup[rev_item['name']] = idx

            for iia_col_name in iia_columns_to_plot_ordered:
                base_name_of_col, _ = parsed_column_details.get(iia_col_name, (iia_col_name, None))
                color_for_line = _iia_color_assignments.get(base_name_of_col, 'black')
                marker_for_line = _iia_marker_assignments.get(base_name_of_col, iia_marker_shape)
                
                rev_idx_for_linestyle = rev_idx_lookup.get(iia_col_name, 0)
                linestyle_for_line = _effective_revision_linestyles[rev_idx_for_linestyle % len(_effective_revision_linestyles)]
                
                iia_line_attributes[iia_col_name] = {
                    'color': color_for_line,
                    'linestyle': linestyle_for_line,
                    'marker': marker_for_line
                }
        else: # Not revision_aware
            for i, iia_col_name in enumerate(iia_columns_to_plot_ordered):
                base_name_of_col, _ = parsed_column_details.get(iia_col_name, (iia_col_name, None))
                color_val = _iia_color_assignments.get(base_name_of_col, 'black')
                marker_val = _iia_marker_assignments.get(base_name_of_col, iia_marker_shape)
                iia_line_attributes[iia_col_name] = {
                    'color': color_val,
                    'linestyle': '-',
                    'marker': marker_val
                }
    # --- End Revision Handling Setup ---

    # Plot accuracy markers
    if not acc_df.empty:
        for model_name_col_acc in acc_df.columns:
            # Determine color for accuracy markers - should be from acc_marker_color (e.g. "red")
            current_acc_marker_color = acc_marker_color if acc_marker_color is not None else 'gray' # Fallback color for marker itself

            # Determine marker shape for accuracy point
            _base_name_acc, rev_tag_acc = parsed_column_details.get(model_name_col_acc, (model_name_col_acc, None))
            current_acc_marker_shape = acc_marker_shape # Default
            if acc_revision_marker_map and rev_tag_acc in acc_revision_marker_map:
                current_acc_marker_shape = acc_revision_marker_map[rev_tag_acc]
            
            # Check if this accuracy column has a corresponding IIA column to decide if it should be plotted
            # This logic is complex because an IIA column might exist for a different revision or not at all
            # For now, let's assume if data is in acc_df, we try to plot it.
            # The original logic tried to link it to an existing IIA point, which might be too restrictive.

            for x_label_str in acc_df.index:
                if x_label_str in x_coords_map and pd.notna(acc_df.loc[x_label_str, model_name_col_acc]):
                    numeric_x_coord = x_coords_map[x_label_str]
                    accuracy_val = acc_df.loc[x_label_str, model_name_col_acc]
                    
                    current_ax.plot(
                        numeric_x_coord,
                        accuracy_val,
                        marker=current_acc_marker_shape, # Use determined shape
                        color=current_acc_marker_color, # Use specified acc_marker_color
                        markersize=acc_marker_size,
                        label='_nolegend_', # Individual points don't get legend entries here
                        linestyle='None', # Ensure no line connects accuracy markers unless intended
                        zorder=3, # Above IIA lines
                    )
                    plotted_any_diamonds = True # Generic flag that some acc markers were plotted

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
            marker_for_line = attrs['marker']

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

                        current_ax.fill_between(
                            plot_x_np, lower_bound, upper_bound,
                            color=line_color,
                            alpha=error_bar_alpha,
                            linewidth=0, # No lines for the boundary of the fill
                            zorder=1.5 # Below line markers but above grid
                        )

                current_ax.plot(
                    plot_x_np,
                    plot_y_means_np,
                    marker=marker_for_line,
                    linestyle=linestyle_for_line, # Use determined linestyle
                    label=original_iia_col_name,
                    color=line_color, # Use determined color
                    linewidth=iia_linewidth,
                    markersize=iia_marker_size,
                    zorder=2, # Ensure line is plotted above fill_between and markers above diamonds
                )

    current_ax.set_xlabel(x_label, fontsize=xlabel_fontsize)
    current_ax.set_ylabel(y_label, fontsize=ylabel_fontsize)

    handles, labels = current_ax.get_legend_handles_labels()
    
    # Store original handles and labels before modifying for accuracy legend
    existing_handles = list(handles)
    existing_labels = list(labels)
    handles, labels = [], [] # Reset for new legend order

    # Add legend entries for accuracy markers (if any were plotted)
    if plotted_any_diamonds:
        legend_acc_color = acc_legend_marker_color_override if acc_legend_marker_color_override is not None else (acc_marker_color if acc_marker_color is not None else 'red')
        
        if acc_revision_marker_map:
            # Create legend entries for each revision defined in the map that had data
            # Need to know which rev_tags were actually present in plotted acc_df
            plotted_acc_rev_tags_with_data = set()
            if not acc_df.empty:
                for acc_col_name_iter in acc_df.columns:
                    _b, rev_tag_iter = parsed_column_details.get(acc_col_name_iter, (None, None))
                    if rev_tag_iter and rev_tag_iter in acc_revision_marker_map:
                        # Check if this column actually has any valid data points
                        if any(pd.notna(acc_df.loc[idx_val, acc_col_name_iter]) for idx_val in acc_df.index if idx_val in x_coords_map):
                            plotted_acc_rev_tags_with_data.add(rev_tag_iter)
            
            # Sort them for consistent legend order, e.g., "Full" then "Init."
            # A more robust sort might be needed if tags are arbitrary. Simple alphabetical for now.
            sorted_tags_for_legend = sorted(list(plotted_acc_rev_tags_with_data), key=lambda t: (str(t).lower() != "full", str(t).lower() != "init.", t))


            for rev_tag in sorted_tags_for_legend:
                marker_shape_for_legend = acc_revision_marker_map[rev_tag]
                label_text = f"{acc_marker_label} ({rev_tag})"
                # Check if this label is already effectively covered by existing_handles/labels
                # This check is tricky if existing_labels are like "non-linear (Full)" from IIA
                # For now, assume acc legend entries are distinct.
                if label_text not in labels: # Avoid adding duplicate legend entries for accuracy
                    entry = Line2D([0], [0], marker=marker_shape_for_legend, color=legend_acc_color,
                                   label=label_text, linestyle='None', markersize=acc_marker_size)
                    handles.append(entry)
                    labels.append(label_text)
        else: # Single accuracy legend entry
            label_text = acc_marker_label
            if label_text not in labels:
                 entry = Line2D([0], [0], marker=acc_marker_shape, color=legend_acc_color,
                               label=label_text, linestyle='None', markersize=acc_marker_size)
                 handles.append(entry)
                 labels.append(label_text)

    # Add back the original IIA legend entries
    for handle, label_text in zip(existing_handles, existing_labels): # Renamed 'label' to 'label_text'
        if label_text not in labels: # Avoid duplicates if IIA lines somehow got into acc legend names
            handles.append(handle)
            labels.append(label_text)

    _created_legend_object = None # To track the legend object

    if show_legend and handles:
        # Initial legend (for layout purposes, then removed)
        initial_legend_obj = current_ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=legend_bbox_initial,
            loc=legend_loc_initial,
            borderaxespad=0.,
            fontsize=legend_font_size,
            labelspacing=legend_label_spacing,
        )
        if initial_legend_obj: # If a legend was actually created
             initial_legend_obj.remove()
        
        # Final legend drawing
        final_legend_obj = current_ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=legend_bbox_final,
            loc=legend_loc_final,
            borderaxespad=0.5,
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
            frameon=legend_frameon,
            fontsize=legend_font_size,
            labelspacing=legend_label_spacing,
            ncol=legend_ncol,
            columnspacing=legend_columnspacing,
            handlelength=legend_handlelength,
            handletextpad=legend_handletextpad,
        )
        if final_legend_obj:
            frame = final_legend_obj.get_frame()
            frame.set_linewidth(1.0)
            frame.set_alpha(1.0) # Original behavior: opaque frame initially
            _created_legend_object = final_legend_obj
            
    elif not show_legend and current_ax.get_legend() is not None: # If not showing, ensure no legend exists
        current_ax.get_legend().remove()


    # Set custom x-ticks
    if x_coords_map:
        sorted_tick_positions = sorted(x_coords_map.values())
        pos_to_label_map = {v: k for k, v in x_coords_map.items()}
        if xtick_label_replace is not None:
            sorted_tick_labels = [xtick_label_replace(pos_to_label_map[pos]) for pos in sorted_tick_positions]
        else:
            sorted_tick_labels = [pos_to_label_map[pos] for pos in sorted_tick_positions]
        current_ax.set_xticks(sorted_tick_positions)
        current_ax.set_xticklabels(sorted_tick_labels, rotation=xtick_label_rotation, ha="right", fontsize=xtick_label_fontsize)
    else:
        # Fallback if no x_coords_map, apply rotation and fontsize to existing ticks
        current_ax.tick_params(axis='x', labelsize=xtick_label_fontsize, labelrotation=xtick_label_rotation)


    current_ax.tick_params(axis='y', labelsize=ytick_label_fontsize) # Ensure ytick_label_fontsize is applied
    current_ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)


    if x_axis_label_pad is not None:
        current_ax.xaxis.labelpad = x_axis_label_pad
    if y_axis_label_pad is not None:
        current_ax.yaxis.labelpad = y_axis_label_pad
    
    if _created_legend_object: # If a legend was created and is intended to be shown
        _created_legend_object.get_frame().set_alpha(legend_background_alpha) # Apply final background alpha

    return fig, current_ax


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
