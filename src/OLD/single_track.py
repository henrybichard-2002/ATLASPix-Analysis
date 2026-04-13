import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union
# Import Rectangle for drawing borders
from matplotlib.patches import Rectangle 

def plot_cluster(data: Dict[str, np.ndarray], 
                 cluster_ids: Union[int, List[int]], 
                 plot_on_single_figure: bool = False):
    """
    Plots a heatmap of cluster data.

    Can either plot each cluster on its own figure (default) or
    plot all requested clusters on a single figure with colored borders.

    Args:
        data (Dict[str, np.ndarray]): Input data dictionary with keys 
            'ClusterID', 'Column', 'Row', 'ToT', 'TS'.
        cluster_ids (Union[int, List[int]]): A single Cluster ID or a list of IDs to plot.
        plot_on_single_figure (bool): If True, plots all clusters on one
            figure. If False (default), creates a new figure for each cluster.
    """

    if 'ClusterID' not in data:
        raise KeyError("Input data dictionary must contain a 'ClusterID' key.")

    # For convenience, if a single int is passed, convert it to a list
    if isinstance(cluster_ids, int):
        cluster_ids = [cluster_ids]

    # --- Check if any of the requested cluster IDs exist at all ---
    master_mask = np.isin(data['ClusterID'], cluster_ids)
    if not np.any(master_mask):
        print(f"Skipping: None of the requested Cluster IDs {cluster_ids} were found.")
        return

    # --- OPTION 1: Plot all clusters on a single figure ---
    if plot_on_single_figure:
        
        # 1. Get data for ALL requested clusters and find global bounds
        all_data = {key: value[master_mask] for key, value in data.items()}
        min_col, max_col = all_data['Column'].min(), all_data['Column'].max()
        min_row, max_row = all_data['Row'].min(), all_data['Row'].max()

        grid_shape = (max_row - min_row + 1, max_col - min_col + 1)
        tot_grid = np.full(grid_shape, np.nan)
        
        # 2. Create the single figure and axes
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get a colormap for the borders (up to 10 distinct colors)
        cmap = plt.cm.get_cmap('tab10') 

        # 3. Loop through each cluster to populate grid and add borders
        for i, cluster_id in enumerate(cluster_ids):
            mask = data['ClusterID'] == cluster_id
            if not np.any(mask):
                print(f"Skipping: Cluster ID {cluster_id} not found.")
                continue

            cluster_data = {key: value[mask] for key, value in data.items()}
            cols = cluster_data['Column']
            rows = cluster_data['Row']
            tots = cluster_data['ToT']
            timestamps = cluster_data['TS']
            
            # Get a unique color for this cluster
            color = cmap(i % 10) # Cycle through the 10 colors
            
            label_set = False # Flag to add legend label only once per cluster

            for j in range(len(cols)):
                # Populate the heatmap grid
                grid_y = rows[j] - min_row
                grid_x = cols[j] - min_col
                tot_grid[grid_y, grid_x] = tots[j]

                # Add timestamp text
                text_color = 'black' if tots[j] > 100 else 'white'
                ax.text(cols[j], rows[j], f"{timestamps[j]}",
                        ha="center", va="center", color=text_color, fontsize=8)

                # Add colored border
                current_label = f"Cluster {cluster_id}" if not label_set else None
                rect = Rectangle((cols[j] - 0.5, rows[j] - 0.5), 1, 1,
                                 edgecolor=color,
                                 facecolor='none',
                                 linewidth=2,
                                 label=current_label)
                ax.add_patch(rect)
                label_set = True # Ensure label is only set for the first hit

        # 4. Draw the heatmap (after grid is fully populated)
        im = ax.imshow(tot_grid, cmap='cividis', origin='lower',
                       vmin=0, vmax=255,
                       extent=[min_col - .5, max_col + .5, min_row - .5, max_row + .5])

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('ToT (Time over Threshold)')
        
        ax.set_aspect(1/3) # y/x aspect ratio
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(f'Heatmap for Clusters: {cluster_ids}')
        
        ax.set_xticks(np.arange(min_col-1, max_col + 2, 1))
        ax.set_yticks(np.arange(min_row-1, max_row + 2, 1))
        ax.set_xticks(np.arange(min_col - 1.5, max_col + 2.5, 1), minor=True)
        ax.set_yticks(np.arange(min_row - 1.5, max_row + 2.5, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.grid(which='major', visible=False)
        
        # 5. Add the legend and show the single plot
        ax.legend()
        plt.tight_layout()
        plt.show()

    else:
        for cluster_id in cluster_ids:
            # 1. Isolate the data for the current cluster
            mask = data['ClusterID'] == cluster_id
            
            if not np.any(mask):
                print(f"Skipping: Cluster ID {cluster_id} not found in the dataset.")
                continue # Move to the next ID in the list

            print(f"Plotting Cluster ID: {cluster_id}...")
            cluster_data = {key: value[mask] for key, value in data.items()}

            # 2. Prepare a 2D grid for the heatmap
            cols = cluster_data['Column']
            rows = cluster_data['Row']
            tots = cluster_data['ToT']
            timestamps = cluster_data['TS']

            min_col, max_col = cols.min(), cols.max()
            min_row, max_row = rows.min(), rows.max()
            
            grid_shape = (max_row - min_row + 1, max_col - min_col + 1)
            tot_grid = np.full(grid_shape, np.nan)

            # Populate the grid with ToT data
            for i in range(len(cols)):
                grid_y = rows[i] - min_row
                grid_x = cols[i] - min_col
                tot_grid[grid_y, grid_x] = tots[i]

            # 3. Create a new plot for each cluster
            fig, ax = plt.subplots(figsize=(10, 8))

            im = ax.imshow(tot_grid, cmap='cividis', origin='lower',
                           vmin=0, vmax=255,
                           extent=[min_col - .5, max_col + .5, min_row - .5, max_row + .5])
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('ToT (Time over Threshold)')
            
            ax.set_aspect(1/3)

            # 4. Label each cell with its TS value
            for i in range(len(cols)):
                # Text color logic from your original code
                text_color = 'black' if tots[i] > 100 else 'white'
                ax.text(cols[i], rows[i], f"{timestamps[i]}",
                        ha="center", va="center", color=text_color, fontsize=8)

            # 5. Configure axes and titles
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            ax.set_title(f'Heatmap for Cluster ID: {cluster_id}')
            
            ax.set_xticks(np.arange(min_col-1, max_col + 2, 1))
            ax.set_yticks(np.arange(min_row-1, max_row + 2, 1))

            ax.set_xticks(np.arange(min_col - 1.5, max_col + 2.5, 1), minor=True)
            ax.set_yticks(np.arange(min_row - 1.5, max_row + 2.5, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            ax.grid(which='major', visible=False)

            # 6. Show the plot (one for each loop iteration)
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.show()
            
            
            
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_clusters_by_length(cluster_data, target_lengths, row_col_ratio=1.0, z_axis='ToT'):
    """
    Identifies and plots clusters that have a specific length (number of hits).

    The plot is a 2D heatmap where:
    - The x-axis represents the 'Row' of the hit.
    - The y-axis represents the 'Column' of the hit.
    - The color of each pixel represents the value specified by 'z_axis'.

    Args:
        cluster_data (dict): A dictionary of numpy arrays containing cluster information.
                             Must include 'TrackID', 'Row', 'Column'.
                             - If z_axis='ToT', must also include 'ToT'.
                             - If z_axis='RelTimestamp', must also include 'Ext_TS'.
        target_lengths (list): A list of integers representing the cluster lengths to plot.
                               (e.g., [12, 13, 14] plots clusters with 12, 13, or 14 hits).
        row_col_ratio (float): The desired ratio of the width of a single row
                               to the width of a single column in the plot.
        z_axis (str): The data to plot on the z-axis (color).
                      Options: 'ToT' (default) or 'RelTimestamp'.
    """
    
    # --- 1. Data Validation ---
    required_keys = ['TrackID', 'Row', 'Column']
    if z_axis == 'ToT':
        required_keys.append('ToT')
    elif z_axis == 'RelTimestamp':
        required_keys.append('ext_TS') 
    else:
        print(f"Error: Invalid z_axis value '{z_axis}'. Must be 'ToT' or 'RelTimestamp'.")
        return

    if not all(k in cluster_data for k in required_keys):
        print(f"Error: The data dictionary must contain {required_keys} for z_axis='{z_axis}'.")
        return

    # --- 2. Find Clusters by Length ---
    # Get all unique IDs and their counts (lengths)
    unique_ids, counts = np.unique(cluster_data['TrackID'], return_counts=True)
    
    # Create a mask where the counts exist in the user-provided target_lengths
    length_mask = np.isin(counts, target_lengths)
    
    # Apply mask to get the IDs
    selected_cluster_ids = unique_ids[length_mask]
    selected_counts = counts[length_mask]

    # Optional: Sort by size descending (largest first) to handle potential overlaps better
    sorted_indices = np.argsort(selected_counts)[::-1]
    clusters_to_plot = selected_cluster_ids[sorted_indices]

    if len(clusters_to_plot) == 0:
        print(f"No clusters found with lengths: {target_lengths}")
        return
        
    # --- 3. Setup the Plot Grid ---
    fig, ax = plt.subplots(figsize=(12, 10))

    min_row_overall = cluster_data['Row'].min()
    max_row_overall = cluster_data['Row'].max()
    min_col_overall = cluster_data['Column'].min()
    max_col_overall = cluster_data['Column'].max()

    grid_height = max_col_overall - min_col_overall + 1
    grid_width = max_row_overall - min_row_overall + 1
    
    # Initialize with NaN for a clear background and correct overlap detection
    plot_grid = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
    
    plotted_clusters_count = 0
    print(f"Found {len(clusters_to_plot)} clusters matching lengths {target_lengths}. Plotting...")

    # --- 4. Populate the Grid ---
    for cluster_id in clusters_to_plot:
        mask = cluster_data['TrackID'] == cluster_id
        
        rows = cluster_data['Row'][mask]
        cols = cluster_data['Column'][mask]

        # Get the grid coordinates
        grid_cols = cols - min_col_overall
        grid_rows = rows - min_row_overall

        # Check for overlap (if any pixel is *not* NaN)
        if np.any(~np.isnan(plot_grid[grid_cols, grid_rows])):
            print(f"Skipping Cluster ID: {cluster_id} because it overlaps with a previously plotted cluster.")
            continue
        
        # Calculate the z-values for plotting
        if z_axis == 'ToT':
            z_values = cluster_data['ToT'][mask]
        elif z_axis == 'RelTimestamp':
            ext_ts_values = cluster_data['ext_TS'][mask] 
            min_ts_cluster = np.min(ext_ts_values)
            z_values = ext_ts_values - min_ts_cluster
            
        # Populate the grid with this cluster's z-values
        plot_grid[grid_cols, grid_rows] = z_values
        plotted_clusters_count += 1
        
    # --- 5. Plotting ---
    
    # Select colormap and label based on z_axis
    if z_axis == 'ToT':
        cmap_name = 'jet'
        cbar_label = 'ToT'
    else: # z_axis == 'RelTimestamp'
        cmap_name = 'viridis' 
        cbar_label = 'Relative Timestamp (from ext_TS)'

    # Get the colormap and set 'nan' values to be black
    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color='black')

    # Use imshow to plot the final grid.
    im = ax.imshow(
        plot_grid,
        cmap=cmap,
        origin='lower',
        extent=[min_row_overall - 0.5, max_row_overall + 0.5, min_col_overall - 0.5, max_col_overall + 0.5],
        aspect=1.0 / row_col_ratio,
        interpolation='nearest' 
    )
    
    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # Add a color bar to the new axes
    fig.colorbar(im, cax=cax, label=cbar_label)
    
    ax.set_title(f'Plot of {plotted_clusters_count} Clusters (Lengths: {target_lengths})')
    ax.set_xlabel('Row')
    ax.set_ylabel('Column')