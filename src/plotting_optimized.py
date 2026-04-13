import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

def _setup_axes(title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    return fig, ax

def _apply_tight_layout(fig):
    fig.tight_layout()
    plt.show()

def _get_bins(data, bins, data_range=None):
    if data_range is None:
        data_min, data_max = np.min(data), np.max(data)
    else:
        data_min, data_max = data_range

    data_range_int = int(data_max - data_min) + 1

    if bins is None:
        return data_range_int
    elif isinstance(bins, int) and bins > data_range_int:
        return data_range_int
    else:
        return bins


def _create_heatmap_on_axis(ax_main, x, y, bins=None, log_z=False, xlabel=None, ylabel=None, title=None,
                            colorscale='viridis', vmin=None, vmax=None):
    """
    Helper: Draws a heatmap with marginals onto an existing axis (ax_main).
    """
    divider = make_axes_locatable(ax_main)
    ax_histx = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    ax_histy = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if bins is None:
        bins = [132, 372]

    # Use cmin=1 to prevent zero-count bins from being drawn (solves the "black everywhere" issue)
    # Using vmin/vmax directly in hist2d works best with the calculated norm
    norm = LogNorm(vmin=max(1, vmin) if vmin else None, vmax=vmax) if log_z else plt.Normalize(vmin=vmin, vmax=vmax)

    mappable = None
    if len(x) > 0 and len(y) > 0:
        counts, xedges, yedges, image = ax_main.hist2d(
            x, y, bins=bins, norm=norm, cmap=colorscale, cmin=1
        )
        mappable = image

        # Marginal histograms
        ax_histx.hist(x, bins=xedges, histtype='step', color='C0', lw=1)
        ax_histy.hist(y, bins=yedges, histtype='step', color='C0', orientation='horizontal', lw=1)

    # Fix: Set title on the top-most axis (ax_histx) to prevent overlap
    ax_histx.set_title(title, fontsize=10, pad=10)
    ax_main.set_xlabel(xlabel, fontsize=8)
    ax_main.set_ylabel(ylabel, fontsize=8)

    # Fix: Log scaling for marginals with nonpositive='clip' to handle zero counts correctly
    if log_z:
        ax_histx.set_yscale('log', nonpositive='clip')
        ax_histy.set_xscale('log', nonpositive='clip')

    return ax_main, mappable

def _plot_matrix_on_axis(ax_main, Z, x_bins, y_bins, x1, y1, x2, y2, label1, label2,
                         cmap, norm=None, title=None):
    """
    Helper: Draws matrix with marginals onto an existing axis.
    """
    divider = make_axes_locatable(ax_main)
    ax_histx = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    ax_histy = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    im = ax_main.pcolormesh(x_bins, y_bins, Z.T, cmap=cmap, norm=norm)
    
    # Fix: Title on ax_histx
    ax_histx.set_title(title, fontsize=10, pad=10)
    ax_main.set_xlabel('Column', fontsize=8)
    ax_main.set_ylabel('Row', fontsize=8)

    if len(x1) > 0: ax_histx.hist(x1, bins=x_bins, histtype='step', color='C0', label=label1)
    if len(x2) > 0: ax_histx.hist(x2, bins=x_bins, histtype='step', color='C1', label=label2)
    
    if len(y1) > 0: ax_histy.hist(y1, bins=y_bins, histtype='step', color='C0', orientation='horizontal')
    if len(y2) > 0: ax_histy.hist(y2, bins=y_bins, histtype='step', color='C1', orientation='horizontal')

    return im

def plot_HeatHitmap(data, xcol, ycol, title=None, xlabel=None, ylabel=None, bins=None,
                    col_pitch=1, row_pitch=1, log_z=True, vmin=None, vmax=None):
    """
    Creates a 2x2 figure containing heatmaps for Layers 0, 1, 2, 3.
    """
    # Robust Layer Identification
    if hasattr(data, 'columns') and 'Layer' in data.columns:
        layers = data['Layer']
    elif hasattr(data, 'keys') and 'Layer' in data.keys():
        layers = data['Layer']
    elif hasattr(data, 'dtype') and 'Layer' in data.dtype.names:
        layers = data['Layer']
    else:
        print("Error: 'Layer' column not found.")
        return

    unique_layers = np.unique(layers)
    plot_layers = unique_layers[:4] 
    
    if vmin is None or vmax is None:
        x_all = data[xcol] * col_pitch
        y_all = data[ycol] * row_pitch
        h, _, _ = np.histogram2d(x_all, y_all, bins=bins or [50, 50])
        if vmax is None: vmax = np.max(h)
        if vmin is None: vmin = 1 if log_z else 0

    current_cmap = 'viridis' if log_z else 'rainbow'
    fig, axes = plt.subplots(2, 2, figsize=(13, 12)) 
    axes_flat = axes.flatten()
    
    fig.suptitle(title or f"Heatmap of {xcol} vs {ycol} by Layer", fontsize=16)
    last_mappable = None

    for i, ax in enumerate(axes_flat):
        if i < len(plot_layers):
            layer_id = plot_layers[i]
            mask = layers == layer_id
            
            # Extract data
            if hasattr(data, 'loc'): d = data.loc[mask]
            elif isinstance(data, np.ndarray): d = data[mask]
            else: d = {k: v[mask] for k, v in data.items()}
            
            x, y = d[xcol] * col_pitch, d[ycol] * row_pitch
            
            _, mappable = _create_heatmap_on_axis(
                ax, x, y, bins=bins, log_z=log_z,
                xlabel=xlabel or xcol, ylabel=ylabel or ycol,
                title=f"Layer {layer_id}",
                vmin=vmin, vmax=vmax, colorscale=current_cmap
            )
            if mappable: last_mappable = mappable
        else:
            ax.axis('off')

    if last_mappable:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        fig.colorbar(last_mappable, cax=cbar_ax, label='Counts')

    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.08, top=0.90, wspace=0.35, hspace=0.4)
    plt.show()

def plot_heatmap_ratio(dataset1, dataset2, label1, label2, log_c=False):
    """
    Generates TWO figures:
    1. 2x2 Residual maps (Signal - Background) per layer
    2. 2x2 Ratio maps (Background / Signal) per layer
    """
    
    # Identify Layers (assuming both datasets have same layers)
    unique_layers = np.unique(dataset1['Layer'])
    plot_layers = unique_layers[:4]

    # Pre-calculate common bins for consistency
    x1, y1 = dataset1['Column'], dataset1['Row']
    x2, y2 = dataset2['Column'], dataset2['Row']
    
    x_min = int(min(x1.min(), x2.min()))
    x_max = int(max(x1.max(), x2.max()))
    y_min = int(min(y1.min(), y2.min()))
    y_max = int(max(y1.max(), y2.max()))

    x_bins = np.arange(x_min - 0.5, x_max + 1.5, 1)
    y_bins = np.arange(y_min - 0.5, y_max + 1.5, 1)
    bins = [x_bins, y_bins]

    # --- Prepare Figures ---
    fig_resid, axes_resid = plt.subplots(2, 2, figsize=(13, 13))
    fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(13, 13))
    
    fig_resid.suptitle(f'Residuals ({label1} - {label2}) by Layer', fontsize=16)
    fig_ratio.suptitle(f'Ratios ({label2} / {label1}) by Layer', fontsize=16)

    last_im_res = None
    last_im_rat = None

    # Loop through layers
    for i, layer_id in enumerate(plot_layers):
        if i >= 4: break
        
        ax_res = axes_resid.flatten()[i]
        ax_rat = axes_ratio.flatten()[i]

        # Filter Data
        mask1 = dataset1['Layer'] == layer_id
        mask2 = dataset2['Layer'] == layer_id
        
        # Handle dict vs struct array vs DataFrame access
        d1_x = dataset1['Column'][mask1] if not hasattr(dataset1, 'loc') else dataset1.loc[mask1, 'Column']
        d1_y = dataset1['Row'][mask1] if not hasattr(dataset1, 'loc') else dataset1.loc[mask1, 'Row']
        d2_x = dataset2['Column'][mask2] if not hasattr(dataset2, 'loc') else dataset2.loc[mask2, 'Column']
        d2_y = dataset2['Row'][mask2] if not hasattr(dataset2, 'loc') else dataset2.loc[mask2, 'Row']

        # Calc histograms
        h1, _, _ = np.histogram2d(d1_x, d1_y, bins=bins)
        h2, _, _ = np.histogram2d(d2_x, d2_y, bins=bins)

        # Calc Metrics
        residual = h1 - h2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = h2 / h1
        ratio[np.isnan(ratio) | np.isinf(ratio)] = 0

        # Plot Residual
        last_im_res = _plot_matrix_on_axis(
            ax_res, residual, x_bins, y_bins, d1_x, d1_y, d2_x, d2_y,
            label1, label2, cmap='coolwarm', norm=CenteredNorm(),
            title=f"Layer {layer_id}"
        )
        
        # Plot Ratio
        last_im_rat = _plot_matrix_on_axis(
            ax_rat, ratio, x_bins, y_bins, d1_x, d1_y, d2_x, d2_y,
            label1, label2, cmap='viridis', norm=LogNorm() if log_c else None,
            title=f"Layer {layer_id}"
        )
    
    # Add shared colorbars and adjust layout manually
    if last_im_res:
        cbar_ax1 = fig_resid.add_axes([0.9, 0.15, 0.02, 0.7])
        fig_resid.colorbar(last_im_res, cax=cbar_ax1, label='Residual')
        fig_resid.subplots_adjust(left=0.07, right=0.87, bottom=0.07, top=0.92, wspace=0.3, hspace=0.3)
        
    if last_im_rat:
        cbar_ax2 = fig_ratio.add_axes([0.9, 0.15, 0.02, 0.7])
        fig_ratio.colorbar(last_im_rat, cax=cbar_ax2, label='Ratio')
        fig_ratio.subplots_adjust(left=0.07, right=0.87, bottom=0.07, top=0.92, wspace=0.3, hspace=0.3)

    plt.show() # Shows both figures

def plot_histogram_ratio(dataset1, dataset2, label1, label2, column_name):
    """
    Modified to plot a 2x2 grid of Signal:Noise ratios, one for each layer.
    """
    unique_layers = np.unique(dataset1['Layer'])
    plot_layers = unique_layers[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle(f'Signal:Noise Ratio for {column_name} by Layer', fontsize=16)

    # Determine global bins for consistency
    all_data = np.concatenate((dataset1[column_name], dataset2[column_name]))
    min_val = int(np.min(all_data))
    max_val = int(np.max(all_data))
    bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

    for i, ax in enumerate(axes):
        if i < len(plot_layers):
            layer_id = plot_layers[i]
            
            # Filter Data
            mask1 = dataset1['Layer'] == layer_id
            mask2 = dataset2['Layer'] == layer_id
            
            # Handle DataFrame vs Dict/Array
            if hasattr(dataset1, 'loc'):
                d1 = dataset1.loc[mask1, column_name]
            else:
                d1 = dataset1[column_name][mask1]
                
            if hasattr(dataset2, 'loc'):
                d2 = dataset2.loc[mask2, column_name]
            else:
                d2 = dataset2[column_name][mask2]

            if len(d1) == 0 or len(d2) == 0:
                ax.text(0.5, 0.5, "No Data", ha='center')
                continue

            hist1, _ = np.histogram(d1, bins=bins)
            hist2, _ = np.histogram(d2, bins=bins)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = hist2 / hist1
            ratio[np.isnan(ratio) | np.isinf(ratio)] = 0

            ax.step(bins[:-1] + 0.5, ratio, where='mid', label=f'Ratio')
            ax.set_title(f'Layer {layer_id}')
            ax.grid(True)
            ax.set_ylim(bottom=0)
            
            if i >= 2: # Bottom row
                ax.set_xlabel(column_name)
            if i % 2 == 0: # Left column
                ax.set_ylabel(f'Ratio ({label1}:{label2})')
        else:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_layer_heatmaps(data, xcol, ycol, common_z_scale=True, **kwargs):
    """
    Plots a separate heatmap for each layer (1, 2, 3, 4) in the dataset.
    
    This function assumes the input 'data' is either a numpy structured array
    or a dictionary of numpy arrays, with a column/key named 'layer' or 'Layer'.
    
    It uses the 'plot_HeatHitmap' function as a helper and passes
    any additional keyword arguments (kwargs) like 'col_pitch', 
    'row_pitch', 'bins', 'log_z', etc., to it.
    
    Args:
        data (np.ndarray or dict): The full dataset.
        xcol (str): The name of the column/key for the x-axis.
        ycol (str): The name of the column/key for the y-axis.
        common_z_scale (bool): If True, all layer plots will share a common z-axis (color) scale.
        **kwargs: Additional arguments to pass to plot_HeatHitmap.
                  For example: col_pitch=1, row_pitch=1, log_z=True, bins=100
    """
    
    # --- Detect data type and find layer column ---
    is_structured_array = False
    is_dict = isinstance(data, dict)
    layer_col_name = None

    if not is_dict and hasattr(data, 'dtype') and data.dtype.names:
        is_structured_array = True
        if 'layer' in data.dtype.names:
            layer_col_name = 'layer'
        elif 'Layer' in data.dtype.names:
            layer_col_name = 'Layer'
    elif is_dict:
        if 'layer' in data:
            layer_col_name = 'layer'
        elif 'Layer' in data:
            layer_col_name = 'Layer'

    if layer_col_name is None:
        print("Error: Could not find a 'layer' or 'Layer' column/key in the dataset.")
        print("       Please ensure your data has one of these.")
        return
    
    if not is_structured_array and not is_dict:
        print("Error: Data type not recognized. Must be a numpy structured array or a dict of arrays.")
        return
        
    # --- Logic for common Z scale ---
    global_vmin = None
    global_vmax = None
    
    # Get kwargs needed for pre-calculation
    col_pitch = kwargs.get('col_pitch', 1)
    row_pitch = kwargs.get('row_pitch', 1)
    log_z = kwargs.get('log_z', True)
    
    # --- 1. Determine common bins ---
    # We must use common bins for a common z-scale to be valid.
    # We'll base this on the min/max of the *entire* dataset,
    # mimicking the binning style in plot_Hitmap_numpy.
    
    # Extract *all* data first
    x_all = None
    y_all = None
    if is_structured_array:
        x_all = data[xcol] * col_pitch
        y_all = data[ycol] * row_pitch
    elif is_dict:
        x_all = data[xcol] * col_pitch
        y_all = data[ycol] * row_pitch
    
    # Use floor/ceil to be safe
    common_x_bins = np.arange(np.floor(x_all.min()), np.ceil(x_all.max()) + 2)
    common_y_bins = np.arange(np.floor(y_all.min()), np.ceil(y_all.max()) + 2)
    common_bins_list = [common_x_bins, common_y_bins]
    
    # Override/set bins in kwargs
    kwargs['bins'] = common_bins_list

    if common_z_scale:
        print("Calculating common z-scale...")
        global_vmin, global_vmax = np.inf, -np.inf
        
        # --- 2. Loop 1: Find global min/max ---
        for layer in [1, 2, 3, 4]:
            layer_data = None
            x_layer = None
            y_layer = None
            
            try:
                if is_structured_array:
                    layer_data = data[data[layer_col_name] == layer]
                    if layer_data.size > 0:
                        x_layer = layer_data[xcol] * col_pitch
                        y_layer = layer_data[ycol] * row_pitch
                
                elif is_dict:
                    mask = (data[layer_col_name] == layer)
                    if np.any(mask):
                        x_layer = data[xcol][mask] * col_pitch
                        y_layer = data[ycol][mask] * row_pitch
                
                if x_layer is not None and x_layer.size > 0:
                    # Calculate histogram counts
                    H, _, _ = np.histogram2d(x_layer, y_layer, bins=common_bins_list)
                    
                    H_flat = H.flatten()
                    if log_z:
                        H_flat = H_flat[H_flat > 0] # Only consider positive counts for log
                    
                    if H_flat.size > 0:
                        global_vmin = min(global_vmin, H_flat.min())
                        global_vmax = max(global_vmax, H_flat.max())

            except Exception as e:
                print(f"Error during z-scale calculation for layer {layer}: {e}")

        # Set vmin/vmax in kwargs for plotting
        if np.isinf(global_vmin) or np.isinf(global_vmax):
             print("Warning: Could not determine valid global z-scale. Plots will autoscale.")
        else:
            print(f"Common z-scale determined: vmin={global_vmin}, vmax={global_vmax}")
            kwargs['vmin'] = global_vmin
            kwargs['vmax'] = global_vmax
            
    # --- End of common Z scale logic ---

    # --- 3. Loop 2: Plotting ---
    for layer in [1, 2, 3, 4]:
        layer_data = None
        
        try:
            if is_structured_array:
                # --- Logic for structured arrays ---
                layer_data = data[data[layer_col_name] == layer]
                if layer_data.size == 0:
                    print(f"No data found for layer {layer}. Skipping heatmap.")
                    continue
            
            elif is_dict:
                # --- Logic for dictionary of arrays ---
                # Create the boolean mask
                mask = (data[layer_col_name] == layer)
                
                # Check if there's any data for this layer
                if not np.any(mask):
                    print(f"No data found for layer {layer}. Skipping heatmap.")
                    continue
                    
                # Create a new dictionary filtering all arrays based on the mask
                layer_data = {}
                for key, arr in data.items():
                    # Check that the item is a numpy array and has the same shape
                    # as the mask (or layer array) to be filterable.
                    if isinstance(arr, np.ndarray) and arr.shape == data[layer_col_name].shape:
                        layer_data[key] = arr[mask]
                    else:
                        # Pass through other keys (e.g., metadata)
                        layer_data[key] = arr 
            
        except Exception as e:
            print(f"Error while filtering data for layer {layer}: {e}")
            print(f"Traceback key: {layer_col_name}")
            continue # Skip to the next layer

        if layer_data is None:
             print(f"No data was processed for layer {layer}. Skipping heatmap.")
             continue
        
        # Create a specific title for this layer's plot
        # Use a provided title_prefix or default to "Heatmap"
        title_prefix = kwargs.get('title_prefix', 'Heatmap')
        layer_title = f"{title_prefix} for Layer {layer} - {xcol} vs {ycol}"
        
        # Make a copy of kwargs to safely set the title for this plot
        plot_kwargs = kwargs.copy()
        plot_kwargs['title'] = layer_title
        
        # Remove 'title_prefix' if it exists, as plot_HeatHitmap doesn't expect it
        if 'title_prefix' in plot_kwargs:
            del plot_kwargs['title_prefix']
            
        print(f"Plotting heatmap for Layer {layer}...")
        
        # Call the helper function with the filtered data and custom title
        plot_HeatHitmap(
            layer_data, 
            xcol, 
            ycol,
            **plot_kwargs
        )


def exp_decay(x, a, b, c):
    """Exponential decay function."""
    return a * np.exp(-b * x) + c

def plot_histograms_with_fits(
    datasets, labels, columns, bins=None, range_hist=None, logy=False, title=None
):
    """
    Plots histograms and exponential fits for 1, 2, or 3 datasets.

    Creates a grid of plots. For each column specified:
    - Top row: Overlaid histograms for all datasets.
    - Middle row: Overlaid data points with exponential fits.
    - Bottom row (IF n_datasets=2): Data-Data Residuals (dataset1 - dataset2).
    
    Args:
        datasets (list): A list of data-containing objects (e.g., pandas DataFrames).
                         Example: [data1, data2]
        labels (list): A list of string labels for each dataset.
                       Example: ["Label 1", "Label 2"]
        columns (list): A list of column names (strings) to plot.
        bins (int, optional): Number of bins for histograms.
        range_hist (tuple, optional): (min, max) range for histograms.
        logy (bool, optional): Whether to use a log scale on the y-axis for histograms.
        title (str, optional): Overall title for the figure.
    """
    n_datasets = len(datasets)
    n_columns = len(columns)
    
    if n_datasets != len(labels):
        raise ValueError("The number of datasets must equal the number of labels.")
    if n_datasets not in [1, 2, 3]:
        raise ValueError("This function supports 1, 2, or 3 datasets.")

    # --- Define colors and markers for plotting ---
    colors = ['C0', 'C1', 'C2']
    markers = ['o', 's', '^']

    # --- Create the figure layout ---
    # Residual plot is only added if n_datasets is exactly 2
    has_residuals = (n_datasets == 2)
    n_rows = 3 if has_residuals else 2
    
    ax_hists = []
    ax_fits = []
    ax_resids = []

    if has_residuals:
        # Use GridSpec for custom row heights (hist, fit, residual)
        fig = plt.figure(figsize=(n_columns * 7, 10))
        # Ratios: 3 for hist, 3 for fit, 1.5 for residual
        gs = fig.add_gridspec(n_rows, n_columns, height_ratios=[3, 3, 1.5], hspace=0.1)
        
        for i in range(n_columns):
            ax_h = fig.add_subplot(gs[0, i])
            ax_f = fig.add_subplot(gs[1, i], sharex=ax_h)
            ax_r = fig.add_subplot(gs[2, i], sharex=ax_h)
            
            # Hide x-labels for upper plots since they share an x-axis
            plt.setp(ax_h.get_xticklabels(), visible=False)
            plt.setp(ax_f.get_xticklabels(), visible=False)
            
            ax_hists.append(ax_h)
            ax_fits.append(ax_f)
            ax_resids.append(ax_r)
    else:
        # Use standard subplots for 1 or 3 datasets (hist, fit)
        fig, axes = plt.subplots(
            n_rows, n_columns, 
            figsize=(n_columns * 7, 8), 
            squeeze=False, 
            sharex='col'
        )
        ax_hists = axes[0, :]
        ax_fits = axes[1, :]
        ax_resids = [None] * n_columns  # No residual plots
        
        # Hide x-labels for upper plots
        for ax_h in ax_hists:
            plt.setp(ax_h.get_xticklabels(), visible=False)

    fig.suptitle(title or "Histograms and Exponential Fits", fontsize=16)

    # --- Loop over each specified column ---
    for i, column in enumerate(columns):
        ax_hist = ax_hists[i]
        ax_fit = ax_fits[i]
        ax_resid = ax_resids[i]  # This will be None if not has_residuals

        # --- Get data for this column ---
        column_data_list = [np.asarray(data[column]) for data in datasets]

        # --- Determine common bins for all datasets ---
        if range_hist is None:
            min_val = min(d.min() for d in column_data_list)
            max_val = max(d.max() for d in column_data_list)
            current_range = (min_val, max_val)
        else:
            current_range = range_hist
        
        if bins is None:
            # Base bins on the first dataset
            current_bins = len(np.unique(column_data_list[0]))
        else:
            current_bins = bins

        # --- Storage for histogram counts ---
        all_counts = []
        bin_centers = None
        bin_edges = None

        # --- Loop over each dataset (1, 2, or 3) ---
        for data_idx in range(n_datasets):
            data_arr = column_data_list[data_idx]
            label = labels[data_idx]
            color = colors[data_idx]
            marker = markers[data_idx]

            # --- Top Plot: Histograms ---
            if data_idx == 0:
                # First dataset defines the bins
                counts, bin_edges = np.histogram(data_arr, bins=current_bins, range=current_range)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            else:
                # Use the same bins for other datasets
                counts, _ = np.histogram(data_arr, bins=bin_edges)
            
            all_counts.append(counts) # Store counts for later (for residuals)
            ax_hist.step(bin_centers, counts, where="mid", lw=1.5, label=label, color=color)

            # --- Bottom Plot: Fits ---
            
            # Use only bins with non-zero counts for fitting
            valid_indices = counts > 0
            if not np.any(valid_indices):
                continue  # Skip if no data
            
            valid_counts = counts[valid_indices]
            valid_bins = bin_centers[valid_indices]
            errors = np.sqrt(valid_counts) # Poisson error
            
            ax_fit.errorbar(
                valid_bins, valid_counts, yerr=errors,
                fmt=marker, capsize=3, label=f'{label} Data', color=color, alpha=0.7
            )

            try:
                p0 = (max(valid_counts), 0.1, min(valid_counts))
                params, _ = curve_fit(
                    exp_decay, valid_bins, valid_counts, p0=p0, maxfev=5000, sigma=errors
                )
                a, b, c = params
                
                # Plot smooth fit line
                x_fit_line = np.linspace(bin_centers.min(), bin_centers.max(), 200)
                y_fit_line = exp_decay(x_fit_line, a, b, c)
                ax_fit.plot(
                    x_fit_line, y_fit_line, linestyle='--', color=color,
                    label=f'{label} Fit (T={1/b:.2f})' if b != 0 else f'{label} Fit'
                )

            except RuntimeError:
                print(f"Could not find an exponential fit for {label} in {column}.")

        # --- Configure Histogram and Fit Axes ---
        ax_hist.set_title(f"{column}")
        ax_hist.set_ylabel("Counts")
        ax_hist.legend()
        if logy:
            ax_hist.set_yscale("log")
        
        ax_fit.set_title("Exponential Fit")
        ax_fit.set_ylabel("Counts")
        ax_fit.legend()
        if not has_residuals:
            ax_fit.set_xlabel(f"{column}") # X-label on fit plot if no residuals

        # --- Residual Plot (only if n_datasets == 2) ---
        if has_residuals and ax_resid:
            ax_resid.axhline(0, color='grey', linestyle='--', lw=1)

            # Get the counts from the two datasets
            counts1 = all_counts[0]
            counts2 = all_counts[1]
            
            # Calculate difference
            residuals = counts1 - counts2
            
            # Propagate Poisson errors: sigma_diff = sqrt(counts1 + counts2)
            errors = np.sqrt(counts1 + counts2)
            
            # Plot only points where at least one bin had data
            valid_indices = (counts1 > 0) | (counts2 > 0)
            
            ax_resid.errorbar(
                bin_centers[valid_indices], 
                residuals[valid_indices], 
                yerr=errors[valid_indices],
                fmt='o', 
                capsize=3, 
                color='k', # Use black for the difference plot
                alpha=0.8
            )
            
            ax_resid.set_xlabel(f"{column}") # X-label goes on the very bottom plot
            ax_resid.set_ylabel(f"{labels[0]} - {labels[1]}") # Label as the difference
            ax_resid.set_ylim(auto=True)


    # --- Final Layout Adjustment ---
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.show()



from typing import Dict, Optional, Tuple


def plot_timing_uniformity(data: Dict[str, np.ndarray],
                                      x_key: str = 'displacement',
                                      y_key: str = 'dTS',
                                      title: str = "Timing Uniformity (Column Normalized)",
                                      log_z: bool = False,
                                      bins=None,
                                      cluster_size_range: Optional[Tuple[int, int]] = None):
    """
    Plots a 2D heatmap where each column is normalized to sum to 1.
    Optionally filters data to only include clusters within a specified size range.

    Args:
        data (Dict[str, np.ndarray]): The full clustered data dictionary. Must contain
                                      'ClusterID' if filtering is used.
        x_key (str): The dictionary key for the x-axis data.
        y_key (str): The dictionary key for the y-axis data.
        title (str): The main title for the plot.
        log_z (bool): Whether to use a logarithmic color scale.
        bins: Pre-calculated bins for the histogram. If None, they are auto-generated.
        cluster_size_range (Optional[Tuple[int, int]]):
            If provided as [a, b], only includes data from clusters with a size
            >= a and < b.
    """
    
    # --- New Filtering Block ---
    if cluster_size_range is not None:
        if 'ClusterID' not in data:
            raise KeyError("To filter by cluster size, the data dictionary must contain a 'ClusterID' key.")
        
        a, b = cluster_size_range
        print(f"Filtering to include clusters with size in range [{a}, {b})...")
        
        # Efficiently get counts for each cluster ID
        ids, counts = np.unique(data['ClusterID'], return_counts=True)
        
        # Find which cluster IDs are in the desired size range
        count_mask = (counts >= a) & (counts < b)
        target_ids = ids[count_mask]
        
        if len(target_ids) == 0:
            print(f"No clusters found in the size range [{a}, {b}). Aborting plot.")
            return
            
        # Create a mask to select all hits from the target clusters
        data_mask = np.isin(data['ClusterID'], target_ids)
        filtered_data = {key: value[data_mask] for key, value in data.items()}
    else:
        # If no range is given, use the original data
        filtered_data = data

    if not filtered_data or len(filtered_data[x_key]) == 0:
        print("No data left to plot after filtering.")
        return

    x = filtered_data[x_key]
    y = filtered_data[y_key]

    # --- Bin Calculation ---
    if bins is None:
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        if len(unique_x) > 1:
            x_edges = np.concatenate(([unique_x[0] - 0.5], (unique_x[:-1] + unique_x[1:]) / 2, [unique_x[-1] + 0.5]))
        else:
            x_edges = np.array([unique_x[0] - 0.5, unique_x[0] + 0.5])
        if len(unique_y) > 1:
            y_edges = np.concatenate(([unique_y[0] - 0.5], (unique_y[:-1] + unique_y[1:]) / 2, [unique_y[-1] + 0.5]))
        else:
            y_edges = np.array([unique_y[0] - 0.5, unique_y[0] + 0.5])
        bins = [x_edges, y_edges]

    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    col_sums = H.sum(axis=1, keepdims=True)
    H_normalized = H / (col_sums + 1e-9)
    H_normalized = np.nan_to_num(H_normalized) 

    # --- Plotting ---
    fig, ax_main = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax_main)
    ax_histx = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    ax_histy = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    norm = LogNorm() if log_z else None
    im = ax_main.pcolormesh(xedges, yedges, H_normalized.T, cmap='viridis', norm=norm)
    fig.colorbar(im, ax=ax_main, label='Normalized Frequency (Column Sum = 1)')

    x_projection = H.sum(axis=1)
    x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
    ax_histx.step(x_bin_centers, x_projection, where='mid', color='C0')
    ax_histx.set_ylabel("Total Counts")

    y_projection = H.sum(axis=0)
    y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
    ax_histy.step(y_projection, y_bin_centers, where='mid', color='C0')
    ax_histy.set_xlabel("Total Counts")

    ax_main.set_xlabel(x_key)
    ax_main.set_ylabel(y_key)
    
    # Update title with filter info
    plot_title = title
    if cluster_size_range is not None:
        plot_title += f'\n(Cluster Size ∈ [{cluster_size_range[0]}, {cluster_size_range[1]}))'
    if plot_title:
        fig.suptitle(plot_title)

    try:
        _apply_tight_layout(fig)
    except NameError:
        fig.tight_layout()

    return fig


def plot_cluster_heatmap(data, xcol, ycol, bins=None, log_z=False, title=None, xlabel=None, ylabel=None):
    if xcol not in data or ycol not in data:
        print(f"Error: One or both columns '{xcol}', '{ycol}' not found in data.")
        return

    x = data[xcol]
    y = data[ycol]

    if bins is None:
        bins_x = len(np.unique(x))
        bins_y = len(np.unique(y))
        bins = [bins_x, bins_y]

    fig = _create_heatmap_with_marginals(
        x, y, bins,
        log_z=log_z,
        xlabel=xlabel or xcol,
        ylabel=ylabel or ycol,
        title=title or f"Heatmap of {xcol} vs {ycol}"
    )
    _apply_tight_layout(fig)
    
    



import numpy as np
from typing import Dict

def plot_cluster_characteristics(sorted_datasets: Dict[str, Dict[str, np.ndarray]], log_y: bool = False):
    """
    Plots three characteristic distributions for the sorted event types.
    Generates a separate figure for each characteristic (Hits, Timescale, ToT),
    with each figure containing a normal scale plot and an optional log scale plot.

    **OPTIMIZED VERSION with Hard-coded Bins:**
    - Calculates stats for Plot 1 (Hit Counts) and Plot 2 (Timescale)
      in a single pass to avoid sorting the large ClusterID array multiple times.
    - Plot 1/2 Bins: Hard-coded to integers 0-50.
    - Plot 3 Bins: Hard-coded to integers 0-255.
    - Plot 3 (ToT) remains optimized with np.histogram and plt.step.
    
    **MODIFIED:**
    - Plots 1 & 2 will plot 'noise' data on *both* linear and log plots.
    - All plots include a 'sum' line (total of all categories).
    - Plot 3 (ToT) left graph now includes an inset zoomed to xlim(0, 50).

    Args:
        sorted_datasets (dict): A dictionary where keys are group names 
            (e.g., 'clusters', 'coupling', 'noise') and values are dictionaries 
            containing data arrays (e.g., {'ClusterID': array(...), 'ToT': array(...)}).
        log_y (bool): If True, sets the y-axis of the *right-hand* histograms 
            to a logarithmic scale. The left plot is always linear.
    """
    
    # --- CHANGED: Simplified colors to match your data ---
    colors = {
        'clusters': 'blue',
        'coupling': 'purple', 
        'noise': 'gray',
        'sum': 'black'  # Added color for the sum plot
    }
    
    # --- 1. Pre-calculation of Statistics for Plots 1 & 2 ---
    group_hit_counts = {}   # For Plot 1
    group_timescales = {}   # For Plot 2
    all_counts_list = []
    all_timescales_list = []

    for name, group_data in sorted_datasets.items():
        if name not in colors:
            # Check if it's a category we care about, skip if not
            if name in ('clusters', 'coupling', 'noise'):
                 print(f"Warning: Skipping group '{name}', not found in 'colors' dictionary.")
            continue
        
        if group_data['ClusterID'].size == 0:
            continue

        # --- NEW Logic to handle 'noise' ---
        if name == 'noise':
            num_noise_hits = group_data['ClusterID'].size
            # Plot 1: Noise events always have 1 hit
            noise_counts = np.ones(num_noise_hits, dtype=int)
            group_hit_counts[name] = noise_counts
            all_counts_list.append(noise_counts)
            
            # Plot 2: Noise events (single hits) have a timescale of 0
            noise_timescales = np.zeros(num_noise_hits, dtype=int)
            group_timescales[name] = noise_timescales
            all_timescales_list.append(noise_timescales)
            
            # Skip the rest of the loop which is for cluster-based stats
            continue 
        
        # --- Existing Logic for 'clusters' and 'coupling' ---
        ids = group_data['ClusterID']
        ts = group_data['ext_TS']

        # Sort by ClusterID *once*
        sort_indices = np.argsort(ids)
        sorted_ids = ids[sort_indices]
        sorted_ts = ts[sort_indices]

        # np.unique on a *sorted* array is O(N) (very fast)
        unique_ids, unique_indices, unique_counts = np.unique(
            sorted_ids, return_index=True, return_counts=True
        )
        
        if unique_indices.size == 0:
            continue

        # --- Stats for Plot 1 ---
        group_hit_counts[name] = unique_counts
        all_counts_list.append(unique_counts)

        # --- Stats for Plot 2 ---
        min_per_group = np.minimum.reduceat(sorted_ts, unique_indices)
        max_per_group = np.maximum.reduceat(sorted_ts, unique_indices)
        
        timescales = max_per_group - min_per_group
        group_timescales[name] = timescales
        all_timescales_list.append(timescales)

    # --- 2. Plot 1: Number of Hits per Event ---
    fig1, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Hard-coded bins for integers 0-50
    # Creates bins [0,1), [1,2), ..., [50,51)
    bins_plot1 = np.arange(0, 52) 
    
    if all_counts_list:
        # Plotting the pre-calculated counts
        for name, counts in group_hit_counts.items():
            if counts.size > 0:
                # Plot on log axis (right) for all groups
                ax_log.hist(counts, bins=bins_plot1, color=colors[name], alpha=0.7, label=name, histtype='step', linewidth=2)
                
                # *** CHANGED: Plot on linear axis (left) for ALL groups ***
                ax_lin.hist(counts, bins=bins_plot1, color=colors[name], alpha=0.7, label=name, histtype='step', linewidth=2)

        # *** NEW: Plot the sum ***
        all_counts_total = np.concatenate(all_counts_list)
        if all_counts_total.size > 0:
            ax_lin.hist(all_counts_total, bins=bins_plot1, color=colors['sum'], label='sum', histtype='step', linewidth=1.5, linestyle='--')
            ax_log.hist(all_counts_total, bins=bins_plot1, color=colors['sum'], label='sum', histtype='step', linewidth=1.5, linestyle='--')

    ax_lin.set_title('Normal Scale')
    ax_lin.set_xlabel('Number of Hits')
    ax_lin.set_ylabel('Number of Events')
    ax_lin.legend()
    ax_lin.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_log.set_title('Log Scale')
    ax_log.set_xlabel('Number of Hits')
    ax_log.set_ylabel('Number of Events')
    if log_y:
        ax_log.set_yscale('log')
    ax_log.legend()
    ax_log.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig1.suptitle('Distribution of Hits per Event', fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 3. Plot 2: Timescale of Events ---
    fig2, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Hard-coded bins for integers 0-50
    bins_plot2 = np.arange(0, 52)
    
    if all_timescales_list:
        # Plotting the pre-calculated timescales
        for name, timescales in group_timescales.items():
            if timescales.size > 0:
                # Plot on log axis (right) for all groups
                ax_log.hist(timescales, bins=bins_plot2, color=colors[name], alpha=0.7, label=name, histtype='step', linewidth=2)

                # *** CHANGED: Plot on linear axis (left) for ALL groups ***
                ax_lin.hist(timescales, bins=bins_plot2, color=colors[name], alpha=0.7, label=name, histtype='step', linewidth=2)
        
        # *** NEW: Plot the sum ***
        all_timescales_total = np.concatenate(all_timescales_list)
        if all_timescales_total.size > 0:
            ax_lin.hist(all_timescales_total, bins=bins_plot2, color=colors['sum'], label='sum', histtype='step', linewidth=1.5, linestyle='--')
            ax_log.hist(all_timescales_total, bins=bins_plot2, color=colors['sum'], label='sum', histtype='step', linewidth=1.5, linestyle='--')


    ax_lin.set_title('Normal Scale')
    ax_lin.set_xlabel('Timescale (max_ts - min_ts)')
    ax_lin.set_ylabel('Number of Events')
    ax_lin.legend()
    ax_lin.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_log.set_title('Log Scale')
    ax_log.set_xlabel('Timescale (max_ts - min_ts)')
    ax_log.set_ylabel('Number of Events')
    if log_y:
        ax_log.set_yscale('log')
    ax_log.legend()
    ax_log.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig2.suptitle('Distribution of Event Timescales (Duration)', fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 4. Plot 3: ToT Spectrum ---
    # This plot *includes* 'noise' hits, as ToT is relevant for all hit types.
    fig3, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- MODIFIED: Create inset axes on the left plot (ax_lin) ---
    # [x, y, width, height] relative to ax_lin's bounding box
    ax_ins = ax_lin.inset_axes([0.4, 0.4, 0.55, 0.55])
    max_y_inset = 0  # To store max y-value for inset's auto-scaling

    all_tots_list = []
    for name, d in sorted_datasets.items():
        if name in colors and name != 'sum' and d['ToT'].size > 0:
             all_tots_list.append(d['ToT'])

    # Hard-coded bins for integers 0-255
    # Creates bins [0,1), [1,2), ..., [255,256)
    bins_plot3 = np.arange(0, 257)

    if all_tots_list:
        # Plotting with pre-binning
        for name, group_data in sorted_datasets.items():
            if name not in colors or name == 'sum':
                continue
                
            if group_data['ToT'].size > 0:
                counts, _ = np.histogram(group_data['ToT'], bins=bins_plot3)
                
                # Plot on main linear axis
                ax_lin.step(bins_plot3[:-1], counts, where='post', color=colors[name], alpha=0.7, label=name, linewidth=2)
                # Plot on main log axis
                ax_log.step(bins_plot3[:-1], counts, where='post', color=colors[name], alpha=0.7, label=name, linewidth=2)
                
                # --- MODIFIED: Plot on inset axis ---
                ax_ins.step(bins_plot3[:-1], counts, where='post', color=colors[name], alpha=0.7, linewidth=2)

        # *** NEW: Plot the sum ***
        all_tots_total = np.concatenate(all_tots_list)
        if all_tots_total.size > 0:
            counts_sum, _ = np.histogram(all_tots_total, bins=bins_plot3)
            
            # Plot sum on main linear axis
            ax_lin.step(bins_plot3[:-1], counts_sum, where='post', color=colors['sum'], label='sum', linewidth=1.5, linestyle='--')
            # Plot sum on main log axis
            ax_log.step(bins_plot3[:-1], counts_sum, where='post', color=colors['sum'], label='sum', linewidth=1.5, linestyle='--')
            
            # --- MODIFIED: Plot sum on inset axis ---
            ax_ins.step(bins_plot3[:-1], counts_sum, where='post', color=colors['sum'], linewidth=1.5, linestyle='--')
            
            # --- MODIFIED: Calculate inset y-limit from the 'sum' data ---
            # We want data from x=0 up to x=50. This is bins 0 to 50 (51 items).
            if counts_sum[0:51].size > 0:
                max_y_inset = np.max(counts_sum[0:51])

    # --- MODIFIED: Configure Inset Axes ---
    ax_ins.set_xlim(0, 30)
    if max_y_inset > 0:
        ax_ins.set_ylim(0, max_y_inset * 1.05) # 5% padding
    else:
        ax_ins.set_ylim(0, 1) # Fallback
        
    ax_ins.set_xlabel('ToT', fontsize=9)
    ax_ins.set_ylabel('Hits', fontsize=9)
    ax_ins.tick_params(axis='both', which='major', labelsize=8)
    ax_ins.grid(True, which='both', linestyle=':', linewidth=0.5)
    # --- END OF INSET MODIFICATION ---

    ax_lin.set_title('Normal Scale')
    ax_lin.set_xlabel('ToT')
    ax_lin.set_ylabel('Number of Hits')
    ax_lin.legend(loc='upper right')
    ax_lin.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- MODIFIED: Add zoom indicator box to main linear plot ---
    ax_lin.indicate_inset_zoom(ax_ins, edgecolor="black", alpha=0.5)

    ax_log.set_title('Log Scale')
    ax_log.set_xlabel('ToT')
    ax_log.set_ylabel('Number of Hits')
    if log_y:
        ax_log.set_yscale('log')
    ax_log.legend()
    ax_log.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig3.suptitle('Time-over-Threshold (ToT) Spectrum', fontsize=16)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()