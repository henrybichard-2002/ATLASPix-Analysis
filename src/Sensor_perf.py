# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 13:41:23 2026

@author: henry
"""

import os
from data_loading_optimized import load_data_numpy
from utils import numpy_to_dataframe, layer_split

# Hardcoded arguments
file_path = r"C:\Users\henry\ATLASpix-analysis\data\202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
file_path2 = r"C:\Users\henry\ATLASpix-analysis\data\202204061308_udp_beamonall_6Gev_kit_0_decode.dat"

n_lines = 50000000 # Load all lines


filename = os.path.splitext(os.path.basename(file_path2))[0]

# Load data
data_raw = load_data_numpy(file_path2, n_lines=n_lines)
if data_raw is None:
    print("File not found")
    exit()
data_raw_df = numpy_to_dataframe(data_raw)
data_layer_split = layer_split(data_raw)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_raw_L4 = data_layer_split[-1]
data_raw_L4_df = numpy_to_dataframe(data_raw_L4)

import matplotlib.pyplot as plt
import numpy as np

def plot_timing_jitter(data_raw, x_mode='Row', y_lim=None, c_lim=None):
    """
    Fast 2D histogram of Timing Jitter vs Variable.
    
    Parameters:
    - data_raw: The dataset dictionary.
    - x_mode: 'Row', 'Column', or 'ToT'.
    - y_lim: Tuple (min, max) for the Timing Difference axis. 
             If None, calculates robust range (1-99 percentile).
    - c_lim: Tuple (min, max) for the colorbar (count) intensity.
    """
    
    # 1. High-Speed Data Extraction (Vectorized)
    # Convert to int64 to handle negative time differences safely
    trigger_ts = data_raw['TriggerTS'].astype(np.int64)
    ext_ts = data_raw['ext_TS'].astype(np.int64)
    layers = data_raw['Layer']
    
    # Calculate Jitter
    jitter = ext_ts - trigger_ts
    
    # Select X-axis Data
    if x_mode == 'Row':
        x_data = data_raw['Row']
        x_bins = np.arange(0, 373) # 0-371
        x_label = 'Row Index'
    elif x_mode == 'ToT':
        x_data = data_raw['ToT']
        x_bins = np.arange(0, 257) # 0-255
        x_label = 'ToT Value'
    elif x_mode == 'Column':
        x_data = data_raw['Column']
        # Dynamic binning for Column as it varies by detector config
        x_bins = np.arange(0, np.max(x_data) + 2) 
        x_label = 'Column Index'
    else:
        raise ValueError("x_mode must be 'Row', 'Column', or 'ToT'")

    # 2. Handle Y-Limits (Robust Auto-scaling if not provided)
    if y_lim is None:
        # Calculate percentiles to ignore massive outliers
        y_min, y_max = np.percentile(jitter, [1, 99])
        # Add 10% breathing room
        pad = (y_max - y_min) * 0.1
        if pad == 0: pad = 10
        y_lim = (y_min - pad, y_max + pad)

    # 3. Sort-and-Split Optimization (Fastest for Large Data)
    # Sorting allows us to slice views rather than copy memory with masks
    sort_idx = np.argsort(layers)
    layers_sorted = layers[sort_idx]
    jitter_sorted = jitter[sort_idx]
    x_data_sorted = x_data[sort_idx]
    
    unique_layers, start_indices = np.unique(layers_sorted, return_index=True)
    n_layers = len(unique_layers)
    
    # 4. Plotting
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 5 * n_layers), constrained_layout=True)
    if n_layers == 1: axes = [axes] # Ensure iterable

    for i, layer_id in enumerate(unique_layers):
        # Slice data for this layer (Instant)
        start = start_indices[i]
        end = start_indices[i+1] if i + 1 < n_layers else None
        
        layer_jitter = jitter_sorted[start:end]
        layer_x = x_data_sorted[start:end]
        
        ax = axes[i]
        
        # Determine Color Limits (vmin, vmax)
        vmin, vmax = c_lim if c_lim else (None, None)

        # Plot 2D Histogram
        # range=[x_range, y_lim] forces the axis to your exact specs
        h = ax.hist2d(layer_x, layer_jitter, 
                      bins=[x_bins, 100], 
                      range=[[x_bins[0], x_bins[-1]], y_lim],
                      cmap='viridis', 
                      cmin=1, # Makes 0-count bins transparent (white)
                      vmin=vmin, 
                      vmax=vmax)
        
        ax.set_title(f'Layer {layer_id} Timing Jitter vs {x_mode}')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Jitter (ext_TS - TriggerTS)')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Colorbar
        cb = plt.colorbar(h[3], ax=ax)
        cb.set_label('Hit Count')

    plt.show()
    
plot_timing_jitter(data_raw, x_mode='Row', y_lim=(-400,2000), c_lim=None)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

def plot_sensor_heatmaps(data_raw, target_layer=None, highlight_outliers=False, c_scales=None):
    """
    Plots 4 heatmaps per layer: Counts, Avg ToT, ToT Std, and Mean Timing Jitter.

    Parameters:
    - highlight_outliers: If True, plots empty maps and highlights 1st/99th percentile pixels.
    - c_scales: Dictionary with keys 'counts', 'tot_avg', 'tot_std', 'jitter' to set (min, max).
      Example: c_scales={'counts': (0, 100), 'jitter': (-5, 5)}
    """
    if c_scales is None: c_scales = {}

    # 1. Load Data
    df = pd.DataFrame(data_raw)

    # Calculate Jitter column first so we can aggregate it
    # Convert to int64 to avoid overflow on subtraction
    df['Jitter'] = df['ext_TS'].astype(np.int64) - df['TriggerTS'].astype(np.int64)

    if target_layer is not None:
        unique_layers = [target_layer]
    else:
        unique_layers = np.sort(df['Layer'].unique())

    for layer in unique_layers:
        layer_df = df[df['Layer'] == layer]

        # 2. Pixel Aggregation
        # Group by (Col, Row) to get per-pixel stats
        pixel_stats = layer_df.groupby(['Column', 'Row']).agg(
            counts=('ToT', 'count'),
            tot_avg=('ToT', 'mean'),
            tot_std=('ToT', 'std'),
            jitter=('Jitter', 'mean')
        ).reset_index()

        # 3. Create Pivot Tables (Heatmap grids)
        # We assume Rows (Y) and Columns (X)
        # fill_value=0/NaN ensures proper empty pixel handling
        grid_counts = pixel_stats.pivot(index='Row', columns='Column', values='counts').fillna(0)
        grid_tot_avg = pixel_stats.pivot(index='Row', columns='Column', values='tot_avg') # keep NaN for empty
        grid_tot_std = pixel_stats.pivot(index='Row', columns='Column', values='tot_std')
        grid_jitter = pixel_stats.pivot(index='Row', columns='Column', values='jitter')

        # Align all grids to the same shape (Max Row x Max Col)
        # This ensures the plots don't shrink if edge pixels are missing
        max_r, max_c = 371, int(df['Column'].max()) 

        # List of matrices and titles for iteration
        maps = [
            ('Hit Counts', grid_counts, 'counts', 'Greens'),
            ('Avg ToT', grid_tot_avg, 'tot_avg', 'plasma'),
            ('ToT Std Dev', grid_tot_std, 'tot_std', 'magma'),
            ('Mean Jitter', grid_jitter, 'jitter', 'jet')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        fig.suptitle(f'Layer {layer} Sensor Analysis', fontsize=16)
        axes = axes.flatten()

        for ax, (title, data_grid, key, cmap_name) in zip(axes, maps):

            # --- OUTLIER MODE ---
            if highlight_outliers:
                # Flatten data to find percentiles (ignoring NaNs)
                flat_data = data_grid.values.flatten()
                flat_data = flat_data[~np.isnan(flat_data)]

                if len(flat_data) > 0:
                    p1 = np.percentile(flat_data, 1)
                    p99 = np.percentile(flat_data, 99)

                    # Create a Custom RGB Map
                    # Start with White (1,1,1)
                    # Use exact dimensions of the data grid
                    rgb_map = np.ones((data_grid.shape[0], data_grid.shape[1], 3))

                    grid_vals = data_grid.values

                    # Red for > 99th, Blue for < 1st
                    mask_high = (grid_vals >= p99)
                    mask_low = (grid_vals <= p1)

                    # Apply colors
                    # Red (1, 0, 0)
                    rgb_map[mask_high] = [1, 0, 0] 
                    # Blue (0, 0, 1)
                    rgb_map[mask_low] = [0, 0, 1]

                    ax.imshow(rgb_map, origin='lower', aspect='auto')

                    # Legend Text
                    stats_text = (f"Outlier Thresholds:\n"
                                  f"High (Red): > {p99:.2f}\n"
                                  f"Low (Blue): < {p1:.2f}")
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                            va='top', ha='left', bbox=dict(facecolor='white', alpha=0.9))
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center')

            # --- STANDARD HEATMAP MODE ---
            else:
                # Get limits from dictionary or auto
                vmin, vmax = c_scales.get(key, (None, None))

                im = ax.imshow(data_grid, origin='lower', aspect='auto', 
                               cmap=cmap_name, vmin=vmin, vmax=vmax)
                cb = plt.colorbar(im, ax=ax)
                cb.set_label(title)

            ax.set_title(title)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')

    plt.show()
    

plot_sensor_heatmaps(data_raw, target_layer=4, highlight_outliers=False, c_scales={'tot_avg':(30,175),
                                                                                   'tot_std':(50,110), 'jitter':(-5000,5000)})

plot_sensor_heatmaps(data_raw, target_layer=4, highlight_outliers=True, c_scales={'tot_avg':(30,175),
                                                                                   'tot_std':(50,110), 'jitter':(-4000,0)})