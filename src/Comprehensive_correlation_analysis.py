# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:56:04 2025

@author: henry
"""
import numpy as np
import os
from data_loading_optimized import load_data_numpy
from utils import numpy_to_dataframe, layer_split, progress_bar

# Hardcoded arguments
file_path = r"C:\Users\henry\ATLASpix-analysis\data\202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
file_path2 = r"C:\Users\henry\ATLASpix-analysis\data\202204061308_udp_beamonall_6Gev_kit_0_decode.dat"
#gamma = 86.5

n_lines = 5000000 # Load all lines


filename = os.path.splitext(os.path.basename(file_path))[0]

# Load data
data_raw = load_data_numpy(file_path, n_lines=n_lines)
if data_raw is None:
    print("File not found")
    exit()
data_raw_df = numpy_to_dataframe(data_raw)
data_layer_split = layer_split(data_raw)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_raw_L4 = data_layer_split[-1]

import pandas as pd

def separate_correlated_hits(data_raw, 
                             dx=30, 
                             dx2=None, 
                             use_trigger_ts=False, 
                             balance_uncorrelated=True, 
                             tot_thresh=0, 
                             ratio_thresh=0.0):
    """
    Separates hits into correlated and uncorrelated datasets.
    
    If dx2 is provided (not None):
        A hit is correlated ONLY if:
        1. It is separated from another hit in the same group by exactly dx2.
        2. BOTH hits have ToT > tot_thresh.
        3. The ToT Ratio (Min/Max) between the pair is > ratio_thresh.
        
    If dx2 is None:
        The original range logic (any neighbor > dx) is used. (ToT/Ratio filters are ignored in this mode).
    
    Args:
        data_raw (dict): Dictionary of numpy arrays.
        dx (int): Minimum row separation (used if dx2 is None).
        dx2 (int, optional): Exact row separation required to be correlated. 
                             Overrides dx if not None.
        use_trigger_ts (bool): If True, group by TriggerTS instead of TriggerID.
        balance_uncorrelated (bool): If True, subsamples the uncorrelated dataset 
                                     to match the length of the correlated dataset.
        tot_thresh (float): Minimum ToT for BOTH hits in a pair (only used if dx2 is set).
        ratio_thresh (float): Minimum ToT Ratio (min/max) for a pair (only used if dx2 is set).
        
    Returns:
        tuple: (correlated_data, uncorrelated_data)
    """
    trigger_key = 'TriggerTS' if use_trigger_ts else 'TriggerID'
    group_keys = ['Layer', 'Column', trigger_key]
    
    print("Building DataFrame for processing...")
    # Create dataframe for calculation. Include ToT for filtering.
    df_keys = pd.DataFrame({
        'Layer': data_raw['Layer'],
        'Column': data_raw['Column'],
        trigger_key: data_raw[trigger_key],
        'Row': data_raw['Row'],
        'ToT': data_raw['ToT']
    })
    
    # --- 1. Determine Correlated Mask ---
    if dx2 is not None:
        print(f"Identifying hits: dRow=={dx2}, ToT>{tot_thresh}, Ratio>{ratio_thresh}...")
        
        # We use a merge strategy here to access properties of the PAIR (ToT, Ratio)
        # Reset index to preserve original indices for the final mask
        df_proc = df_keys.reset_index()
        
        # Optimization: Pre-filter by ToT if threshold exists
        if tot_thresh > 0:
            df_proc = df_proc[df_proc['ToT'] > tot_thresh]
            
        # Self Merge to find neighbors
        # Inner join only keeps hits that have potential partners
        merged = pd.merge(df_proc, df_proc, on=group_keys, suffixes=('_1', '_2'))
        
        # Filter 1: Unique pairs (Index A < Index B) to avoid self-matches and duplicates
        merged = merged[merged['index_1'] < merged['index_2']]
        
        # Filter 2: Exact Spatial Separation
        # Use int cast to avoid uint overflow
        d_row = np.abs(merged['Row_1'].astype(int) - merged['Row_2'].astype(int))
        merged = merged[d_row == dx2]
        
        # Filter 3: Ratio Threshold
        if ratio_thresh > 0 and not merged.empty:
            t1 = merged['ToT_1'].values.astype(float)
            t2 = merged['ToT_2'].values.astype(float)
            
            # Avoid div/0
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.minimum(t1, t2) / np.maximum(t1, t2)
                ratios = np.nan_to_num(ratios, nan=0.0)
            
            merged = merged[ratios > ratio_thresh]
            
        # Collect Indices
        if not merged.empty:
            corr_idx = np.union1d(merged['index_1'].values, merged['index_2'].values)
            mask_correlated = np.zeros(len(df_keys), dtype=bool)
            mask_correlated[corr_idx] = True
        else:
            mask_correlated = np.zeros(len(df_keys), dtype=bool)
        
    else:
        # Original "Range" Logic (separation > dx)
        print(f"Identifying hits separated by > {dx}...")
        grouped = df_keys.groupby(group_keys)['Row']
        min_rows = grouped.transform('min')
        max_rows = grouped.transform('max')
        
        mask_correlated = (
            ((df_keys['Row'] - min_rows) > dx) | 
            ((max_rows - df_keys['Row']) > dx)
        ).values

    # --- 2. Process Indices ---
    total_indices = np.arange(len(df_keys))
    correlated_indices = total_indices[mask_correlated]
    candidate_uncorrelated_indices = total_indices[~mask_correlated]
    
    # Default to all candidates
    final_uncorrelated_indices = candidate_uncorrelated_indices

    # Optional Balancing
    if balance_uncorrelated:
        n_corr = len(correlated_indices)
        n_avail_uncorr = len(candidate_uncorrelated_indices)
        
        print(f"Correlated hits: {n_corr}. Balancing uncorrelated dataset...")

        if n_corr == 0:
            print("Warning: No correlated hits found matching criteria.")
            final_uncorrelated_indices = np.array([], dtype=int)
        elif n_avail_uncorr == 0:
            print("Warning: All hits are correlated. Cannot create uncorrelated dataset.")
            final_uncorrelated_indices = np.array([], dtype=int)
        else:
            should_replace = n_avail_uncorr < n_corr
            # Randomly sample to match size
            final_uncorrelated_indices = np.random.choice(
                candidate_uncorrelated_indices, 
                size=n_corr, 
                replace=should_replace
            )
    else:
        print(f"Correlated hits: {len(correlated_indices)}. Uncorrelated hits: {len(candidate_uncorrelated_indices)} (No balancing).")

    # --- 3. Slice Data ---
    correlated_data = {}
    uncorrelated_data = {}
    
    iterator = progress_bar(
        data_raw.items(), 
        description="Slicing Datasets", 
        total=len(data_raw)
    )
    
    for key, array in iterator:
        correlated_data[key] = array[correlated_indices]
        uncorrelated_data[key] = array[final_uncorrelated_indices]
        
    return correlated_data, uncorrelated_data

correlated_data, uncorrelated_data = separate_correlated_hits(data_raw, use_trigger_ts=False, dx = 5, dx2 = None, balance_uncorrelated=False)

import matplotlib.pyplot as plt
import numpy as np


def plot_tot_spectra(data_raw, data_corr, data_uncorr, log_scale=False, normalize=True):
    """
    Plots ToT spectra for each layer comparing Raw, Correlated, and Uncorrelated data.
    Optionally normalizes Raw and Uncorrelated data to the total count of Correlated data.
    
    Includes a marginal subplot showing the Ratio of Uncorrelated / Correlated Hits.
    
    Args:
        data_raw (dict): Dictionary containing the full raw dataset.
        data_corr (dict): Dictionary containing the correlated hits.
        data_uncorr (dict): Dictionary containing the uncorrelated hits.
        log_scale (bool): If True, sets the y-axis of the main spectra to logarithmic scale.
        normalize (bool): If True (default), scales Raw and Uncorrelated histograms to match 
                          the total area (count) of the Correlated histogram.
    """
    # Identify unique layers from the raw data
    layers = np.unique(data_raw['Layer'])
    layers.sort()
    
    # Check if we have more layers than expected
    num_layers = len(layers)
    if num_layers > 4:
        print(f"Notice: Found {num_layers} layers. Plotting first 4 only.")
        layers = layers[:4]

    # Setup the figure
    fig = plt.figure(figsize=(14, 12))
    
    # Create an outer 2x2 grid for the layers
    # hspace/wspace controls gap between the 4 main layer groups
    outer_grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    # Loop through each layer and plot
    for i, layer_id in enumerate(layers):
        # Create a nested grid for this specific layer: 
        # Top panel (3) for Spectra, Bottom panel (1) for Ratio
        inner_grid = outer_grid[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
        
        ax_main = fig.add_subplot(inner_grid[0])
        ax_ratio = fig.add_subplot(inner_grid[1], sharex=ax_main)
        
        # 1. Create Masks for the current layer
        mask_raw = data_raw['Layer'] == layer_id
        mask_corr = data_corr['Layer'] == layer_id
        mask_uncorr = data_uncorr['Layer'] == layer_id
        
        # 2. Extract ToT data
        tot_raw = data_raw['ToT'][mask_raw]
        tot_corr = data_corr['ToT'][mask_corr]
        tot_uncorr = data_uncorr['ToT'][mask_uncorr]
        
        if len(tot_raw) == 0:
            ax_main.text(0.5, 0.5, 'No Data', transform=ax_main.transAxes, ha='center')
            ax_main.set_title(f"Layer {layer_id}")
            continue

        # 3. Calculate Normalization Weights
        n_corr = len(tot_corr)
        n_uncorr = len(tot_uncorr)
        n_raw = len(tot_raw)
        
        w_uncorr = None
        w_raw = None
        
        lbl_raw = 'Raw Data'
        lbl_uncorr = 'Uncorrelated'
        title_suffix = ''
        y_label = 'Count'

        if normalize:
            # Target N is the number of correlated hits
            # Avoid division by zero
            w_uncorr = np.ones(n_uncorr) * (n_corr / n_uncorr) if n_uncorr > 0 else None
            w_raw = np.ones(n_raw) * (n_corr / n_raw) if n_raw > 0 else None
            
            lbl_raw = 'Raw (Norm)'
            lbl_uncorr = 'Uncorr (Norm)'
            title_suffix = f' (Normalized to N={n_corr})'
            y_label = 'Weighted Counts'
        
        bins = np.arange(-0.5, 256.5, 1)
        
        # 5. Plot Main Spectra
        if log_scale:
            ax_main.set_yscale('log')
            ax_ratio.set_yscale('log')
            
        # Raw Data (Background)
        ax_main.hist(tot_raw, bins=bins, weights=w_raw, color='lightgray', label=lbl_raw, 
                     alpha=0.7, edgecolor='none', zorder=1)
        
        # Correlated & Uncorrelated (Outlines)
        ax_main.hist(tot_corr, bins=bins, histtype='step', linewidth=2, 
                     label='Correlated', color='#1f77b4', zorder=3)
        
        ax_main.hist(tot_uncorr, bins=bins, weights=w_uncorr, histtype='step', linewidth=2, 
                     label=lbl_uncorr, color='#ff7f0e', linestyle='--', zorder=2)

        # 6. Calculate and Plot Ratio (Uncorrelated / Correlated)
        # Weights used here ensure ratio reflects the visual plot (normalized or not)
        counts_corr, _ = np.histogram(tot_corr, bins=bins) # weights are 1 implicitly
        counts_uncorr, _ = np.histogram(tot_uncorr, bins=bins, weights=w_uncorr)
        
        # Calculate ratio, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.true_divide(counts_uncorr, counts_corr)
            ratio[~np.isfinite(ratio)] = np.nan  # Hide points where Corr is 0
        
        # Plot ratio
        ax_ratio.hist(bins[:-1], bins=bins, weights=ratio, 
                      histtype='step', color='purple', label='Uncorr / Corr')
        
        # Reference line at 1.0 (where shapes/counts match)
        ax_ratio.axhline(1.0, color='gray', linewidth=0.8, linestyle=':', alpha=0.8)

        # Formatting
        ax_main.set_title(f'Layer {layer_id} ToT Spectrum{title_suffix}')
        ax_main.set_ylabel(y_label)
        ax_main.legend(loc='upper right', fontsize='small')
        ax_main.grid(True, alpha=0.3, linestyle=':')
        ax_main.set_xlim(-0.5, 255.5)
        
        # Remove x-tick labels from the top plot (ax_main)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        
        # Format the ratio plot
        ax_ratio.set_ylabel('Ratio\n(Uncorr / Corr)', fontsize=9)
        ax_ratio.set_xlabel('ToT Value')
        ax_ratio.grid(True, alpha=0.3, linestyle=':')
        
        # Optional: Align y-labels horizontally
        ax_main.yaxis.set_label_coords(-0.15, 0.5)
        ax_ratio.yaxis.set_label_coords(-0.15, 0.5)

    plt.show()
    
plot_tot_spectra(data_raw, correlated_data, uncorrelated_data, normalize=False)
plot_tot_spectra(data_raw, correlated_data, uncorrelated_data, normalize=False, log_scale = True)
from plotting_optimized import plot_HeatHitmap
plot_HeatHitmap(data_raw, xcol = 'Column', ycol = 'Row', title='Raw_Hitdata', log_z= False, vmin = 0, vmax = 100)
plot_HeatHitmap(correlated_data, xcol = 'Column', ycol = 'Row', title='correlated_Hitdata', log_z= False, vmin = 0, vmax = 100)
plot_HeatHitmap(uncorrelated_data, xcol = 'Column', ycol = 'Row', title='uncorrelated_Hitdata', log_z= False, vmin = 0, vmax = 100)
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_correlated_ratios(layer_data, tot_thresh=15):
    """
    Calculates ToT ratios (min/max) for unique pairs of hits sharing a TriggerID.
    Uses dynamic batching to prevent MemoryErrors on dense datasets.
    """
    # Lightweight DF
    df = pd.DataFrame({
        'TriggerID': layer_data['TriggerID'],
        'ToT': layer_data['ToT']
    })
    
    # Identify multi-hit triggers
    trigger_counts = df['TriggerID'].value_counts()
    valid_counts = trigger_counts[trigger_counts > 1]
    
    if len(valid_counts) == 0:
        return np.array([])

    results = []
    
    # Memory Safety Settings
    MAX_PAIRS_PER_BATCH = 10_000_000 
    MAX_HITS_SINGLE_TRIGGER = 3000 
    
    batch_triggers = []
    batch_pair_sum = 0
    
    for trigger_id, count in valid_counts.items():
        if count > MAX_HITS_SINGLE_TRIGGER:
            continue
            
        n_pairs = count * count
        
        # Batch processing trigger
        if batch_pair_sum + n_pairs > MAX_PAIRS_PER_BATCH:
            _process_ratio_batch(df, batch_triggers, results, tot_thresh)
            batch_triggers = []
            batch_pair_sum = 0
            
        batch_triggers.append(trigger_id)
        batch_pair_sum += n_pairs
        
    # Process final batch
    if batch_triggers:
        _process_ratio_batch(df, batch_triggers, results, tot_thresh)

    return np.concatenate(results) if results else np.array([])

def _process_ratio_batch(df, batch_ids, results_list, tot_thresh):
    """Helper: processes a batch of TriggerIDs for ratio calculation."""
    df_subset = df[df['TriggerID'].isin(batch_ids)].reset_index()
    
    # Vectorized Self-Merge
    merged = pd.merge(df_subset, df_subset, on='TriggerID', suffixes=('_1', '_2'))
    
    # Filter unique pairs (A < B)
    unique_pairs = merged[merged['index_1'] < merged['index_2']]
    if unique_pairs.empty:
        return

    t1 = unique_pairs['ToT_1'].values.astype(float)
    t2 = unique_pairs['ToT_2'].values.astype(float)
    
    # Filters: Non-zero and Threshold
    mask = (t1 > 0) & (t2 > 0) & ((t1 >= tot_thresh) | (t2 >= tot_thresh))
    t1, t2 = t1[mask], t2[mask]
    
    if len(t1) == 0:
        return

    # Calculate Ratios
    min_t = np.minimum(t1, t2)
    max_t = np.maximum(t1, t2)
    
    # Final zero check just in case
    valid = max_t > 0
    results_list.append(min_t[valid] / max_t[valid])

def get_random_ratios(tot_array, num_samples, tot_thresh=15):
    """
    Generates random ToT ratios from the dataset to match a target sample size.
    """
    tot_array = tot_array[tot_array > 0] # Filter zeros upfront
    if len(tot_array) < 2 or num_samples == 0:
        return np.array([])
    
    collected = []
    needed = num_samples
    
    while needed > 0:
        # Oversample to account for threshold filtering
        batch = int(needed * 1.5) + 100
        pairs = np.random.choice(tot_array, size=(batch, 2), replace=True)
        
        t1, t2 = pairs[:, 0], pairs[:, 1]
        
        # Keep if at least one hit passes threshold
        mask = (t1 >= tot_thresh) | (t2 >= tot_thresh)
        valid_pairs = pairs[mask]
        
        if len(valid_pairs) == 0:
            if len(tot_array) < 10: break 
            continue

        mins = valid_pairs.min(axis=1)
        maxs = valid_pairs.max(axis=1)
        
        # Calculate ratios
        with np.errstate(divide='ignore', invalid='ignore'):
            r = mins / maxs
            r = r[np.isfinite(r)]
            
        collected.append(r)
        needed -= len(r)
        
        # Safety break for tiny datasets
        if len(tot_array) < 50 and len(collected) > 5:
            break

    if not collected:
        return np.array([])

    return np.concatenate(collected)[:num_samples]

# --- Spatial / Heatmap Functions ---

def get_spatial_pairs_corr(layer_data, dx=50):
    """
    Finds pairs sharing TriggerID AND Column.
    Returns: r1, r2, t1, t2
    """
    df = pd.DataFrame({
        'TriggerID': layer_data['TriggerID'],
        'Column': layer_data['Column'],
        'Row': layer_data['Row'],
        'ToT': layer_data['ToT']
    })
    
    # Filter for groups with > 1 hit
    counts = df.groupby(['TriggerID', 'Column']).size()
    valid_groups = counts[counts > 1].index
    
    if len(valid_groups) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    # Filter DF to relevant triggers only
    vc = df['TriggerID'].value_counts()
    valid_triggers = vc[vc > 1].index
    df_reduced = df[df['TriggerID'].isin(valid_triggers)].reset_index()
    
    # Self-merge on TriggerID and Column
    merged = pd.merge(df_reduced, df_reduced, on=['TriggerID', 'Column'], suffixes=('_1', '_2'))
    
    # Unique pairs only
    unique = merged[merged['index_1'] < merged['index_2']]
    
    r1 = unique['Row_1'].values
    r2 = unique['Row_2'].values
    t1 = unique['ToT_1'].values
    t2 = unique['ToT_2'].values
    
    return _filter_by_dx(r1, r2, t1, t2, dx)

def get_spatial_pairs_random(layer_data, n_required, dx=50):
    """
    Generates random pairs sharing Column (any TriggerID).
    Returns: r1, r2, t1, t2
    """
    if n_required <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    df = pd.DataFrame({
        'Column': layer_data['Column'],
        'Row': layer_data['Row'],
        'ToT': layer_data['ToT']
    })
    
    col_counts = df['Column'].value_counts()
    valid_cols = col_counts[col_counts >= 2]
    
    if len(valid_cols) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    # Weight sampling by combinatorial frequency (nC2)
    weights = valid_cols * (valid_cols - 1) / 2
    weights = weights / weights.sum()
    
    # Oversample to account for dx filtering
    n_oversample = int(n_required * 3.0) + 200
    chosen_cols = np.random.choice(valid_cols.index, size=n_oversample, p=weights.values)
    
    # Group data by column for fast lookup
    # {col: (rows, tots)}
    grouped = df.groupby('Column')
    col_map = {c: (grp['Row'].values, grp['ToT'].values) for c, grp in grouped}
    
    r1_list, r2_list, t1_list, t2_list = [], [], [], []
    
    # Iterate over counts of chosen columns (much faster than iterating samples)
    for col, count in pd.Series(chosen_cols).value_counts().items():
        rows, tots = col_map[col]
        n_hits = len(rows)
        
        # Random pairs
        idx1 = np.random.randint(0, n_hits, size=count)
        offset = np.random.randint(1, n_hits, size=count)
        idx2 = (idx1 + offset) % n_hits
        
        r1_list.append(rows[idx1])
        r2_list.append(rows[idx2])
        t1_list.append(tots[idx1])
        t2_list.append(tots[idx2])
            
    if not r1_list:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Concatenate results
    r1 = np.concatenate(r1_list)
    r2 = np.concatenate(r2_list)
    t1 = np.concatenate(t1_list)
    t2 = np.concatenate(t2_list)
    
    # Filter by dx
    r1, r2, t1, t2 = _filter_by_dx(r1, r2, t1, t2, dx)
    
    # Limit to requested size
    if len(r1) > n_required:
        # Shuffle to ensure random subset
        perm = np.random.permutation(len(r1))[:n_required]
        return r1[perm], r2[perm], t1[perm], t2[perm]
        
    return r1, r2, t1, t2

def _filter_by_dx(r1, r2, t1, t2, dx):
    """Filters pairs where row displacement is <= dx."""
    diff = np.abs(r1.astype(int) - r2.astype(int))
    mask = diff > dx
    return r1[mask], r2[mask], t1[mask], t2[mask]

def _compute_heatmap_grid(x_val, y_val, t1, t2, x_edges, y_edges, mode):
    """Computes the 2D grid values based on the color mode."""
    if mode == 'counts':
        H, _, _ = np.histogram2d(x_val, y_val, bins=[x_edges, y_edges])
        # Mask zeros for visualization if needed, handled by cmin in hist2d usually
        return H, H.max(), 1 
    
    # Prepare Z values
    if mode == 'ratio':
        with np.errstate(divide='ignore', invalid='ignore'):
            z_vals = np.minimum(t1, t2) / np.maximum(t1, t2)
            z_vals = np.nan_to_num(z_vals, nan=0.0)
    elif mode == 'sum':
        z_vals = t1 + t2
    
    # Compute Weighted Average per bin
    H_sum, _, _ = np.histogram2d(x_val, y_val, bins=[x_edges, y_edges], weights=z_vals)
    H_count, _, _ = np.histogram2d(x_val, y_val, bins=[x_edges, y_edges])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = H_sum / H_count
        # Set empty bins to NaN for transparency
        H_avg[H_count < 1] = np.nan
        
    valid_vals = H_avg[~np.isnan(H_avg)]
    v_max = valid_vals.max() if len(valid_vals) > 0 else 1
    v_min = valid_vals.min() if len(valid_vals) > 0 else 0
    
    return H_avg.T, v_max, v_min

# --- Plotting Functions ---

def plot_tot_ratio_distribution(data_raw, data_corr, data_uncorr, tot_thresh=15, n_bins=100, log_yscale=False):
    """
    Plots ToT ratio distributions (Linear and Log X-axis) with marginal Ratio plots.
    """
    layers = np.unique(data_raw['Layer'])
    layers.sort()
    layers = layers[:4] if len(layers) > 4 else layers
    
    # Setup Figures
    def create_fig(title):
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(title, fontsize=14)
        outer = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        axes = []
        for i in range(4):
            inner = outer[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
            ax_m = fig.add_subplot(inner[0])
            ax_r = fig.add_subplot(inner[1], sharex=ax_m)
            plt.setp(ax_m.get_xticklabels(), visible=False)
            axes.append((ax_m, ax_r))
        return fig, axes

    fig_lin, ax_lin = create_fig(f"ToT Ratio Dist (Linear) - Thresh {tot_thresh}")
    fig_log, ax_log = create_fig(f"ToT Ratio Dist (Log) - Thresh {tot_thresh}")
    
    bins_lin = np.linspace(0, 1, n_bins + 1)
    bins_log = np.logspace(np.log10(1/255), np.log10(1), n_bins + 1)
    
    # State tracking for uniform scaling
    gmax_lin = 0
    gmax_log = 0
    valid_idx = []

    for i, lid in enumerate(progress_bar(layers, "Processing Layers")):
        # Slice Data
        def get_l(d): return {k: v[d['Layer'] == lid] for k,v in d.items()}
        l_corr, l_uncorr, l_raw = get_l(data_corr), get_l(data_uncorr), get_l(data_raw)
        
        # Calculate Ratios
        r_corr = get_correlated_ratios(l_corr, tot_thresh)
        n = len(r_corr)
        
        # Helper to plot one subplot set
        def plot_sub(ax_m, ax_r, bins, is_log_x):
            if n == 0:
                ax_m.text(0.5, 0.5, 'Insufficient Pairs', transform=ax_m.transAxes, ha='center')
                ax_m.set_title(f"Layer {lid}")
                return 0
            
            r_uncorr = get_random_ratios(l_uncorr['ToT'], n, tot_thresh)
            r_raw = get_random_ratios(l_raw['ToT'], n, tot_thresh)
            
            if log_yscale: ax_m.set_yscale('log')
            if is_log_x: 
                ax_m.set_xscale('log')
                ax_r.set_xscale('log')
                
            kw = dict(histtype='step', linewidth=1.5)
            # Main Histograms
            nr, _, _ = ax_m.hist(r_raw, bins, color='gray', label='Raw', alpha=0.6, **kw)
            nu, _, _ = ax_m.hist(r_uncorr, bins, color='#ff7f0e', label='Uncorr', linestyle='--', **kw)
            nc, _, _ = ax_m.hist(r_corr, bins, color='#1f77b4', label='Corr', **kw)
            
            ax_m.set_title(f"Layer {lid} (N={n})")
            ax_m.set_ylabel("Count")
            ax_m.legend(loc='upper left', fontsize='small')
            if not is_log_x: ax_m.set_xlim(0, 1)

            # Ratio Plot
            with np.errstate(divide='ignore', invalid='ignore'):
                rat_ur = np.divide(nu, nr)
                rat_cr = np.divide(nc, nr)
                
            ax_r.hist(bins[:-1], bins, weights=rat_ur, histtype='step', color='purple', linestyle='--', label='Uncorr/Raw')
            ax_r.hist(bins[:-1], bins, weights=rat_cr, histtype='step', color='green', linestyle=':', label='Corr/Raw')
            ax_r.axhline(1, color='black', alpha=0.3)
            ax_r.set_ylabel("Ratio")
            ax_r.legend(fontsize='x-small')
            
            return max(nr.max(), nu.max(), nc.max())

        # Plot both linear and log versions
        ml = plot_sub(ax_lin[i][0], ax_lin[i][1], bins_lin, False)
        mg = plot_sub(ax_log[i][0], ax_log[i][1], bins_log, True)
        
        if ml > gmax_lin: gmax_lin = ml
        if mg > gmax_log: gmax_log = mg
        valid_idx.append(i)

    # Uniform Limits
    bottom = 0.5 if log_yscale else 0
    for i in range(4):
        if i in valid_idx:
            ax_lin[i][0].set_ylim(bottom, gmax_lin * 1.1)
            ax_log[i][0].set_ylim(bottom, gmax_log * 1.1)
        else:
            for axes in [ax_lin, ax_log]:
                axes[i][0].set_visible(False)
                axes[i][1].set_visible(False)

    # Removed tight_layout calls to avoid warnings with GridSpec
    plt.show()

def plot_correlation_heatmaps(data_raw, data_corr, data_uncorr, bin_size=2, log_cscale=False, dx=50, color_mode='counts', raw_corr_mode='random'):
    """
    Plots 3 figures (Correlated, Uncorrelated, Raw) showing spatial correlation heatmaps.
    X-axis: Abs Displacement |r1-r2|. Y-axis: Sum r1+r2.
    
    Args:
        raw_corr_mode (str): 'random' (default) finds random background pairs for Raw.
                             'trigger' finds actual spatial correlations (TriggerID) for Raw.
    """
    datasets = {'Correlated': data_corr, 'Uncorrelated': data_uncorr, 'Raw': data_raw}
    layers = np.unique(data_raw['Layer'])
    layers.sort()
    layers = layers[:4] if len(layers) > 4 else layers

    # Configuration based on mode
    cmaps = {'counts': 'viridis', 'ratio': 'nipy_spectral', 'sum': 'turbo'}
    z_labels = {'counts': 'Counts', 'ratio': 'Avg ToT Ratio', 'sum': 'Avg ToT Sum'}
    cmap = cmaps.get(color_mode, 'viridis')
    z_lbl = z_labels.get(color_mode, 'Value')

    # Pre-calculate Correlated N
    print("Calculating Baseline Correlated Pairs...")
    n_pairs_map = {}
    corr_pairs_map = {} 
    
    for lid in layers:
        l_data = {k: v[data_corr['Layer'] == lid] for k, v in data_corr.items()}
        r1, r2, t1, t2 = get_spatial_pairs_corr(l_data, dx=dx)
        n_pairs_map[lid] = len(r1)
        corr_pairs_map[lid] = (r1, r2, t1, t2)
        print(f"Layer {lid}: {len(r1)} pairs")

    # Fixed Geometry Bins (Detector Rows 0-371)
    x_edges = np.arange(-0.5, 372.5, bin_size)
    y_edges = np.arange(-0.5, 743.5, bin_size)

    for ds_name, ds_data in datasets.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{ds_name} - {z_lbl} Heatmap (dx > {dx})", fontsize=16)
        axes = axes.flatten()
        
        # Store processed grids to unify scale
        grids = {} 
        g_max, g_min = -np.inf, np.inf
        
        # 1. Calculation Pass
        for lid in layers:
            n_target = n_pairs_map.get(lid, 0)
            if n_target == 0 and ds_name != 'Raw': continue
            # If Raw in trigger mode, we proceed even if correlated is 0? 
            # Sticking to valid layers defined by correlations for layout consistency.
            if n_target == 0: continue

            if ds_name == 'Correlated':
                r1, r2, t1, t2 = corr_pairs_map[lid]
            elif ds_name == 'Raw' and raw_corr_mode == 'trigger':
                l_data = {k: v[ds_data['Layer'] == lid] for k, v in ds_data.items()}
                r1, r2, t1, t2 = get_spatial_pairs_corr(l_data, dx=dx)
            else:
                # Uncorrelated OR Raw (random mode)
                l_data = {k: v[ds_data['Layer'] == lid] for k, v in ds_data.items()}
                r1, r2, t1, t2 = get_spatial_pairs_random(l_data, n_target, dx)
            
            if len(r1) == 0: continue
            
            x_val = np.abs(r1.astype(int) - r2.astype(int))
            y_val = r1.astype(int) + r2.astype(int)
            
            grid, l_max, l_min = _compute_heatmap_grid(x_val, y_val, t1, t2, x_edges, y_edges, color_mode)
            
            grids[lid] = (x_val, y_val, grid)
            if l_max > g_max: g_max = l_max
            if l_min < g_min: g_min = l_min

        # 2. Plotting Pass
        if g_max == -np.inf: g_max = 1
        if g_min == np.inf: g_min = 0
        vmin = 1 if color_mode == 'counts' else g_min
        
        # Norm safety
        if log_cscale and vmin <= 0: vmin = 1e-3
        norm = LogNorm(vmin=vmin, vmax=g_max) if log_cscale else Normalize(vmin=vmin, vmax=g_max)
        
        mappable = None
        
        for i, lid in enumerate(layers):
            ax = axes[i]
            if lid not in grids:
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                continue
                
            x_v, y_v, Z = grids[lid]
            
            # Draw Heatmap
            if color_mode == 'counts':
                h = ax.hist2d(x_v, y_v, bins=[x_edges, y_edges], cmap=cmap, cmin=1, norm=norm)
                mappable = h[3]
            else:
                X, Y = np.meshgrid(x_edges, y_edges)
                mappable = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)
            
            # Draw Marginals
            div = make_axes_locatable(ax)
            ax_x = div.append_axes("top", "20%", pad=0.05, sharex=ax)
            ax_y = div.append_axes("right", "20%", pad=0.05, sharey=ax)
            
            ax_x.hist(x_v, x_edges, color='gray', alpha=0.7, histtype='stepfilled')
            ax_y.hist(y_v, y_edges, orientation='horizontal', color='gray', alpha=0.7, histtype='stepfilled')
            
            # Styling
            ax_x.set_title(f"Layer {lid} (N={len(x_v)})")
            plt.setp(ax_x.get_xticklabels(), visible=False)
            plt.setp(ax_y.get_yticklabels(), visible=False)
            
            ax.set_xlim(0, 372)
            ax.set_ylim(0, 743)
            ax.set_xlabel("|Row1 - Row2|")
            ax.set_ylabel("Row1 + Row2")

        if mappable:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(mappable, cax=cbar_ax, label=z_lbl)

        plt.subplots_adjust(right=0.9, wspace=0.3, hspace=0.3)
        plt.show()
plot_tot_ratio_distribution(data_raw, correlated_data, uncorrelated_data)

plot_correlation_heatmaps(data_raw, correlated_data, uncorrelated_data, bin_size = 2, log_cscale=False, dx=30, color_mode='counts', raw_corr_mode='random')

plot_correlation_heatmaps(data_raw, correlated_data, uncorrelated_data, bin_size = 2, log_cscale=False, dx=30, color_mode='ratio')
plot_correlation_heatmaps(data_raw, correlated_data, uncorrelated_data, bin_size = 2, log_cscale=False, dx=30, color_mode='sum')


def plot_tot_vs_displacement_heatmaps(data_raw, data_corr, data_uncorr, bin_size=1, log_cscale=False, dx=50, raw_corr_mode='random', normalize_x=False, ylim=(0, 256)):
    """
    Plots ToT vs Displacement heatmaps.
    X-axis: Abs Displacement |r1-r2|. Y-axis: Individual ToT (t1 and t2).
    
    Args:
        data_raw, data_corr, data_uncorr: Input datasets.
        bin_size (int): Bin size for the displacement axis (X).
        log_cscale (bool): Use logarithmic color scale.
        dx (int): Ignore correlations where displacement |Row1 - Row2| <= dx.
        raw_corr_mode (str): 'random' (background) or 'trigger' (signal) for Raw dataset.
        normalize_x (bool): If True, normalizes each x-bin to sum to 1 (Probability Density).
        ylim (tuple): Y-axis limits (ToT). Default (0, 256).
    """
    datasets = {'Correlated': data_corr, 'Uncorrelated': data_uncorr, 'Raw': data_raw}
    layers = np.unique(data_raw['Layer'])
    layers.sort()
    layers = layers[:4] if len(layers) > 4 else layers

    # Pre-calculate Correlated N for normalization
    print("Calculating Baseline Correlated Pairs...")
    n_pairs_map = {}
    corr_pairs_map = {} 
    
    for lid in layers:
        l_data = {k: v[data_corr['Layer'] == lid] for k, v in data_corr.items()}
        r1, r2, t1, t2 = get_spatial_pairs_corr(l_data, dx=dx)
        n_pairs_map[lid] = len(r1)
        corr_pairs_map[lid] = (r1, r2, t1, t2)
    
    # Bins
    x_edges = np.arange(-0.5, 372.5, bin_size)
    y_edges = np.arange(-0.5, 256.5, 1) # ToT range 0-255

    for ds_name, ds_data in datasets.items():
        title_suffix = " (Column Normalized)" if normalize_x else ""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{ds_name} - ToT vs Displacement {title_suffix} (dx > {dx})", fontsize=16)
        axes = axes.flatten()
        
        grids = {}
        g_max = -np.inf
        g_min = np.inf # Use inf for proper min finding
        
        # 1. Calculation Phase
        for lid in layers:
            n_target = n_pairs_map.get(lid, 0)
            
            # Skip if no target pairs, unless we are looking for actual triggers in Raw
            if n_target == 0 and not (ds_name == 'Raw' and raw_corr_mode == 'trigger'): 
                continue

            # Select Data Source
            if ds_name == 'Correlated':
                r1, r2, t1, t2 = corr_pairs_map[lid]
            elif ds_name == 'Raw' and raw_corr_mode == 'trigger':
                l_data = {k: v[ds_data['Layer'] == lid] for k, v in ds_data.items()}
                r1, r2, t1, t2 = get_spatial_pairs_corr(l_data, dx=dx)
            else:
                # Random background generation
                if n_target == 0: continue
                l_data = {k: v[ds_data['Layer'] == lid] for k, v in ds_data.items()}
                r1, r2, t1, t2 = get_spatial_pairs_random(l_data, n_target, dx)
            
            if len(r1) == 0: continue
            
            # Prepare Data: Each pair contributes 2 points (x, t1) and (x, t2)
            x_val = np.abs(r1.astype(int) - r2.astype(int))
            # Stack data to plot both ToT values against the displacement
            X = np.concatenate([x_val, x_val])
            Y = np.concatenate([t1, t2])
            
            # Compute Histogram
            H, _, _ = np.histogram2d(X, Y, bins=[x_edges, y_edges])
            
            if normalize_x:
                # Normalize by column sum
                col_sums = H.sum(axis=1, keepdims=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    H = np.divide(H, col_sums, where=col_sums!=0)
                    
            grids[lid] = (X, Y, H.T) # Transpose for pcolormesh
            
            # Update Globals
            c_max = H.max()
            if c_max > g_max: g_max = c_max
            
            # Find min non-zero value
            valid = H[H > 0]
            if len(valid) > 0:
                c_min = valid.min()
                if c_min < g_min: g_min = c_min

        # 2. Plotting Phase
        if g_max == -np.inf: g_max = 1
        if g_min == np.inf: g_min = 1e-4 if normalize_x else 1 # Default if no data
        
        vmin = g_min
        vmax = g_max
        
        # Ensure valid limits for LogNorm
        if log_cscale:
            if vmin <= 0: vmin = 1e-4
            if vmax <= vmin: vmax = vmin * 10
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        mappable = None
        
        for i, lid in enumerate(layers):
            ax = axes[i]
            if lid not in grids:
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                ax.set_title(f"Layer {lid}")
                continue
                
            X_all, Y_all, Z = grids[lid]
            
            # Heatmap
            X_mesh, Y_mesh = np.meshgrid(x_edges, y_edges)
            mappable = ax.pcolormesh(X_mesh, Y_mesh, Z, cmap='jet', norm=norm)
            
            # Marginals
            div = make_axes_locatable(ax)
            ax_x = div.append_axes("top", "20%", pad=0.05, sharex=ax)
            ax_y = div.append_axes("right", "20%", pad=0.05, sharey=ax)
            
            # Marginal X (Displacement Distribution)
            # We take just the first half of X_all to represent the pairs (avoid double counting)
            x_unique = X_all[:len(X_all)//2]
            
            # Use log=log_cscale to handle log axis automatically for histograms
            ax_x.hist(x_unique, x_edges, color='gray', alpha=0.7, histtype='stepfilled', log=log_cscale)
            
            # Marginal Y (ToT Distribution)
            ax_y.hist(Y_all, y_edges, orientation='horizontal', color='gray', alpha=0.7, histtype='stepfilled', log=log_cscale)
            
            # --- Fix for Log Scale Overflow ---
            if log_cscale:
                # Explicitly set lower limit for log count axis to avoid 0 (which is -inf)
                # Counts are integers, so anything < 1 is effectively 0. Safe floor is 0.5.
                ax_x.set_ylim(bottom=0.5) 
                ax_y.set_xlim(left=0.5) 
            
            ax_x.set_title(f"Layer {lid} (N Pairs={len(x_unique)})")
            # Hide tick labels on inner axes
            plt.setp(ax_x.get_xticklabels(), visible=False)
            plt.setp(ax_y.get_yticklabels(), visible=False)
            
            ax.set_xlim(0, 372)
            ax.set_ylim(ylim)
            ax.set_xlabel("|Row1 - Row2|")
            ax.set_ylabel("ToT")

        if mappable:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            
            if normalize_x and log_cscale:
                lbl = "Frequency Density"
            elif normalize_x:
                lbl = "Probability Density"
            else:
                lbl = "Counts"
                
            fig.colorbar(mappable, cax=cbar_ax, label=lbl)

        plt.subplots_adjust(right=0.9, wspace=0.3, hspace=0.3)
        plt.show()
plot_tot_vs_displacement_heatmaps(data_raw, correlated_data, uncorrelated_data, bin_size = 1, 
                                  log_cscale=True, dx=5, ylim = (0,256), normalize_x=False)


def plot_tot_ratio_vs_displacement_heatmaps(data_raw, data_corr, data_uncorr, bin_size=1, n_ybins=100, log_cscale=False, log_yscale=False, dx=20, raw_corr_mode='random', normalize_x=False):
    """
    Plots ToT Ratio vs Displacement heatmaps.
    X-axis: Abs Displacement |r1-r2|. 
    Y-axis: ToT Ratio (min/max).
    
    Args:
        data_raw, data_corr, data_uncorr: Input datasets.
        bin_size (int): Bin size for the displacement axis (X).
        n_ybins (int): Number of bins for the Y-axis (Ratio).
        log_cscale (bool): Use logarithmic color scale.
        log_yscale (bool): Use logarithmic Y-axis bins (0.001 to 1) instead of linear (0 to 1).
        dx (int): Ignore correlations where displacement |Row1 - Row2| <= dx.
        raw_corr_mode (str): 'random' (background) or 'trigger' (signal) for Raw dataset.
        normalize_x (bool): If True, normalizes each x-bin to sum to 1 (Probability Density).
    """
    datasets = {'Correlated': data_corr, 'Uncorrelated': data_uncorr, 'Raw': data_raw}
    layers = np.unique(data_raw['Layer'])
    layers.sort()
    layers = layers[:4] if len(layers) > 4 else layers

    # Pre-calculate Correlated N for normalization/matching counts
    print("Calculating Baseline Correlated Pairs...")
    n_pairs_map = {}
    corr_pairs_map = {} 
    
    for lid in layers:
        l_data = {k: v[data_corr['Layer'] == lid] for k, v in data_corr.items()}
        r1, r2, t1, t2 = get_spatial_pairs_corr(l_data, dx=dx)
        n_pairs_map[lid] = len(r1)
        corr_pairs_map[lid] = (r1, r2, t1, t2)
    
    # Define Bins
    x_edges = np.arange(-0.5, 372.5, bin_size)
    
    if log_yscale:
        y_edges = np.logspace(np.log10(1/255), np.log10(1), n_ybins + 1)
    else:
        y_edges = np.linspace(0, 1, n_ybins + 1)

    for ds_name, ds_data in datasets.items():
        title_suffix = " (Col Norm)" if normalize_x else ""
        y_scale_str = " (Log Y)" if log_yscale else ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"{ds_name} - Ratio vs Disp{title_suffix}{y_scale_str} (dx > {dx})", fontsize=16)
        axes = axes.flatten()
        
        grids = {}
        g_max = -np.inf
        g_min = np.inf 
        
        # 1. Calculation Phase
        for lid in layers:
            n_target = n_pairs_map.get(lid, 0)
            
            # Skip logic
            if n_target == 0 and not (ds_name == 'Raw' and raw_corr_mode == 'trigger'): 
                continue

            # Select Data Source
            if ds_name == 'Correlated':
                r1, r2, t1, t2 = corr_pairs_map[lid]
            elif ds_name == 'Raw' and raw_corr_mode == 'trigger':
                l_data = {k: v[ds_data['Layer'] == lid] for k, v in ds_data.items()}
                r1, r2, t1, t2 = get_spatial_pairs_corr(l_data, dx=dx)
            else:
                if n_target == 0: continue
                l_data = {k: v[ds_data['Layer'] == lid] for k, v in ds_data.items()}
                r1, r2, t1, t2 = get_spatial_pairs_random(l_data, n_target, dx)
            
            if len(r1) == 0: continue
            
            # Prepare Data
            x_val = np.abs(r1.astype(int) - r2.astype(int))
            
            # Calculate Ratios
            with np.errstate(divide='ignore', invalid='ignore'):
                y_val = np.minimum(t1, t2) / np.maximum(t1, t2)
                y_val = np.nan_to_num(y_val, nan=0.0)
            
            # Compute Histogram
            H, _, _ = np.histogram2d(x_val, y_val, bins=[x_edges, y_edges])
            
            if normalize_x:
                # Normalize by column sum
                col_sums = H.sum(axis=1, keepdims=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    H = np.divide(H, col_sums, where=col_sums!=0)
                    
            grids[lid] = (x_val, y_val, H.T)
            
            # Update Globals
            c_max = H.max()
            if c_max > g_max: g_max = c_max
            
            valid = H[H > 0]
            if len(valid) > 0:
                c_min = valid.min()
                if c_min < g_min: g_min = c_min

        # 2. Plotting Phase
        if g_max == -np.inf: g_max = 1
        if g_min == np.inf: g_min = 1e-4 if normalize_x else 1 
        
        vmin = g_min
        vmax = g_max
        
        if log_cscale:
            # Fix for incredibly small vmin (e.g. 1e-279) caused by float artifacts
            if vmin < 1e-3: vmin = 1e-3
            if vmax <= vmin: vmax = vmin * 10
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        
        mappable = None
        
        for i, lid in enumerate(layers):
            ax = axes[i]
            if lid not in grids:
                ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
                ax.set_title(f"Layer {lid}")
                continue
                
            X_all, Y_all, Z = grids[lid]
            
            if log_yscale:
                ax.set_yscale('log')
                
            # Heatmap
            X_mesh, Y_mesh = np.meshgrid(x_edges, y_edges)
            mappable = ax.pcolormesh(X_mesh, Y_mesh, Z, cmap='jet', norm=norm)
            
            # Marginals
            div = make_axes_locatable(ax)
            ax_x = div.append_axes("top", "20%", pad=0.05, sharex=ax)
            ax_y = div.append_axes("right", "20%", pad=0.05, sharey=ax)
            
            # Use 'step' instead of 'stepfilled' for Log Scale to avoid filling to -infinity (OverflowError)
            h_type = 'step' if log_cscale else 'stepfilled'
            
            # Marginal X (Displacement) 
            ax_x.hist(X_all, x_edges, color='gray', alpha=0.7, histtype=h_type, log=log_cscale)
            
            # Marginal Y (Ratio)
            ax_y.hist(Y_all, y_edges, orientation='horizontal', color='gray', alpha=0.7, histtype=h_type, log=log_cscale)
            
            # Formatting
            if log_yscale:
                ax_y.set_yscale('log')
                
            # Log Scale Overflow Fix for Marginals
            if log_cscale:
                ax_x.set_ylim(bottom=0.5) 
                ax_y.set_xlim(left=0.5)
            
            ax_x.set_title(f"Layer {lid} (N={len(X_all)})")
            plt.setp(ax_x.get_xticklabels(), visible=False)
            plt.setp(ax_y.get_yticklabels(), visible=False)
            
            ax.set_xlim(dx, 372) # Start x-axis at dx
            if not log_yscale:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(1/255, 1)
                
            ax.set_xlabel("|Row1 - Row2|")
            ax.set_ylabel("ToT Ratio")

        if mappable:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            
            if normalize_x and log_cscale:
                lbl = "Frequency Density"
            elif normalize_x:
                lbl = "Probability Density"
            else:
                lbl = "Counts"
                
            fig.colorbar(mappable, cax=cbar_ax, label=lbl)

        plt.subplots_adjust(right=0.9, wspace=0.3, hspace=0.3)
        plt.show()
plot_tot_ratio_vs_displacement_heatmaps(data_raw, correlated_data, uncorrelated_data, bin_size = 1, normalize_x=False,log_yscale=False, log_cscale = False)