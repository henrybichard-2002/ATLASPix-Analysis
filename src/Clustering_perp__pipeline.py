# -*- coding: utf-8 -*-
"""
Pipeline: Clustering + Crosstalk Labeling + Cleaning + Diagnostics
Output: Dictionary of NumPy Arrays
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from numba import njit
from matplotlib.colors import LogNorm


@njit(fastmath=True)
def _find_root(parent, i):
    if parent[i] == i:
        return i
    parent[i] = _find_root(parent, parent[i])
    return parent[i]

@njit(fastmath=True)
def _union(parent, i, j):
    root_i = _find_root(parent, i)
    root_j = _find_root(parent, j)
    if root_i != root_j:
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

@njit(fastmath=True)
def _anisotropic_cluster_kernel(cols, rows, times, time_window, 
                                pitch_x, pitch_y, search_radius_sq):
    n = len(times)
    parent = np.arange(n) 
    max_d_col = (search_radius_sq**0.5) / pitch_x + 0.5
    max_d_row = (search_radius_sq**0.5) / pitch_y + 0.5
    
    for i in range(n):
        c_curr = cols[i]
        r_curr = rows[i]
        for j in range(i - 1, -1, -1):
            if times[i] - times[j] > time_window: break 
            
            dc = c_curr - cols[j]
            dr = r_curr - rows[j]
            
            # Optimization: fast rejection based on index difference
            if abs(dc) > max_d_col or abs(dr) > max_d_row: continue
                
            dist_sq = (dc*pitch_x)**2 + (dr*pitch_y)**2
            if dist_sq <= search_radius_sq:
                _union(parent, i, j)
    
    for i in range(n):
        parent[i] = _find_root(parent, i)
    return parent

# ==========================================
# 2. CORE PROCESSING STEPS
# ==========================================

def assign_cluster_ids(data, time_window=30, search_radius=2, 
                       pitch_ratio_x=3.0, pitch_ratio_y=1.0, subset_indices=None):
    """
    Applies clustering logic.
    Args:
        subset_indices: If provided, only processes hits at these indices 
                        and appends new Cluster IDs starting from current max.
    """
    t0 = time.time()
    
    if subset_indices is not None:
        # --- Partial Mode (Re-clustering) ---
        print(f"   -> Re-clustering subset of {len(subset_indices)} hits...")
        df = pd.DataFrame({
            'Layer': data['Layer'][subset_indices],
            'Col': data['Column'][subset_indices],
            'Row': data['Row'][subset_indices],
            'ext_TS': data['ext_TS'][subset_indices]
        })

        df['global_index'] = subset_indices
        
        # Start IDs after the current maximum to ensure uniqueness
        current_max_id = np.max(data['clusterID']) + 1
    else:
        print(f"\n[Step 1] Clustering Hits (Win={time_window}, Rad={search_radius})...")
        df = pd.DataFrame({
            'Layer': data['Layer'],
            'Col': data['Column'],
            'Row': data['Row'],
            'ext_TS': data['ext_TS']
        })
        df['global_index'] = np.arange(len(df))
        current_max_id = 0
        
        # Reset IDs
        data['clusterID'] = np.full(len(df), -1, dtype=np.int64)

    # 2. Run Kernel Layer-by-Layer
    local_ids = np.full(len(df), -1, dtype=np.int64)
    
    for layer in sorted(df['Layer'].unique()):
        mask = (df['Layer'] == layer)
        if not mask.any(): continue
        
        # Sort by time (required for kernel efficiency)
        sub = df[mask].sort_values('ext_TS')
        
        parents = _anisotropic_cluster_kernel(
            sub['Col'].values.astype(np.float32), 
            sub['Row'].values.astype(np.float32), 
            sub['ext_TS'].values.astype(np.int64),
            time_window, pitch_ratio_x, pitch_ratio_y, search_radius**2
        )
        
        _, seq_ids = np.unique(parents, return_inverse=True)
        
        # Shift IDs to global range
        global_ids = seq_ids + current_max_id
        
        # Assign back to local storage
        local_ids[sub.index] = global_ids
        
        if len(global_ids) > 0:
            current_max_id = global_ids.max() + 1
            
    # 3. Write Back to Main Dictionary
    data['clusterID'][df['global_index'].values] = local_ids
    
    if subset_indices is None:
        print(f"   -> Found {current_max_id} clusters ({time.time()-t0:.2f}s).")
    
    return data

def label_crosstalk_hits(hits_dict, dTS=5, ratio_threshold=0.1):
    print(f"\n[Step 2] Labeling Crosstalk (dTS={dTS}, Ratio<={ratio_threshold})...")
    t0 = time.time()
    
    df = pd.DataFrame({k: hits_dict[k] for k in ['Layer', 'Column', 'ext_TS', 'ToT', 'clusterID']})
    xtalk_type = np.zeros(len(df), dtype=np.uint8)
    
    valid = (df['clusterID'] != -1)
    sub = df[valid].sort_values(['Layer', 'Column', 'ext_TS'])
    
    # 1. Define 'Events' (Hits on same pixel nearby in time)
    loc_change = (sub['Layer'].diff() != 0) | (sub['Column'].diff() != 0)
    time_gap = sub['ext_TS'].diff() > dTS
    sub['eventID'] = (loc_change | time_gap).cumsum()
    
    # 2. Filter Multi-Cluster Events
    evt_counts = sub.groupby('eventID')['clusterID'].nunique()
    target_evts = evt_counts[evt_counts >= 2].index
    
    if len(target_evts) > 0:
        relevant_hits = sub[sub['eventID'].isin(target_evts)].reset_index()
        pairs = pd.merge(relevant_hits, relevant_hits, on='eventID', suffixes=('_1', '_2'))
        
        # Filter self-pairs and same-cluster pairs
        pairs = pairs[
            (pairs['index_1'] != pairs['index_2']) & 
            (pairs['clusterID_1'] != pairs['clusterID_2'])
        ].copy()
        
        # Energy Ratios
        t1, t2 = pairs['ToT_1'].values, pairs['ToT_2'].values
        min_t, max_t = np.minimum(t1, t2), np.maximum(t1, t2)
        max_t[max_t == 0] = 1 
        pairs['ratio'] = min_t / max_t
        
        is_low = pairs['ratio'] <= ratio_threshold
        is_small = t1 < t2
        
        # Scoring
        pairs['score'] = 0
        pairs.loc[~is_low, 'score'] = 5             # Ambiguous
        pairs.loc[is_low & is_small, 'score'] = 10  # Victim
        
        final = pairs.groupby('index_1')['score'].max()
        label_map = {10: 1, 5: 2, 0: 0}
        xtalk_type[final.index] = final.map(label_map).values
    
    hits_dict['xtalk_type'] = xtalk_type
    
    counts = dict(zip(*np.unique(xtalk_type, return_counts=True)))
    print(f"   -> Labeling Complete ({time.time()-t0:.2f}s).")
    print(f"      Type 0 (Clean):     {counts.get(0,0)}")
    print(f"      Type 1 (Victim):    {counts.get(1,0)}")
    print(f"      Type 2 (Ambiguous): {counts.get(2,0)}")
    
    return hits_dict

def clean_and_recluster(data, strict_separation=False, min_hits_threshold=10, 
                        time_window=30, search_radius=2, 
                        pitch_ratio_x=3.0, pitch_ratio_y=1.0):
    """
    Step 2.5: Splits clusters that are artificially bridged by crosstalk.
    
    Args:
        strict_separation (bool): If True, reclusters ALL clusters (regardless of size)
                                  that contain mixed data to ensure no clean hit is 
                                  linked to a crosstalk hit.
    """
    if strict_separation:
        print("\n[Step 2.5] STRICT CLEANING ENABLED: Removing crosstalk links in all mixed clusters.")
        eff_threshold = 0 # Process everything
    else:
        print(f"\n[Step 2.5] Standard Cleaning (Size >= {min_hits_threshold})...")
        eff_threshold = min_hits_threshold

    t0 = time.time()
    
    # 1. Identify Candidate Clusters
    df = pd.DataFrame({
        'clusterID': data['clusterID'],
        'xtalk': data['xtalk_type'],
        'idx': np.arange(len(data['clusterID']))
    })
    df = df[df['clusterID'] != -1]
    
    g = df.groupby('clusterID')
    stats = g.agg({'xtalk': ['max', 'min', 'count']})
    stats.columns = ['max_xtalk', 'min_xtalk', 'size']
    
    # Criteria: 
    # 1. Size Threshold (0 if strict_separation is True)
    # 2. Mixed: Contains both clean (0) and dirty (>0)
    cond_size = stats['size'] >= eff_threshold
    cond_mixed = (stats['max_xtalk'] > 0) & (stats['min_xtalk'] == 0)
    
    dirty_cluster_ids = stats.index[cond_size & cond_mixed].values
    
    if len(dirty_cluster_ids) == 0:
        print("   -> No mixed clusters found to clean. Skipping.")
        return data
        
    print(f"   -> Found {len(dirty_cluster_ids)} mixed clusters to separate.")
    
    # 2. Separate Hits
    hits_in_clusters = df[df['clusterID'].isin(dirty_cluster_ids)]
    
    clean_indices = hits_in_clusters.loc[hits_in_clusters['xtalk'] == 0, 'idx'].values
    dirty_indices = hits_in_clusters.loc[hits_in_clusters['xtalk'] > 0, 'idx'].values
    
    # 3. Action
    # A. Remove dirty hits from these clusters (Set ID to -1)
    data['clusterID'][dirty_indices] = -1
    
    # B. Re-cluster clean hits (Assign NEW IDs)
    data = assign_cluster_ids(
        data, 
        time_window=time_window, 
        search_radius=search_radius,
        pitch_ratio_x=pitch_ratio_x, 
        pitch_ratio_y=pitch_ratio_y,
        subset_indices=clean_indices
    )
    
    print(f"   -> Cleaning finished ({time.time()-t0:.2f}s).")
    return data

# ==========================================
# 3. AGGREGATION & DIAGNOSTICS
# ==========================================

def generate_optimized_cluster_dataset(hits_dict: dict) -> dict:
    print("\n[Step 3] Generating Optimized Cluster Dataset (Dict of Arrays)...")
    t0 = time.time()
    
    df = pd.DataFrame({
        'clusterID': hits_dict['clusterID'],
        'Layer': hits_dict['Layer'].astype(np.uint8),
        'Col': hits_dict['Column'].astype(np.uint16),
        'Row': hits_dict['Row'].astype(np.uint16),
        'TS': hits_dict['ext_TS'].astype(np.uint64),
        'ToT': hits_dict['ToT'].astype(np.uint16),
        'pToF': hits_dict['pToF'].astype(np.int16), 
        'xtalk': hits_dict['xtalk_type'].astype(np.uint8)
    })
    
    df = df[df['clusterID'] != -1].sort_values('clusterID')
    if df.empty: return {}
    
    df['w_col'] = (df['Col'] * df['ToT']).astype(np.float32)
    df['w_row'] = (df['Row'] * df['ToT']).astype(np.float32)
    
    print("   Aggregating scalars...")
    g = df.groupby('clusterID')
    stats = g.agg({
        'Layer': 'first',
        'Col': ['min', 'max', 'mean'],
        'Row': ['min', 'max', 'mean'],
        'TS': ['min', 'max'],
        'ToT': ['count', 'mean', 'sum'],
        'w_col': 'sum',
        'w_row': 'sum',
        'xtalk': ['min', 'max'],
        'pToF': ['min', 'max']    
    })
    stats.columns = [f"{c[0]}_{c[1]}" for c in stats.columns]
    
    tot_sum = stats['ToT_sum'].values
    cog_col = stats['Col_mean'].values.astype(np.float32)
    cog_row = stats['Row_mean'].values.astype(np.float32)
    has_nrg = tot_sum > 0
    np.divide(stats['w_col_sum'].values, tot_sum, out=cog_col, where=has_nrg)
    np.divide(stats['w_row_sum'].values, tot_sum, out=cog_row, where=has_nrg)
    
    result_dict = {
        'clusterID': stats.index.values.astype(np.int64),
        'Layer': stats['Layer_first'].values.astype(np.uint8),
        'col_min': stats['Col_min'].values.astype(np.uint16),
        'col_max': stats['Col_max'].values.astype(np.uint16),
        'row_min': stats['Row_min'].values.astype(np.uint16),
        'row_max': stats['Row_max'].values.astype(np.uint16),
        'width_col': (stats['Col_max'] - stats['Col_min'] + 1).values.astype(np.uint16),
        'width_row': (stats['Row_max'] - stats['Row_min'] + 1).values.astype(np.uint16),
        'cog_col': cog_col,
        'cog_row': cog_row,
        'ts_start': stats['TS_min'].values.astype(np.uint64),
        'ts_stop':  stats['TS_max'].values.astype(np.uint64),
        'duration': (stats['TS_max'] - stats['TS_min']).values.astype(np.uint64),
        'n_hits':  stats['ToT_count'].values.astype(np.uint16),
        'sum_ToT': stats['ToT_sum'].values.astype(np.float32),
        'avg_ToT': stats['ToT_mean'].values.astype(np.float32),
    }
    
    # Handle Smart Lists for mixed clusters
    def process_smart_column(col_name, source_col):
        min_vals = stats[f'{source_col}_min'].values
        max_vals = stats[f'{source_col}_max'].values
        final_arr = min_vals.astype(object)
        is_mixed = min_vals != max_vals
        if np.any(is_mixed):
            mixed_ids = stats.index[is_mixed]
            print(f"   Optimizing mixed '{col_name}' ({len(mixed_ids)} found)...")
            mixed_series = df[df['clusterID'].isin(mixed_ids)].groupby('clusterID')[source_col].apply(list)
            mixed_dict = mixed_series.to_dict()
            indices = np.nonzero(is_mixed)[0]
            ids = stats.index.values[indices]
            for i, cid in zip(indices, ids):
                final_arr[i] = mixed_dict.get(cid, final_arr[i])
        result_dict[col_name] = final_arr

    process_smart_column('xtalk_type', 'xtalk')
    process_smart_column('pToF', 'pToF')
    
    print(f"--- Finished ({time.time() - t0:.2f}s) ---")
    return result_dict

def plot_cluster_structure_heatmaps(hits_dict, window_size=4, aspect_ratio=3.0):
    """
    Generates 2 Heatmaps per Layer:
    1. Hit Density (Log Scale, Jet)
    2. Average Energy (Linear, Nipy_Spectral)
    """
    print("\n[Diagnostics] Generating Cluster Structure Heatmaps...")
    t0 = time.time()
    
    # 1. Prepare Data
    df = pd.DataFrame({
        'Layer': hits_dict['Layer'].astype(np.uint8),
        'clusterID': hits_dict['clusterID'],
        'col': hits_dict['Column'].astype(np.int32),
        'row': hits_dict['Row'].astype(np.int32),
        'tot': hits_dict['ToT'].astype(np.float32)
    })
    
    df = df[df['clusterID'] != -1]
    if df.empty: return

    # 2. Find Center (Max ToT Hit)
    df.sort_values(['clusterID', 'tot'], ascending=[True, False], inplace=True)
    df['seed_col'] = df.groupby('clusterID')['col'].transform('first')
    df['seed_row'] = df.groupby('clusterID')['row'].transform('first')
    
    # Calculate Relative Positions
    df['d_col'] = df['col'] - df['seed_col']
    df['d_row'] = df['row'] - df['seed_row']
    
    # 3. Define Ranges
    x_range = window_size
    bins_x = np.arange(-x_range - 0.5, x_range + 1.5, 1)
    
    y_range = int(window_size * aspect_ratio)
    bins_y = np.arange(-y_range - 0.5, y_range + 1.5, 1)
    
    extent = [
        -x_range - 0.5, x_range + 0.5, 
        -y_range - 0.5, y_range + 0.5    
    ]

    # 4. Plot Layer by Layer
    for L in sorted(df['Layer'].unique()):
        layer_data = df[df['Layer'] == L]
        if layer_data.empty: continue
        
        counts, _, _ = np.histogram2d(
            layer_data['d_row'], layer_data['d_col'], 
            bins=[bins_y, bins_x] 
        )
        
        tot_sum, _, _ = np.histogram2d(
            layer_data['d_row'], layer_data['d_col'], 
            bins=[bins_y, bins_x], 
            weights=layer_data['tot']
        )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_tot = tot_sum / counts
            avg_tot[counts == 0] = np.nan

        counts[counts == 0] = np.nan 

        fig, axes = plt.subplots(1, 2, figsize=(14, 7)) 
        
        # Left: Hit Density
        ax1 = axes[0]
        im1 = ax1.imshow(
            counts, 
            origin='lower', 
            extent=extent, 
            cmap='jet', 
            norm=LogNorm(), 
            interpolation='nearest',
            aspect='auto' 
        )
        ax1.set_box_aspect(1) 
        fig.colorbar(im1, ax=ax1, label='Hit Count', shrink=0.8)
        ax1.set_title(f"Layer {L}: Hit Density (Log)")
        ax1.set_xlabel(r"$\Delta$ Column")
        ax1.set_ylabel(r"$\Delta$ Row")
        
        # Right: Avg Energy
        ax2 = axes[1]
        im2 = ax2.imshow(
            avg_tot, 
            origin='lower', 
            extent=extent, 
            cmap='nipy_spectral', 
            interpolation='nearest',
            aspect='auto',
            vmin=0,    
            vmax=175   
        )
        ax2.set_box_aspect(1) 
        fig.colorbar(im2, ax=ax2, label='Avg ToT', shrink=0.8)
        ax2.set_title(f"Layer {L}: Avg Energy Profile")
        ax2.set_xlabel(r"$\Delta$ Column")
        
        plt.tight_layout()
        plt.show()

    print(f"   -> Plots generated ({time.time()-t0:.2f}s).")

def run_processing_pipeline(
    data: dict,
    time_window: int = 30,
    search_radius: float = 3.2,
    pitch_ratio_x: float = 3.0,
    pitch_ratio_y: float = 1.0,
    xtalk_ratio: float = 0.2,
    xtalk_dTS: int = 3,
    strict_clean_separation: bool = False,
    recluster_size_threshold: int = 5
):
    print("=== STARTING PIPELINE ===")
    
    # Step 1: Initial Clustering
    # Adjusted search_radius default in args to ensure neighbor connectivity
    assign_cluster_ids(data, time_window, search_radius, pitch_ratio_x, pitch_ratio_y)
    
    # Step 2: Crosstalk Labeling
    label_crosstalk_hits(data, dTS=xtalk_dTS, ratio_threshold=xtalk_ratio)
    
    # Step 2.5: Cleaning & Re-clustering
    # 'strict_clean_separation' forces cleaning of ALL mixed clusters
    if strict_clean_separation or recluster_size_threshold > 0:
        clean_and_recluster(
            data, 
            strict_separation=strict_clean_separation,
            min_hits_threshold=recluster_size_threshold,
            time_window=time_window,
            search_radius=search_radius,
            pitch_ratio_x=pitch_ratio_x,
            pitch_ratio_y=pitch_ratio_y
        )
    
    # Step 3: Aggregate
    cluster_data = generate_optimized_cluster_dataset(data)
    
    # Step 4: Diagnostics
    try:
        plot_cluster_structure_heatmaps(data, window_size=7, aspect_ratio = pitch_ratio_x)
    except Exception as e:
        print(f"Warning: Plotting failed ({e}).")
    
    print("=== PIPELINE FINISHED ===")
    return data, cluster_data


# === ADJUSTED INPUT PARAMETERS ===
# 1. strict_clean_separation=True -> Ensures no clean data remains clustered with xtalk
# 2. search_radius=3.2 -> For pitch_x=3.0, distance to diagonal neighbor is sqrt(3^2+1^2)=3.16. 
#                         3.2 is the tightest fit to include only touching neighbors.

labelled_hits, final_clusters = run_processing_pipeline(
     processed_data,
     time_window=10,
     search_radius=3.2,
     xtalk_ratio=0.2,
     strict_clean_separation=True
)

cluster_df = pd.DataFrame(final_clusters)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_position_resolution_by_cluster_size(cluster_dict: dict):
    """
    Plots internal position resolution (CoG Deviation from Geometric Center).
    
    Corrections:
    - Casts to float32 before Geometric Center calc to prevent uint16 overflow.
    - X-Range set to +/- 1.2 pixels.
    - Excludes '0' bin (single-pixel-width clusters).
    """
    if not cluster_dict: return
    print("\n--- Plotting Internal Position Resolution (Zero-Suppressed) ---")
    
    sizes_to_plot = [2, 3, 4, 5]
    layers = sorted(np.unique(cluster_dict['Layer']))
    
    # 1. Safe Calculation of Geometric Centers
    c_min = cluster_dict['col_min'].astype(np.float32)
    c_max = cluster_dict['col_max'].astype(np.float32)
    geo_col = (c_min + c_max) / 2.0
    
    r_min = cluster_dict['row_min'].astype(np.float32)
    r_max = cluster_dict['row_max'].astype(np.float32)
    geo_row = (r_min + r_max) / 2.0
    
    # 2. Calculate Residuals
    res_col = cluster_dict['cog_col'] - geo_col
    res_row = cluster_dict['cog_row'] - geo_row
    
    # 3. Plotting
    for L in layers:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        layer_mask = (cluster_dict['Layer'] == L)
        
        for i, n_hits in enumerate(sizes_to_plot):
            ax = axes[i]
            
            # Filter
            mask = layer_mask & (cluster_dict['n_hits'] == n_hits)
            
            # Extract
            dx = res_col[mask]
            dy = res_row[mask]
            
            epsilon = 1e-4
            dx_clean = dx[np.abs(dx) > epsilon]
            dy_clean = dy[np.abs(dy) > epsilon]
            
            # Plot Column (X) - Blue Filled
            if len(dx_clean) > 5:
                sns.histplot(dx_clean, ax=ax, color='royalblue', label=f'Column (N={len(dx_clean)})', 
                             element="step", fill=True, alpha=0.2, stat='density', bins=40, binrange=(-1.2, 1.2))
            
            # Plot Row (Y) - Red Step (No Fill)
            if len(dy_clean) > 5:
                sns.histplot(dy_clean, ax=ax, color='crimson', label=f'Row (N={len(dy_clean)})', 
                             element="step", fill=False, linewidth=1.5, stat='density', bins=40, binrange=(-1.2, 1.2))
            
            # Formatting
            ax.set_title(f"Cluster Size = {n_hits} Hits")
            ax.set_xlabel("Deviation from Geometric Center (Pixels)")
            ax.set_xlim(-1.2, 1.2)
            
            # Vertical reference
            ax.axvline(0, color='black', linestyle='--', alpha=0.3)
            
            # Add legends/text
            if len(dx_clean) < 5 and len(dy_clean) < 5:
                ax.text(0.5, 0.5, "Insufficient Data\n(Zeros Excluded)", 
                        ha='center', va='center', transform=ax.transAxes, color='gray')
            else:
                ax.legend(loc='upper right', fontsize='small')
                
            ax.grid(True, alpha=0.2)
            
        plt.suptitle(f"Layer {L}: CoG Sub-Pixel Distribution", fontsize=16)
        plt.tight_layout()
        
        
        plt.show()

# --- USAGE ---
plot_position_resolution_by_cluster_size(final_clusters)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

def plot_cluster_structure_6panel(hits_dict, window_size=4, aspect_ratio=3.0):
    """
    Generates a 3x2 Grid of Heatmaps per Layer:
    - Row 1: All Clusters
    - Row 2: Clusters containing Type 1 (Victim) Hits
    - Row 3: Clusters containing Type 2 (Ambiguous) Hits
    
    Columns:
    - Left: Hit Density (Log Scale, Jet)
    - Right: Avg Energy (Linear, Nipy_Spectral, 0-255)
    """
    print("\n[Diagnostics] Generating 6-Panel Cluster Structure Maps...")
    t0 = time.time()
    
    # 1. Prepare Data
    df = pd.DataFrame({
        'Layer': hits_dict['Layer'].astype(np.uint8),
        'clusterID': hits_dict['clusterID'],
        'col': hits_dict['Column'].astype(np.int32),
        'row': hits_dict['Row'].astype(np.int32),
        'tot': hits_dict['ToT'].astype(np.float32),
        'xtalk': hits_dict['xtalk_type'].astype(np.uint8)
    })
    
    df = df[df['clusterID'] != -1]
    if df.empty: return

    # 2. Centering Logic (Seed = Max ToT)
    df.sort_values(['clusterID', 'tot'], ascending=[True, False], inplace=True)
    
    # Broadcast Seed position
    df['seed_col'] = df.groupby('clusterID')['col'].transform('first')
    df['seed_row'] = df.groupby('clusterID')['row'].transform('first')
    
    # Calculate Relative Positions
    df['d_col'] = df['col'] - df['seed_col']
    df['d_row'] = df['row'] - df['seed_row']
    
    # 3. Pre-Identify Crosstalk Clusters
    # Find IDs of clusters that contain at least one hit of Type 1 or Type 2
    ids_with_t1 = df[df['xtalk'] == 1]['clusterID'].unique()
    ids_with_t2 = df[df['xtalk'] == 2]['clusterID'].unique()
    
    # 4. Define Bins & Extent
    x_range = window_size
    y_range = int(window_size * aspect_ratio)
    
    bins_x = np.arange(-x_range - 0.5, x_range + 1.5, 1)
    bins_y = np.arange(-y_range - 0.5, y_range + 1.5, 1)
    
    extent = [
        -x_range - 0.5, x_range + 0.5, 
        -y_range - 0.5, y_range + 0.5
    ]

    # 5. Plotting Loop (Per Layer)
    for L in sorted(df['Layer'].unique()):
        layer_df = df[df['Layer'] == L]
        if layer_df.empty: continue
        
        # Create 3x2 Grid
        fig, axes = plt.subplots(3, 2, figsize=(12, 18), constrained_layout=True)
        
        # Define the 3 datasets to plot
        datasets = [
            ("All Clusters", layer_df),
            ("Clusters w/ Victim (Type 1)", layer_df[layer_df['clusterID'].isin(ids_with_t1)]),
            ("Clusters w/ Ambiguous (Type 2)", layer_df[layer_df['clusterID'].isin(ids_with_t2)])
        ]
        
        # Iterate over rows
        for row_idx, (label, data) in enumerate(datasets):
            ax_left = axes[row_idx, 0]
            ax_right = axes[row_idx, 1]
            
            if data.empty:
                ax_left.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax_right.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            # --- Calculation ---
            # 1. Hit Counts
            counts, _, _ = np.histogram2d(
                data['d_row'], data['d_col'], bins=[bins_y, bins_x]
            )
            
            # 2. Avg ToT
            tot_sum, _, _ = np.histogram2d(
                data['d_row'], data['d_col'], bins=[bins_y, bins_x], weights=data['tot']
            )
            
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_tot = tot_sum / counts
                avg_tot[counts == 0] = np.nan
            
            counts[counts == 0] = np.nan # Mask 0 for Log Scale

            # --- Left Plot: Hit Density ---
            im1 = ax_left.imshow(
                counts, origin='lower', extent=extent, 
                cmap='jet', norm=LogNorm(), interpolation='nearest', aspect='auto'
            )
            ax_left.set_box_aspect(1) # Force Square
            fig.colorbar(im1, ax=ax_left, label='Hit Count', shrink=0.9)
            ax_left.set_title(f"{label}\nHit Density (Log)")
            ax_left.set_ylabel(r"$\Delta$ Row")
            
            # --- Right Plot: Energy Profile ---
            im2 = ax_right.imshow(
                avg_tot, origin='lower', extent=extent, 
                cmap='nipy_spectral', interpolation='nearest', aspect='auto',
                vmin=0, vmax=255 # Fixed Scale
            )
            ax_right.set_box_aspect(1) # Force Square
            fig.colorbar(im2, ax=ax_right, label='Avg ToT', shrink=0.9)
            ax_right.set_title(f"{label}\nAvg Energy")
            
            # X-Labels only on bottom row
            if row_idx == 2:
                ax_left.set_xlabel(r"$\Delta$ Column")
                ax_right.set_xlabel(r"$\Delta$ Column")
            
            # Mark Center Seed
            for ax in [ax_left, ax_right]:
                ax.text(0, 0, "+", color='white', ha='center', va='center', fontsize=14, fontweight='bold')

        plt.suptitle(f"Layer {L} Cluster Structure Analysis", fontsize=16, fontweight='bold')
        
        
        plt.show()

    print(f"   -> Plots generated ({time.time()-t0:.2f}s).")

# --- USAGE ---
plot_cluster_structure_6panel(labelled_hits, window_size=7, aspect_ratio=3.0)