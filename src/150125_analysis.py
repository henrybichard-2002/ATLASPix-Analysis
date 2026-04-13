import pandas as pd
import numpy as np

def find_candidate_tracks_vectors(data_raw, n_packages=None, 
                                  d_row=20, d_col=20, d_ts=2, min_hits=4):
    """
    Finds candidates for 6GeV electrons and calculates geometric direction vectors.
    
    Returns DataFrame with:
    - 'Segment_Vectors': List of normalized [vx, vy, vz] between adjacent layers.
    - 'Global_Vector': Normalized [Vx, Vy, Vz] of the best fit line.
    - 'Chi2_Spatial_mm2': Physical Chi2 using 150x50um pixel weights.
    """
    # --- Physical Constants ---
    PITCH_COL = 0.150  # mm (150 um)
    PITCH_ROW = 0.050  # mm (50 um)
    DIST_Z    = 25.4   # mm (Layer spacing)

    df = pd.DataFrame(data_raw)
    packages = df['PackageID'].unique()
    if n_packages:
        packages = packages[:n_packages]
    
    results = []

    for pid in packages:
        p_data = df[df['PackageID'] == pid].copy()
        
        # 1. DFS Search (Find all hit combinations)
        package_candidates = []
        
        def dfs_search(current_track):
            current_hit = current_track[-1]
            current_layer = int(current_hit['Layer'])
            
            if current_layer == 1:
                if len(current_track) >= min_hits:
                    package_candidates.append(current_track)
                return

            found_next = False
            for next_layer in range(current_layer - 1, 0, -1):
                # 6GeV electrons are ultra-relativistic => d_ts window is very tight (default 2)
                mask = (
                    (p_data['Layer'] == next_layer) &
                    (p_data['Column'].between(current_hit['Column'] - d_col, current_hit['Column'] + d_col)) &
                    (p_data['Row'].between(current_hit['Row'] - d_row, current_hit['Row'] + d_row)) &
                    (p_data['ext_TS'].between(current_hit['ext_TS'] - d_ts, current_hit['ext_TS'] + d_ts))
                )
                matches = p_data[mask]
                for _, match in matches.iterrows():
                    found_next = True
                    dfs_search(current_track + [match])
            
            if not found_next and len(current_track) >= min_hits:
                package_candidates.append(current_track)

        seeds = p_data[p_data['Layer'] == 4]
        for _, seed in seeds.iterrows():
            dfs_search([seed])

        # 2. Physics & Vector Calculation
        for track_hits in package_candidates:
            cand = pd.DataFrame(track_hits)
            
            # Coordinates in mm (Z=0 at Layer 4)
            layers = cand['Layer'].values
            z_mm = (4 - layers) * DIST_Z 
            col_mm = cand['Column'].values * PITCH_COL
            row_mm = cand['Row'].values * PITCH_ROW
            
            # --- A. Segment Vectors (Layer to Layer) ---
            segment_vectors = []
            
            for i in range(len(cand) - 1):
                # Displacement
                dx = col_mm[i+1] - col_mm[i]
                dy = row_mm[i+1] - row_mm[i]
                dz = z_mm[i+1] - z_mm[i] # Always positive (downstream)
                
                # Normalize to unit vector
                magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
                if magnitude > 0:
                    v_seg = [dx/magnitude, dy/magnitude, dz/magnitude]
                else:
                    v_seg = [0.0, 0.0, 0.0]
                
                segment_vectors.append(v_seg)
            
            # --- B. Global Vector (Best Fit) ---
            # Fit X vs Z and Y vs Z
            slope_c, int_c = np.polyfit(z_mm, col_mm, 1) # dx/dz
            slope_r, int_r = np.polyfit(z_mm, row_mm, 1) # dy/dz
            
            # The vector along the line is (dx, dy, dz) = (slope_c, slope_r, 1)
            # We normalize this to get the global direction
            global_mag = np.sqrt(slope_c**2 + slope_r**2 + 1**2)
            global_vector = [slope_c/global_mag, slope_r/global_mag, 1.0/global_mag]
            
            # --- C. Chi2 Metrics (mm^2) ---
            res_c = np.sum((col_mm - (slope_c * z_mm + int_c))**2)
            res_r = np.sum((row_mm - (slope_r * z_mm + int_r))**2)
            
            dof = max(1, len(cand) - 2)
            chi2_spatial = (res_c + res_r) / dof
            
            # Time Uniformity
            ts_vals = cand['ext_TS'].values.astype(float)
            slope_t, int_t = np.polyfit(z_mm, ts_vals, 1)
            res_t = np.sum((ts_vals - (slope_t * z_mm + int_t))**2)
            chi2_time = res_t / dof

            results.append({
                'PackageID': pid,
                'Length': len(cand),
                
                # Sorting Metrics
                'Chi2_Spatial_mm2': chi2_spatial,
                'Chi2_Time': chi2_time,
                
                # Geometric Vectors
                'Global_Vector': global_vector,    # [Vx, Vy, Vz] normalized
                'Segment_Vectors': segment_vectors, # List of normalized vectors
                
                # Raw Data
                'Hits_Data': cand.to_dict('records')
            })

    return pd.DataFrame(results)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

def plot_all_candidate_tracks_with_tot(candidates_df, title="Candidate Tracks with ToT"):
    """
    Plots candidate tracks with ToT visualized by marker size and color.
    - Marker Size: Proportional to ToT.
    - Marker Color: Maps to ToT value (lighter = low ToT, darker = high ToT).
    """
    if candidates_df.empty:
        print("No candidates to plot.")
        return

    # Physical Constants
    PITCH_COL = 0.150  # mm
    PITCH_ROW = 0.050  # mm
    DIST_Z    = 25.4   # mm

    # Setup Figure: 2 Subplots + space for colorbar on the right
    fig, (ax_col, ax_row) = plt.subplots(2, 1, figsize=(11, 8), dpi=150, sharex=True)
    
    # Create a colormap for ToT values
    # 'viridis' or 'plasma' are good for intensity; 'Reds' or 'Blues' for single color scale
    cmap = plt.cm.viridis 
    norm = plt.Normalize(vmin=0, vmax=255) # Assuming ToT is 8-bit (0-255)

    # Loop through every candidate track
    for idx, row in candidates_df.iterrows():
        hits = pd.DataFrame(row['Hits_Data'])
        
        # Convert to Physical Coordinates (mm)
        z_mm = (4 - hits['Layer']) * DIST_Z
        x_mm = hits['Column'] * PITCH_COL
        y_mm = hits['Row'] * PITCH_ROW
        tot  = hits['ToT'].values
        
        # Determine Marker Sizes (scale factor can be adjusted)
        # Base size 20 + ToT value ensures even 0 ToT is visible
        sizes = 20 + (tot * 1.5) 
        
        # Determine Colors based on ToT
        colors = cmap(norm(tot))
        
        # -- Top View (Column vs Z) --
        # Plot lines first (faint gray to show connectivity)
        ax_col.plot(z_mm, x_mm, linestyle='-', linewidth=0.8, alpha=0.3, color='gray')
        # Plot Scatter Points (ToT encoded)
        sc_col = ax_col.scatter(z_mm, x_mm, s=sizes, c=colors, alpha=0.9, edgecolors='k', linewidth=0.5)
        
        # -- Side View (Row vs Z) --
        ax_row.plot(z_mm, y_mm, linestyle='-', linewidth=0.8, alpha=0.3, color='gray')
        sc_row = ax_row.scatter(z_mm, y_mm, s=sizes, c=colors, alpha=0.9, edgecolors='k', linewidth=0.5)
        
        # Annotate ToT values directly on the plot for precision
        for i, val in enumerate(tot):
            # Offset text slightly to right of marker
            ax_col.text(z_mm[i]+1.5, x_mm[i], str(int(val)), fontsize=7, alpha=0.8)
            ax_row.text(z_mm[i]+1.5, y_mm[i], str(int(val)), fontsize=7, alpha=0.8)

    # --- Formatting ---
    
    # Ax1: Column (X) vs Z
    ax_col.set_title(f"{title}: X-Z Top View", fontweight='bold')
    ax_col.set_ylabel("Position X (mm) [Columns]", fontweight='bold')
    ax_col.grid(True, which='major', alpha=0.5)
    ax_col.minorticks_on()
    
    # Ax2: Row (Y) vs Z
    ax_row.set_title("Y-Z Side View", fontweight='bold')
    ax_row.set_ylabel("Position Y (mm) [Rows]", fontweight='bold')
    ax_row.set_xlabel("Position Z (mm) [Beam Direction]", fontweight='bold')
    ax_row.grid(True, which='major', alpha=0.5)
    ax_row.minorticks_on()

    # X-Axis ticks (Z-distance)
    layer_z = [0, 25.4, 50.8, 76.2]
    layer_labels = ["L4\n(0mm)", "L3\n(25.4mm)", "L2\n(50.8mm)", "L1\n(76.2mm)"]
    ax_row.set_xticks(layer_z)
    ax_row.set_xticklabels(layer_labels)

    # Add Colorbar for ToT
    # We use the scatter object from the last plot to define the color scale
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('ToT Value (Energy)', fontweight='bold')

    plt.subplots_adjust(right=0.90) # Make room for colorbar
    plt.show()

# --- Usage Example ---
# candidates = find_candidate_tracks_vectors(data_raw, n_packages=5)
# plot_all_candidate_tracks_with_tot(candidates)



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

def _create_joint_plot(fig, sub_spec, x_data, y_data, title, xlabel, ylabel, 
                       color, x_log=False, xlims=None, bins=50):
    """
    Internal helper: Creates a scatter plot with top and right marginal histograms.
    """
    # Grid: Scatter (3x3), Top Hist (1x3), Right Hist (3x1)
    gs_inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=sub_spec, 
                                                wspace=0.05, hspace=0.05,
                                                width_ratios=[3, 0.5, 0.1, 0.1], 
                                                height_ratios=[0.5, 3, 0.1, 0.1])
    
    ax_main = fig.add_subplot(gs_inner[1, 0])
    ax_top  = fig.add_subplot(gs_inner[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs_inner[1, 1], sharey=ax_main)

    # 1. Main Scatter
    ax_main.scatter(x_data, y_data, alpha=0.4, s=5, c=color, edgecolors='none')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title, fontsize=11, fontweight='bold', loc='left')

    # 2. Scale Handling
    if x_log:
        ax_main.set_xscale('log')
        if xlims:
            ax_main.set_xlim(xlims)
        # Log bins for top histogram
        if xlims:
            hist_bins = np.geomspace(xlims[0], xlims[1], bins)
        else:
            hist_bins = bins
    else:
        hist_bins = bins
        if xlims:
            ax_main.set_xlim(xlims)
        # Zero reference for Difference plots
        ax_main.axvline(0, color='k', ls=':', alpha=0.3) 

    if x_log:
        # Unity reference for Ratio plots
        ax_main.axvline(1.0, color='k', ls=':', alpha=0.3)

    # 3. Marginals
    # Top (X Distribution)
    ax_top.hist(x_data, bins=hist_bins, color=color, alpha=0.6, density=True)
    ax_top.axis('off')

    # Right (Y Distribution)
    ax_right.hist(y_data, bins=bins, orientation='horizontal', color=color, alpha=0.6, density=True)
    ax_right.axis('off')
    
    
def plot_tot_difference_analysis(candidates_df, bins=50):
    """
    Figure 1: Correlations between Vector Residuals and Signed ToT Difference.
    """
    if candidates_df.empty:
        print("No candidate data.")
        return

    # Extract Data
    res_x, res_y, tot_diff = [], [], []
    for idx, row in candidates_df.iterrows():
        global_v = row['Global_Vector']
        segments = row['Segment_Vectors']
        hits = row['Hits_Data']
        for i, seg_v in enumerate(segments):
            if i+1 < len(hits):
                res_x.append(seg_v[0] - global_v[0])
                res_y.append(seg_v[1] - global_v[1])
                tot_diff.append(float(hits[i+1]['ToT']) - float(hits[i]['ToT']))

    # Setup Figure
    fig = plt.figure(figsize=(16, 7), dpi=150)
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.25) # 1 Row, 2 Cols

    # Plot 1: Residual X
    _create_joint_plot(fig, outer_grid[0], tot_diff, res_x,
                       title=r"Residual X vs Signed $\Delta$ToT",
                       xlabel=r"ToT$_{next}$ - ToT$_{curr}$",
                       ylabel="Residual X (Column-wise)",
                       color='royalblue', bins=bins)

    # Plot 2: Residual Y
    _create_joint_plot(fig, outer_grid[1], tot_diff, res_y,
                       title=r"Residual Y vs Signed $\Delta$ToT",
                       xlabel=r"ToT$_{next}$ - ToT$_{curr}$",
                       ylabel="Residual Y (Row-wise)",
                       color='crimson', bins=bins)

    fig.suptitle(f"Analysis A: Energy Difference (N={len(res_x)})", fontsize=14, fontweight='bold', y=0.98)
    plt.show()
    
def plot_tot_ratio_analysis(candidates_df, bins=50):
    """
    Figure 2: Correlations between Vector Residuals and ToT Ratio (Log Scale).
    X-Axis limits: 1/255 to Max Ratio.
    """
    if candidates_df.empty:
        print("No candidate data.")
        return

    # Extract Data
    res_x, res_y, tot_ratio = [], [], []
    for idx, row in candidates_df.iterrows():
        global_v = row['Global_Vector']
        segments = row['Segment_Vectors']
        hits = row['Hits_Data']
        for i, seg_v in enumerate(segments):
            if i+1 < len(hits):
                t_curr = float(hits[i]['ToT'])
                t_next = float(hits[i+1]['ToT'])
                
                if t_curr > 0:
                    ratio = t_next / t_curr
                    res_x.append(seg_v[0] - global_v[0])
                    res_y.append(seg_v[1] - global_v[1])
                    tot_ratio.append(ratio)

    # Setup Figure
    fig = plt.figure(figsize=(16, 7), dpi=150)
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.25)

    # Log Scale Limits
    min_limit = 1.0 / 255.0
    max_limit = max(tot_ratio) if tot_ratio else 10.0
    # Add slight padding to max
    max_limit = max(2.0, max_limit * 1.2)

    # Plot 1: Residual X
    _create_joint_plot(fig, outer_grid[0], tot_ratio, res_x,
                       title="Residual X vs ToT Ratio",
                       xlabel=r"Ratio (ToT$_{next}$ / ToT$_{curr}$)",
                       ylabel="Residual X (Column-wise)",
                       color='royalblue', 
                       x_log=True, xlims=(min_limit, max_limit), bins=bins)

    # Plot 2: Residual Y
    _create_joint_plot(fig, outer_grid[1], tot_ratio, res_y,
                       title="Residual Y vs ToT Ratio",
                       xlabel=r"Ratio (ToT$_{next}$ / ToT$_{curr}$)",
                       ylabel="Residual Y (Row-wise)",
                       color='crimson', 
                       x_log=True, xlims=(min_limit, max_limit), bins=bins)

    fig.suptitle(f"Analysis B: Energy Ratio (Log Scale, N={len(res_x)})", fontsize=14, fontweight='bold', y=0.98)
    plt.show()
    
# Generate Difference Plots
plot_tot_difference_analysis(candidate_tracks_all)

# Generate Ratio Plots
plot_tot_ratio_analysis(candidate_tracks_all)



import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

def plot_heatmap_events(data, row_diff=None, nTS=1, limit=250, x_lim=None, y_lim=None, cmap='jet'):
    """
    Plots a 2D Heatmap with aligned marginals.
    * Adjustment: Added .astype(int) to prevent unsigned integer underflow during subtraction.
    """
    # --- 1. SEARCH & FILTER ---
    print("Filtering and sorting data...")
    
    valid_mask = data['ToT'] > 0
    clean_data = {k: v[valid_mask] for k, v in data.items()}
    
    sort_idx = np.argsort(clean_data['ext_TS'])
    sorted_data = {k: v[sort_idx] for k, v in clean_data.items()}
    
    ts_array = sorted_data['ext_TS']
    time_diffs = np.diff(ts_array)
    split_indices = np.where(time_diffs > nTS)[0] + 1
    event_bounds = np.concatenate(([0], split_indices, [len(ts_array)]))
    
    valid_slices = []
    
    print(f"Scanning {len(event_bounds)-1} clusters for Gap={row_diff}...")
    
    for i in range(len(event_bounds) - 1):
        if len(valid_slices) >= limit:
            break
            
        start, end = event_bounds[i], event_bounds[i+1]
        if end - start < 2: continue
            
        evt_cols = sorted_data['Column'][start:end]
        evt_rows = sorted_data['Row'][start:end]
        
        should_plot = False
        if row_diff is not None:
            unique_cols = np.unique(evt_cols)
            for col in unique_cols:
                rows_in_col = evt_rows[evt_cols == col]
                if len(rows_in_col) < 2: continue
                
                # --- ADJUSTMENT: SAFETY CAST ---
                # Cast to signed int before sorting/diff to prevent uint underflow
                rows_signed = rows_in_col.astype(int)
                
                if np.any(np.abs(np.diff(np.sort(rows_signed))) == row_diff):
                    should_plot = True
                    break
        else:
            should_plot = True 

        if should_plot:
            valid_slices.append((start, end))

    print(f"Found {len(valid_slices)} events. Generating Heatmap...")

    if not valid_slices:
        print("No events found.")
        return

    # --- 2. GRID GENERATION ---
    MAX_ROW = 372
    MAX_COL = 132
    sensor_grid = np.full((MAX_ROW, MAX_COL), np.nan)
    
    for start, end in valid_slices:
        cols = sorted_data['Column'][start:end]
        rows = sorted_data['Row'][start:end]
        tots = sorted_data['ToT'][start:end]
        
        for r, c, t in zip(rows, cols, tots):
            if 0 <= r < MAX_ROW and 0 <= c < MAX_COL:
                current_val = sensor_grid[r, c]
                if np.isnan(current_val) or t > current_val:
                    sensor_grid[r, c] = t

    # --- CALCULATE MARGINALS ---
    x_tot_sum = np.nansum(sensor_grid, axis=0)
    y_tot_sum = np.nansum(sensor_grid, axis=1)

    hit_grid = ~np.isnan(sensor_grid)
    x_hit_count = np.sum(hit_grid, axis=0)
    y_hit_count = np.sum(hit_grid, axis=1)

    # --- 3. PLOTTING SETUP ---
    fig = plt.figure(figsize=(16, 14))
    
    gs = gridspec.GridSpec(2, 3, 
                           width_ratios=[6, 1.2, 0.2], 
                           height_ratios=[1.2, 6],     
                           wspace=0.08, hspace=0.08)

    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    cax      = fig.add_subplot(gs[1, 2])

    if isinstance(cmap, str):
        current_cmap = plt.get_cmap(cmap).copy()
    else:
        current_cmap = cmap.copy()
    current_cmap.set_bad(color='white')

    # --- 4. DRAW PLOTS ---
    im = ax_main.imshow(
        sensor_grid, 
        origin='lower', 
        cmap=current_cmap, 
        interpolation='nearest',
        aspect=1/3, 
        extent=[-0.5, MAX_COL - 0.5, -0.5, MAX_ROW - 0.5]
    )

    # Top Marginal
    ax_top.fill_between(np.arange(MAX_COL), x_tot_sum, step='mid', color='gray', alpha=0.4)
    ax_top.set_ylabel('Sum ToT', color='gray', fontsize=12, fontweight='bold')
    ax_top.tick_params(axis='y', labelcolor='gray', labelsize=10)
    ax_top.tick_params(axis='x', labelbottom=False)
    
    ax_top_hits = ax_top.twinx()
    ax_top_hits.step(np.arange(MAX_COL), x_hit_count, where='mid', color='black', linewidth=1.5)
    ax_top_hits.set_ylabel('Hits', color='black', fontsize=12, fontweight='bold')
    ax_top_hits.tick_params(axis='y', labelcolor='black', labelsize=10)
    
    # Right Marginal
    ax_right.fill_betweenx(np.arange(MAX_ROW), y_tot_sum, step='mid', color='gray', alpha=0.4)
    ax_right.set_xlabel('Sum ToT', color='gray', fontsize=12, fontweight='bold')
    ax_right.tick_params(axis='x', labelcolor='gray', labelsize=10)
    ax_right.tick_params(axis='y', labelleft=False)
    
    ax_right_hits = ax_right.twiny()
    ax_right_hits.step(y_hit_count, np.arange(MAX_ROW), where='mid', color='black', linewidth=1.5)
    ax_right_hits.set_xlabel('Hits', color='black', fontsize=12, fontweight='bold')
    ax_right_hits.tick_params(axis='x', labelcolor='black', labelsize=10)

    # --- 5. FORMATTING ---
    x_min, x_max = -0.5, MAX_COL - 0.5
    y_min, y_max = -0.5, MAX_ROW - 0.5
    
    final_xlim = x_lim if x_lim else (x_min, x_max)
    final_ylim = y_lim if y_lim else (y_min, y_max)
    
    ax_main.set_xlim(final_xlim)
    ax_main.set_ylim(final_ylim)
    ax_top.set_xlim(final_xlim)
    ax_top_hits.set_xlim(final_xlim)
    ax_right.set_ylim(final_ylim)
    ax_right_hits.set_ylim(final_ylim)

    ax_main.set_xlabel('Column ID', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Row ID', fontsize=14, fontweight='bold')
    ax_main.tick_params(axis='both', which='major', labelsize=12)

    fig.suptitle(f'ToT Heatmap: {len(valid_slices)} Events (Gap={row_diff})', y=0.95, fontsize=16, fontweight='bold')

    ax_main.xaxis.set_major_locator(MultipleLocator(5)) 
    ax_main.yaxis.set_major_locator(MultipleLocator(10))
    ax_main.grid(which='major', color='gray', alpha=0.3, linewidth=1)
    
    cbar = plt.colorbar(im, cax=cax)
    cax.set_ylabel('ToT Value', fontsize=14, fontweight='bold')
    cax.tick_params(labelsize=10)

    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_3d_gap_analysis(data, nTS=1, limit=None, cmap='jet'):
    """
    Plots a 3D Bar Graph of Event Profiles vs Gap Size.
    
    * If 'is_crosstalk' exists in data: Plots TWO graphs (All Data vs Clean Data).
    * Otherwise: Plots ONE graph (All Data).
    """
    
    # --- Helper to Run the Analysis logic ---
    def _generate_3d_plot(subset_data, title_prefix):
        print(f"\n--- Processing {title_prefix} ---")
        
        # 1. FILTER & SORT
        valid_mask = subset_data['ToT'] > 0
        clean_data = {k: v[valid_mask] for k, v in subset_data.items()}
        
        if len(clean_data['ToT']) == 0:
            print("No valid hits found.")
            return

        sort_idx = np.argsort(clean_data['ext_TS'])
        sorted_data = {k: v[sort_idx] for k, v in clean_data.items()}
        
        ts_array = sorted_data['ext_TS']
        split_indices = np.where(np.diff(ts_array) > nTS)[0] + 1
        event_bounds = np.concatenate(([0], split_indices, [len(ts_array)]))
        
        gap_map = {}
        count_clusters = 0
        
        print(f"Scanning {len(event_bounds)-1} clusters...")
        
        for i in range(len(event_bounds) - 1):
            if limit and count_clusters >= limit: break
            
            start, end = event_bounds[i], event_bounds[i+1]
            if end - start < 2: continue 
            
            c_cols = sorted_data['Column'][start:end]
            c_rows = sorted_data['Row'][start:end]
            c_tots = sorted_data['ToT'][start:end]
            
            # --- A. Find All Gaps in this Cluster ---
            all_gaps_in_event = set()
            unique_cols = np.unique(c_cols)
            
            for col in unique_cols:
                mask = (c_cols == col)
                if np.sum(mask) < 2: continue
                
                # Safety Cast for gap calculation
                r_in_col = np.sort(c_rows[mask]).astype(int)
                diffs = np.diff(r_in_col)
                all_gaps_in_event.update(diffs)
                
            # --- B. Classify Event by Max Gap ---
            if all_gaps_in_event:
                count_clusters += 1
                max_gap = max(all_gaps_in_event)
                
                # Store every hit in the event under this Max Gap key
                for r, t in zip(c_rows, c_tots):
                    key = (max_gap, r)
                    if key not in gap_map:
                        gap_map[key] = []
                    gap_map[key].append(t)

        if not gap_map:
            print(f"No vertical gaps found in {title_prefix}.")
            return

        # 2. PREPARE PLOTTING ARRAYS
        x_gaps = []
        y_rows = []
        z_counts = []
        c_means = []

        for (gap, row), tots in gap_map.items():
            x_gaps.append(gap)
            y_rows.append(row)
            z_counts.append(len(tots))        
            c_means.append(np.mean(tots))    

        x = np.array(x_gaps)
        y = np.array(y_rows)
        z = np.zeros_like(x)
        
        dx = np.ones_like(x) * 0.8  
        dy = np.ones_like(y) * 1.0   
        dz = np.array(z_counts)      
        
        # 3. COLOR MAPPING
        norm = colors.Normalize(vmin=min(c_means), vmax=max(c_means))
        if isinstance(cmap, str):
            m = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        else:
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            
        bar_colors = m.to_rgba(c_means)

        # 4. PLOTTING
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        print(f"Generating plot with {len(x)} bars...")
        
        ax.bar3d(x, y, z, dx, dy, dz, color=bar_colors, shade=True)

        ax.set_xlabel('Row Separation (Max Gap in Cluster)', fontsize=12, labelpad=10)
        ax.set_ylabel('Row ID', fontsize=12, labelpad=10)
        ax.set_zlabel('Total Counts', fontsize=12, labelpad=10)
        
        ax.set_title(f'{title_prefix}: Event Profile vs Max Gap Size\n(Z=Counts, Color=Mean ToT)', fontsize=14)
        ax.set_ylim(0, 372)

        cbar = plt.colorbar(m, ax=ax, shrink=0.6, aspect=20, pad=0.05)
        cbar.set_label('Mean ToT', fontsize=11)

        ax.view_init(elev=25, azim=-60)

        plt.tight_layout()
        plt.show()

    # --- MAIN LOGIC ---
    
    # 1. Plot ALL Data (Original Behavior)
    _generate_3d_plot(data, "All Hits (Original)")

    # 2. Plot CLEAN Data (If crosstalk column exists)
    if 'is_crosstalk' in data:
        print("\nFound 'is_crosstalk' flag. Generating second plot...")
        
        # Create mask for GOOD hits (NOT crosstalk)
        # Ensure it's a boolean array
        xtalk_mask = np.array(data['is_crosstalk'], dtype=bool)
        good_mask = ~xtalk_mask
        
        # Filter the dictionary
        clean_subset = {k: v[good_mask] for k, v in data.items() if k != 'is_crosstalk'}
        
        # Add 'is_crosstalk' back purely for structure consistency, though strictly not needed
        # clean_subset['is_crosstalk'] = np.zeros(np.sum(good_mask), dtype=bool) 
        
        _generate_3d_plot(clean_subset, "Clean Hits (Crosstalk Removed)")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_displacement_vs_tot(data, nTS=1, y_range=25, normalize=False, cmap='jet'):
    """
    Plots Heatmaps of Signed Row Displacement vs Main Hit ToT.
    
    * Mode 1: All Hits
    * Mode 2: Clean Hits (Crosstalk Removed) - Only if 'is_crosstalk' exists.
    """
    
    # --- INTERNAL HELPER FOR PLOTTING ---
    def _generate_plot(subset_data, title_prefix):
        print(f"\nProcessing {title_prefix}...")

        grid_height = (y_range * 2) + 1
        max_tot = 256
        heatmap_grid = np.zeros((grid_height, max_tot))

        # --- DATA PREP ---
        valid_mask = subset_data['ToT'] > 0
        clean_data = {k: v[valid_mask] for k, v in subset_data.items()}
        
        if len(clean_data['ToT']) == 0:
            print("No valid hits found.")
            return

        sort_idx = np.argsort(clean_data['ext_TS'])
        sorted_data = {k: v[sort_idx] for k, v in clean_data.items()}

        ts_array = sorted_data['ext_TS']
        split_indices = np.where(np.diff(ts_array) > nTS)[0] + 1
        event_bounds = np.concatenate(([0], split_indices, [len(ts_array)]))

        print(f"Scanning {len(event_bounds)-1} clusters...")
        count_clusters = 0

        for i in range(len(event_bounds) - 1):
            start, end = event_bounds[i], event_bounds[i+1]
            c_cols = sorted_data['Column'][start:end]
            c_rows = sorted_data['Row'][start:end]
            c_tots = sorted_data['ToT'][start:end]

            if len(c_tots) == 0: continue

            # --- Identify Main Hit ---
            main_idx = np.argmax(c_tots)
            main_tot = c_tots[main_idx]
            main_row = c_rows[main_idx]
            main_col = c_cols[main_idx]

            if main_tot >= max_tot: continue

            # --- Calculate Signed Displacements ---
            # Only consider hits in the same column for this specific plot
            col_mask = (c_cols == main_col)
            rows_in_col = c_rows[col_mask]

            # SAFE MATH
            deltas = rows_in_col.astype(int) - int(main_row)

            for delta, t_val in zip(deltas, c_tots):
                # Only count secondary hits or normalize properly?
                # Usually we count all, but delta=0 is the main hit itself.
                if -y_range <= delta <= y_range:
                    grid_idx = int(delta + y_range)
                    heatmap_grid[grid_idx, int(main_tot)] += 1

            count_clusters += 1

        # --- NORMALIZATION ---
        if normalize:
            col_sums = heatmap_grid.sum(axis=0)
            nonzero_mask = col_sums > 0
            heatmap_grid[:, nonzero_mask] /= col_sums[nonzero_mask]

        # --- PLOTTING ---
        fig, ax = plt.subplots(figsize=(12, 9))
        
        norm_scale = None
        if not normalize and heatmap_grid.max() > 0:
            norm_scale = LogNorm(vmin=1, vmax=heatmap_grid.max())

        extent_bounds = [-0.5, 255.5, -y_range - 0.5, y_range + 0.5]

        im = ax.imshow(
            heatmap_grid,
            origin='lower',
            aspect='auto',
            cmap=cmap,
            norm=norm_scale,
            interpolation='nearest',
            extent=extent_bounds
        )

        norm_lbl = " (Normalized)" if normalize else " (Counts)"
        ax.set_title(f'{title_prefix}: Displacement vs Main Hit ToT{norm_lbl}\n(Center Y=0 is Main Hit)', fontsize=14)
        ax.set_xlabel('Main Hit ToT Value', fontsize=12)
        ax.set_ylabel('Signed Row Distance (Hit Row - Main Row)', fontsize=12)
        ax.set_ylim(-y_range - 0.5, y_range + 0.5)

        ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability' if normalize else 'Hit Count', fontsize=11)

        plt.tight_layout()
        plt.show()

    # --- MAIN EXECUTION ---
    # 1. Plot All Data
    _generate_plot(data, "All Hits")

    # 2. Plot Clean Data (if 'is_crosstalk' exists)
    if 'is_crosstalk' in data:
        print("\nFound 'is_crosstalk' flag. Generating filtered plot...")
        # Ensure we filter ALL arrays in the dict
        xtalk_mask = np.array(data['is_crosstalk'], dtype=bool)
        good_mask = ~xtalk_mask
        
        clean_subset = {k: v[good_mask] for k, v in data.items() if len(v) == len(good_mask)}
        _generate_plot(clean_subset, "Clean Hits (Crosstalk Removed)")
    
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_displacement_vs_tot_3d(data, nTS=1, y_range=25, log_scale=False, cmap='jet'):
    """
    Plots a 3D Bar Graph of Signed Row Displacement vs Main Hit ToT.
    
    * Update: PLOTS EVERY HIT. 
      - Removes column constraints. 
      - Calculates vertical displacement for ALL hits in a cluster relative 
        to the Main Hit, regardless of column alignment.
    """
    print("Generating 3D Displacement Plot (All Hits)...")

    # --- 1. SETUP GRID ---
    grid_height = (y_range * 2) + 1
    max_tot = 256
    
    # Grid Z (Counts) and Color (Sum ToT)
    count_grid = np.zeros((grid_height, max_tot))
    tot_sum_grid = np.zeros((grid_height, max_tot))

    # --- 2. DATA PROCESSING ---
    valid_mask = data['ToT'] > 0
    clean_data = {k: v[valid_mask] for k, v in data.items()}
    
    sort_idx = np.argsort(clean_data['ext_TS'])
    sorted_data = {k: v[sort_idx] for k, v in clean_data.items()}
    
    ts_array = sorted_data['ext_TS']
    split_indices = np.where(np.diff(ts_array) > nTS)[0] + 1
    event_bounds = np.concatenate(([0], split_indices, [len(ts_array)]))
    
    print(f"Scanning {len(event_bounds)-1} clusters...")
    
    count_clusters = 0
    total_hits_plotted = 0
    
    for i in range(len(event_bounds) - 1):
        start, end = event_bounds[i], event_bounds[i+1]
        
        c_rows = sorted_data['Row'][start:end]
        c_tots = sorted_data['ToT'][start:end]
        
        if len(c_tots) == 0: continue

        # --- Identify Main Hit ---
        # The anchor point for the entire cluster
        main_idx = np.argmax(c_tots)
        main_tot = c_tots[main_idx]
        main_row = c_rows[main_idx]
        
        # Safety: Ensure we don't crash if ToT is out of bounds
        x_idx = int(main_tot)
        if x_idx >= max_tot: x_idx = max_tot - 1

        # --- Process ALL Hits in Cluster ---
        # Previously we filtered by column. Now we take everything.
        # This ensures every hit gets a Z-entry.
        
        # Calculate vertical displacement for every hit relative to Main Hit
        deltas = c_rows.astype(int) - int(main_row)
        
        for delta, t_val in zip(deltas, c_tots):
            # Check if the hit is within the vertical viewing window
            if -y_range <= delta <= y_range:
                # Map signed delta to grid index
                # -y_range -> index 0
                y_idx = int(delta + y_range)
                
                count_grid[y_idx, x_idx] += 1
                tot_sum_grid[y_idx, x_idx] += t_val
                total_hits_plotted += 1
                
        count_clusters += 1
        
    print(f"Total Hits Plotted: {total_hits_plotted}")

    # --- 3. PREPARE PLOTTING DATA ---
    y_indices, x_indices = np.where(count_grid > 0)
    
    if len(x_indices) == 0:
        print("No data to plot.")
        return

    # Extract values
    z_counts = count_grid[y_indices, x_indices]
    z_sums = tot_sum_grid[y_indices, x_indices]
    c_means = z_sums / z_counts

    # Map indices to Axis Coordinates
    x = x_indices  # Main Hit ToT
    y = y_indices - y_range # Signed Delta
    z = np.zeros_like(x)
    
    dx = np.ones_like(x) * 1.0
    dy = np.ones_like(y) * 0.8
    
    if log_scale:
        dz = np.log10(z_counts)
        dz[dz == 0] = 0.1 # Visibility fix for count=1
        z_label = 'Log10(Total Hits)'
    else:
        dz = z_counts
        z_label = 'Total Hits'

    # --- 4. COLOR MAPPING ---
    norm = colors.Normalize(vmin=np.min(c_means), vmax=np.max(c_means))
    if isinstance(cmap, str):
        m = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    else:
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        
    bar_colors = m.to_rgba(c_means)

    # --- 5. PLOTTING ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Generating 3D plot with {len(x)} bars...")
    
    ax.bar3d(x, y, z, dx, dy, dz, color=bar_colors, shade=True)

    # Formatting
    ax.set_xlabel('Main Hit ToT', fontsize=12, labelpad=10)
    ax.set_ylabel('Signed Row Distance', fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    
    scale_txt = " (Log Scale)" if log_scale else ""
    ax.set_title(f'Cluster Displacement vs Main Hit ToT{scale_txt}\n(Color = Mean ToT of Hits)', fontsize=14)
    
    ax.set_xlim(0, 256)
    ax.set_ylim(-y_range, y_range)
    
    cbar = plt.colorbar(m, ax=ax, shrink=0.6, aspect=20, pad=0.05)
    cbar.set_label('Mean ToT of Bin', fontsize=11)
    
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_absolute_position_vs_main_tot_3d(data, nTS=1, nrows=None, y_axis_mode='row', log_scale=False, cmap='jet'):
    """
    Plots a 3D Bar Graph of Absolute Position vs Main Hit ToT.
    
    Parameters:
    - nrows (int): If provided, ONLY plots clusters containing at least one pair 
      of hits in the same column separated by exactly this row distance.
    - y_axis_mode: 'row' (0-372) or 'col' (0-132).
    - log_scale: Logarithmic Z-axis.
    """
    
    # --- 1. CONFIGURE AXES ---
    if y_axis_mode.lower() == 'col':
        MAX_Y = 132
        y_data_key = 'Column'
        y_label_str = 'Absolute Column ID'
        print(f"Generating 3D Plot: Column ID vs Main Hit ToT...")
    else:
        MAX_Y = 372
        y_data_key = 'Row'
        y_label_str = 'Absolute Row ID'
        print(f"Generating 3D Plot: Row ID vs Main Hit ToT...")

    MAX_TOT = 256
    
    # Grid Z (Counts) and Color (Sum ToT)
    count_grid = np.zeros((MAX_Y, MAX_TOT))
    tot_sum_grid = np.zeros((MAX_Y, MAX_TOT))

    # --- 2. DATA PROCESSING ---
    valid_mask = data['ToT'] > 0
    clean_data = {k: v[valid_mask] for k, v in data.items()}
    
    sort_idx = np.argsort(clean_data['ext_TS'])
    sorted_data = {k: v[sort_idx] for k, v in clean_data.items()}
    
    ts_array = sorted_data['ext_TS']
    split_indices = np.where(np.diff(ts_array) > nTS)[0] + 1
    event_bounds = np.concatenate(([0], split_indices, [len(ts_array)]))
    
    print(f"Scanning {len(event_bounds)-1} clusters...")
    if nrows is not None:
        print(f"-> Filtering for clusters containing a vertical gap of exactly {nrows} rows.")
    
    count_clusters = 0
    total_hits_plotted = 0
    
    for i in range(len(event_bounds) - 1):
        start, end = event_bounds[i], event_bounds[i+1]
        
        c_cols = sorted_data['Column'][start:end]
        c_rows = sorted_data['Row'][start:end]
        c_tots = sorted_data['ToT'][start:end]
        
        if len(c_tots) < 2: continue # Need at least 2 hits to form a gap

        # --- A. FILTER LOGIC (If nrows is set) ---
        if nrows is not None:
            has_target_gap = False
            unique_cols = np.unique(c_cols)
            
            for col in unique_cols:
                # Get rows in this column
                mask = (c_cols == col)
                if np.sum(mask) < 2: continue
                
                # Check separations
                r_in_col = np.sort(c_rows[mask]).astype(int) # Safety cast
                diffs = np.diff(r_in_col)
                
                # If ANY gap matches nrows, this event is valid
                if np.any(diffs == nrows):
                    has_target_gap = True
                    break
            
            # Skip this entire cluster if it doesn't meet the criteria
            if not has_target_gap:
                continue

        # --- B. PLOT LOGIC ---
        # Identify Main Hit (X-Coordinate)
        main_idx = np.argmax(c_tots)
        main_tot = c_tots[main_idx]
        x_idx = int(main_tot)
        if x_idx >= MAX_TOT: x_idx = MAX_TOT - 1

        # Get Y-axis data (Row or Col) based on mode
        c_pos = sorted_data[y_data_key][start:end]

        # Process ALL Hits in Cluster
        for pos, t in zip(c_pos, c_tots):
            y_idx = int(pos)
            
            if 0 <= y_idx < MAX_Y:
                count_grid[y_idx, x_idx] += 1
                tot_sum_grid[y_idx, x_idx] += t
                total_hits_plotted += 1
                
        count_clusters += 1

    print(f"Clusters Found: {count_clusters}")
    print(f"Total Hits Plotted: {total_hits_plotted}")

    # --- 3. PREPARE PLOTTING DATA ---
    y_indices, x_indices = np.where(count_grid > 0)
    
    if len(x_indices) == 0:
        print("No events matched the criteria.")
        return

    # Extract values
    z_counts = count_grid[y_indices, x_indices]
    z_sums = tot_sum_grid[y_indices, x_indices]
    c_means = z_sums / z_counts

    # Map to plotting coordinates
    x = x_indices  # Main Hit ToT
    y = y_indices  # Absolute Position
    z = np.zeros_like(x)
    
    dx = np.ones_like(x) * 1.0
    dy = np.ones_like(y) * 1.0
    
    if log_scale:
        dz = np.log10(z_counts)
        dz[dz == 0] = 0.1 
        z_label = 'Log10(Total Hits)'
    else:
        dz = z_counts
        z_label = 'Total Hits'

    # --- 4. COLOR MAPPING ---
    norm = colors.Normalize(vmin=np.min(c_means), vmax=np.max(c_means))
    if isinstance(cmap, str):
        m = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    else:
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        
    bar_colors = m.to_rgba(c_means)

    # --- 5. PLOTTING ---
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Generating 3D plot with {len(x)} bars...")
    
    ax.bar3d(x, y, z, dx, dy, dz, color=bar_colors, shade=True)

    # Formatting
    ax.set_xlabel('Main Hit ToT (Event Energy)', fontsize=12, labelpad=10)
    ax.set_ylabel(y_label_str, fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    
    filter_txt = f" (Filtered: Row Gap={nrows})" if nrows is not None else ""
    scale_txt = " (Log Scale)" if log_scale else ""
    
    ax.set_title(f'{y_label_str} vs Main Hit ToT{filter_txt}{scale_txt}\n(Color = Mean ToT)', fontsize=14)
    
    ax.set_xlim(0, 256)
    ax.set_ylim(0, MAX_Y)
    
    cbar = plt.colorbar(m, ax=ax, shrink=0.6, aspect=20, pad=0.05)
    cbar.set_label('Mean ToT', fontsize=11)
    
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.show()
    
    


def filter_and_sort_data(data, hithresh = 255, lothresh = 0):
    valid_mask = (data['ToT'] < hithresh) & (data['ToT'] > lothresh)
    if not np.any(valid_mask):
        print("Warning: All data filtered out!")
        return {k: np.array([]) for k in data}
    filtered_data = {k: v[valid_mask] for k, v in data.items()}
    
    print(f"After Filtering: {len(filtered_data['ToT'])} hits")
    if 'ext_TS' in filtered_data:
        sort_idx = np.argsort(filtered_data['ext_TS'])
        final_data = {k: v[sort_idx] for k, v in filtered_data.items()}
        return final_data
    
    return filtered_data