import numpy as np
import time
import sys
import gc
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.optimize import curve_fit
from numba import njit


# --- 1. Fast Index Matching (Numba) ---
@njit(cache=True)
def _expand_match_indices(u_indices, start_idxs, end_idxs):
    """
    Expands one-to-many time matches into aligned index arrays for vectorized operations.
    """
    n_hits = len(u_indices)
    total_matches = 0
    for i in range(n_hits):
        count = end_idxs[i] - start_idxs[i]
        if count > 0: total_matches += count

    idx_u = np.empty(total_matches, dtype=np.int64)
    idx_d = np.empty(total_matches, dtype=np.int64)

    cursor = 0
    for i in range(n_hits):
        count = end_idxs[i] - start_idxs[i]
        if count > 0:
            for k in range(count):
                idx_u[cursor + k] = u_indices[i]
            start = start_idxs[i]
            for k in range(count):
                idx_d[cursor + k] = start + k
            cursor += count

    return idx_u, idx_d

from scipy.optimize import curve_fit

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_residual_peak(residuals, pitch_scale=1.0):
    """
    Fits a Gaussian to the residual distribution to find the true peak (offset).
    Falls back to median if fit fails or data is too sparse.
    """
    # 1. Filter huge outliers first (coarse cleanup)
    # limit to +/- 50 pixels/units to avoid random background skewing the binning
    clean = residuals[np.abs(residuals) < 50 * pitch_scale]
    
    if len(clean) < 50:
        return 0.0 # Not enough data
        
    # 2. Binning
    # We want fine bins for precision. 
    # For X (cols), use pitch_scale. For Y (rows), use 1.0.
    bin_width = 1.0 * pitch_scale
    limit = 30 * pitch_scale
    
    # Create bins centered on the expected resolution
    bins = np.arange(-limit, limit + bin_width, bin_width)
    counts, edges = np.histogram(clean, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # 3. Initial Guesses for Optimizer
    p0_amp = np.max(counts)
    p0_mean = np.median(clean) # Median is a great guess for the peak
    p0_sigma = np.std(clean)
    if p0_sigma == 0: p0_sigma = 1.0
    
    try:
        # 4. Curve Fit
        # Bounds: Amp > 0, Mean within range, Sigma reasonable
        popt, _ = curve_fit(gaussian, centers, counts, p0=[p0_amp, p0_mean, p0_sigma], maxfev=2000)
        
        peak_val = popt[1] # The 'x0' parameter
        
        # Sanity check: if the fit found a peak wildly far from median, reject it
        if abs(peak_val - p0_mean) > 10 * pitch_scale:
            return p0_mean
            
        return peak_val
        
    except (RuntimeError, ValueError):
        # Fallback to median if fit fails
        return np.median(clean)

def calculate_translational_misalignment_robust(
    cluster_dict: dict, 
    ref_layer: int = 4, 
    pitch_ratio_col: float = 3.0,
    time_window: int = 10, 
    jitter_tol: int = 1
):
    print(f"\n[Alignment] Computing ROBUST Gaussian-fitted parameters (Ref: L{ref_layer})...")

    # Data Prep (Same as before)
    df = pd.DataFrame({
        'Layer': cluster_dict['Layer'],
        'PhysRow': cluster_dict['cog_row'], 
        'PhysCol': cluster_dict['cog_col'] * pitch_ratio_col, 
        'TS': cluster_dict['ts_start'].astype(np.int64)
    })
    df = df[cluster_dict['clusterID'] != -1].sort_values('TS')

    layers_data = {}
    available_layers = sorted(df['Layer'].unique())
    for L in available_layers:
        sub = df[df['Layer'] == L]
        layers_data[L] = {'r': sub['PhysRow'].values, 'c': sub['PhysCol'].values, 't': sub['TS'].values}

    if ref_layer == max(available_layers):
        layer_order = sorted(available_layers, reverse=True)
    else:
        layer_order = sorted(available_layers)
        
    pairs = list(zip(layer_order[:-1], layer_order[1:]))
    results_list = []
    chunk_size = 50000
    
    for (fixed_L, moving_L) in pairs:
        data_ref = layers_data[fixed_L]
        data_mov = layers_data[moving_L]
        n_ref = len(data_ref['t'])
        matches_r, matches_c = [], []
        
        if n_ref > 0 and len(data_mov['t']) > 0:
            for i in range(0, n_ref, chunk_size):
                end_i = min(i + chunk_size, n_ref)
                t_ref_chunk = data_ref['t'][i:end_i]
                start_idxs = np.searchsorted(data_mov['t'], t_ref_chunk - jitter_tol, side='left')
                end_idxs   = np.searchsorted(data_mov['t'], t_ref_chunk + time_window, side='right')
                idx_ref_rel, idx_mov = _expand_match_indices(np.arange(len(t_ref_chunk)), start_idxs, end_idxs)
                
                if len(idx_ref_rel) > 0:
                    ref_r = data_ref['r'][i:end_i][idx_ref_rel]
                    ref_c = data_ref['c'][i:end_i][idx_ref_rel]
                    mov_r = data_mov['r'][idx_mov]
                    mov_c = data_mov['c'][idx_mov]
                    matches_r.append(ref_r - mov_r)
                    matches_c.append(ref_c - mov_c)
        
        if not matches_r:
            dx, dy = 0.0, 0.0
        else:
            diff_r = np.concatenate(matches_r)
            diff_c = np.concatenate(matches_c)
            
            # --- IMPROVEMENT: Gaussian Fitting ---
            # Pass pitch_ratio_col for columns to scale binning correctly
            dx = fit_residual_peak(diff_c, pitch_scale=pitch_ratio_col)
            dy = fit_residual_peak(diff_r, pitch_scale=1.0)
            
        results_list.append({
            'Layer_to_Align': moving_L, 'Ref_Layer': fixed_L,
            'dx_col': dx, 'dy_row': dy
        })

    df_res = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print(f"{'ROBUST MISALIGNMENT (GAUSSIAN FIT)':^80}")
    print("="*80)
    print(f"{'Layer':<6} | {'Ref Layer':<10} | {'dx (Column)':<15} | {'dy (Row)':<15}")
    print("-" * 80)
    for _, r in df_res.iterrows():
        print(f"L{int(r['Layer_to_Align']):<5} | L{int(r['Ref_Layer']):<9} | {r['dx_col']:<15.3f} | {r['dy_row']:<15.3f}")
    print("="*80 + "\n")
    
    return df_res
# --- 3. Diagnostic Plotting ---
def plot_misalignment_diagnostics(cluster_dict, alignment_df, pitch_ratio_col=3.0, time_window=10):
    """
    Generates a single figure with 2D histograms of residuals.
    Bins are scaled to pitch_ratio_col on X axis to avoid striping.
    """
    print("\n[Diagnostics] Generating alignment plots...")

    df = pd.DataFrame({
        'Layer': cluster_dict['Layer'],
        'PhysRow': cluster_dict['cog_row'], 
        'PhysCol': cluster_dict['cog_col'] * pitch_ratio_col, 
        'TS': cluster_dict['ts_start'].astype(np.int64)
    })
    df = df[cluster_dict['clusterID'] != -1].sort_values('TS')

    layers_data = {}
    for L in df['Layer'].unique():
        sub = df[df['Layer'] == L]
        layers_data[L] = {'r': sub['PhysRow'].values, 'c': sub['PhysCol'].values, 't': sub['TS'].values}

    # Setup Plot
    n_pairs = len(alignment_df)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs), squeeze=False)
    fig.suptitle("Layer Misalignment Diagnostics (2D Residuals)", fontsize=16)

    chunk_size = 50000

    for idx, row_data in alignment_df.iterrows():
        mov_L = int(row_data['Layer_to_Align'])
        ref_L = int(row_data['Ref_Layer'])
        dx = row_data['dx_col']
        dy = row_data['dy_row']

        # Get residuals
        data_ref = layers_data[ref_L]
        data_mov = layers_data[mov_L]

        matches_r, matches_c = [], []
        n_ref = len(data_ref['t'])

        # Limit samples for plotting speed
        sample_limit = 2000000 
        processed = 0

        for i in range(0, n_ref, chunk_size):
            if processed >= sample_limit: break

            end_i = min(i + chunk_size, n_ref)
            t_ref_chunk = data_ref['t'][i:end_i]

            start_idxs = np.searchsorted(data_mov['t'], t_ref_chunk - 1, side='left')
            end_idxs   = np.searchsorted(data_mov['t'], t_ref_chunk + time_window, side='right')

            idx_ref_rel, idx_mov = _expand_match_indices(np.arange(len(t_ref_chunk)), start_idxs, end_idxs)

            if len(idx_ref_rel) > 0:
                ref_r = data_ref['r'][i:end_i][idx_ref_rel]
                ref_c = data_ref['c'][i:end_i][idx_ref_rel]
                mov_r = data_mov['r'][idx_mov]
                mov_c = data_mov['c'][idx_mov]

                matches_r.append(ref_r - mov_r)
                matches_c.append(ref_c - mov_c)
                processed += len(idx_ref_rel)

        if not matches_r: continue

        d_r = np.concatenate(matches_r)
        d_c = np.concatenate(matches_c)

        # --- Adjusted Binning Logic ---
        # Plot Range (+/- 10 pixels roughly)
        r_lim = 10
        c_lim = 10 * pitch_ratio_col

        # Y Bins: Step 1.0, centered on integers
        # Edges: -10.5, -9.5, ..., 10.5
        bins_y = np.arange(-r_lim - 0.5, r_lim + 1.5, 1.0)

        # X Bins: Step = pitch_ratio_col, centered on multiples of pitch
        # Edges: (N * pitch) - 0.5*pitch ...
        half_pitch = 0.5 * pitch_ratio_col
        # We start slightly before -c_lim to align with the step grid
        bins_x = np.arange(-c_lim - half_pitch, c_lim + half_pitch + pitch_ratio_col, pitch_ratio_col)

        # LEFT: Uncorrected
        ax_raw = axes[idx, 0]
        h1 = ax_raw.hist2d(d_c, d_r, bins=[bins_x, bins_y], cmap='inferno', norm=LogNorm())
        ax_raw.set_title(f"Raw: L{ref_L} - L{mov_L}")
        ax_raw.set_xlabel("dCol (Phys)")
        ax_raw.set_ylabel("dRow (Phys)")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.axvline(0, color='cyan', linestyle='--', alpha=0.5)
        ax_raw.axhline(0, color='cyan', linestyle='--', alpha=0.5)
        plt.colorbar(h1[3], ax=ax_raw)

        # RIGHT: Corrected
        d_c_corr = d_c - dx
        d_r_corr = d_r - dy

        ax_corr = axes[idx, 1]
        h2 = ax_corr.hist2d(d_c_corr, d_r_corr, bins=[bins_x, bins_y], cmap='viridis', norm=LogNorm())
        ax_corr.set_title(f"Aligned: L{ref_L} - L{mov_L} (dx={dx:.2f}, dy={dy:.2f})")
        ax_corr.set_xlabel("dCol (Phys)")
        ax_corr.set_ylabel("dRow (Phys)")
        ax_corr.grid(True, alpha=0.3)
        ax_corr.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax_corr.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(h2[3], ax=ax_corr)

    plt.tight_layout()
    plt.show()

# Example Usage:
df_align = calculate_translational_misalignment(final_clusters, ref_layer=4, pitch_ratio_col=3.0)
plot_misalignment_diagnostics(final_clusters, df_align, pitch_ratio_col=3.0)


def tracking(
    cluster_data: dict,
    alignment_df: pd.DataFrame = None, 
    target_n_tracks: int = 2000,  
    subset_factor: int = 100,      
    search_radius: float = 10.0, 
    pitch_ratio_col: float = 3.0, 
    time_window: int = 10,        
    jitter_tol: int = 1,
    max_hits_per_cluster: int = 5,
    chunk_size: int = 2000,
    min_hits: int = 4,           
    xtalk_filter: str = 'all'    
) -> dict:
    
    print(f"--- Track Search (Min Hits: {min_hits}, Xtalk: {xtalk_filter}) ---")
    t0 = time.time()
    
    # 1. Load Alignment Parameters
    corrections = {L: {'dr': 0.0, 'dc': 0.0} for L in [4, 3, 2, 1]}

    if alignment_df is not None and not alignment_df.empty:
        print("   -> Loading alignment parameters...")
        def get_pars(u, d):
            r = alignment_df[(alignment_df['Ref_Layer'] == u) & (alignment_df['Layer_to_Align'] == d)]
            if r.empty: return 0.0, 0.0
            # Handle variable column names
            dy = r['Row_Misalign_Phys'].values[0] if 'Row_Misalign_Phys' in r else r['dy_row'].values[0]
            dx = r['Col_Misalign_Phys'].values[0] if 'Col_Misalign_Phys' in r else r['dx_col'].values[0]
            return dy, dx

        dr43, dc43 = get_pars(4, 3)
        dr32, dc32 = get_pars(3, 2)
        dr21, dc21 = get_pars(2, 1)

        corrections[3] = {'dr': dr43, 'dc': dc43}
        corrections[2] = {'dr': dr43 + dr32, 'dc': dc43 + dc32}
        corrections[1] = {'dr': dr43 + dr32 + dr21, 'dc': dc43 + dc32 + dc21}

    # 2. Data Prep & Filtering
    try:
        mask = (cluster_data['clusterID'] != -1) & (cluster_data['n_hits'] <= max_hits_per_cluster)
        
        # --- ROBUST XTALK FILTERING ---
        if xtalk_filter != 'all':
            xt_arr = cluster_data['xtalk_type']
            # Since dtype=object, we must be careful. 
            # We assume "Clean" is 0, '0', or [0]. Everything else is Xtalk.
            
            # Create a boolean mask safely for objects
            # This lambda checks if x is "effectively zero"
            is_zero = lambda x: (isinstance(x, (int, float)) and x == 0) or \
                                (isinstance(x, str) and x == '0') or \
                                (isinstance(x, list) and len(x) == 1 and x[0] == 0)

            # Vectorize is slow but safe for mixed object types
            clean_mask = np.vectorize(is_zero)(xt_arr)

            if xtalk_filter == 'clean_only':
                mask = mask & clean_mask
            elif xtalk_filter == 'xtalk_only':
                mask = mask & (~clean_mask)

        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            print("No clusters match the filter criteria.")
            return {}

        valid_ts = cluster_data['ts_start'][valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_ts)]
        
    except KeyError as e:
        print(f"Missing Key in cluster data: {e}")
        return {}

    limit = min(len(sorted_indices), target_n_tracks * subset_factor) if target_n_tracks else len(sorted_indices)
    active_indices = sorted_indices[:limit]

    # --- Helper Functions ---
    def apply_trans_correction(px, py, layer_mask, layer_id):
        pars = corrections[layer_id]
        px[layer_mask] += pars['dc']
        py[layer_mask] += pars['dr']
        return px, py

    def _find_tracks_in_chunk(chunk_idxs):
        c_layer = cluster_data['Layer'][chunk_idxs]
        c_ts    = cluster_data['ts_start'][chunk_idxs]
        c_id    = cluster_data['clusterID'][chunk_idxs]
        
        phys_x = cluster_data['cog_col'][chunk_idxs] * pitch_ratio_col
        phys_y = cluster_data['cog_row'][chunk_idxs].copy()

        # Apply Alignment
        for L in [3, 2, 1]:
            mask = (c_layer == L)
            if np.any(mask):
                phys_x, phys_y = apply_trans_correction(phys_x, phys_y, mask, L)

        layers = {}
        for L in [4, 3, 2, 1]:
            mask = (c_layer == L)
            if not np.any(mask): 
                layers[L] = None
                continue
            
            idxs = np.where(mask)[0]
            coords = np.column_stack((phys_x[idxs], phys_y[idxs]))
            tree = cKDTree(coords)
            
            layers[L] = {
                'tree': tree, 'coords': coords,
                'ts': c_ts[idxs].astype(np.int64),
                'local_idx': idxs 
            }

        adj_list = {}
        transitions = list(combinations([4, 3, 2, 1], 2)) 
        
        for up, down in transitions:
            if layers[up] is None or layers[down] is None: continue
            
            tree_u = layers[up]['tree']
            tree_d = layers[down]['tree']
            matches = tree_u.query_ball_tree(tree_d, r=search_radius)
            ts_u = layers[up]['ts']; ts_d = layers[down]['ts']
            
            for i, match_indices in enumerate(matches):
                if not match_indices: continue
                u_idx = layers[up]['local_idx'][i]
                t_start = ts_u[i]
                
                valid_matches = []
                for j in match_indices:
                    dt = ts_d[j] - t_start
                    if -jitter_tol <= dt <= time_window:
                        d_idx = layers[down]['local_idx'][j]
                        valid_matches.append((down, d_idx))
                
                if valid_matches:
                    node = (up, u_idx)
                    if node not in adj_list: adj_list[node] = []
                    adj_list[node].extend(valid_matches)

        tracks_found = []
        
        def dfs(current_node, current_path):
            children = adj_list.get(current_node, [])
            curr_layer = current_node[0]
            
            if len(current_path) + (curr_layer - 1) < min_hits: return 

            is_terminal = (curr_layer == 1) or (not children)
            
            if is_terminal:
                if len(current_path) >= min_hits:
                    track_entry = [-1, -1, -1, -1]
                    for (L, idx) in current_path:
                        track_entry[4-L] = c_id[idx]
                    tracks_found.append(track_entry)
                return

            for child in children:
                if child[0] < curr_layer:
                    dfs(child, current_path + [child])

        # Seeding
        for L in [4, 3, 2]:
            if layers[L]:
                for i in range(len(layers[L]['local_idx'])):
                    dfs((L, layers[L]['local_idx'][i]), [(L, layers[L]['local_idx'][i])])

        return np.array(tracks_found, dtype=np.int64) if tracks_found else None

    # Execution Loop
    all_tracks_arr = []
    for i in range(0, len(active_indices), chunk_size):
        end = min(i+chunk_size, len(active_indices))
        w_end = min(end+2000, len(active_indices)) if end < len(active_indices) else end
        res = _find_tracks_in_chunk(active_indices[i:w_end])
        if res is not None and len(res) > 0: all_tracks_arr.append(res)
        if target_n_tracks and sum(len(x) for x in all_tracks_arr) >= target_n_tracks*1.2: break
            
    if not all_tracks_arr: return {}
    
    full_stack = np.vstack(all_tracks_arr)
    final_ids = np.unique(full_stack, axis=0)
    
    hit_counts = np.sum(final_ids != -1, axis=1)
    final_ids = final_ids[hit_counts >= min_hits]

    # 3. Output
    results = {}
    lookup_map = pd.Series(active_indices, index=cluster_data['clusterID'][active_indices])
    
    # --- FIXED EXTRACT COL ---
    def extract_col(key):
        valid_mask = (final_ids != -1)
        valid_ids_flat = final_ids[valid_mask]
        mapped_idxs = lookup_map.reindex(valid_ids_flat).values.astype(int)
        vals = cluster_data[key][mapped_idxs]
        
        # Check type - if object/list, initialize object array
        if vals.dtype == object:
            col_data = np.empty(final_ids.shape, dtype=object)
            fill_val = None
        else:
            col_data = np.zeros(final_ids.shape, dtype=float)
            fill_val = np.nan
        
        col_data[valid_mask] = vals
        col_data[~valid_mask] = fill_val
        return col_data

    for k, L in enumerate([4, 3, 2, 1]):
        results[f'L{L}_ID'] = final_ids[:, k]
        results[f'x{L}'] = extract_col('cog_col')[:, k]
        results[f'y{L}'] = extract_col('cog_row')[:, k]
        for p in ['ts_start', 'sum_ToT', 'n_hits', 'pToF', 'xtalk_type']:
             out_key = {'ts_start':'t', 'sum_ToT':'tot', 'n_hits':'nhits', 'pToF':'pToF', 'xtalk_type':'xtalk'}[p]
             results[f'{out_key}{L}'] = extract_col(p)[:, k]

    # Chi2 Calculation
    # Must explicitly cast to float for math, ignoring None/Objects
    # Using 'x' and 'y' which are definitely floats
    px_stack = np.column_stack([results[f'x{L}'].astype(float) * pitch_ratio_col for L in [4,3,2,1]])
    py_stack = np.column_stack([results[f'y{L}'].astype(float) for L in [4,3,2,1]])
    
    # Apply Corrections
    px_stack[:, 1] += corrections[3]['dc']; py_stack[:, 1] += corrections[3]['dr']
    px_stack[:, 2] += corrections[2]['dc']; py_stack[:, 2] += corrections[2]['dr']
    px_stack[:, 3] += corrections[1]['dc']; py_stack[:, 3] += corrections[1]['dr']
    
    dz = np.array([-1.5, -0.5, 0.5, 1.5])
    chi2_arr = []
    
    # Simple check for NaN (missing hits)
    valid = ~np.isnan(px_stack)
    
    for i in range(len(final_ids)):
        msk = valid[i]
        if np.sum(msk) < 2: 
            chi2_arr.append(999.0) 
            continue
            
        z_i = dz[msk]; x_i = px_stack[i][msk]; y_i = py_stack[i][msk]
        
        # Fit X
        A = np.vstack([z_i, np.ones(len(z_i))]).T
        m_x, c_x = np.linalg.lstsq(A, x_i, rcond=None)[0]
        resid_x = x_i - (m_x * z_i + c_x)
        
        # Fit Y
        m_y, c_y = np.linalg.lstsq(A, y_i, rcond=None)[0]
        resid_y = y_i - (m_y * z_i + c_y)
        chi2_arr.append(np.sum(resid_x**2 + resid_y**2))

    results['chi2'] = np.array(chi2_arr)
    
    sort_idx = np.argsort(results['chi2'])
    if target_n_tracks: sort_idx = sort_idx[:target_n_tracks]
    
    final_dict = {k: v[sort_idx] for k, v in results.items()}
    print(f"--- Finished. Found {len(final_dict['chi2'])} tracks (Filter={xtalk_filter}, MinHits={min_hits}) in {time.time()-t0:.2f}s ---")
    return final_dict

def check_alignment_residuals(
    tracks_dict: dict, 
    alignment_df: pd.DataFrame = None, 
    ref_layer: int = 4,
    pitch_ratio_col: float = 3.0,
    show_plots: bool = True
):
    if not tracks_dict: 
        print("No tracks provided for residual check.")
        return

    print(f"\n[Alignment Check] Analyzing Residuals (Ref: L{ref_layer})...")

    available_layers = [int(k[1]) for k in tracks_dict.keys() if k.startswith('L') and k.endswith('_ID')]
    available_layers = sorted(available_layers, reverse=True) 
    
    df_phys = pd.DataFrame()
    for L in available_layers:
        df_phys[f'x{L}'] = tracks_dict[f'x{L}'].astype(float) * pitch_ratio_col
        df_phys[f'y{L}'] = tracks_dict[f'y{L}'].astype(float)

    df_aln = df_phys.copy()

    if alignment_df is not None and not alignment_df.empty:
        def get_pars(u, d):
            r = alignment_df[(alignment_df['Ref_Layer'] == u) & (alignment_df['Layer_to_Align'] == d)]
            if r.empty: return 0.0, 0.0
            dy = r['Row_Misalign_Phys'].values[0] if 'Row_Misalign_Phys' in r else r['dy_row'].values[0]
            dx = r['Col_Misalign_Phys'].values[0] if 'Col_Misalign_Phys' in r else r['dx_col'].values[0]
            return dy, dx

        corrections = {L: (0.0, 0.0) for L in available_layers}
        if ref_layer == max(available_layers):
            cum_dx, cum_dy = 0.0, 0.0
            for i in range(len(available_layers)-1):
                curr, next_l = available_layers[i], available_layers[i+1]
                dy, dx = get_pars(curr, next_l)
                cum_dx += dx; cum_dy += dy
                corrections[next_l] = (cum_dx, cum_dy)

        for L, (dx, dy) in corrections.items():
            if L != ref_layer:
                df_aln[f'x{L}'] += dx; df_aln[f'y{L}'] += dy

    z_map = {L: (max(available_layers) - L) for L in available_layers} 
    
    def get_linear_fit_residuals(df_in):
        # We need to handle NaNs (missing hits)
        # Vectorized nan-aware linear regression is complex, so we iterate for safety
        # or use simple diffs if 4 layers
        
        residuals = {f'x{L}': np.full(len(df_in), np.nan) for L in available_layers}
        residuals.update({f'y{L}': np.full(len(df_in), np.nan) for L in available_layers})
        
        X_mat = np.column_stack([df_in[f'x{L}'] for L in available_layers])
        Y_mat = np.column_stack([df_in[f'y{L}'] for L in available_layers])
        Z_vec = np.array([z_map[L] for L in available_layers])
        
        # Iterate over tracks
        for i in range(len(df_in)):
            valid = ~np.isnan(X_mat[i])
            if np.sum(valid) < 3: continue # Need at least 3 points for a meaningful residual check
            
            z = Z_vec[valid]; x = X_mat[i][valid]; y = Y_mat[i][valid]
            
            # Fit
            A = np.vstack([z, np.ones(len(z))]).T
            mx, cx = np.linalg.lstsq(A, x, rcond=None)[0]
            my, cy = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Calc residuals for ALL layers (even those included in fit)
            # We predict pos for ALL available layers
            all_z = Z_vec
            x_pred = mx * all_z + cx
            y_pred = my * all_z + cy
            
            for j, L in enumerate(available_layers):
                if not np.isnan(X_mat[i][j]): # Only store if hit exists
                    residuals[f'x{L}'][i] = X_mat[i][j] - x_pred[j]
                    residuals[f'y{L}'][i] = Y_mat[i][j] - y_pred[j]
                    
        return residuals

    res_raw = get_linear_fit_residuals(df_phys)
    res_aln = get_linear_fit_residuals(df_aln)

    if not show_plots: return
    layers_to_plot = [L for L in available_layers if L != ref_layer]
    
    limit = 10
    bins_hist = np.arange(-limit - 0.5, limit + 1.5, 1.0) 
    n_rows = len(layers_to_plot)
    fig2, axes2 = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1: axes2 = np.expand_dims(axes2, 0)
    plt.subplots_adjust(hspace=0.4)

    for i, L in enumerate(layers_to_plot):
        ax = axes2[i, 0]
        # Drop NaNs before plotting
        d_raw = res_raw[f'x{L}']; d_raw = d_raw[~np.isnan(d_raw)]
        d_aln = res_aln[f'x{L}']; d_aln = d_aln[~np.isnan(d_aln)]
        
        sns.histplot(d_raw, bins=bins_hist, ax=ax, color='red', element='step', fill=False, label='Raw', stat='count')
        sns.histplot(d_aln, bins=bins_hist, ax=ax, color='blue', element='step', fill=True, alpha=0.3, label='Aligned', stat='count')
        
        if len(d_aln) > 0:
            mu, sig = np.mean(d_aln), np.std(d_aln)
            ax.text(0.95, 0.95, rf"$\mu$={mu:.2f}, $\sigma$={sig:.2f}", transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f"L{L} X-Residual (Ref L{ref_layer})")
        ax.set_xlim(-limit, limit)
        if i==0: ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes2[i, 1]
        d_raw = res_raw[f'y{L}']; d_raw = d_raw[~np.isnan(d_raw)]
        d_aln = res_aln[f'y{L}']; d_aln = d_aln[~np.isnan(d_aln)]
        
        sns.histplot(d_raw, bins=bins_hist, ax=ax, color='red', element='step', fill=False, stat='count')
        sns.histplot(d_aln, bins=bins_hist, ax=ax, color='blue', element='step', fill=True, alpha=0.3, stat='count')
        
        if len(d_aln) > 0:
            mu, sig = np.mean(d_aln), np.std(d_aln)
            ax.text(0.95, 0.95, rf"$\mu$={mu:.2f}, $\sigma$={sig:.2f}", transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f"L{L} Y-Residual (Ref L{ref_layer})")
        ax.set_xlim(-limit, limit)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Translational Residuals (Target: 0.0)", fontsize=16)
    plt.show()

   
    
tracks = tracking(
    cluster_data = final_clusters,
    alignment_df=df_align,
    target_n_tracks = None,  
    subset_factor = 100,      
    search_radius = 15.0, 
    pitch_ratio_col = 3.0, 
    time_window = 18,        
    jitter_tol = 1,
    max_hits_per_cluster = 5,
    chunk_size = 2000,
    min_hits= 4,           # Min track length (e.g. 3 allows 4-3-2)
    xtalk_filter = 'all'
)
import numpy as np
import copy

def extract_best_unique_data(tracks_dict: dict, clusters_dict: dict):
    """
    Filters for the 'Best Unique Tracks' and their associated Clusters.
    
    Ranking Criteria (Hierarchy):
    1. Completeness: Higher hit count is better.
    2. Straightness: Lower Reduced Chi2 is better.
    3. Timing: Lower standard deviation of timestamps is better.
    4. Perpendicularity: Lower incidence angle (slope magnitude) is better.
    
    Returns:
        best_tracks (dict): Filtered track dictionary.
        best_clusters (dict): Filtered cluster dictionary containing only hits from best tracks.
    """
    print("--- Extracting Best Unique Tracks & Clusters ---")
    
    if not tracks_dict:
        return {}, {}

    n_total = len(tracks_dict['L4_ID'])
    
    # --- 1. Calculate Metrics for Sorting ---
    score_list = []
    
    # Pre-fetch track arrays
    ids_map = {L: np.array(tracks_dict[f'L{L}_ID']) for L in [4, 3, 2, 1]}
    ts_map = {L: np.array(tracks_dict[f't{L}']) for L in [4, 3, 2, 1]}
    chi2_arr = np.array(tracks_dict['chi2'])
    
    # Pre-fetch Coordinate arrays for Slope Calculation
    x_map = {L: np.array(tracks_dict[f'x{L}']) for L in [4, 3, 2, 1]}
    y_map = {L: np.array(tracks_dict[f'y{L}']) for L in [4, 3, 2, 1]}
    
    for i in range(n_total):
        # A. Hit Count
        valid_L = [L for L in [4,3,2,1] if ids_map[L][i] != -1]
        n_hits = len(valid_L)
        
        if n_hits < 3: continue 
            
        # B. Reduced Chi2
        dof = 2 * n_hits - 4
        red_chi2 = chi2_arr[i] / dof if dof > 0 else 0.0
        if np.isnan(red_chi2): red_chi2 = 9999.0
            
        # C. Timing Uniformity
        times = [ts_map[L][i] for L in valid_L]
        t_std = np.std(times) if len(times) > 1 else 9999.0
        
        # D. Perpendicularity (Slope Magnitude)
        top_L, bot_L = valid_L[0], valid_L[-1]
        dz = top_L - bot_L
        
        if dz > 0:
            dx = x_map[top_L][i] - x_map[bot_L][i]
            dy = y_map[top_L][i] - y_map[bot_L][i]
            angle_mag = np.sqrt(dx**2 + dy**2) / dz
        else:
            angle_mag = 9999.0
        
        # Append Score: (-Hits, Red_Chi2, Time_Std, Angle_Mag)
        score_list.append( ((-n_hits, red_chi2, t_std, angle_mag), i) )

    # --- 2. Greedy Filter (Uniqueness) ---
    score_list.sort(key=lambda x: x[0]) 
    
    used_clusters = set()
    kept_track_indices = []
    
    for score, idx in score_list:
        current_clusters = [ids_map[L][idx] for L in [4,3,2,1] if ids_map[L][idx] != -1]
        
        # Check conflicts
        if not any(cid in used_clusters for cid in current_clusters):
            kept_track_indices.append(idx)
            used_clusters.update(current_clusters)
            
    kept_track_indices.sort()
    print(f"  Selected {len(kept_track_indices)} best unique tracks.")

    # --- 3. Construct Output Dictionaries ---
    
    # A. Best Tracks Dictionary
    best_tracks = {}
    idx_arr = np.array(kept_track_indices)
    
    for key, val in tracks_dict.items():
        if isinstance(val, (np.ndarray, list)) and len(val) == n_total:
            best_tracks[key] = np.array(val)[idx_arr]
        else:
            best_tracks[key] = copy.deepcopy(val)

    # B. Best Clusters Dictionary
    best_clusters = {}
    
    # Find indices of clusters that are in 'used_clusters'
    all_cluster_ids = np.array(clusters_dict['clusterID'])
    
    # Create a boolean mask for clusters to keep
    # np.isin is efficient for this
    cluster_mask = np.isin(all_cluster_ids, list(used_clusters))
    
    n_cl_total = len(all_cluster_ids)
    print(f"  Extracted {np.sum(cluster_mask)} associated clusters.")

    for key, val in clusters_dict.items():
        if isinstance(val, (np.ndarray, list)) and len(val) == n_cl_total:
            best_clusters[key] = np.array(val)[cluster_mask]
        else:
            best_clusters[key] = copy.deepcopy(val)

    return best_tracks, best_clusters

# Example Usage:
best_trks_dict, best_clsts_dict = extract_best_unique_data(tracks, final_clusters)


check_alignment_residuals(tracks, alignment_df=df_align, ref_layer=4, pitch_ratio_col=3.0)

import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import pandas as pd 
import seaborn as sns


# ==========================================
# 1. HEATMAP PLOTTING
# ==========================================
def plot_track_heatmaps(tracks_dict: dict, x_lim=None, y_lim=None, cmap='jet'):
    """
    Plots a 2D Heatmap with aligned marginals for each layer using Dictionary Input.
    """
    if not tracks_dict or 'L4_ID' not in tracks_dict or len(tracks_dict['L4_ID']) == 0:
        print("No tracks to plot.")
        return

    print("\n--- Generating Track Heatmaps (High-Res Style) ---")

    MAX_COL = 132
    MAX_ROW = 372

    for L in [4, 3, 2, 1]:
        print(f"  Plotting Layer {L}...")

        # EXTRACT DATA DIRECTLY FROM DICT
        cols = tracks_dict[f'x{L}'].astype(int)
        rows = tracks_dict[f'y{L}'].astype(int)
        tots = tracks_dict[f'tot{L}']

        if len(cols) == 0:
            print(f"    Skipping Layer {L} (No hits)")
            continue

        # Create Grids
        sensor_grid, _, _ = np.histogram2d(
            rows, cols, 
            bins=[MAX_ROW, MAX_COL], 
            range=[[0, MAX_ROW], [0, MAX_COL]], 
            weights=tots
        )

        hit_grid, _, _ = np.histogram2d(
            rows, cols, 
            bins=[MAX_ROW, MAX_COL], 
            range=[[0, MAX_ROW], [0, MAX_COL]]
        )

        sensor_grid[sensor_grid == 0] = np.nan # Mask zeros for better visibility

        # Marginals
        x_tot_sum = np.nansum(sensor_grid, axis=0)
        x_hit_count = np.sum(hit_grid, axis=0)
        y_tot_sum = np.nansum(sensor_grid, axis=1)
        y_hit_count = np.sum(hit_grid, axis=1)

        # Plot Setup
        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(2, 3, 
                               width_ratios=[6, 1.2, 0.2], 
                               height_ratios=[1.2, 6],      
                               wspace=0.08, hspace=0.08)

        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        cax      = fig.add_subplot(gs[1, 2])

        # Colormap
        if isinstance(cmap, str):
            current_cmap = plt.get_cmap(cmap).copy()
        else:
            current_cmap = cmap.copy()
        current_cmap.set_bad(color='white')

        # Main Image
        im = ax_main.imshow(
            sensor_grid, 
            origin='lower', 
            cmap=current_cmap, 
            interpolation='nearest',
            aspect=1/3, 
            extent=[-0.5, MAX_COL - 0.5, -0.5, MAX_ROW - 0.5],
            vmin=0,    
            vmax=255   
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

        # Formatting
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

        fig.suptitle(f'Layer {L} Reconstruction Density (N={len(cols)})', y=0.95, fontsize=16, fontweight='bold')

        ax_main.xaxis.set_major_locator(MultipleLocator(20)) 
        ax_main.yaxis.set_major_locator(MultipleLocator(50))
        ax_main.grid(which='major', color='gray', alpha=0.3, linewidth=1)

        cbar = plt.colorbar(im, cax=cax)
        cax.set_ylabel('Sum ToT', fontsize=14, fontweight='bold')
        cax.tick_params(labelsize=10)

        plt.show()


def plot_track_statistics(tracks_dict: dict):
    if not tracks_dict: return
    tracks_df = pd.DataFrame(tracks_dict)

    # --- Pre-Calculation ---
    tracks_df['dx'] = tracks_df['x4'] - tracks_df['x1']
    tracks_df['dy'] = tracks_df['y4'] - tracks_df['y1']

    # Calculate N-Hits per cluster
    for L in [4, 3, 2, 1]:
        col = f'pToF{L}'
        if f'nhits{L}' in tracks_df.columns: pass 
        elif col in tracks_df.columns:
            tracks_df[f'nhits{L}'] = tracks_df[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1)
        else:
            tracks_df[f'nhits{L}'] = 1

    dt_data = tracks_df[['t3', 't4', 't2', 't1']].copy()
    dt_data['dt_43'] = dt_data['t3'] - dt_data['t4']
    dt_data['dt_32'] = dt_data['t2'] - dt_data['t3']
    dt_data['dt_21'] = dt_data['t1'] - dt_data['t2']

    # --- Chi2 Limit ---
    all_chi2 = tracks_df['chi2']
    limit_high = np.percentile(all_chi2, 98) * 1.2 if len(all_chi2) > 0 else 100
    df_zoom = tracks_df[tracks_df['chi2'] <= limit_high]

    # --- Setup Grid ---
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    def add_minor_grid(ax):
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

    # --- PANEL 1: Chi-Squared ---
    ax1 = fig.add_subplot(gs[0, 0])
    if len(df_zoom) > 0:
        sns.histplot(df_zoom['chi2'], bins=100, kde=True, ax=ax1, color='teal', stat='density')
    ax1.set_title(r"Track Straightness ($\chi^2$)") 
    ax1.set_xlabel(r"$\chi^2$ (Linear Scale)")
    ax1.set_xlim(0, limit_high)
    add_minor_grid(ax1)

    # Inset (Log)
    ax1_ins = ax1.inset_axes([0.55, 0.45, 0.4, 0.4]) 
    sns.histplot(tracks_df['chi2'], bins=100, ax=ax1_ins, color='teal', log_scale=True, element="step", fill=False)
    ax1_ins.set_title("Full Range (Log)", fontsize=9)
    ax1_ins.set_xticks([]); ax1_ins.set_yticks([]) # Clean look
    add_minor_grid(ax1_ins)

    # --- PANEL 2: Energy vs Hits ---
    ax2 = fig.add_subplot(gs[0, 1])
    frames = []
    for L in [4, 3, 2, 1]:
        temp = tracks_df[[f'tot{L}', f'nhits{L}']].copy()
        temp.columns = ['ToT', 'Hits']
        temp['Layer'] = f'Layer {L}'
        frames.append(temp)
    correlation_data = pd.concat(frames)

    sns.boxplot(data=correlation_data, x='Hits', y='ToT', hue='Layer',
                palette="viridis", ax=ax2, showfliers=False, linewidth=1.2)
    ax2.set_title("Energy Deposition vs Cluster Size")
    ax2.set_ylabel("Sum ToT")
    ax2.legend(loc='upper left', title=None)
    add_minor_grid(ax2)

    # --- PANEL 3: Track Angle (ZOOMED) ---
    ax3 = fig.add_subplot(gs[1, 0])

    # Force Zoom Limits
    x_lim = 50
    y_lim = 150

    bins_x = np.arange(-x_lim, x_lim + 1, 1)
    bins_y = np.arange(-y_lim, y_lim + 1, 1)

    h = ax3.hist2d(tracks_df['dx'], tracks_df['dy'], bins=[bins_x, bins_y], cmap='inferno', norm=LogNorm())
    fig.colorbar(h[3], ax=ax3, label='Track Count')
    ax3.set_title(f"Track Angle Distribution (Zoomed)")
    ax3.set_xlabel(r"$\Delta$ Column (L4 - L1)")
    ax3.set_ylabel(r"$\Delta$ Row (L4 - L1)")
    ax3.set_xlim(-x_lim, x_lim)
    ax3.set_ylim(-y_lim, y_lim)
    add_minor_grid(ax3)

    # --- PANEL 4: Timing ---
    ax4 = fig.add_subplot(gs[1, 1])
    dt_melt = dt_data[['dt_43', 'dt_32', 'dt_21']].melt(var_name='Gap', value_name='Time Delta')
    sns.histplot(data=dt_melt, x='Time Delta', hue='Gap', element="step", bins=np.arange(-2.5, 12.5, 1), ax=ax4)
    ax4.set_title("Inter-Layer Time Differences")
    add_minor_grid(ax4)

    plt.suptitle(f"Track Reconstruction Summary (N={len(tracks_df)})", fontsize=16)

    plt.show()

# ==========================================
# 2. CROSSTALK IMPACT (Linear Inset + Stacked Bars)
# ==========================================
def plot_crosstalk_impact(tracks_dict: dict):
    if not tracks_dict: return
    tracks_df = pd.DataFrame(tracks_dict)

    # 1. Identify Dirty Tracks
    mask_dirty = pd.Series(False, index=tracks_df.index)
    for col in ['xtalk4', 'xtalk3', 'xtalk2', 'xtalk1']:
        mask_dirty |= tracks_df[col].astype(str).str.contains(r'1|2')

    df_clean = tracks_df[~mask_dirty]
    df_dirty = tracks_df[mask_dirty]

    # Scale Limit
    all_chi2 = tracks_df['chi2']
    limit_high = np.percentile(all_chi2, 98) * 1.2 if len(all_chi2) > 0 else 100

    clean_zoom = df_clean[df_clean['chi2'] <= limit_high]
    dirty_zoom = df_dirty[df_dirty['chi2'] <= limit_high]

    # Create Figure
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)

    def add_minor_grid(ax):
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

    # --- PANEL 1: Impact on Straightness ---
    ax1 = fig.add_subplot(gs[0])

    if len(clean_zoom) > 0: 
        sns.kdeplot(clean_zoom['chi2'], ax=ax1, color='blue', label='Clean', fill=True, alpha=0.3)
    if len(dirty_zoom) > 0: 
        sns.kdeplot(dirty_zoom['chi2'], ax=ax1, color='red', label='Contaminated', fill=True, alpha=0.3)

    ax1.set_title(r"Impact on Straightness ($\chi^2$)")
    ax1.set_xlabel(r"$\chi^2$ (Linear)")
    ax1.set_xlim(0, limit_high)
    ax1.legend()
    add_minor_grid(ax1)

    # --- INSET: LINEAR Hist + Residuals ---
    ax_ins_main = ax1.inset_axes([0.5, 0.5, 0.45, 0.35]) 

    # Linear bins for inset (Zoomed area)
    bins_lin = np.linspace(0, limit_high, 50)

    # Compute densities (using the zoomed data to match main plot context)
    h_clean, _ = np.histogram(clean_zoom['chi2'], bins=bins_lin, density=True)
    h_dirty, _ = np.histogram(dirty_zoom['chi2'], bins=bins_lin, density=True)
    centers = (bins_lin[:-1] + bins_lin[1:]) / 2

    # Top Inset: Distributions
    ax_ins_main.step(centers, h_clean, where='mid', color='blue', label='Clean')
    ax_ins_main.step(centers, h_dirty, where='mid', color='red', label='Dirty')
    ax_ins_main.set_xticks([]) # Hide x ticks for top plot
    ax_ins_main.set_title("Detail & Residual (Linear)", fontsize=9)
    add_minor_grid(ax_ins_main)

    # Bottom Inset: Residual (Dirty - Clean)
    ax_ins_res = ax1.inset_axes([0.5, 0.35, 0.45, 0.15], sharex=ax_ins_main)
    residual = h_dirty - h_clean
    ax_ins_res.bar(centers, residual, width=np.diff(bins_lin)[0], color='black', alpha=0.5)
    ax_ins_res.axhline(0, color='black', linewidth=0.5)
    ax_ins_res.set_ylabel(r"$\Delta$", fontsize=8)
    ax_ins_res.set_xlabel(r"$\chi^2$", fontsize=8)
    ax_ins_res.tick_params(labelsize=7)
    add_minor_grid(ax_ins_res)

    # --- PANEL 2: Composition (Stacked Bars) ---
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1], hspace=0.3, wspace=0.3)

    if len(df_dirty) > 0:
        for i, L in enumerate([4, 3, 2, 1]):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs_right[row, col])

            # Prepare Data: Hits vs Type
            l_data = pd.DataFrame({
                'Type': df_dirty[f'xtalk{L}'].astype(str),
                'Hits': df_dirty[f'nhits{L}']
            })

            # Aggregate for Stacked Bar
            # We want: Index=Hits, Columns=Type, Values=Count
            ct = pd.crosstab(l_data['Hits'], l_data['Type'])

            # Ensure columns exist even if count is 0
            for t in ['0', '1', '2']:
                if t not in ct.columns: ct[t] = 0

            # Rename for legend
            ct = ct.rename(columns={'0': 'Clean', '1': 'Victim', '2': 'Ambiguous'})
            ct = ct[['Clean', 'Victim', 'Ambiguous']] # Order

            # Plot Stacked Bar
            ct.plot(kind='bar', stacked=True, ax=ax, 
                    color=['gray', 'crimson', 'orange'], width=0.8)

            ax.set_title(f"Layer {L}")
            ax.set_xlabel("Hits/Cluster")
            ax.set_ylabel("Count") if col == 0 else ax.set_ylabel("")
            ax.tick_params(axis='x', rotation=0)

            if i == 1: 
                ax.legend(title=None, fontsize='small')
            else:
                ax.get_legend().remove()

            add_minor_grid(ax)

    plt.suptitle("Crosstalk Impact Analysis", fontsize=16)

    plt.show()

# ==========================================
# 5. SPATIAL RESOLUTION COMPARISON
# ==========================================
def compare_spatial_resolution(tracks_dict: dict, cluster_dict: dict):
    """
    Compares CoG vs Geometric Center using Dict Inputs.
    """
    if not tracks_dict or 'L4_ID' not in tracks_dict: return
    print("\n--- Spatial Resolution Analysis (CoG vs Geometric) ---")
    
    # 1. Prepare Data
    # Look up cluster properties. cluster_dict['clusterID'] is sorted.
    global_ids = cluster_dict['clusterID']
    data = {}
    
    for L in [4, 3, 2, 1]:
        # Existing CoG
        data[f'x{L}'] = tracks_dict[f'x{L}']
        data[f'y{L}'] = tracks_dict[f'y{L}']
        
        # Lookup Geo Center using IDs
        ids = tracks_dict[f'L{L}_ID']
        indices = np.searchsorted(global_ids, ids)
        indices = np.clip(indices, 0, len(global_ids)-1)
        
        cmin = cluster_dict['col_min'][indices]
        cmax = cluster_dict['col_max'][indices]
        rmin = cluster_dict['row_min'][indices]
        rmax = cluster_dict['row_max'][indices]
        
        data[f'geo_x{L}'] = (cmin + cmax) / 2.0
        data[f'geo_y{L}'] = (rmin + rmax) / 2.0

    # 2. Predictions & Residuals
    residuals = {}
    def interp(vu, vl, zu, zl, zt):
        return vl + (vu - vl) * (zt - zl) / (zu - zl)
    def extrap(vm, vf, zm, zf, zt):
        return vm + (vm - vf) * (zt - zm) / (zm - zf)

    # Layer 3 (Interp 4->2)
    pred_x3 = interp(data['x4'], data['x2'], 3, 1, 2)
    pred_y3 = interp(data['y4'], data['y2'], 3, 1, 2)
    residuals[3] = {'cog_x': data['x3']-pred_x3, 'geo_x': data['geo_x3']-pred_x3, 
                    'cog_y': data['y3']-pred_y3, 'geo_y': data['geo_y3']-pred_y3}

    # Layer 2 (Interp 3->1)
    pred_x2 = interp(data['x3'], data['x1'], 2, 0, 1)
    pred_y2 = interp(data['y3'], data['y1'], 2, 0, 1)
    residuals[2] = {'cog_x': data['x2']-pred_x2, 'geo_x': data['geo_x2']-pred_x2, 
                    'cog_y': data['y2']-pred_y2, 'geo_y': data['geo_y2']-pred_y2}
    
    # Layer 1 (Extrap 4->2)
    pred_x1 = extrap(data['x2'], data['x4'], 1, 3, 0)
    pred_y1 = extrap(data['y2'], data['y4'], 1, 3, 0)
    residuals[1] = {'cog_x': data['x1']-pred_x1, 'geo_x': data['geo_x1']-pred_x1, 
                    'cog_y': data['y1']-pred_y1, 'geo_y': data['geo_y1']-pred_y1}
    
    # Layer 4 (Extrap 2->3)
    pred_x4 = extrap(data['x3'], data['x2'], 2, 1, 3)
    pred_y4 = extrap(data['y3'], data['y2'], 2, 1, 3)
    residuals[4] = {'cog_x': data['x4']-pred_x4, 'geo_x': data['geo_x4']-pred_x4, 
                    'cog_y': data['y4']-pred_y4, 'geo_y': data['geo_y4']-pred_y4}
    
    # 3. Plotting
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    plt.subplots_adjust(hspace=0.4)
    summary = []
    
    for idx, L in enumerate([4, 3, 2, 1]):
        res = residuals[L]
        def clean(arr): return arr[np.abs(arr) < 5.0]
        
        # Plot X
        ax_x = axes[idx, 0]
        sns.kdeplot(clean(res['geo_x']), ax=ax_x, color='gray', linestyle='--', label='Geometric', fill=True, alpha=0.1)
        sns.kdeplot(clean(res['cog_x']), ax=ax_x, color='blue', label='CoG', fill=True, alpha=0.2)
        ax_x.set_title(f"Layer {L} X Residuals")
        ax_x.set_xlim(-3, 3)
        ax_x.legend()
        
        # Plot Y
        ax_y = axes[idx, 1]
        sns.kdeplot(clean(res['geo_y']), ax=ax_y, color='gray', linestyle='--', label='Geometric', fill=True, alpha=0.1)
        sns.kdeplot(clean(res['cog_y']), ax=ax_y, color='crimson', label='CoG', fill=True, alpha=0.2)
        ax_y.set_title(f"Layer {L} Y Residuals")
        ax_y.set_xlim(-3, 3)
        ax_y.legend()
        
        summary.append({'Layer': L, 
                        'Col_Geo': np.std(res['geo_x']), 'Col_CoG': np.std(res['cog_x']),
                        'Row_Geo': np.std(res['geo_y']), 'Row_CoG': np.std(res['cog_y'])})
        
    plt.suptitle("Resolution Comparison", fontsize=16)
    plt.show()
    
    print("\n--- Resolution Summary (Sigma in Pixels) ---")
    sdf = pd.DataFrame(summary).set_index('Layer')
    print(sdf.round(3))

# ==========================================
# 6. CLEAN vs MAJOR CROSSTALK
# ==========================================
def compare_clean_vs_major_crosstalk(tracks_dict: dict):
    """
    Compares clean tracks vs those with heavy crosstalk (dict input).
    """
    if not tracks_dict: return
    tracks_df = pd.DataFrame(tracks_dict)
    print("\n--- Clean vs. Major Crosstalk Analysis ---")

    xtalk_cols = ['xtalk4', 'xtalk3', 'xtalk2', 'xtalk1']
    is_layer_dirty = tracks_df[xtalk_cols].apply(lambda col: col.astype(str).str.contains(r'1|2'))
    dirty_count = is_layer_dirty.sum(axis=1)

    df_clean = tracks_df[dirty_count == 0].copy()
    df_major = tracks_df[dirty_count >= 3].copy()

    print(f"Total Tracks: {len(tracks_df)}")
    print(f"  > Pure Clean:       {len(df_clean)} ({len(df_clean)/len(tracks_df):.1%})")
    print(f"  > Major Crosstalk:  {len(df_major)} ({len(df_major)/len(tracks_df):.1%})")

    if len(df_major) < 10: print("Warning: Not enough 'Major Crosstalk' tracks."); return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # 1. Straightness
    ax1 = axes[0, 0]
    sns.kdeplot(df_clean['chi2'], ax=ax1, color='blue', fill=True, alpha=0.3, label='Clean', common_norm=False)
    sns.kdeplot(df_major['chi2'], ax=ax1, color='red', fill=True, alpha=0.3, label='Major Xtalk', common_norm=False)
    ax1.set_title(r"Track Quality ($\chi^2$)")
    ax1.set_xscale('log'); ax1.legend()

    # 2. Energy
    ax2 = axes[0, 1]
    df_clean['mean_tot'] = df_clean[['tot4', 'tot3', 'tot2', 'tot1']].mean(axis=1)
    df_major['mean_tot'] = df_major[['tot4', 'tot3', 'tot2', 'tot1']].mean(axis=1)
    sns.kdeplot(df_clean['mean_tot'], ax=ax2, color='blue', fill=True, alpha=0.3, label='Clean')
    sns.kdeplot(df_major['mean_tot'], ax=ax2, color='red', fill=True, alpha=0.3, label='Major Xtalk')
    ax2.set_title("Energy Signature"); ax2.legend()

    # 3. Spatial Residuals
    ax3 = axes[1, 0]
    def get_resid_l2(df):
        slope = (df['x4'] - df['x1']) / 3.0
        return df['x2'] - (df['x1'] + slope)
    sns.kdeplot(get_resid_l2(df_clean), ax=ax3, color='blue', label='Clean', linewidth=2)
    sns.kdeplot(get_resid_l2(df_major), ax=ax3, color='red', label='Major Xtalk', linewidth=2, linestyle='--')
    ax3.set_title("Spatial Accuracy (L2 Residuals)"); ax3.set_xlim(-5, 5); ax3.legend()

    # 4. Composition
    ax4 = axes[1, 1]
    all_vals = []
    for col in xtalk_cols: all_vals.extend(df_major[col].astype(str).values)
    s = pd.Series(all_vals)
    c = [s.str.contains('0').sum(), s.str.contains('1').sum(), s.str.contains('2').sum()]
    ax4.bar(['Clean', 'Victim', 'Ambiguous'], c, color=['gray', 'crimson', 'orange'])
    ax4.set_title("Composition of Major Xtalk Tracks")

    plt.suptitle("Pure vs. Contaminated Track Comparison", fontsize=16)
    plt.show()

# --- USAGE ---

plot_track_heatmaps(tracks)
plot_track_statistics(tracks)
plot_crosstalk_impact(tracks)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_track_energy_distribution(track_data: dict, title: str = "Track Energy Distribution Analysis"):
    """
    Plots the energy (Total ToT) distribution for four categories derived from a single track dataset:
      1. All Tracks (Gray)
      2. Purely Clean (Blue): All hits are clean (xtalk == 0).
      3. Purely Contaminated (Red): All hits are crosstalk (xtalk != 0).
      4. Mixed (Orange): Contains at least one clean hit AND at least one crosstalk hit.
    """
    if not track_data or len(track_data['chi2']) == 0:
        print("No track data to plot.")
        return

    # --- 1. Calculate Total Energy (Sum ToT) ---
    tots = []
    for L in [4, 3, 2, 1]:
        key = f'tot{L}'
        if key in track_data:
            col = track_data[key].copy()
            # Handle object arrays (lists/None) safely
            if col.dtype == object:
                col[col == None] = 0
            col = col.astype(float)
            col[np.isnan(col)] = 0
            tots.append(col)

    if not tots:
        print("No ToT data found in track dictionary.")
        return

    # Sum across layers -> Energy per track
    total_energy = np.sum(np.column_stack(tots), axis=1)

    # --- 2. Determine Track Categories ---
    # Definitions:
    # - Purely Clean: ALL valid hits are clean.
    # - Purely Contam: ALL valid hits are xtalk.
    # - Mixed: NOT Purely Clean AND NOT Purely Contam.

    n_tracks = len(total_energy)
    mask_clean = np.ones(n_tracks, dtype=bool)   # Start assuming everything is clean
    mask_contam = np.ones(n_tracks, dtype=bool)  # Start assuming everything is contaminated

    # Helper to check if a value is "clean" (0)
    def is_clean_val(x):
        if isinstance(x, (int, float, np.number)): return x == 0
        if isinstance(x, str): return x == '0'
        if isinstance(x, list): return len(x) == 1 and x[0] == 0
        return False 

    # Helper to check if a value is "xtalk" (non-0)
    def is_xtalk_val(x):
        if isinstance(x, (int, float, np.number)): return x != 0
        if isinstance(x, str): return x != '0'
        if isinstance(x, list): return any(v != 0 for v in x)
        return False

    for L in [4, 3, 2, 1]:
        xt_key = f'xtalk{L}'
        id_key = f'L{L}_ID'

        if xt_key in track_data and id_key in track_data:
            xt_col = track_data[xt_key]
            id_col = track_data[id_key]

            for i in range(n_tracks):
                if id_col[i] != -1: # Only check existing hits
                    val = xt_col[i]

                    # If we find a non-clean hit, it can't be purely clean
                    if not is_clean_val(val):
                        mask_clean[i] = False

                    # If we find a non-xtalk (clean) hit, it can't be purely contaminated
                    if not is_xtalk_val(val):
                        mask_contam[i] = False

    # The "Mixed" category is simply whatever is left over
    mask_mixed = (~mask_clean) & (~mask_contam)

    # --- 3. Plotting ---
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Define bins
    max_val = np.percentile(total_energy, 99) if len(total_energy) > 0 else 100
    bins = np.linspace(0, max_val, 60)

    # 1. All Tracks (Gray Background)
    sns.histplot(total_energy, bins=bins, element="step", fill=True, 
                 color="gray", alpha=0.2, label=f"All Tracks (N={n_tracks})", stat='density')

    # 2. Purely Clean (Blue)
    clean_E = total_energy[mask_clean]
    if len(clean_E) > 0:
        sns.histplot(clean_E, bins=bins, element="step", fill=False, 
                     color="#1f77b4", linewidth=2.5, label=f"Purely Clean (N={len(clean_E)})", stat='density')

    # 3. Purely Contaminated (Red)
    contam_E = total_energy[mask_contam]
    if len(contam_E) > 0:
        sns.histplot(contam_E, bins=bins, element="step", fill=False, 
                     color="#d62728", linewidth=2.5, linestyle='--', label=f"Purely Xtalk (N={len(contam_E)})", stat='density')

    # 4. Mixed (Orange)
    mixed_E = total_energy[mask_mixed]
    if len(mixed_E) > 0:
        sns.histplot(mixed_E, bins=bins, element="step", fill=False, 
                     color="#ff7f0e", linewidth=2.5, linestyle=':', label=f"Mixed (N={len(mixed_E)})", stat='density')

    plt.xlabel("Total Track Energy (Sum ToT)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example Usage:
plot_track_energy_distribution(tracks)
