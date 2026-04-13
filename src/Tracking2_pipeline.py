import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import gc
import sys
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from numba import njit
from itertools import combinations
import copy
import warnings


# Use seaborn for nicer histograms if available
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# --- Progress Bar Handling ---
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None, **kwargs):
        if total is None and hasattr(iterable, '__len__'):
            total = len(iterable)
        print(f"{desc}...")
        for i, item in enumerate(iterable):
            if i % 10 == 0 and total:
                sys.stdout.write(f"\rProgress: {i}/{total} chunks")
                sys.stdout.flush()
            yield item
        print(f"\nDone.")

# ==========================================
# 1. HELPER: FAST INDEX MATCHING
# ==========================================
@njit
def _expand_match_indices(u_indices, start_idxs, end_idxs):
    """Expands one-to-many time matches into aligned index arrays."""
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

# ==========================================
# 2. ALIGNMENT LOGIC
# ==========================================
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_residual_peak(residuals, pitch_scale=1.0):
    clean = residuals[np.abs(residuals) < 50 * pitch_scale]
    if len(clean) < 50: return 0.0 
        
    bin_width = 1.0 * pitch_scale
    limit = 30 * pitch_scale
    bins = np.arange(-limit, limit + bin_width, bin_width)
    counts, edges = np.histogram(clean, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    p0_amp = np.max(counts)
    p0_mean = np.median(clean)
    p0_sigma = np.std(clean) if np.std(clean) > 0 else 1.0
    
    try:
        popt, _ = curve_fit(gaussian, centers, counts, p0=[p0_amp, p0_mean, p0_sigma], maxfev=2000)
        peak_val = popt[1]
        if abs(peak_val - p0_mean) > 10 * pitch_scale: return p0_mean
        return peak_val
    except:
        return np.median(clean)

def calculate_translational_misalignment_robust(cluster_dict, ref_layer=4, pitch_ratio_col=3.0, time_window=10, jitter_tol=1):
    print(f"\n[Alignment] Computing ROBUST parameters (Ref: L{ref_layer})...")
    
    # Minimal DF creation to save memory
    df = pd.DataFrame({
        'Layer': cluster_dict['Layer'],
        'PhysRow': cluster_dict['cog_row'], 
        'PhysCol': cluster_dict['cog_col'] * pitch_ratio_col, 
        'TS': cluster_dict['ts_start'].astype(np.int64)
    })
    # Filter inplace
    mask = cluster_dict['clusterID'] != -1
    df = df[mask].sort_values('TS')
    
    # Store arrays directly
    layers_data = {}
    available_layers = sorted(df['Layer'].unique())
    for L in available_layers:
        sub = df[df['Layer'] == L]
        layers_data[L] = {'r': sub['PhysRow'].values, 'c': sub['PhysCol'].values, 't': sub['TS'].values}
    
    del df
    gc.collect()

    layer_order = sorted(available_layers, reverse=True) if ref_layer == max(available_layers) else sorted(available_layers)
    pairs = list(zip(layer_order[:-1], layer_order[1:]))
    results_list = []
    chunk_size = 50000
    
    for (fixed_L, moving_L) in tqdm(pairs, desc="Aligning Layers"):
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
            dx = fit_residual_peak(diff_c, pitch_scale=pitch_ratio_col)
            dy = fit_residual_peak(diff_r, pitch_scale=1.0)
            del matches_r, matches_c, diff_r, diff_c
            
        results_list.append({
            'Layer_to_Align': moving_L, 'Ref_Layer': fixed_L,
            'dx_col': dx, 'dy_row': dy
        })
        gc.collect()

    return pd.DataFrame(results_list)

# ==========================================
# 3. DIAGNOSTICS: ALIGNMENT PLOTS
# ==========================================
def check_alignment_residuals(tracks_dict, alignment_df=None, ref_layer=4, pitch_ratio_col=3.0, show_plots=True):
    if not tracks_dict or not show_plots: return
    print(f"\n[Alignment Check] Generating Residual Plots (Ref: L{ref_layer})...")

    available_layers = [int(k[1]) for k in tracks_dict.keys() if k.startswith('L') and k.endswith('_ID')]
    available_layers = sorted(available_layers, reverse=True)
    
    df_raw = pd.DataFrame()
    for L in available_layers:
        df_raw[f'x{L}'] = tracks_dict[f'x{L}'].astype(float) * pitch_ratio_col
        df_raw[f'y{L}'] = tracks_dict[f'y{L}'].astype(float)
        
    df_aln = df_raw.copy()
    
    if alignment_df is not None and not alignment_df.empty:
        align_lookup = {(r.Ref_Layer, r.Layer_to_Align): r for r in alignment_df.itertuples()}
        corrections = {L: (0.0, 0.0) for L in available_layers}
        
        if ref_layer == max(available_layers):
            cum_dx, cum_dy = 0.0, 0.0
            for i in range(len(available_layers)-1):
                curr, next_l = available_layers[i], available_layers[i+1]
                r = align_lookup.get((curr, next_l))
                dx, dy = (r.dx_col, r.dy_row) if r else (0.0, 0.0)
                cum_dx += dx; cum_dy += dy
                corrections[next_l] = (cum_dx, cum_dy)

        for L, (dx, dy) in corrections.items():
            if L != ref_layer:
                df_aln[f'x{L}'] += dx; df_aln[f'y{L}'] += dy

    def get_residuals(df_in):
        residuals = {f'x{L}': [] for L in available_layers}
        residuals.update({f'y{L}': [] for L in available_layers})
        
        z_map = {L: (max(available_layers) - L) for L in available_layers}
        Z_vec = np.array([z_map[L] for L in available_layers])
        X_mat = df_in[[f'x{L}' for L in available_layers]].values
        Y_mat = df_in[[f'y{L}' for L in available_layers]].values
        
        for i in range(len(df_in)):
            valid = ~np.isnan(X_mat[i])
            if np.sum(valid) < 3: continue 
            
            z = Z_vec[valid]
            x = X_mat[i][valid]
            y = Y_mat[i][valid]
            
            A = np.vstack([z, np.ones(len(z))]).T
            mx, cx = np.linalg.lstsq(A, x, rcond=None)[0]
            my, cy = np.linalg.lstsq(A, y, rcond=None)[0]
            
            for j, L in enumerate(available_layers):
                if not np.isnan(X_mat[i][j]):
                    pred_x = mx * Z_vec[j] + cx
                    pred_y = my * Z_vec[j] + cy
                    residuals[f'x{L}'].append(X_mat[i][j] - pred_x)
                    residuals[f'y{L}'].append(Y_mat[i][j] - pred_y)
        return residuals

    res_raw = get_residuals(df_raw)
    res_aln = get_residuals(df_aln)

    layers_to_plot = [L for L in available_layers if L != ref_layer]
    limit = 15.0
    bins_hist = np.linspace(-limit, limit, 60)
    
    n_rows = len(layers_to_plot)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1: axes = np.expand_dims(axes, 0)
    plt.subplots_adjust(hspace=0.4)

    for i, L in enumerate(layers_to_plot):
        for col_idx, direction in enumerate(['x', 'y']):
            ax = axes[i, col_idx]
            data_raw = np.array(res_raw[f'{direction}{L}'])
            data_aln = np.array(res_aln[f'{direction}{L}'])
            
            if HAS_SNS:
                sns.histplot(data_raw, bins=bins_hist, ax=ax, color='red', element='step', fill=False, label='Raw', stat='density')
                sns.histplot(data_aln, bins=bins_hist, ax=ax, color='blue', element='step', fill=True, alpha=0.3, label='Aligned', stat='density')
            else:
                ax.hist(data_raw, bins=bins_hist, color='red', histtype='step', label='Raw', density=True)
                ax.hist(data_aln, bins=bins_hist, color='blue', alpha=0.3, label='Aligned', density=True)

            if len(data_aln) > 0:
                mu, sig = np.mean(data_aln), np.std(data_aln)
                ax.text(0.95, 0.95, rf"$\mu$={mu:.2f}, $\sigma$={sig:.2f}", transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))

            ax.set_title(f"L{L} {direction.upper()}-Residual (Ref L{ref_layer})")
            ax.set_xlim(-limit, limit)
            if i == 0 and col_idx == 0: ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"Translational Residuals (Zero is perfect)", fontsize=16)
    plt.show()

# ==========================================
# 4. TRACKING LOGIC
# ==========================================
def tracking2(
    cluster_data: dict,
    alignment_df: pd.DataFrame = None, 
    target_n_tracks: int = 2000,  
    subset_factor: int = 100,      
    search_radius: float = 10.0, 
    pitch_ratio_col: float = 3.0, 
    time_window: int = 15,        
    jitter_tol: int = 1,
    max_hits_per_cluster: int = 5,
    chunk_size: int = 20000,
    min_hits: int = 4,             
    xtalk_filter: str = 'all'    
) -> dict:
    
    print(f"--- Track Search (Min Hits: {min_hits}, Xtalk: {xtalk_filter}) ---")
    t0 = time.time()
    
    corrections = {L: {'dr': 0.0, 'dc': 0.0} for L in [4, 3, 2, 1]}
    if alignment_df is not None and not alignment_df.empty:
        align_lookup = {(r.Ref_Layer, r.Layer_to_Align): r for r in alignment_df.itertuples()}
        def get_pars(u, d):
            r = align_lookup.get((u, d))
            return (r.dy_row, r.dx_col) if r else (0.0, 0.0)

        dr43, dc43 = get_pars(4, 3)
        dr32, dc32 = get_pars(3, 2)
        dr21, dc21 = get_pars(2, 1)

        corrections[3] = {'dr': dr43, 'dc': dc43}
        corrections[2] = {'dr': dr43 + dr32, 'dc': dc43 + dc32}
        corrections[1] = {'dr': dr43 + dr32 + dr21, 'dc': dc43 + dc32 + dc21}

    try:
        if max_hits_per_cluster:
            mask = (cluster_data['clusterID'] != -1) & (cluster_data['n_hits'] <= max_hits_per_cluster)
        else: 
            mask = (cluster_data['clusterID'] != -1)
            
        if xtalk_filter != 'all':
            xt_arr = cluster_data['xtalk_type']
            if np.issubdtype(xt_arr.dtype, np.number):
                clean_mask = (xt_arr == 0)
            else:
                is_zero = lambda x: (isinstance(x, (int, float)) and x == 0) or \
                                    (isinstance(x, str) and x == '0') or \
                                    (isinstance(x, list) and len(x) == 1 and x[0] == 0)
                clean_mask = np.vectorize(is_zero)(xt_arr)

            if xtalk_filter == 'clean_only': mask &= clean_mask
            elif xtalk_filter == 'xtalk_only': mask &= (~clean_mask)

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0: return {}

        valid_ts = cluster_data['ts_start'][valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_ts)]
        del valid_ts, mask 
        
    except KeyError as e:
        print(f"Missing Key: {e}")
        return {}

    limit = min(len(sorted_indices), target_n_tracks * subset_factor) if target_n_tracks else len(sorted_indices)
    active_indices = sorted_indices[:limit]

    # --- Pre-calculate Alignment Arrays ---
    global_x = cluster_data['cog_col'] * pitch_ratio_col
    global_y = cluster_data['cog_row'].copy()
    
    layer_arr = cluster_data['Layer']
    for L in [3, 2, 1]:
        l_mask = (layer_arr == L)
        if np.any(l_mask):
            global_x[l_mask] += corrections[L]['dc']
            global_y[l_mask] += corrections[L]['dr']

    # --- Helper: Chunk Processing ---
    def _find_tracks_in_chunk(chunk_idxs):
        c_layer = cluster_data['Layer'][chunk_idxs]
        c_ts    = cluster_data['ts_start'][chunk_idxs]
        c_id    = cluster_data['clusterID'][chunk_idxs]
        phys_x = global_x[chunk_idxs]
        phys_y = global_y[chunk_idxs]

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
                'tree': tree, 
                'ts': c_ts[idxs].astype(np.int64),
                'local_idx': idxs 
            }

        adj_list = {}
        transitions = [(4,3), (4,2), (3,2), (3,1), (2,1), (4,1)]
        
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
                dts = ts_d[match_indices] - t_start
                valid_mask = (dts >= -jitter_tol) & (dts <= time_window)
                
                if np.any(valid_mask):
                    valid_match_idxs = np.array(match_indices)[valid_mask]
                    d_idxs = layers[down]['local_idx'][valid_match_idxs]
                    node = (up, u_idx)
                    if node not in adj_list: adj_list[node] = []
                    adj_list[node].extend([(down, d) for d in d_idxs])

        tracks_found = []
        def dfs(curr_layer, curr_idx, path):
            children = adj_list.get((curr_layer, curr_idx), [])
            if len(path) + (curr_layer - 1) < min_hits: return 
            is_terminal = (curr_layer == 1) or (not children)
            if is_terminal:
                if len(path) >= min_hits:
                    track_entry = [-1, -1, -1, -1]
                    for (L, idx) in path:
                        track_entry[4-L] = c_id[idx]
                    tracks_found.append(track_entry)
                return
            for child_layer, child_idx in children:
                if child_layer < curr_layer:
                    dfs(child_layer, child_idx, path + [(child_layer, child_idx)])

        for L in [4, 3, 2]:
            if layers[L]:
                for i in range(len(layers[L]['local_idx'])):
                    u_idx = layers[L]['local_idx'][i]
                    dfs(L, u_idx, [(L, u_idx)])

        return np.array(tracks_found, dtype=np.int64) if tracks_found else None

    # --- Main Loop ---
    all_tracks_arr = []
    n_chunks = (len(active_indices) + chunk_size - 1) // chunk_size
    
    with tqdm(total=n_chunks, desc="Finding Tracks") as pbar:
        for i in range(0, len(active_indices), chunk_size):
            end = min(i+chunk_size, len(active_indices))
            w_end = min(end + 5000, len(active_indices)) if end < len(active_indices) else end
            res = _find_tracks_in_chunk(active_indices[i:w_end])
            if res is not None and len(res) > 0: all_tracks_arr.append(res)
            pbar.update(1)
            if target_n_tracks and sum(len(x) for x in all_tracks_arr) >= target_n_tracks*1.2: break
    
    if not all_tracks_arr: return {}
    
    print("   Combining track segments...")
    full_stack = np.vstack(all_tracks_arr)
    del all_tracks_arr
    gc.collect() 
    
    final_ids = np.unique(full_stack, axis=0)
    hit_counts = np.sum(final_ids != -1, axis=1)
    final_ids = final_ids[hit_counts >= min_hits]

    # --- Output Construction ---
    print("   Extracting properties...")
    results = {}
    max_id = np.max(cluster_data['clusterID'])
    id_to_idx_map = np.full(max_id + 1, -1, dtype=np.int32)
    valid_mask = cluster_data['clusterID'] != -1
    id_to_idx_map[cluster_data['clusterID'][valid_mask]] = np.where(valid_mask)[0].astype(np.int32)
    
    def extract_col(key, dtype=float, fill_val=np.nan):
        source_arr = cluster_data[key]
        if source_arr.dtype == object:
            dtype = object
            if fill_val is np.nan: fill_val = None
            
        flat_ids = final_ids.ravel()
        valid_mask = flat_ids != -1
        valid_ids = flat_ids[valid_mask]
        source_indices = id_to_idx_map[valid_ids]
        
        out_flat = np.full(flat_ids.shape, fill_val, dtype=dtype)
        out_flat[valid_mask] = source_arr[source_indices]
        return out_flat.reshape(final_ids.shape)

    for k, L in enumerate([4, 3, 2, 1]):
        results[f'L{L}_ID'] = final_ids[:, k]
        results[f'x{L}'] = extract_col('cog_col')[:, k]
        results[f'y{L}'] = extract_col('cog_row')[:, k]
        results[f't{L}'] = extract_col('ts_start', dtype=np.float64)[:, k]
        results[f'xtalk{L}'] = extract_col('xtalk_type', dtype=np.float64, fill_val=-1)[:, k] 

    del id_to_idx_map
    gc.collect()

    # Chi2 Calculation
    px_stack = np.column_stack([results[f'x{L}'] * pitch_ratio_col for L in [4,3,2,1]])
    py_stack = np.column_stack([results[f'y{L}'] for L in [4,3,2,1]])
    
    px_stack[:, 1] += corrections[3]['dc']; py_stack[:, 1] += corrections[3]['dr']
    px_stack[:, 2] += corrections[2]['dc']; py_stack[:, 2] += corrections[2]['dr']
    px_stack[:, 3] += corrections[1]['dc']; py_stack[:, 3] += corrections[1]['dr']
    
    dz = np.array([-1.5, -0.5, 0.5, 1.5])
    mask = (final_ids != -1)
    
    @njit(fastmath=True)
    def fast_chi2_loop(x_stack, y_stack, mask, z_pos):
        n = x_stack.shape[0]
        chi2_out = np.empty(n, dtype=np.float32)
        for i in range(n):
            n_hits = 0
            for j in range(4):
                if mask[i, j]: n_hits += 1
            if n_hits < 2:
                chi2_out[i] = 999.0
                continue
            sx, sy, sz, sz2, sxz, syz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for j in range(4):
                if mask[i, j]:
                    z = z_pos[j]
                    x = x_stack[i, j]; y = y_stack[i, j]
                    sx += x; sy += y; sz += z; sz2 += z*z; sxz += x*z; syz += y*z
            denom = n_hits * sz2 - sz * sz
            if abs(denom) < 1e-9:
                chi2_out[i] = 999.0
                continue
            mx = (n_hits * sxz - sx * sz) / denom
            cx = (sx * sz2 - sxz * sz) / denom
            my = (n_hits * syz - sy * sz) / denom
            cy = (sy * sz2 - syz * sz) / denom
            r_sq = 0.0
            for j in range(4):
                if mask[i, j]:
                    z = z_pos[j]
                    pred_x = mx * z + cx
                    pred_y = my * z + cy
                    r_sq += (x_stack[i, j] - pred_x)**2 + (y_stack[i, j] - pred_y)**2
            chi2_out[i] = r_sq
        return chi2_out

    results['chi2'] = fast_chi2_loop(px_stack, py_stack, mask, dz)
    
    sort_idx = np.argsort(results['chi2'])
    if target_n_tracks: sort_idx = sort_idx[:target_n_tracks]
    
    final_dict = {k: v[sort_idx] for k, v in results.items()}
    print(f"--- Finished. Found {len(final_dict['chi2'])} tracks ({time.time()-t0:.2f}s) ---")
    return final_dict

# ==========================================
# 5. DATA PACKAGING (MODIFIED)
# ==========================================
def _map_track_chi2_to_clusters(cluster_data, tracks_dict):
    """Internal helper to map Track Chi2 onto the Clusters."""
    # 1. Build Mapping: ClusterID -> TrackChi2
    # Since a cluster can technically belong to multiple tracks in 'all_tracks', 
    # we take the best (lowest) chi2 if there's a conflict, or just the last one encountered.
    # In 'best_tracks', mapping is 1-to-1.
    
    print("   Mapping Track Chi2 to Clusters...")
    
    # Create a mapping dictionary
    cid_to_chi2 = {}
    
    # We iterate over the tracks_dict
    chi2s = tracks_dict['chi2']
    
    for L in [4, 3, 2, 1]:
        ids = tracks_dict[f'L{L}_ID']
        # Boolean mask for valid IDs
        valid = ids != -1
        
        # Loop over valid IDs and assign Chi2
        # Note: In pure python this is slow for 1M+ tracks.
        # But 'ids' and 'chi2s' are numpy arrays, so we can zip them.
        for cid, c2 in zip(ids[valid], chi2s[valid]):
            # If cluster already visited, keep the lower chi2 (better fit)
            if cid in cid_to_chi2:
                if c2 < cid_to_chi2[cid]:
                    cid_to_chi2[cid] = c2
            else:
                cid_to_chi2[cid] = c2
                
    c_ids = cluster_data['clusterID']
    
    # Map using pandas
    s_ids = pd.Series(c_ids)
    mapped_chi2 = s_ids.map(cid_to_chi2).fillna(-1.0).values
    
    cluster_data['track_chi2'] = mapped_chi2
    return cluster_data

def subset_clusters_by_ids(cluster_data: dict, valid_ids: np.ndarray) -> dict:
    """Returns a new cluster dictionary containing only the clusters with IDs in valid_ids."""
    if len(valid_ids) == 0: return {}
    
    print("   Subsetting clusters (indexing)...")
    all_cluster_ids = np.array(cluster_data['clusterID'])
    
    mask = np.isin(all_cluster_ids, valid_ids)
    
    new_data = {}
    n_total = len(all_cluster_ids)
    
    for k, v in cluster_data.items():
        if isinstance(v, (np.ndarray, list)) and len(v) == n_total:
            new_data[k] = np.array(v)[mask]
        else:
            new_data[k] = copy.copy(v) 
            
    return new_data

def extract_best_unique_data(tracks_dict: dict, clusters_dict: dict):
    print("--- Extracting Best Unique Tracks ---")
    if not tracks_dict: return {}, {}

    n_total = len(tracks_dict['L4_ID'])
    
    valid_mask = (tracks_dict['L4_ID'] != -1).astype(int) + \
                 (tracks_dict['L3_ID'] != -1).astype(int) + \
                 (tracks_dict['L2_ID'] != -1).astype(int) + \
                 (tracks_dict['L1_ID'] != -1).astype(int)
    
    chi2 = tracks_dict['chi2']
    dof = 2 * valid_mask - 4
    dof[dof <= 0] = 1
    red_chi2 = chi2 / dof
    
    dtype = [('hits', 'i4'), ('chi2', 'f4'), ('index', 'i4')]
    metrics = np.zeros(n_total, dtype=dtype)
    metrics['hits'] = -valid_mask 
    metrics['chi2'] = red_chi2
    metrics['index'] = np.arange(n_total)
    
    sorted_metrics = np.sort(metrics, order=['hits', 'chi2'])
    
    used_clusters = set()
    kept_indices = []
    
    l4 = tracks_dict['L4_ID']; l3 = tracks_dict['L3_ID']
    l2 = tracks_dict['L2_ID']; l1 = tracks_dict['L1_ID']
    
    for i in tqdm(range(n_total), desc="Filtering Unique"):
        idx = sorted_metrics[i]['index']
        cids = []
        if l4[idx] != -1: cids.append(l4[idx])
        if l3[idx] != -1: cids.append(l3[idx])
        if l2[idx] != -1: cids.append(l2[idx])
        if l1[idx] != -1: cids.append(l1[idx])
        
        collision = False
        for c in cids:
            if c in used_clusters:
                collision = True; break
        
        if not collision:
            kept_indices.append(idx)
            used_clusters.update(cids)

    kept_indices.sort()
    print(f"   Selected {len(kept_indices)} unique tracks.")
    
    idx_arr = np.array(kept_indices)
    best_tracks = {}
    for k, v in tracks_dict.items():
        if isinstance(v, (np.ndarray, list)) and len(v) == n_total:
            best_tracks[k] = np.array(v)[idx_arr]
        else:
            best_tracks[k] = v

    best_clusters = subset_clusters_by_ids(clusters_dict, list(used_clusters))
    
    # *** ADDED: Map Track Chi2 to Clusters ***
    best_clusters = _map_track_chi2_to_clusters(best_clusters, best_tracks)
    
    return best_tracks, best_clusters

# ==========================================
# 6. PIPELINE RUNNER (UPDATED)
# ==========================================
def run_full_tracking_pipeline2(
    cluster_data: dict,
    target_n_tracks: int = None,
    ref_layer: int = 4,
    pitch_ratio_col: float = 3.0,
    search_radius: float = 15.0,
    time_window: int = 18,
    min_hits: int = 4,
    xtalk_filter: str = 'all'
):
    print("=== STARTING MEMORY-OPTIMIZED PIPELINE ===")
    gc.collect()

    # 1. Alignment
    df_align = calculate_translational_misalignment_robust(
        cluster_data, ref_layer=ref_layer, pitch_ratio_col=pitch_ratio_col
    )
    gc.collect()

    # 2. Tracking
    all_tracks = tracking2(
        cluster_data=cluster_data,
        target_n_tracks = target_n_tracks,
        alignment_df=df_align,
        search_radius=search_radius,
        pitch_ratio_col=pitch_ratio_col,
        time_window=time_window,
        min_hits=min_hits,
        xtalk_filter=xtalk_filter
    )
    gc.collect()

    if not all_tracks:
        return {}, {}, {}, {}

    # 3. Diagnostics 
    check_alignment_residuals(
        all_tracks, alignment_df=df_align, ref_layer=ref_layer,
        pitch_ratio_col=pitch_ratio_col, show_plots=True
    )

    # 4. Extract Clusters for ALL Tracks
    print("--- Extracting Clusters for All Tracks ---")
    all_used_ids = np.unique(np.concatenate([
        all_tracks[f'L{L}_ID'][all_tracks[f'L{L}_ID'] != -1] for L in [1,2,3,4]
    ]))
    all_track_clusters = subset_clusters_by_ids(cluster_data, all_used_ids)
    
    # *** ADDED: Map Track Chi2 to ALL Clusters ***
    all_track_clusters = _map_track_chi2_to_clusters(all_track_clusters, all_tracks)
    
    gc.collect()

    # 5. Extract Best Tracks & Clusters (Function now handles mapping internally)
    best_tracks, best_track_clusters = extract_best_unique_data(all_tracks, cluster_data)
    gc.collect()

    print("=== PIPELINE FINISHED ===")
    return all_tracks, all_track_clusters, best_tracks, best_track_clusters

# Example call
all_trks, all_clsts, best_trks, best_clsts = run_full_tracking_pipeline2(
    final_clusters,
    target_n_tracks = None,
    ref_layer=4,
    pitch_ratio_col=3.0,
    search_radius=300,
    time_window=4,
    min_hits=4,
    xtalk_filter='all'
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# ==========================================
# 0. HELPER: DATA ENRICHMENT
# ==========================================
def _enrich_track_df(tracks_dict: dict, cluster_dict: dict = None):
    """
    Creates a DataFrame from tracks_dict and ensures 'tot{L}' and 'nhits{L}' 
    exist by looking them up in cluster_dict if missing.
    """
    df = pd.DataFrame(tracks_dict)

    # If we don't have cluster data to look up, return what we have
    if cluster_dict is None: 
        return df

    # Create a lookup map for cluster properties
    # Mapping: ClusterID -> Index in cluster_dict arrays
    max_id = np.max(cluster_dict['clusterID'])
    id_map = np.full(max_id + 1, -1, dtype=int)
    id_map[cluster_dict['clusterID']] = np.arange(len(cluster_dict['clusterID']))

    for L in [4, 3, 2, 1]:
        # 1. Enrich ToT (Time over Threshold)
        tot_col = f'tot{L}'
        if tot_col not in df.columns:
            # Look up sum_ToT from cluster_dict
            ids = df[f'L{L}_ID'].values
            valid = ids != -1

            # Initialize with 0.0
            tots = np.zeros(len(df), dtype=float)

            # Get indices for valid IDs
            # Clip to avoid index errors on -1, then mask later
            lookup_idxs = id_map[np.clip(ids, 0, max_id)]

            # Where valid ID exists and is found in map
            valid_map = valid & (lookup_idxs != -1)
            tots[valid_map] = cluster_dict['sum_ToT'][lookup_idxs[valid_map]]
            df[tot_col] = tots

        # 2. Enrich Hit Counts (from pToF or n_hits)
        nhit_col = f'nhits{L}'
        if nhit_col not in df.columns:
            ids = df[f'L{L}_ID'].values
            valid = ids != -1

            counts = np.ones(len(df), dtype=int) # Default to 1 hit
            lookup_idxs = id_map[np.clip(ids, 0, max_id)]
            valid_map = valid & (lookup_idxs != -1)

            if 'n_hits' in cluster_dict:
                 counts[valid_map] = cluster_dict['n_hits'][lookup_idxs[valid_map]]

            df[nhit_col] = counts

    return df

# ==========================================
# 1. HEATMAP PLOTTING
# ==========================================
def plot_track_heatmaps(tracks_dict: dict, cluster_dict: dict = None, x_lim=None, y_lim=None, cmap='jet', clims= (0,700)):
    """
    Plots a 2D Heatmap. Will fetch ToT from cluster_dict if missing in tracks.
    """
    if not tracks_dict or 'L4_ID' not in tracks_dict:
        print("No tracks to plot.")
        return

    print("\n--- Generating Track Heatmaps ---")

    # Enrich data to ensure we have 'tot'
    df = _enrich_track_df(tracks_dict, cluster_dict)

    MAX_COL = 132
    MAX_ROW = 372

    for L in [4, 3, 2, 1]:
        print(f"  Plotting Layer {L}...")

        cols = df[f'x{L}'].fillna(-1).astype(int).values
        rows = df[f'y{L}'].fillna(-1).astype(int).values

        # Use ToT if available, otherwise weight=1
        if f'tot{L}' in df.columns:
            tots = df[f'tot{L}'].fillna(0).values
        else:
            tots = np.ones(len(df))

        # Filter invalid hits
        mask = (cols >= 0) & (rows >= 0)
        cols, rows, tots = cols[mask], rows[mask], tots[mask]

        if len(cols) == 0: continue

        # Create Grids
        sensor_grid, _, _ = np.histogram2d(
            rows, cols, bins=[MAX_ROW, MAX_COL], 
            range=[[0, MAX_ROW], [0, MAX_COL]], weights=tots
        )
        hit_grid, _, _ = np.histogram2d(
            rows, cols, bins=[MAX_ROW, MAX_COL], 
            range=[[0, MAX_ROW], [0, MAX_COL]]
        )

        sensor_grid[sensor_grid == 0] = np.nan 

        # Marginals
        x_tot = np.nansum(sensor_grid, axis=0); x_hits = np.sum(hit_grid, axis=0)
        y_tot = np.nansum(sensor_grid, axis=1); y_hits = np.sum(hit_grid, axis=1)

        # Plot Setup
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(2, 3, width_ratios=[6, 1.2, 0.2], height_ratios=[1.2, 6], wspace=0.08, hspace=0.08)

        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        cax      = fig.add_subplot(gs[1, 2])

        im = ax_main.imshow(
            sensor_grid, origin='lower', cmap=cmap, interpolation='nearest',
            aspect=1/3, extent=[-0.5, MAX_COL - 0.5, -0.5, MAX_ROW - 0.5],
            vmin=clims[0], vmax=clims[-1]
        )

        # Top Marginal
        ax_top.fill_between(np.arange(MAX_COL), x_tot, step='mid', color='gray', alpha=0.4)
        ax_top_hits = ax_top.twinx()
        ax_top_hits.step(np.arange(MAX_COL), x_hits, where='mid', color='black', linewidth=1)
        ax_top.set_ylabel('ToT', color='gray'); ax_top.tick_params(labelbottom=False)

        # Right Marginal
        ax_right.fill_betweenx(np.arange(MAX_ROW), y_tot, step='mid', color='gray', alpha=0.4)
        ax_right.tick_params(labelleft=False)
        ax_right_hits = ax_right.twiny()
        ax_right_hits.step(y_hits, np.arange(MAX_ROW), where='mid', color='black', linewidth=1)
        ax_right_hits.set_xlabel('Hits')

        ax_main.set_xlabel('Column'); ax_main.set_ylabel('Row')
        ax_main.grid(alpha=0.3)
        plt.colorbar(im, cax=cax, label='Sum ToT')
        fig.suptitle(f'Layer {L} Heatmap (N={len(cols)})', y=0.95)
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

# ==========================================
# 1. LANDAU / MOYAL DISTRIBUTION DEFINITION
# ==========================================
def landau_pdf(x, mpv, eta, amp):
    """
    Approximation of Landau distribution using Moyal distribution.
    """
    # Standard Moyal form: f(x) = (1/sqrt(2*pi)) * exp( -0.5 * (L + exp(-L)) )
    # where L = (x - mpv) / eta
    if eta == 0:
        return np.zeros_like(x)
    L = (x - mpv) / eta
    term = -0.5 * (L + np.exp(-L))
    return (amp / np.sqrt(2 * np.pi)) * np.exp(term)

# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================
def plot_track_statistics(tracks_dict: dict, cluster_dict: dict = None):
    if not tracks_dict: return
    
    # 1. Enrich Data (Assuming _enrich_track_df is defined elsewhere)
    # If testing without the helper, you can comment this out and pass a DataFrame directly
    df = _enrich_track_df(tracks_dict, cluster_dict)

    # 2. Calculate Variables
    df['dx'] = df['x4'] - df['x1']
    df['dy'] = df['y4'] - df['y1']
    
    # Time Differences
    dt_cols = []
    for (t_start, t_end, name) in [('t4','t3','dt_43'), ('t3','t2','dt_32'), ('t2','t1','dt_21')]:
        if t_start in df and t_end in df:
            df[name] = df[t_end] - df[t_start]
            dt_cols.append(name)

    # ---------------------------------------------------------
    # FIGURE 1: TRACK GEOMETRY & TIMING
    # ---------------------------------------------------------
    fig1 = plt.figure(figsize=(16, 12))
    
    # Updated Layout: Chi2 on top (spanning), Angle and Time split on bottom
    # height_ratios=[1.2, 1] gives the main graph slightly more vertical space
    gs1 = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
    
    # --- PANEL 1.1: Track Straightness (Chi2) - MAIN GRAPH ---
    ax1 = fig1.add_subplot(gs1[0, :]) # Spans both columns
    
    # Fixed Limits and Bins (Centered on Integers)
    chi2_min, chi2_max = 0, 20
    # Bins from -0.5 to 20.5 with step 1 ensure integers fall in the center
    bins_chi2 = np.arange(chi2_min - 0.5, chi2_max + 1.5, 1)
    
    # Filter data for fit range
    data_fit = df[(df['chi2'] >= chi2_min) & (df['chi2'] <= chi2_max)]['chi2']
    
    # Histogram (Step Graph)
    counts, edges, _ = ax1.hist(data_fit, bins=bins_chi2, 
                                histtype='step', linewidth=2,
                                color='teal', label='Data')
    
    # Calculate centers for fitting (should be 0.0, 1.0, 2.0 ...)
    centers = (edges[:-1] + edges[1:]) / 2

    # Fit Landau
    try:
        # Initial guess: Peak position, Width=1, Max Count
        p0 = [centers[np.argmax(counts)], 1.0, np.max(counts)]
        popt, pcov = curve_fit(landau_pdf, centers, counts, p0=p0, maxfev=5000)
        
        # Plot smooth fit line
        x_fit = np.linspace(chi2_min, chi2_max, 200)
        y_fit = landau_pdf(x_fit, *popt)
        ax1.plot(x_fit, y_fit, 'r-', lw=2, label=f'Landau Fit\nMPV={popt[0]:.2f}\n$\eta$={popt[1]:.2f}')
    except Exception as e:
        print(f"Fit failed: {e}")
    
    ax1.set_title(r"Track Straightness ($\chi^2$)")
    ax1.set_xlabel(r"$\chi^2$")
    ax1.set_xlim(chi2_min, chi2_max)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MultipleLocator(1)) # Tick every integer
    
    # Inset: Full Range Log Scale
    ax1_ins = ax1.inset_axes([0.8, 0.4, 0.18, 0.35]) # Moved to right side
    ax1_ins.hist(df['chi2'], bins=100, color='gray', log=True, histtype='step')
    ax1_ins.set_title("Full Range (Log)", fontsize=8)
    ax1_ins.tick_params(labelsize=8)

    # --- PANEL 1.2: Track Angle (2D Hist) ---
    ax2 = fig1.add_subplot(gs1[1, 0])
    
    max_dx = max(np.abs(df['dx'].max()), 10) if not df['dx'].empty else 10
    max_dy = max(np.abs(df['dy'].max()), 10) if not df['dy'].empty else 10
    
    bins_x = np.arange(-max_dx - 0.5, max_dx + 1.5, 1)
    bins_y = np.arange(-max_dy - 0.5, max_dy + 1.5, 1)
    
    h = ax2.hist2d(df['dx'], df['dy'], bins=[bins_x, bins_y], 
                   cmap='inferno', norm=LogNorm())
    fig1.colorbar(h[3], ax=ax2, label='Count')
    
    ax2.set_title("Track Angle (L4 - L1)")
    ax2.set_xlabel(r"$\Delta$ Column (pixels)")
    ax2.set_ylabel(r"$\Delta$ Row (pixels)")
    ax2.grid(True, alpha=0.2)

    # --- PANEL 1.3: Time Differences (Separated) ---
    # Stacked vertically in the bottom-right quadrant
    gs_time = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs1[1, 1], hspace=0.4)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    if dt_cols:
        t_min = np.floor(df[dt_cols].min().min()) if not df[dt_cols].empty else -5
        t_max = np.ceil(df[dt_cols].max().max()) if not df[dt_cols].empty else 5
        bins_time = np.arange(t_min - 0.5, t_max + 1.5, 1)

        for i, col in enumerate(['dt_43', 'dt_32', 'dt_21']):
            ax_t = fig1.add_subplot(gs_time[i, 0])
            if col in df:
                # Plot
                ax_t.hist(df[col].dropna(), bins=bins_time, color=colors[i], alpha=0.7, 
                         edgecolor='black', histtype='stepfilled')
                
                # Stats text
                mu, sig = df[col].mean(), df[col].std()
                ax_t.text(0.98, 0.9, f"$\mu$={mu:.2f}, $\sigma$={sig:.2f}", 
                          transform=ax_t.transAxes, ha='right', va='top', 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
                
            ax_t.set_title(f"Time Delta: {col}", fontsize=10, pad=3)
            ax_t.grid(True, alpha=0.3)
            ax_t.xaxis.set_major_locator(MultipleLocator(1))
            
            # Only label bottom axis
            if i < 2:
                ax_t.set_xticklabels([])
            else:
                ax_t.set_xlabel("$\Delta$TS (clock cycles)")

    fig1.suptitle(f"Track Geometry & Timing (N={len(df)})", fontsize=16)
    plt.show()

    # ---------------------------------------------------------
    # FIGURE 2: CLUSTER ENERGETICS
    # ---------------------------------------------------------
    fig2 = plt.figure(figsize=(12, 6))
    # Corrected GridSpec: 1 row, 2 columns for side-by-side plots
    gs2 = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # --- PANEL 2.1: Avg Energy per Pixel (Overlayed) ---
    ax_en = fig2.add_subplot(gs2[0, 0])
    
    bins_energy = np.arange(-0.5, 256.5, 2) # 0-255 integer centered
    layer_colors = {4: '#d62728', 3: '#2ca02c', 2: '#ff7f0e', 1: '#1f77b4'}
    
    for L in [4, 3, 2, 1]:
        if f'tot{L}' in df and f'nhits{L}' in df:
            # Calculate Avg Energy/Pixel for this layer
            hits = df[f'nhits{L}'].replace(0, 1)
            avg_tot = df[f'tot{L}'] / hits
            
            # Plot Step Histogram
            ax_en.hist(avg_tot, bins=bins_energy, histtype='step', 
                       color=layer_colors[L], linewidth=2, label=f'Layer {L}',
                       density=True) # Normalized for comparison
    
    ax_en.set_title("Average Energy Deposited per Pixel (Normalized)")
    ax_en.set_xlabel("Average ToT (ADC)")
    ax_en.set_ylabel("Density")
    ax_en.set_xlim(0, 255)
    ax_en.legend()
    ax_en.grid(True, alpha=0.3)

    # --- PANEL 2.2: Energy vs Cluster Size (Box Plot) ---
    ax_box = fig2.add_subplot(gs2[0, 1])
    
    frames = []
    for L in [4, 3, 2, 1]:
        if f'tot{L}' in df and f'nhits{L}' in df:
            temp = df[[f'tot{L}', f'nhits{L}']].copy()
            temp.columns = ['ToT', 'Hits']
            temp['Layer'] = f'Layer {L}'
            frames.append(temp)
    
    if frames:
        corr_data = pd.concat(frames)
        sns.boxplot(data=corr_data, x='Hits', y='ToT', hue='Layer', 
                    palette=layer_colors.values(), ax=ax_box, showfliers=False,
                    boxprops=dict(alpha=.7)) 
        
        ax_box.set_title("Cluster Energy (Sum ToT) vs Size")
        ax_box.grid(True, alpha=0.3, axis='y')
    
    fig2.suptitle(f"Cluster Energetics", fontsize=16)
    plt.show()

import numpy as np
import pandas as pd
import time
import sys
import gc
import copy
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from numba import njit

# ==========================================
# 1. NUMBA KERNELS (Must be defined first)
# ==========================================
@njit(cache=True)
def _build_adjacency_csr(n_nodes, u_indices, d_indices):
    """Builds Compressed Sparse Row adjacency."""
    counts = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(len(u_indices)):
        counts[u_indices[i]] += 1
    
    offsets = np.zeros(n_nodes + 2, dtype=np.int32)
    acc = 0
    for i in range(n_nodes + 1):
        offsets[i] = acc
        acc += counts[i]
    offsets[-1] = acc

    targets = np.zeros(len(u_indices), dtype=np.int32)
    current_ptr = offsets.copy()
    
    for i in range(len(u_indices)):
        u, d = u_indices[i], d_indices[i]
        pos = current_ptr[u]
        targets[pos] = d
        current_ptr[u] += 1
        
    return offsets, targets

@njit(cache=True)
def _dfs_iterative(offsets, targets, seeds, c_ids, layers, min_hits):
    """Fast iterative DFS for track finding."""
    # Dummy init to fix Numba typing
    dummy = np.zeros(4, dtype=np.int64)
    found = [dummy]
    found.pop()
    
    path = np.full(4, -1, dtype=np.int32)
    for seed in seeds:
        _dfs_recurse(seed, 4, offsets, targets, c_ids, layers, min_hits, path, found)
        
    if len(found) == 0: return np.zeros((0, 4), dtype=np.int64)
    
    out = np.empty((len(found), 4), dtype=np.int64)
    for i in range(len(found)):
        out[i] = found[i]
    return out

@njit(cache=True)
def _dfs_recurse(u_idx, u_layer, offsets, targets, c_ids, layers, min_hits, path, found):
    path[4 - u_layer] = u_idx # Map L4->0
    
    start = offsets[u_idx]
    end = offsets[u_idx+1]
    
    # Terminal node if Layer 1 or no children
    if u_layer == 1 or start == end:
        hits = 0
        for k in range(4): 
            if path[k] != -1: hits += 1
            
        if hits >= min_hits:
            t = np.full(4, -1, dtype=np.int64)
            for k in range(4):
                if path[k] != -1: t[k] = c_ids[path[k]]
            found.append(t)
    
    # Recurse
    for i in range(start, end):
        v_idx = targets[i]
        v_layer = layers[v_idx]
        if v_layer < u_layer:
            _dfs_recurse(v_idx, v_layer, offsets, targets, c_ids, layers, min_hits, path, found)
            
    # Backtrack
    path[4 - u_layer] = -1

@njit(fastmath=True)
def fast_chi2_loop(x_stack, y_stack, mask, z_pos):
    n = x_stack.shape[0]
    chi2_out = np.empty(n, dtype=np.float32)
    for i in range(n):
        n_hits = 0
        for j in range(4):
            if mask[i, j]: n_hits += 1
        
        if n_hits < 2:
            chi2_out[i] = 999.0; continue
            
        sx, sy, sz, sz2, sxz, syz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for j in range(4):
            if mask[i, j]:
                z = z_pos[j]; x = x_stack[i, j]; y = y_stack[i, j]
                sx += x; sy += y; sz += z; sz2 += z*z; sxz += x*z; syz += y*z
        
        denom = n_hits * sz2 - sz * sz
        if abs(denom) < 1e-9: chi2_out[i] = 999.0; continue
            
        mx = (n_hits * sxz - sx * sz) / denom
        cx = (sx * sz2 - sxz * sz) / denom
        my = (n_hits * syz - sy * sz) / denom
        cy = (sy * sz2 - syz * sz) / denom
        
        r_sq = 0.0
        for j in range(4):
            if mask[i, j]:
                z = z_pos[j]
                r_sq += (x_stack[i, j] - (mx*z + cx))**2 + (y_stack[i, j] - (my*z + cy))**2
        chi2_out[i] = r_sq
    return chi2_out

# ==========================================
# 2. TRACKING FUNCTION (Optimized Chunk Size)
# ==========================================
def tracking_fast(
    cluster_data, alignment_df=None, target_n_tracks=None, 
    search_radius=15.0, pitch_ratio_col=3.0, time_window=18, min_hits=4
):
    print(f"--- Fast Tracking (MinHits={min_hits}) ---")
    t0 = time.time()
    
    # 1. Filter valid hits
    mask = (cluster_data['clusterID'] != -1)
    valid_indices = np.where(mask)[0]
    
    # Sort by time
    valid_ts = cluster_data['ts_start'][valid_indices]
    sorted_sub = np.argsort(valid_ts)
    active_indices = valid_indices[sorted_sub]
    
    if target_n_tracks: active_indices = active_indices[:target_n_tracks*100] # Over-sample slightly

    # 2. Prepare Data Arrays
    phys_x = cluster_data['cog_col'][active_indices] * pitch_ratio_col
    phys_y = cluster_data['cog_row'][active_indices].copy()
    layers = cluster_data['Layer'][active_indices].astype(np.int32)
    ts     = cluster_data['ts_start'][active_indices].astype(np.int64)
    c_ids  = cluster_data['clusterID'][active_indices]

    # Apply Alignment
    corrections = {L: {'dr': 0.0, 'dc': 0.0} for L in [4, 3, 2, 1]}
    if alignment_df is not None and not alignment_df.empty:
        align_lookup = {(r.Ref_Layer, r.Layer_to_Align): r for r in alignment_df.itertuples()}
        def get_pars(u, d):
            r = align_lookup.get((u, d))
            return (r.dy_row, r.dx_col) if r else (0.0, 0.0)
        dr43, dc43 = get_pars(4, 3); dr32, dc32 = get_pars(3, 2); dr21, dc21 = get_pars(2, 1)
        corrections[3] = {'dr': dr43, 'dc': dc43}
        corrections[2] = {'dr': dr43+dr32, 'dc': dc43+dc32}
        corrections[1] = {'dr': dr43+dr32+dr21, 'dc': dc43+dc32+dc21}
    
    for L in [3, 2, 1]:
        mask_l = (layers == L)
        if np.any(mask_l):
            phys_x[mask_l] += corrections[L]['dc']
            phys_y[mask_l] += corrections[L]['dr']

    # 3. Chunked Graph Search
    # REDUCED CHUNK SIZE TO PREVENT HANGS
    chunk_size = 2000 
    all_tracks_list = []
    total_hits = len(active_indices)
    
    print(f"Processing {total_hits} hits in chunks of {chunk_size}...")
    
    for start_i in range(0, total_hits, chunk_size):
        # Progress indicator every 10 chunks
        if (start_i // chunk_size) % 10 == 0:
            sys.stdout.write(f"\r  -> Progress: {start_i}/{total_hits} hits")
            sys.stdout.flush()

        sl = slice(start_i, min(start_i + chunk_size + 2000, total_hits)) # +Buffer for lookahead
        loc_x, loc_y = phys_x[sl], phys_y[sl]
        loc_l, loc_t = layers[sl], ts[sl]
        loc_id = c_ids[sl]
        
        # Build KDTree per layer
        trees, indices_map = {}, {}
        for L in [4,3,2,1]:
            mask_l = (loc_l == L)
            idxs = np.where(mask_l)[0]
            if len(idxs) > 0:
                trees[L] = cKDTree(np.column_stack((loc_x[idxs], loc_y[idxs])))
                indices_map[L] = idxs
        
        edge_u, edge_d = [], []
        
        for u, d in [(4,3), (4,2), (3,2), (3,1), (2,1), (4,1)]:
            if u not in trees or d not in trees: continue
            
            matches = trees[u].query_ball_tree(trees[d], r=search_radius)
            u_idxs = indices_map[u]
            d_idxs = indices_map[d]
            
            for i, d_match in enumerate(matches):
                if not d_match: continue
                u_local = u_idxs[i]
                t_start = loc_t[u_local]
                
                # Time filtering using numpy vectorization for speed
                d_real = d_idxs[d_match]
                dt = loc_t[d_real] - t_start
                valid = (dt >= -1) & (dt <= time_window)
                
                if np.any(valid):
                    valid_d = d_real[valid]
                    edge_u.extend([u_local] * len(valid_d))
                    edge_d.extend(valid_d)
                    
        if not edge_u: continue
        
        offsets, targets = _build_adjacency_csr(len(loc_x), np.array(edge_u, dtype=np.int32), np.array(edge_d, dtype=np.int32))
        
        seeds = []
        if 4 in indices_map: seeds.extend(indices_map[4])
        if 3 in indices_map: seeds.extend(indices_map[3])
        
        if seeds:
            tracks = _dfs_iterative(offsets, targets, np.array(seeds, dtype=np.int32), loc_id, loc_l, min_hits)
            if len(tracks) > 0: all_tracks_list.append(tracks)

    sys.stdout.write("\n")
    if not all_tracks_list: return {}
    
    full_stack = np.vstack(all_tracks_list)
    final_ids = np.unique(full_stack, axis=0)
    
    # 4. Extract & Format
    results = {}
    
    # ID Map
    max_id = np.max(cluster_data['clusterID'])
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    mask_c = cluster_data['clusterID'] != -1
    id_map[cluster_data['clusterID'][mask_c]] = np.where(mask_c)[0].astype(np.int32)
    
    def get_col(key, dtype=float, fill=np.nan):
        vals = cluster_data[key]
        flat_ids = final_ids.ravel()
        valid = flat_ids != -1
        indices = id_map[flat_ids[valid]]
        out = np.full(flat_ids.shape, fill, dtype=dtype)
        out[valid] = vals[indices]
        return out.reshape(final_ids.shape)
    
    for k, L in enumerate([4, 3, 2, 1]):
        results[f'L{L}_ID'] = final_ids[:, k]
        results[f'x{L}'] = get_col('cog_col')[:, k]
        results[f'y{L}'] = get_col('cog_row')[:, k]
        results[f't{L}'] = get_col('ts_start', dtype=np.float64)[:, k]
        results[f'xtalk{L}'] = get_col('xtalk_type', dtype=object, fill=0)[:, k]

    # Chi2
    px = np.column_stack([results[f'x{L}'] * pitch_ratio_col for L in [4,3,2,1]]).astype(np.float32)
    py = np.column_stack([results[f'y{L}'] for L in [4,3,2,1]]).astype(np.float32)
    
    px[:, 1] += corrections[3]['dc']; py[:, 1] += corrections[3]['dr']
    px[:, 2] += corrections[2]['dc']; py[:, 2] += corrections[2]['dr']
    px[:, 3] += corrections[1]['dc']; py[:, 3] += corrections[1]['dr']
    
    results['chi2'] = fast_chi2_loop(px, py, final_ids != -1, np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32))
    
    print(f"--- Found {len(final_ids)} tracks in {time.time()-t0:.2f}s ---")
    return results

# ==========================================
# 3. HELPERS AND WRAPPER
# ==========================================
def filter_cluster_data(cluster_data, allowed_types):
    xt_arr = cluster_data['xtalk_type']
    if np.issubdtype(xt_arr.dtype, np.number):
        mask = np.isin(xt_arr, allowed_types)
    else:
        def check_type(val):
            v = val[0] if isinstance(val, list) and len(val) > 0 else val
            try: return int(v) in allowed_types
            except: return False
        mask = np.vectorize(check_type)(xt_arr)
    
    filtered = {}
    for k, v in cluster_data.items():
        if len(v) == len(mask): filtered[k] = v[mask]
        else: filtered[k] = v 
    return filtered

def subset_clusters_by_ids(cluster_data, valid_ids):
    if len(valid_ids) == 0: return {}
    mask = np.isin(cluster_data['clusterID'], valid_ids)
    new_data = {}
    n_total = len(cluster_data['clusterID'])
    for k, v in cluster_data.items():
        if isinstance(v, (np.ndarray, list)) and len(v) == n_total:
            new_data[k] = np.array(v)[mask]
        else:
            new_data[k] = copy.copy(v)
    return new_data

def _map_track_chi2_to_clusters(cluster_data, tracks_dict):
    if 'chi2' not in tracks_dict or len(tracks_dict['chi2']) == 0:
        cluster_data['track_chi2'] = np.full(len(cluster_data['clusterID']), -1.0, dtype=np.float32)
        return cluster_data

    cid_to_chi2 = {}
    chi2s = tracks_dict['chi2']
    
    for L in [4, 3, 2, 1]:
        ids = tracks_dict[f'L{L}_ID']
        valid = ids != -1
        for cid, c2 in zip(ids[valid], chi2s[valid]):
            if cid not in cid_to_chi2 or c2 < cid_to_chi2[cid]:
                cid_to_chi2[cid] = c2

    s_ids = pd.Series(cluster_data['clusterID'])
    out_chi2 = s_ids.map(cid_to_chi2).fillna(-1.0).values.astype(np.float32)
    cluster_data['track_chi2'] = out_chi2
    return cluster_data


def separate_competing_tracks_fast(all_tracks, final_clusters):
    print("   -> Mode 2: O(N) Selection (Clean vs Dirty)...")
    n = len(all_tracks['L4_ID'])
    if n == 0: return {}, {}
    
    hits = np.sum([all_tracks[f'L{L}_ID'] != -1 for L in [4,3,2,1]], axis=0)
    order = np.lexsort((all_tracks['chi2'], -hits))
    ids_map = {L: all_tracks[f'L{L}_ID'] for L in [4,3,2,1]}
    
    claimed = {} 
    clean_indices = []
    dirty_candidates = []
    
    for idx in order:
        cids = [ids_map[L][idx] for L in [4,3,2,1] if ids_map[L][idx] != -1]
        if any(c in claimed for c in cids): dirty_candidates.append(idx)
        else:
            clean_indices.append(idx)
            for c in cids: claimed[c] = idx
            
    pair_map = {} 
    for d_idx in dirty_candidates:
        cids = [ids_map[L][d_idx] for L in [4,3,2,1] if ids_map[L][d_idx] != -1]
        owner = next((claimed[c] for c in cids if c in claimed), None)
        if owner is not None and owner not in pair_map: pair_map[owner] = d_idx
            
    final_clean = []
    final_dirty = []
    for c_idx in clean_indices:
        final_clean.append(c_idx)
        final_dirty.append(pair_map.get(c_idx, c_idx)) 
        
    def subset(idx_list):
        return {k: np.array(v)[idx_list] for k, v in all_tracks.items() if len(v) == n}
        
    return subset(final_clean), subset(final_dirty)

def analyze_xtalk_impact(final_clusters, xtalk_mode=2, ref_layer=4, pitch_ratio_col=3.0, search_radius=15.0, time_window=18, min_hits=4):
    print(f"\n=== XTALK ANALYSIS (Mode {xtalk_mode}) [OPTIMIZED] ===")
    
    # 1. Alignment (Clean only)
    align_df = calculate_translational_misalignment_robust(
        filter_cluster_data(final_clusters, [0]), 
        ref_layer=ref_layer, pitch_ratio_col=pitch_ratio_col
    )
    
    params = {
        'alignment_df': align_df,
        'pitch_ratio_col': pitch_ratio_col,
        'search_radius': search_radius,
        'time_window': time_window,
        'min_hits': min_hits
    }
    
    clean_trks, dirty_trks = {}, {}
    
    if xtalk_mode == 2:
        xt = final_clusters['xtalk_type']
        if np.issubdtype(xt.dtype, np.number): mask = np.isin(xt, [0, 2])
        else: mask = np.array([x in [0, 2] or (isinstance(x, list) and x[0] in [0,2]) for x in xt])
        
        data_mixed = {k: v[mask] for k,v in final_clusters.items()}
        raw = tracking_fast(data_mixed, **params)
        clean_trks, dirty_trks = separate_competing_tracks_fast(raw, final_clusters)
        
    elif xtalk_mode == 0:
        xt = final_clusters['xtalk_type']
        if np.issubdtype(xt.dtype, np.number): mask = (xt == 0)
        else: mask = np.array([x == 0 or (isinstance(x, list) and x[0] == 0) for x in xt])
        data_clean = {k: v[mask] for k,v in final_clusters.items()}
        
        raw = tracking_fast(data_clean, **params)
        clean_trks, _ = separate_competing_tracks_fast(raw, final_clusters)
        dirty_trks = {}

    print("Extracting clusters...")
    
    def get_cl(trks):
        if not trks: return {}
        all_ids = []
        for L in [4,3,2,1]:
            c = trks[f'L{L}_ID']
            all_ids.append(c[c!=-1])
        if len(all_ids) == 0: return {}
        u_ids = np.unique(np.concatenate(all_ids))
        cl = subset_clusters_by_ids(final_clusters, u_ids)
        return _map_track_chi2_to_clusters(cl, trks)
        
    clean_cl = get_cl(clean_trks)
    dirty_cl = get_cl(dirty_trks)
    
    n_c = len(clean_trks.get('chi2', []))
    n_d = len(dirty_trks.get('chi2', []))
    print(f"Done. Clean: {n_c}, Dirty: {n_d}")
    
    return clean_trks, clean_cl, dirty_trks, dirty_cl


import pickle

print("Saving golden datasets to disk...")

# Save the raw combinatorial tracks
with open('all_trks_R300.pkl', 'wb') as f:
    pickle.dump(all_trks, f)

# Save the filtered best tracks
with open('best_trks_R300.pkl', 'wb') as f:
    pickle.dump(best_trks, f)

print("Saved successfully! You can now load these instantly in the future.")