import numpy as np
import time
import sys
from scipy.spatial import cKDTree
from numba import njit
import pandas as pd


# --- NUMBA KERNELS (Must be defined before the function) ---

@njit(cache=True)
def _build_adjacency_csr(n_nodes, u_indices, d_indices):
    """Builds Compressed Sparse Row adjacency for fast graph traversal."""
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
    found.pop() # Empty list with correct type signature
    
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

# --- MAIN TRACKING FUNCTION ---

def tracking_fast(
    cluster_data, 
    alignment_df=None, 
    target_n_tracks=None, 
    search_radius=15.0, 
    pitch_ratio_col=3.0, 
    time_window=18, 
    min_hits=4
):
    print(f"--- Fast Tracking (MinHits={min_hits}) ---")
    t0 = time.time()
    
    # 1. Filter valid hits
    mask = (cluster_data['clusterID'] != -1)
    valid_indices = np.where(mask)[0]
    
    valid_ts = cluster_data['ts_start'][valid_indices]
    sorted_sub = np.argsort(valid_ts)
    active_indices = valid_indices[sorted_sub]
    
    if target_n_tracks: active_indices = active_indices[:target_n_tracks*100]

    # 2. Prepare Data Arrays
    # Note: We use RAW COG here and apply alignment if provided
    phys_x = cluster_data['cog_col'][active_indices] * pitch_ratio_col
    phys_y = cluster_data['cog_row'][active_indices].copy()
    layers = cluster_data['Layer'][active_indices].astype(np.int32)
    ts     = cluster_data['ts_start'][active_indices].astype(np.int64)
    c_ids  = cluster_data['clusterID'][active_indices]

    # Apply Alignment
    corrections = {L: {'dr': 0.0, 'dc': 0.0} for L in [4, 3, 2, 1]}
    if alignment_df is not None and not alignment_df.empty:
        # Assuming typical alignment DF structure
        align_lookup = {(r.Ref_Layer, r.Layer_to_Align): r for r in alignment_df.itertuples()}
        def get_pars(u, d):
            r = align_lookup.get((u, d))
            # Handle column names flexibly
            dx = r.dx_col if hasattr(r, 'dx_col') else getattr(r, 'Col_Misalign_Phys', 0.0)
            dy = r.dy_row if hasattr(r, 'dy_row') else getattr(r, 'Row_Misalign_Phys', 0.0)
            return (dy, dx)
            
        dr43, dc43 = get_pars(4, 3); dr32, dc32 = get_pars(3, 2); dr21, dc21 = get_pars(2, 1)
        corrections[3] = {'dr': dr43, 'dc': dc43}
        corrections[2] = {'dr': dr43+dr32, 'dc': dc43+dc32}
        corrections[1] = {'dr': dr43+dr32+dr21, 'dc': dc43+dc32+dc21}
    
    for L in [3, 2, 1]:
        mask_l = (layers == L)
        if np.any(mask_l):
            phys_x[mask_l] += corrections[L]['dc']
            phys_y[mask_l] += corrections[L]['dr']

    # 3. Chunked Graph Search (Chunk Size 2000 to prevent Hangs)
    chunk_size = 2000
    all_tracks_list = []
    total_hits = len(active_indices)
    
    print(f"Processing {total_hits} hits in chunks of {chunk_size}...")
    
    for start_i in range(0, total_hits, chunk_size):
        if (start_i // chunk_size) % 10 == 0:
            sys.stdout.write(f"\r  -> Progress: {start_i}/{total_hits} hits")
            sys.stdout.flush()

        sl = slice(start_i, min(start_i + chunk_size + 2000, total_hits))
        loc_x, loc_y = phys_x[sl], phys_y[sl]
        loc_l, loc_t = layers[sl], ts[sl]
        loc_id = c_ids[sl]
        
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
            u_idxs = indices_map[u]; d_idxs = indices_map[d]
            
            for i, d_match in enumerate(matches):
                if not d_match: continue
                u_local = u_idxs[i]; t_start = loc_t[u_local]
                d_real = d_idxs[d_match]; dt = loc_t[d_real] - t_start
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

    px = np.column_stack([results[f'x{L}'] * pitch_ratio_col for L in [4,3,2,1]]).astype(np.float32)
    py = np.column_stack([results[f'y{L}'] for L in [4,3,2,1]]).astype(np.float32)
    
    # Apply corrections for Chi2 calc
    px[:, 1] += corrections[3]['dc']; py[:, 1] += corrections[3]['dr']
    px[:, 2] += corrections[2]['dc']; py[:, 2] += corrections[2]['dr']
    px[:, 3] += corrections[1]['dc']; py[:, 3] += corrections[1]['dr']
    
    results['chi2'] = fast_chi2_loop(px, py, final_ids != -1, np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32))
    
    print(f"--- Found {len(final_ids)} tracks in {time.time()-t0:.2f}s ---")
    return results

# ==========================================
# 2. HELPERS (Format & Data)
# ==========================================
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

def filter_cluster_data(cluster_data, allowed_types):
    xt_arr = cluster_data['xtalk_type']
    if np.issubdtype(xt_arr.dtype, np.number): mask = np.isin(xt_arr, allowed_types)
    else: mask = np.vectorize(lambda x: x in allowed_types if isinstance(x, (int, float)) else x[0] in allowed_types)(xt_arr)
    
    filtered = {}
    for k, v in cluster_data.items():
        if len(v) == len(mask): filtered[k] = v[mask]
        else: filtered[k] = v 
    return filtered

@njit(cache=True)
def _resolve_conflicts_greedy_strip(
    h_track_ids, h_chi2s, h_ts, h_cols, h_layers,
    time_win, col_rad, n_total_tracks
):
    """
    Status Codes:
    0 = Clean (No Conflict) -> IGNORE (No crosstalk found)
    1 = Clean (Winner)      -> KEEP (Had crosstalk, but was better)
    2 = Dirty (Loser)       -> KEEP (The crosstalk ghost)
    """
    track_status = np.zeros(n_total_tracks, dtype=np.int8)
    n_hits = h_ts.shape[0]
    
    for i in range(n_hits):
        t_id_i = h_track_ids[i]
        ref_t, ref_c, ref_l = h_ts[i], h_cols[i], h_layers[i]
        chi_i = h_chi2s[i]
        
        # Look ahead
        for j in range(i + 1, n_hits):
            if h_ts[j] > ref_t + time_win: break
            if h_layers[j] != ref_l: continue
            if abs(h_cols[j] - ref_c) > col_rad: continue
            
            t_id_j = h_track_ids[j]
            if t_id_i == t_id_j: continue
            
            # CONFLICT FOUND
            chi_j = h_chi2s[j]
            
            if chi_i < chi_j:
                # Track I wins, J loses
                track_status[t_id_j] = 2 
                # Mark I as Winner (if it hasn't already lost elsewhere)
                if track_status[t_id_i] != 2: 
                    track_status[t_id_i] = 1
            else:
                # Track J wins, I loses
                track_status[t_id_i] = 2
                # Mark J as Winner
                if track_status[t_id_j] != 2: 
                    track_status[t_id_j] = 1
                    
    return track_status

def analyze_xtalk_competition_strip(
    final_clusters, 
    xtalk_type=2, 
    clock_cycles=5,         
    search_radius_col=15.0, 
    pitch_ratio_col=3.0,
    ref_layer=4,
    time_window_track=18,
    min_hits=4,
    tracking_func=None,
    alignment_func=None
):
    print(f"\n=== XTALK COMPETITION (Type {xtalk_type}) [FILTERED] ===")
    
    # 1. Align & Filter
    if alignment_func:
        # Note: We filter for type 0 (Clean) to calculate alignment
        align_df = alignment_func(filter_cluster_data(final_clusters, [0]), ref_layer, pitch_ratio_col)
    else: 
        align_df = None
    
    # Filter: Allow Clean (0) AND Target Xtalk Type
    data_mixed = filter_cluster_data(final_clusters, [0, xtalk_type])
    
    # 2. Track
    if tracking_func is None: 
        raise ValueError("tracking_func required (pass tracking_fast)")
        
    print("1. Tracking on Mixed Data...")
    
    # --- FIX: Use Keyword Arguments to match tracking_fast signature ---
    raw_tracks = tracking_func(
        cluster_data=data_mixed, 
        alignment_df=align_df, 
        target_n_tracks=None,   # Process all tracks
        # subset_factor=100,    # REMOVED: tracking_fast doesn't use this
        search_radius=search_radius_col, 
        pitch_ratio_col=pitch_ratio_col, 
        time_window=time_window_track, 
        min_hits=min_hits
    )
    
    if not raw_tracks or len(raw_tracks) == 0: 
        print("No tracks found.")
        return {}, {}, {}, {}
        
    n_tracks = len(raw_tracks['L4_ID'])
    print(f"   -> Found {n_tracks} total candidate tracks.")

    # 3. Flatten Hits for Conflict Check
    # (Same logic as before...)
    
    # Pre-fetch arrays to avoid dict lookup in loop
    ids_map = {L: raw_tracks[f'L{L}_ID'] for L in [4,3,2,1]}
    cols_map = {L: raw_tracks[f'x{L}'] * pitch_ratio_col for L in [4,3,2,1]} 
    ts_map = {L: raw_tracks[f't{L}'] for L in [4,3,2,1]}
    chi2_arr = raw_tracks['chi2']
    
    h_tid, h_l, h_col, h_t, h_chi2 = [], [], [], [], []
    
    # We iterate tracks
    for i in range(n_tracks):
        c2 = chi2_arr[i]
        for L in [4,3,2,1]:
            if ids_map[L][i] != -1:
                # Store hit data
                h_tid.append(i)
                h_l.append(L)
                h_col.append(cols_map[L][i]) # Already physical units
                h_t.append(ts_map[L][i])
                h_chi2.append(c2)
                
    arr_tid = np.array(h_tid, dtype=np.int32)
    arr_l   = np.array(h_l, dtype=np.int32)
    arr_col = np.array(h_col, dtype=np.float32)
    arr_t   = np.array(h_t, dtype=np.int64)
    arr_chi2 = np.array(h_chi2, dtype=np.float32)
    
    # Sort hits by time for the Numba kernel
    sort_idx = np.argsort(arr_t)

    # 4. Resolve Conflicts
    print("2. Resolving Conflicts (Strip Search)...")
    track_status = _resolve_conflicts_greedy_strip(
        arr_tid[sort_idx], arr_chi2[sort_idx], arr_t[sort_idx], 
        arr_col[sort_idx], arr_l[sort_idx],
        clock_cycles, search_radius_col, n_tracks
    )
    
    # 5. Filter: Only Winners(1) and Losers(2)
    clean_indices = np.where(track_status == 1)[0]
    dirty_indices = np.where(track_status == 2)[0]
    
    def subset(idxs):
        # Helper to subset the dictionary arrays
        return {k: np.array(v)[idxs] for k, v in raw_tracks.items() if len(v) == n_tracks}
    
    clean2_trks = subset(clean_indices)
    dirty2_trks = subset(dirty_indices)
    
    # 6. Clusters
    print("3. Packaging Clusters...")
    def get_cl(trks):
        if not trks or 'L4_ID' not in trks or len(trks['L4_ID'])==0: return {}
        all_ids = []
        for L in [4,3,2,1]:
            c = trks[f'L{L}_ID']
            all_ids.append(c[c!=-1])
        if len(all_ids) == 0: return {}
        u_ids = np.unique(np.concatenate(all_ids))
        cl = subset_clusters_by_ids(final_clusters, u_ids)
        return _map_track_chi2_to_clusters(cl, trks)

    clean2_clsts = get_cl(clean2_trks)
    dirty2_clsts = get_cl(dirty2_trks)
    
    print(f"Done. Clean (Winners): {len(clean_indices)}, Dirty (Losers): {len(dirty_indices)}")
    
    return clean2_trks, clean2_clsts, dirty2_trks, dirty2_clsts


# 1. Define Parameters
# --------------------
target_xtalk_type = 2       # The type of crosstalk you are investigating (e.g., Type 2)
clock_cycles_limit = 5      # Conflict Window: Max time difference to consider tracks "competing"
strip_width_col = 15.0      # Conflict Window: "Long Thin Window" width (Columns)

# 2. Run the Competition Analysis
# -------------------------------
clean2_trks, clean2_clsts, dirty2_trks, dirty2_clsts = analyze_xtalk_competition_strip(
    final_clusters,                  # Your main cluster dictionary
    xtalk_type=target_xtalk_type,    # 2
    clock_cycles=clock_cycles_limit, # 5
    search_radius_col=strip_width_col, # 15.0
    pitch_ratio_col=3.0,
    
    # Standard Tracking Params
    ref_layer=4,
    time_window_track=18,
    min_hits=4,
    
    # *** IMPORTANT: Inject the helper functions ***
    tracking_func=tracking_fast,
    alignment_func=None
)

# 3. View Results
# ---------------
n_clean = len(clean2_trks.get('chi2', []))
n_dirty = len(dirty2_trks.get('chi2', []))

print("\n=== RESULTS ===")
print(f"Clean (Winner) Tracks: {n_clean}")
print(f"Dirty (Loser) Tracks:  {n_dirty}")

if n_dirty > 0:
    print(f"Avg Chi2 Clean: {np.mean(clean2_trks['chi2']):.3f}")
    print(f"Avg Chi2 Dirty: {np.mean(dirty2_trks['chi2']):.3f}")
    
    
    
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. HELPER: PAIRING & PHYSICS
# ==========================================
def get_paired_features(clean_trks, dirty_trks, pitch_ratio=3.0):
    """
    Pairs tracks and calculates comparative physics features.
    Returns: DataFrame-like dictionary with paired stats.
    """
    print("--- Pairing Tracks & Calculating Unbiased Residuals ---")

    # 1. Map Clean Tracks by Hits
    # Map: ClusterID -> List of Clean Track Indices
    cluster_map = {}
    n_clean = len(clean_trks['L4_ID'])
    for i in range(n_clean):
        for L in [4,3,2,1]:
            cid = clean_trks[f'L{L}_ID'][i]
            if cid != -1:
                if cid not in cluster_map: cluster_map[cid] = []
                cluster_map[cid].append(i)

    # 2. Iterate Dirty Tracks to find parents
    pairs_data = {
        'layer': [], 'c_idx': [], 'd_idx': [],
        'win_res': [], 'lose_res': [],
        'win_chi2': [], 'lose_chi2': [],
        'cx': [], 'cy': [], 'dx': [], 'dy': [] # For vector plot
    }

    z_map = {4: -1.5, 3: -0.5, 2: 0.5, 1: 1.5}
    n_dirty = len(dirty_trks['L4_ID'])

    for i in range(n_dirty):
        # Voting for best clean match
        votes = {}
        d_ids = {L: dirty_trks[f'L{L}_ID'][i] for L in [4,3,2,1]}

        for L, cid in d_ids.items():
            if cid != -1 and cid in cluster_map:
                for c_idx in cluster_map[cid]:
                    votes[c_idx] = votes.get(c_idx, 0) + 1

        # Match requirement: Share exactly 3 hits (so 1 is swapped)
        best_match = -1
        for c_idx, count in votes.items():
            if count == 3: # Strictly 3 shared, 1 swapped
                best_match = c_idx
                break

        if best_match == -1: continue

        # Identify the Swap Layer
        swap_L = -1
        for L in [4,3,2,1]:
            if clean_trks[f'L{L}_ID'][best_match] != d_ids[L]:
                swap_L = L
                break

        if swap_L == -1: continue # Should not happen if count==3

        # --- PHYSICS CALCULATION ---
        # Unbiased Fit: Use the 3 SHARED hits to define the "True Line"
        hits_x, hits_y, hits_z = [], [], []
        for L in [4,3,2,1]:
            if L == swap_L: continue
            if clean_trks[f'L{L}_ID'][best_match] == -1: continue

            hits_x.append(clean_trks[f'x{L}'][best_match] * pitch_ratio)
            hits_y.append(clean_trks[f'y{L}'][best_match])
            hits_z.append(z_map[L])

        if len(hits_z) < 2: continue # Cannot fit line

        # Fit Line
        A = np.vstack([hits_z, np.ones(len(hits_z))]).T
        mx, cx = np.linalg.lstsq(A, hits_x, rcond=None)[0]
        my, cy = np.linalg.lstsq(A, hits_y, rcond=None)[0]

        # Predict at Swap Layer
        z_tgt = z_map[swap_L]
        pred_x = mx * z_tgt + cx
        pred_y = my * z_tgt + cy

        # Get Coordinates
        c_x = clean_trks[f'x{swap_L}'][best_match] * pitch_ratio
        c_y = clean_trks[f'y{swap_L}'][best_match]
        d_x = dirty_trks[f'x{swap_L}'][i] * pitch_ratio
        d_y = dirty_trks[f'y{swap_L}'][i]

        # Calculate Residuals
        res_win = np.sqrt((c_x - pred_x)**2 + (c_y - pred_y)**2)
        res_lose = np.sqrt((d_x - pred_x)**2 + (d_y - pred_y)**2)

        # Store
        pairs_data['layer'].append(swap_L)
        pairs_data['c_idx'].append(best_match)
        pairs_data['d_idx'].append(i)
        pairs_data['win_res'].append(res_win)
        pairs_data['lose_res'].append(res_lose)
        pairs_data['win_chi2'].append(clean_trks['chi2'][best_match])
        pairs_data['lose_chi2'].append(dirty_trks['chi2'][i])

        # Coords for Vector Plot
        pairs_data['cx'].append(c_x); pairs_data['cy'].append(c_y)
        pairs_data['dx'].append(d_x); pairs_data['dy'].append(d_y)

    # Convert to arrays
    for k in pairs_data: pairs_data[k] = np.array(pairs_data[k])

    print(f"   -> Analyzed {len(pairs_data['layer'])} pairs.")
    return pairs_data

# ==========================================
# 2. PLOT: VECTORS (2x2)
# ==========================================
def plot_xtalk_vectors_2x2(data):


    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, L in enumerate([4, 3, 2, 1]):
        ax = axes[i]
        mask = (data['layer'] == L)

        if np.sum(mask) == 0:
            ax.text(0.5, 0.5, f"No Swaps L{L}", ha='center')
            continue

        cx = data['cx'][mask]; cy = data['cy'][mask]
        dx = data['dx'][mask]; dy = data['dy'][mask]

        # Subsample if huge
        if len(cx) > 300:
            idx = np.random.choice(len(cx), 300, replace=False)
            cx, cy, dx, dy = cx[idx], cy[idx], dx[idx], dy[idx]

        # Draw Lines
        for j in range(len(cx)):
            ax.plot([cx[j], dx[j]], [cy[j], dy[j]], c='gray', alpha=0.3, lw=0.8)

        ax.scatter(cx, cy, c='blue', s=20, label='Winner', edgecolors='none', alpha=0.7)
        ax.scatter(dx, dy, c='red', s=20, label='Loser', edgecolors='none', alpha=0.7)

        ax.set_title(f"Layer {L} Shift Vectors")
        ax.set_xlabel("Col (Phys)")
        ax.set_ylabel("Row")
        ax.axis('equal')
        if i==0: ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 3. PLOT: DISTRIBUTIONS & SCATTERS
# ==========================================
def plot_advanced_comparisons(data):
    if len(data['win_chi2']) == 0: return

    # Features
    # Log Chi2
    w_chi2 = np.log10(data['win_chi2'] + 1e-4)
    l_chi2 = np.log10(data['lose_chi2'] + 1e-4)
    # Residuals
    w_res = data['win_res']
    l_res = data['lose_res']

    # --- FIGURE 1: DISTRIBUTIONS (2x1) ---


    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chi2 Hist
    ax = axes[0]
    bins = np.linspace(min(w_chi2.min(), l_chi2.min()), max(w_chi2.max(), l_chi2.max()), 50)
    ax.hist(w_chi2, bins=bins, density=True, histtype='step', lw=2, color='blue', label='Winner')
    ax.hist(l_chi2, bins=bins, density=True, histtype='step', lw=2, color='red', label='Loser')
    ax.set_title(r"Log$_{10}(\chi^2)$ Distribution")
    ax.set_xlabel("Log Chi2")
    ax.legend()

    # Residual Hist
    ax = axes[1]
    bins_r = np.linspace(0, np.percentile(l_res, 98), 50) # Cut off huge outliers
    ax.hist(w_res, bins=bins_r, density=True, histtype='step', lw=2, color='blue', label='Winner')
    ax.hist(l_res, bins=bins_r, density=True, histtype='step', lw=2, color='red', label='Loser')
    ax.set_title("Unbiased Residual Distribution (at Swap Layer)")
    ax.set_xlabel("Residual Magnitude (Phys Units)")

    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: TRIPLE SCATTER (Raw, White, Manifold) ---


    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))

    # Prepare Data for ML (Stacked)
    # Class 0 = Winner, Class 1 = Loser
    X_win = np.column_stack((w_chi2, w_res))
    X_lose = np.column_stack((l_chi2, l_res))

    # 1. RAW SCATTER
    ax = axes2[0]
    # Scatter points (alpha for density)
    ax.scatter(w_chi2, w_res, c='blue', s=10, alpha=0.3, label='Winner')
    ax.scatter(l_chi2, l_res, c='red', s=10, alpha=0.3, label='Loser')
    ax.set_title("1. Raw Features")
    ax.set_xlabel("Log Chi2")
    ax.set_ylabel("Residual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. WHITENING (PCA)
    # Fit PCA only on WINNERS to define "Normality"
    pca = PCA(whiten=True)
    pca.fit(X_win)

    Xw_win = pca.transform(X_win)
    Xw_lose = pca.transform(X_lose)

    ax = axes2[1]
    ax.scatter(Xw_win[:,0], Xw_win[:,1], c='blue', s=10, alpha=0.3)
    ax.scatter(Xw_lose[:,0], Xw_lose[:,1], c='red', s=10, alpha=0.3)

    # Sigma Circles
    circ = plt.Circle((0,0), 3, fill=False, ls='--', color='k', label='3$\sigma$')
    ax.add_artist(circ)

    ax.set_title("2. Whitened (Normalized to Winner Dist)")
    ax.set_xlabel("PC1 ($\sigma$)")
    ax.set_ylabel("PC2 ($\sigma$)")
    ax.legend([circ], ['3$\sigma$ Limit'])
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # 3. MANIFOLD (Isomap)
    # Note: Isomap is slow. Subsample if N > 2000
    X_combined = np.vstack((X_win, X_lose))
    y_combined = np.concatenate((np.zeros(len(X_win)), np.ones(len(X_lose))))

    if len(X_combined) > 2000:
        idx = np.random.choice(len(X_combined), 2000, replace=False)
        X_sub = X_combined[idx]
        y_sub = y_combined[idx]
    else:
        X_sub, y_sub = X_combined, y_combined

    # Scale first! Manifold learning is sensitive to scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    print("Running Isomap (this might take a moment)...")
    iso = Isomap(n_components=2, n_neighbors=15)
    X_iso = iso.fit_transform(X_scaled)

    ax = axes2[2]
    ax.scatter(X_iso[y_sub==0, 0], X_iso[y_sub==0, 1], c='blue', s=10, alpha=0.4, label='Winner')
    ax.scatter(X_iso[y_sub==1, 0], X_iso[y_sub==1, 1], c='red', s=10, alpha=0.4, label='Loser')

    ax.set_title("3. Manifold Learning (Isomap)")
    ax.set_xlabel("Iso 1")
    ax.set_ylabel("Iso 2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. MASTER RUNNER
# ==========================================
def run_full_visualizations(clean_trks, dirty_trks, pitch_ratio=3.0):
    # 1. Process Data
    data = get_paired_features(clean_trks, dirty_trks, pitch_ratio)

    if not data['layer'].size:
        print("No paired data found.")
        return

    # 2. Vector Plot
    plot_xtalk_vectors_2x2(data)

    # 3. Distributions & Scatters
    plot_advanced_comparisons(data)

# EXECUTE
run_full_visualizations(clean2_trks, dirty2_trks)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib.colors import ListedColormap

# ==========================================
# 1. DATA PREPARATION ENGINE
# ==========================================
def prepare_feature_spaces(clean_trks, dirty_trks, pairs, pitch_ratio=3.0):
    """
    Generates the three feature sets: Raw, Whitened, Manifold.
    Returns X_raw, X_white, X_iso, y
    """
    print("--- Preparing Feature Spaces ---")
    
    # 1. Extract Base Features (Raw)
    data = []
    z_map = {4: -1.5, 3: -0.5, 2: 0.5, 1: 1.5}
    
    for (c_idx, d_idx, swap_L) in pairs:
        # Fit Line to N-1 Clean Hits
        hits_x, hits_y, hits_z = [], [], []
        for L in [4,3,2,1]:
            if L == swap_L: continue
            if clean_trks[f'L{L}_ID'][c_idx] == -1: continue
            hits_x.append(clean_trks[f'x{L}'][c_idx] * pitch_ratio)
            hits_y.append(clean_trks[f'y{L}'][c_idx])
            hits_z.append(z_map[L])
            
        if len(hits_z) < 2: continue
        
        # Fit
        A = np.vstack([hits_z, np.ones(len(hits_z))]).T
        mx, cx = np.linalg.lstsq(A, hits_x, rcond=None)[0]
        my, cy = np.linalg.lstsq(A, hits_y, rcond=None)[0]
        
        # Predict at Swap Layer
        z_tgt = z_map[swap_L]
        px, py = mx * z_tgt + cx, my * z_tgt + cy
        
        # --- Feature Extraction ---
        # 1. Winner (Class 0)
        wx = clean_trks[f'x{swap_L}'][c_idx] * pitch_ratio
        wy = clean_trks[f'y{swap_L}'][c_idx]
        w_res = np.sqrt((wx - px)**2 + (wy - py)**2)
        w_chi = np.log10(clean_trks['chi2'][c_idx] + 1e-4)
        data.append([w_chi, w_res, 0])
        
        # 2. Loser (Class 1)
        lx = dirty_trks[f'x{swap_L}'][d_idx] * pitch_ratio
        ly = dirty_trks[f'y{swap_L}'][d_idx]
        l_res = np.sqrt((lx - px)**2 + (ly - py)**2)
        l_chi = np.log10(dirty_trks['chi2'][d_idx] + 1e-4)
        data.append([l_chi, l_res, 1])
        
    arr = np.array(data)
    X_raw = arr[:, :2]
    y = arr[:, 2].astype(int)
    
    # 2. Whitened Space (PCA)
    print("   -> Computing PCA (Whitening)...")
    pca = PCA(n_components=2, whiten=True)
    X_white = pca.fit_transform(X_raw)
    
    # 3. Manifold Space (Isomap)
    print("   -> Computing Manifold (Isomap)... (Subsampling if needed)")
    if len(X_raw) > 3000:
        # Train manifold on subset to save time, transform all
        idx = np.random.choice(len(X_raw), 3000, replace=False)
        iso = Isomap(n_components=2, n_neighbors=20)
        iso.fit(X_raw[idx]) # Fit on subset
        X_iso = iso.transform(X_raw) # Transform all
    else:
        iso = Isomap(n_components=2, n_neighbors=20)
        X_iso = iso.fit_transform(X_raw)
        
    return {'Raw': X_raw, 'Whitened': X_white, 'Manifold': X_iso}, y

# ==========================================
# 2. DIAGNOSTIC PLOTTING ENGINE
# ==========================================
def plot_decision_boundary(ax, model, X, y, title):
    """Draws decision boundary and scatter plot."""
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100),
                         np.arange(y_min, y_max, (y_max-y_min)/100))
    
    # Predict
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Contour
    cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
    
    # Scatter
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=10, alpha=0.5, label='Winner')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=10, alpha=0.5, label='Loser')
    
    ax.set_title(title)
    ax.legend(loc='upper left')

def run_model_diagnostics(name, X, y, model_type='LDA'):
    """
    Runs one model on one dataset and produces the 3-plot diagnostic panel.
    """
    # 1. Train Model
    if model_type == 'LDA':
        clf = LinearDiscriminantAnalysis()
    else:
        clf = QuadraticDiscriminantAnalysis()
        
    clf.fit(X, y)
    
    # Get scores/probs
    # LDA has 'decision_function' (projection to 1D), QDA only 'predict_proba'
    if hasattr(clf, 'decision_function'):
        scores = clf.decision_function(X)
    else:
        # For QDA, log-ratio of probabilities is effectively the score
        probs = clf.predict_proba(X)
        epsilon = 1e-10
        scores = np.log(probs[:, 1] + epsilon) - np.log(probs[:, 0] + epsilon)
        
    # Get ROC
    probs_pos = clf.predict_proba(X)[:, 1]
    fpr, tpr, thresh = roc_curve(y, probs_pos)
    roc_auc = auc(fpr, tpr)
    
    # Statistics
    J = tpr - fpr
    idx_J = np.argmax(J)
    dist01 = np.sqrt((1-tpr)**2 + fpr**2)
    idx_D = np.argmin(dist01)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Scatter with Boundary
    plot_decision_boundary(axes[0], clf, X, y, f"{model_type} Decision Boundary")
    axes[0].set_xlabel("Feature 1"); axes[0].set_ylabel("Feature 2")
    
    # 2. Discriminant Score Distribution
    
    axes[1].hist(scores[y==0], bins=40, density=True, histtype='stepfilled', alpha=0.3, color='blue', label='Winner')
    axes[1].hist(scores[y==0], bins=40, density=True, histtype='step', lw=2, color='blue')
    
    axes[1].hist(scores[y==1], bins=40, density=True, histtype='stepfilled', alpha=0.3, color='red', label='Loser')
    axes[1].hist(scores[y==1], bins=40, density=True, histtype='step', lw=2, color='red')
    
    # Plot Optimal Cutoff (Youden) mapping back to score is hard directly from ROC array index
    # We approximate by finding the score percentile
    axes[1].set_title(f"Discriminant Scores (Separation)")
    axes[1].set_xlabel("Log-Likelihood Ratio / Score")
    axes[1].legend()
    
    # 3. ROC Curve
    
    axes[2].plot(fpr, tpr, lw=2, color='darkorange', label=f'AUC = {roc_auc:.3f}')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Markers
    axes[2].scatter(fpr[idx_J], tpr[idx_J], s=100, c='green', marker='o', label=f"Youden J={J[idx_J]:.2f}")
    axes[2].scatter(fpr[idx_D], tpr[idx_D], s=100, c='purple', marker='x', label=f"Closest(0,1) d={dist01[idx_D]:.2f}")
    
    axes[2].set_title(f"ROC Curve ({name})")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f"Diagnostics: {model_type} on {name} Data", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return {
        'Dataset': name,
        'Model': model_type,
        'AUC': roc_auc,
        'Youden_J': J[idx_J],
        'TPR_at_J': tpr[idx_J],
        'FPR_at_J': fpr[idx_J]
    }

# ==========================================
# 3. MASTER RUNNER
# ==========================================
def run_full_lda_qda_suite(clean_trks, dirty_trks, pairs, pitch_ratio=3.0):
    # 1. Get Data
    datasets, y = prepare_feature_spaces(clean_trks, dirty_trks, pairs, pitch_ratio)
    
    results = []
    
    # 2. Loop Combinations
    for name, X in datasets.items():
        print(f"\nProcessing {name} Dataset...")
        
        # Run LDA
        res_lda = run_model_diagnostics(name, X, y, 'LDA')
        results.append(res_lda)
        
        # Run QDA
        res_qda = run_model_diagnostics(name, X, y, 'QDA')
        results.append(res_qda)
        
    # 3. Summary Table
    print("\n=== FINAL SUMMARY ===")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res

# EXECUTE (Assuming pairs are already generated from previous step)
# pairs = pair_clean_and_dirty_tracks(clean2_trks, dirty2_trks)
# run_full_lda_qda_suite(clean2_trks, dirty2_trks, pairs)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# ==========================================
# 1. DATA PREPARATION (Reuse Pairing Logic)
# ==========================================
def prepare_gbdt_data(clean_trks, dirty_trks, pitch_ratio=3.0):
    """
    Extracts raw physics features for GBDT.
    GBDTs don't need scaling or whitening.
    """
    print("--- Preparing Data for Gradient Boosting ---")
    
    # 1. Map Clean Tracks
    cluster_map = {}
    n_clean = len(clean_trks['L4_ID'])
    for i in range(n_clean):
        for L in [4,3,2,1]:
            cid = clean_trks[f'L{L}_ID'][i]
            if cid != -1:
                if cid not in cluster_map: cluster_map[cid] = []
                cluster_map[cid].append(i)
                
    data = []
    z_map = {4: -1.5, 3: -0.5, 2: 0.5, 1: 1.5}
    n_dirty = len(dirty_trks['L4_ID'])
    
    for i in range(n_dirty):
        # Voting for match
        votes = {}
        d_ids = {L: dirty_trks[f'L{L}_ID'][i] for L in [4,3,2,1]}
        for L, cid in d_ids.items():
            if cid != -1 and cid in cluster_map:
                for c_idx in cluster_map[cid]: votes[c_idx] = votes.get(c_idx, 0) + 1
                    
        # Strict Match (3 shared)
        best_match = -1
        for c_idx, count in votes.items():
            if count == 3: 
                best_match = c_idx; break
        
        if best_match == -1: continue
        
        # Find Swap Layer
        swap_L = -1
        for L in [4,3,2,1]:
            if clean_trks[f'L{L}_ID'][best_match] != d_ids[L]:
                swap_L = L; break
        if swap_L == -1: continue

        # Unbiased Fit (N-1 hits)
        hx, hy, hz = [], [], []
        for L in [4,3,2,1]:
            if L == swap_L: continue
            if clean_trks[f'L{L}_ID'][best_match] == -1: continue
            hx.append(clean_trks[f'x{L}'][best_match] * pitch_ratio)
            hy.append(clean_trks[f'y{L}'][best_match])
            hz.append(z_map[L])
            
        if len(hz) < 2: continue
        
        # Fit & Predict
        A = np.vstack([hz, np.ones(len(hz))]).T
        mx, cx = np.linalg.lstsq(A, hx, rcond=None)[0]
        my, cy = np.linalg.lstsq(A, hy, rcond=None)[0]
        z_tgt = z_map[swap_L]
        px, py = mx * z_tgt + cx, my * z_tgt + cy
        
        # --- Extract Features ---
        # Winner (Class 0)
        wx = clean_trks[f'x{swap_L}'][best_match] * pitch_ratio
        wy = clean_trks[f'y{swap_L}'][best_match]
        w_res = np.sqrt((wx - px)**2 + (wy - py)**2)
        w_chi = np.log10(clean_trks['chi2'][best_match] + 1e-4)
        data.append([w_chi, w_res, 0])
        
        # Loser (Class 1)
        lx = dirty_trks[f'x{swap_L}'][i] * pitch_ratio
        ly = dirty_trks[f'y{swap_L}'][i]
        l_res = np.sqrt((lx - px)**2 + (ly - py)**2)
        l_chi = np.log10(dirty_trks['chi2'][i] + 1e-4)
        data.append([l_chi, l_res, 1])
        
    arr = np.array(data)
    X = arr[:, :2] # [LogChi2, Residual]
    y = arr[:, 2].astype(int)
    return X, y

# ==========================================
# 2. GBDT DIAGNOSTICS ENGINE
# ==========================================
def run_gbdt_diagnostics(X, y):
    print("--- Training Gradient Boosted Decision Tree ---")
    
    # Split (Standard ML practice to avoid overfit display)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train GBDT
    # n_estimators=100, learning_rate=0.1, max_depth=3 is a robust default
    gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbdt.fit(X_train, y_train)
    
    # Predict
    probs = gbdt.predict_proba(X_test)[:, 1]
    
    # --- METRICS ---
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    # Youden
    J = tpr - fpr
    ix = np.argmax(J)
    youden_thresh = thresholds[ix]
    
    # Closest (0,1)
    d = np.sqrt((1-tpr)**2 + fpr**2)
    ix_d = np.argmin(d)
    
    # --- PLOTTING (4-Panel) ---
    
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Decision Boundary (Contour)
    ax = axes[0, 0]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5 # Residuals can be large, maybe clip?
    # Clip view for detail if outliers exist
    y_max = min(y_max, np.percentile(X[:, 1], 99) * 1.2)
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = gbdt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    cmap = ListedColormap(['#AAAAFF', '#FFAAAA']) # Blue=Clean, Red=Dirty
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
    
    # Plot Test Data Scatter
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', s=10, alpha=0.6, label='Winner')
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', s=10, alpha=0.6, label='Loser')
    ax.set_title("GBDT Decision Boundary (Non-Linear)")
    ax.set_xlabel("Log10(Chi2)")
    ax.set_ylabel("Residual (Phys)")
    ax.legend()
    
    # 2. Feature Importance
    ax = axes[0, 1]
    features = ['Log10(Chi2)', 'Residual']
    importances = gbdt.feature_importances_
    sns.barplot(x=features, y=importances, ax=ax, palette=['purple', 'orange'])
    ax.set_title("Feature Importance")
    ax.set_ylabel("Relative Importance")
    for i, v in enumerate(importances):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
    # 3. Probability Distribution (Separation)
    ax = axes[1, 0]
    ax.hist(probs[y_test==0], bins=30, density=True, alpha=0.5, color='blue', label='Winner')
    ax.hist(probs[y_test==1], bins=30, density=True, alpha=0.5, color='red', label='Loser')
    ax.axvline(youden_thresh, color='green', ls='--', label=f'Cut > {youden_thresh:.2f}')
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("P(Is Dirty)")
    ax.legend()
    
    # 4. ROC Curve
    ax = axes[1, 1]
    ax.plot(fpr, tpr, lw=2, color='darkorange', label=f'GBDT (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.scatter(fpr[ix], tpr[ix], s=100, c='green', marker='o', label=f'Youden J={J[ix]:.2f}')
    ax.scatter(fpr[ix_d], tpr[ix_d], s=100, c='purple', marker='x', label=f'Close(0,1)')
    
    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR (Dirty Accepted)")
    ax.set_ylabel("TPR (Clean Accepted)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'AUC': roc_auc,
        'Youden_J': J[ix],
        'TPR': tpr[ix],
        'FPR': fpr[ix],
        'Importances': dict(zip(features, importances))
    }

# ==========================================
# 3. RUNNER
# ==========================================
def run_full_gbdt_analysis(clean_trks, dirty_trks, pitch_ratio=3.0):
    X, y = prepare_gbdt_data(clean_trks, dirty_trks, pitch_ratio)
    
    if len(X) == 0:
        print("No paired data found.")
        return
        
    results = run_gbdt_diagnostics(X, y)
    
    print("\n=== GBDT RESULTS ===")
    print(f"AUC: {results['AUC']:.4f}")
    print(f"Optimal Operating Point:")
    print(f"  Efficiency (TPR): {results['TPR']:.2%}")
    print(f"  Contamination (FPR): {results['FPR']:.2%}")
    print(f"Feature Importance: {results['Importances']}")

# EXECUTE
run_full_gbdt_analysis(clean2_trks, dirty2_trks)



