# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:40:16 2026

@author: henry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import copy
from scipy.spatial import cKDTree
from numba import njit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import ListedColormap
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ==============================================================================
# PART 1: NUMBA ACCELERATED TRACKING & CONFLICT LOGIC
# ==============================================================================

@njit(cache=True)
def _build_adjacency_csr(n_nodes, u_indices, d_indices):
    counts = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(len(u_indices)): counts[u_indices[i]] += 1
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
    dummy = np.zeros(4, dtype=np.int64)
    found = [dummy]; found.pop()
    path = np.full(4, -1, dtype=np.int32)
    for seed in seeds:
        _dfs_recurse(seed, 4, offsets, targets, c_ids, layers, min_hits, path, found)
    if len(found) == 0: return np.zeros((0, 4), dtype=np.int64)
    out = np.empty((len(found), 4), dtype=np.int64)
    for i in range(len(found)): out[i] = found[i]
    return out

@njit(cache=True)
def _dfs_recurse(u_idx, u_layer, offsets, targets, c_ids, layers, min_hits, path, found):
    path[4 - u_layer] = u_idx
    start, end = offsets[u_idx], offsets[u_idx+1]
    if u_layer == 1 or start == end:
        hits = 0
        for k in range(4): 
            if path[k] != -1: hits += 1
        if hits >= min_hits:
            t = np.full(4, -1, dtype=np.int64)
            for k in range(4):
                if path[k] != -1: t[k] = c_ids[path[k]]
            found.append(t)
    for i in range(start, end):
        v_idx = targets[i]
        if layers[v_idx] < u_layer:
            _dfs_recurse(v_idx, layers[v_idx], offsets, targets, c_ids, layers, min_hits, path, found)
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

@njit(cache=True)
def _resolve_conflicts_greedy_strip(h_track_ids, h_chi2s, h_ts, h_cols, h_layers, time_win, col_rad, n_total_tracks):
    # 0=Ignore, 1=Winner, 2=Loser
    track_status = np.zeros(n_total_tracks, dtype=np.int8)
    n_hits = h_ts.shape[0]
    for i in range(n_hits):
        t_id_i = h_track_ids[i]
        ref_t, ref_c, ref_l, chi_i = h_ts[i], h_cols[i], h_layers[i], h_chi2s[i]
        for j in range(i + 1, n_hits):
            if h_ts[j] > ref_t + time_win: break
            if h_layers[j] != ref_l: continue
            if abs(h_cols[j] - ref_c) > col_rad: continue
            t_id_j = h_track_ids[j]
            if t_id_i == t_id_j: continue
            chi_j = h_chi2s[j]
            if chi_i < chi_j:
                track_status[t_id_j] = 2
                if track_status[t_id_i] != 2: track_status[t_id_i] = 1
            else:
                track_status[t_id_i] = 2
                if track_status[t_id_j] != 2: track_status[t_id_j] = 1
    return track_status

# ==============================================================================
# PART 2: TRACKING WRAPPERS
# ==============================================================================

def tracking_fast(cluster_data, alignment_df=None, target_n_tracks=None, search_radius=15.0, pitch_ratio_col=3.0, time_window=18, min_hits=4):
    print(f"   -> Tracking... (Radius={search_radius}, MinHits={min_hits})")
    mask = (cluster_data['clusterID'] != -1)
    valid_indices = np.where(mask)[0]
    valid_ts = cluster_data['ts_start'][valid_indices]
    active_indices = valid_indices[np.argsort(valid_ts)]
    
    phys_x = cluster_data['cog_col'][active_indices] * pitch_ratio_col
    phys_y = cluster_data['cog_row'][active_indices].copy()
    layers = cluster_data['Layer'][active_indices].astype(np.int32)
    ts     = cluster_data['ts_start'][active_indices].astype(np.int64)
    c_ids  = cluster_data['clusterID'][active_indices]

    corrections = {L: {'dr': 0.0, 'dc': 0.0} for L in [4, 3, 2, 1]}
    if alignment_df is not None:
        # Simplified alignment application
        pass # Add your alignment logic here if needed

    chunk_size = 2000
    all_tracks_list = []
    
    for start_i in range(0, len(active_indices), chunk_size):
        sl = slice(start_i, min(start_i + chunk_size + 2000, len(active_indices)))
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
                if np.any((dt >= -1) & (dt <= time_window)):
                    valid_d = d_real[(dt >= -1) & (dt <= time_window)]
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

    if not all_tracks_list: return {}
    full_stack = np.vstack(all_tracks_list)
    final_ids = np.unique(full_stack, axis=0)
    
    results = {}
    max_id = np.max(cluster_data['clusterID'])
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    id_map[cluster_data['clusterID'][cluster_data['clusterID']!=-1]] = np.where(cluster_data['clusterID']!=-1)[0].astype(np.int32)
    
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
    
    px = np.column_stack([results[f'x{L}'] * pitch_ratio_col for L in [4,3,2,1]]).astype(np.float32)
    py = np.column_stack([results[f'y{L}'] for L in [4,3,2,1]]).astype(np.float32)
    results['chi2'] = fast_chi2_loop(px, py, final_ids != -1, np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32))
    return results

def get_subset(trks, idxs):
    n = len(trks['L4_ID'])
    return {k: np.array(v)[idxs] for k, v in trks.items() if len(v) == n}

# ==============================================================================
# PART 3: ANALYSIS & FEATURE EXTRACTION
# ==============================================================================

def extract_features_and_pair(clean_trks, dirty_trks, full_clusters, pitch_ratio=3.0):
    print("--- Pairing Tracks & Extracting Features ---")
    
    # 1. Cluster Prop Lookup
    max_id = np.max(full_clusters['clusterID'])
    id_map = np.full(max_id + 1, -1, dtype=np.int32)
    id_map[full_clusters['clusterID']] = np.arange(len(full_clusters['clusterID']))
    
    def get_props(cid):
        idx = id_map[cid]
        if idx == -1: return None
        return {'tot': full_clusters['sum_ToT'][idx], 'w_col': full_clusters['width_col'][idx], 'w_row': full_clusters['width_row'][idx]}

    # 2. Map Clean
    cluster_to_clean = {}
    n_clean = len(clean_trks['L4_ID'])
    for i in range(n_clean):
        for L in [4,3,2,1]:
            cid = clean_trks[f'L{L}_ID'][i]
            if cid != -1:
                if cid not in cluster_to_clean: cluster_to_clean[cid] = []
                cluster_to_clean[cid].append(i)
                
    data = []
    vector_data = {'layer':[], 'cx':[], 'cy':[], 'dx':[], 'dy':[]}
    z_map = {4: -1.5, 3: -0.5, 2: 0.5, 1: 1.5}
    n_dirty = len(dirty_trks['L4_ID'])
    
    for i in range(n_dirty):
        votes = {}
        d_ids = {L: dirty_trks[f'L{L}_ID'][i] for L in [4,3,2,1]}
        for L, cid in d_ids.items():
            if cid != -1 and cid in cluster_to_clean:
                for c_idx in cluster_to_clean[cid]: votes[c_idx] = votes.get(c_idx, 0) + 1
        
        best_match = -1
        for c_idx, count in votes.items():
            if count == 3: best_match = c_idx; break
        if best_match == -1: continue
        
        swap_L = -1
        for L in [4,3,2,1]:
            if clean_trks[f'L{L}_ID'][best_match] != d_ids[L]: swap_L = L; break
        if swap_L == -1: continue

        # Unbiased Fit
        hx, hy, hz = [], [], []
        track_ts = []
        for L in [4,3,2,1]:
            if clean_trks[f'L{L}_ID'][best_match] != -1:
                track_ts.append(clean_trks[f't{L}'][best_match])
                if L != swap_L:
                    hx.append(clean_trks[f'x{L}'][best_match] * pitch_ratio)
                    hy.append(clean_trks[f'y{L}'][best_match])
                    hz.append(z_map[L])
        if len(hz) < 2: continue
        
        A = np.vstack([hz, np.ones(len(hz))]).T
        mx, cx = np.linalg.lstsq(A, hx, rcond=None)[0]
        my, cy = np.linalg.lstsq(A, hy, rcond=None)[0]
        z_tgt = z_map[swap_L]
        px, py = mx * z_tgt + cx, my * z_tgt + cy
        duration = max(track_ts) - min(track_ts) if track_ts else 0
        
        # Save Vectors
        cx, cy = clean_trks[f'x{swap_L}'][best_match] * pitch_ratio, clean_trks[f'y{swap_L}'][best_match]
        dx, dy = dirty_trks[f'x{swap_L}'][i] * pitch_ratio, dirty_trks[f'y{swap_L}'][i]
        vector_data['layer'].append(swap_L)
        vector_data['cx'].append(cx); vector_data['cy'].append(cy)
        vector_data['dx'].append(dx); vector_data['dy'].append(dy)

        # Extract
        for label, trk_dict, idx in [(0, clean_trks, best_match), (1, dirty_trks, i)]:
            x = trk_dict[f'x{swap_L}'][idx] * pitch_ratio
            y = trk_dict[f'y{swap_L}'][idx]
            res = np.sqrt((x - px)**2 + (y - py)**2)
            chi2 = np.log10(trk_dict['chi2'][idx] + 1e-4)
            props = get_props(trk_dict[f'L{swap_L}_ID'][idx])
            if props:
                tot = props['tot']
                size = props['w_col'] * props['w_row']
                aspect = props['w_row'] / props['w_col'] if props['w_col'] > 0 else 0
                data.append([chi2, res, tot, size, aspect, duration, label])

    cols = ['LogChi2', 'Residual', 'ToT', 'Size', 'Aspect', 'Duration', 'Label']
    return pd.DataFrame(data, columns=cols), vector_data

# ==============================================================================
# PART 4: MODEL BENCHMARKING & VISUALIZATION
# ==============================================================================

def benchmark_models(df):
    print("--- Benchmarking Discriminators ---")
    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'LDA': Pipeline([('sc', StandardScaler()), ('clf', LinearDiscriminantAnalysis())]),
        'QDA': Pipeline([('sc', StandardScaler()), ('clf', QuadraticDiscriminantAnalysis())]),
        'GBDT': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    results = []
    
    plt.figure(figsize=(10, 8))
    plt.plot([0,1], [0,1], 'k--')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresh = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        J = tpr - fpr
        ix = np.argmax(J)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
        plt.scatter(fpr[ix], tpr[ix], s=50, label=f'{name} J={J[ix]:.2f}')
        
        results.append({
            'Model': name, 'AUC': roc_auc, 'Youden_J': J[ix], 
            'TPR': tpr[ix], 'FPR': fpr[ix], 'Obj': model, 
            'X_test': X_test, 'y_test': y_test
        })
        
    plt.title("Model Comparison: ROC Curves")
    plt.xlabel("FPR (Dirty Accepted)")
    plt.ylabel("TPR (Clean Accepted)")
    plt.legend()
    plt.show()
    
    res_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
    print("\n=== BENCHMARK RESULTS ===")
    print(res_df[['Model', 'AUC', 'Youden_J', 'TPR', 'FPR']].to_string(index=False))
    
    return res_df.iloc[0] # Return best row

def plot_best_model_diagnostics(best_row, df):
    model = best_row['Obj']
    name = best_row['Model']
    X_test = best_row['X_test']
    y_test = best_row['y_test']
    
    print(f"\n--- Visualizing Best Model: {name} ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Feature Importance (GBDT only) or Coeffs (LDA)
    ax = axes[0]
    if name == 'GBDT':
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feat_names = X_test.columns
        sns.barplot(x=importances[indices], y=feat_names[indices], ax=ax, palette='viridis')
        ax.set_title("Feature Importance")
        top_f1, top_f2 = feat_names[indices[0]], feat_names[indices[1]]
    else:
        # Fallback for LDA/QDA - just plot distribution of top physics features
        sns.kdeplot(data=df, x='LogChi2', hue='Label', ax=ax, fill=True, palette={0:'blue', 1:'red'})
        ax.set_title("LogChi2 Separation")
        top_f1, top_f2 = 'LogChi2', 'Residual'

    # 2. Decision Boundary on Top 2 Features
    ax = axes[1]
    x_dat = X_test[top_f1]
    y_dat = X_test[top_f2]
    
    x_min, x_max = x_dat.min(), x_dat.max()
    y_min, y_max = y_dat.min(), y_dat.max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Grid for prediction (fill others with mean)
    grid_df = pd.DataFrame(np.zeros((xx.size, len(X_test.columns))), columns=X_test.columns)
    for c in X_test.columns: grid_df[c] = X_test[c].mean()
    grid_df[top_f1] = xx.ravel()
    grid_df[top_f2] = yy.ravel()
    
    Z = model.predict(grid_df).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=ListedColormap(['#AAAAFF', '#FFAAAA']), alpha=0.3)
    ax.scatter(X_test[y_test==0][top_f1], X_test[y_test==0][top_f2], c='blue', s=10, alpha=0.5, label='Winner')
    ax.scatter(X_test[y_test==1][top_f1], X_test[y_test==1][top_f2], c='red', s=10, alpha=0.5, label='Loser')
    ax.set_xlabel(top_f1); ax.set_ylabel(top_f2)
    ax.set_title(f"Decision Boundary ({name})")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_vectors(v_data):
    if len(v_data['layer']) == 0: return
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, L in enumerate([4,3,2,1]):
        ax = axes[i]
        mask = (np.array(v_data['layer']) == L)
        if np.sum(mask) == 0:
            ax.text(0.5, 0.5, f"No Swaps L{L}", ha='center'); continue
            
        cx = np.array(v_data['cx'])[mask]; cy = np.array(v_data['cy'])[mask]
        dx = np.array(v_data['dx'])[mask]; dy = np.array(v_data['dy'])[mask]
        
        if len(cx) > 200:
            idx = np.random.choice(len(cx), 200, replace=False)
            cx, cy, dx, dy = cx[idx], cy[idx], dx[idx], dy[idx]
            
        for j in range(len(cx)): ax.plot([cx[j], dx[j]], [cy[j], dy[j]], c='gray', alpha=0.3)
        ax.scatter(cx, cy, c='blue', s=10, label='Winner' if i==0 else "")
        ax.scatter(dx, dy, c='red', s=10, label='Loser' if i==0 else "")
        ax.set_title(f"Layer {L} Shifts"); ax.axis('equal')
        if i==0: ax.legend()
    plt.tight_layout()
    plt.show()

# ==============================================================================
# PART 5: MASTER ORCHESTRATOR
# ==============================================================================

def find_best_xtalk_discriminator(final_clusters, xtalk_type=2, pitch_ratio=3.0):
    print(f"\n=== XTALK DISCRIMINATOR PIPELINE (Type {xtalk_type}) ===")
    
    # 1. Filter Data
    xt = final_clusters['xtalk_type']
    if np.issubdtype(xt.dtype, np.number): mask = np.isin(xt, [0, xtalk_type])
    else: mask = np.vectorize(lambda x: x in [0, xtalk_type] if isinstance(x, (int, float)) else x[0] in [0, xtalk_type])(xt)
    data = {k: v[mask] for k,v in final_clusters.items()}
    
    # 2. Track & Resolve Conflicts
    raw_trks = tracking_fast(data, pitch_ratio_col=pitch_ratio)
    if not raw_trks: print("No tracks."); return
    
    # Flatten & Resolve (Inline Logic for speed)
    n = len(raw_trks['L4_ID'])
    ids = {L: raw_trks[f'L{L}_ID'] for L in [4,3,2,1]}
    cols = {L: raw_trks[f'x{L}']*pitch_ratio for L in [4,3,2,1]}
    ts = {L: raw_trks[f't{L}'] for L in [4,3,2,1]}
    chi = raw_trks['chi2']
    
    h_tid, h_l, h_c, h_t, h_chi = [], [], [], [], []
    for i in range(n):
        c2 = chi[i]
        for L in [4,3,2,1]:
            if ids[L][i] != -1:
                h_tid.append(i); h_l.append(L); h_c.append(cols[L][i]); h_t.append(ts[L][i]); h_chi.append(c2)
                
    sort = np.argsort(np.array(h_t))
    status = _resolve_conflicts_greedy_strip(
        np.array(h_tid)[sort].astype(np.int32), np.array(h_chi)[sort].astype(np.float32), 
        np.array(h_t)[sort].astype(np.int64), np.array(h_c)[sort].astype(np.float32), 
        np.array(h_l)[sort].astype(np.int32), 
        5, 15.0, n
    )
    
    clean_trks = get_subset(raw_trks, np.where(status==1)[0])
    dirty_trks = get_subset(raw_trks, np.where(status==2)[0])
    print(f"Clean: {len(clean_trks['L4_ID'])}, Dirty: {len(dirty_trks['L4_ID'])}")
    
    # 3. Features & Visuals
    df, v_data = extract_features_and_pair(clean_trks, dirty_trks, final_clusters, pitch_ratio)
    if len(df) == 0: print("No pairs."); return
    
    
    plot_vectors(v_data)
    
    # 4. Benchmark
    best_res = benchmark_models(df)
    
    # 5. Best Model Diagnostics
    
    plot_best_model_diagnostics(best_res, df)

# Usage
# find_best_xtalk_discriminator(final_clusters, xtalk_type=2)