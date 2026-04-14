import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import progress_bar

def _extract_correlated_pairs(layer_data, dx=50, dx2=None):
    """
    Core function to identify correlated pairs in a layer.
    
    Args:
        dx (int): Minimum separation threshold (used if dx2 is None).
        dx2 (int or list/tuple): 
            If int: Only pairs with exactly |dRow| == dx2 are kept.
            If list [min, max]: Only pairs with min <= |dRow| <= max are kept.
            
    Returns:
        total_hits (int): Total number of hits in the layer.
        pairs_df (DataFrame): DataFrame containing details of correlated pairs 
                              (index_1, index_2, TriggerID, dRow, Ratio, ToT_1, ToT_2, ...).
    """
    # Check for optional TS columns
    has_ts = 'TS' in layer_data
    has_ts2 = 'TS2' in layer_data

    data_dict = {
        'TriggerID': layer_data['TriggerID'],
        'Column': layer_data['Column'],
        'Row': layer_data['Row'],
        'ToT': layer_data['ToT']
    }
    if has_ts: data_dict['TS'] = layer_data['TS']
    if has_ts2: data_dict['TS2'] = layer_data['TS2']

    df = pd.DataFrame(data_dict)
    
    total_hits = len(df)
    
    # 1. Quick Filter: Triggers with > 1 hit
    vc = df['TriggerID'].value_counts()
    valid_triggers = vc[vc > 1].index
    
    if len(valid_triggers) == 0:
        return total_hits, pd.DataFrame()

    # 2. Subset & Merge
    # Keep original index to map back to boolean masks later
    df_subset = df[df['TriggerID'].isin(valid_triggers)].reset_index()
    
    # This merge creates a Cartesian product for hits in the same Trigger/Column.
    # It naturally allows 1-to-many correlations (Hit A can match with Hit B AND Hit C).
    merged = pd.merge(df_subset, df_subset, on=['TriggerID', 'Column'], suffixes=('_1', '_2'))
    
    # 3. Filter Unique Spatial Pairs
    # index_1 < index_2 ensures uniqueness of the PAIR (A-B is same as B-A) and avoids self-pairs (A-A).
    merged = merged[merged['index_1'] < merged['index_2']]
    
    if merged.empty:
        return total_hits, pd.DataFrame()
        
    r1 = merged['Row_1'].values.astype(int)
    r2 = merged['Row_2'].values.astype(int)
    merged['dRow'] = np.abs(r1 - r2)
    
    # --- UPDATED LOGIC FOR DX / DX2 ---
    if dx2 is not None:
        # Check if dx2 is iterable (list or tuple) for range filtering
        if isinstance(dx2, (list, tuple, np.ndarray)) and len(dx2) == 2:
            # Range Bound: dx2[0] <= dRow <= dx2[1]
            mask_spatial = (merged['dRow'] >= dx2[0]) & (merged['dRow'] <= dx2[1])
        else:
            # Exact Match: dRow == dx2
            mask_spatial = (merged['dRow'] == dx2)
    else:
        # Default Lower Bound: dRow > dx
        mask_spatial = merged['dRow'] > dx

    pairs_df = merged[mask_spatial].copy()
    
    if pairs_df.empty:
        return total_hits, pd.DataFrame()

    # 4. Calculate Ratios
    t1 = pairs_df['ToT_1'].values.astype(float)
    t2 = pairs_df['ToT_2'].values.astype(float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.minimum(t1, t2) / np.maximum(t1, t2)
        pairs_df['Ratio'] = np.nan_to_num(ratios, nan=0.0)
        
    # Return columns needed for downstream analysis
    cols = ['index_1', 'index_2', 'TriggerID', 'dRow', 'Ratio', 'ToT_1', 'ToT_2']
    if has_ts: cols.extend(['TS_1', 'TS_2'])
    if has_ts2: cols.extend(['TS2_1', 'TS2_2'])
        
    return total_hits, pairs_df[cols]

def _generate_mask_from_pairs(total_hits, pairs_df):
    """Generates a boolean mask of length total_hits where True indicates a correlated hit."""
    mask = np.zeros(total_hits, dtype=bool)
    if not pairs_df.empty:
        mask[pairs_df['index_1']] = True
        mask[pairs_df['index_2']] = True
    return mask

# --- Plotting Helpers (Unchanged) ---

def _calculate_and_plot_series(ax, mask, step, label_suffix=""):
    """Calculates cumulative probability with uncertainty and plots it."""
    cum_total = np.arange(1, len(mask) + 1)
    cum_corr = np.cumsum(mask)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        prob = cum_corr / cum_total
        prob = np.nan_to_num(prob, nan=0.0)
        prob = np.clip(prob, 0, 1)
        # Standard Error
        unc = np.sqrt(prob * (1 - prob) / cum_total)
        unc = np.nan_to_num(unc, nan=0.0)
    
    # Downsample for performance
    indices = np.arange(0, len(prob), step)
    if len(prob) > 0 and indices[-1] != len(prob) - 1:
        indices = np.append(indices, len(prob) - 1)
        
    if len(indices) == 0: return

    x = cum_total[indices]
    y = prob[indices]
    y_err = unc[indices]
    
    final_pct = y[-1] * 100
    line, = ax.plot(x, y, linewidth=2, label=f'{label_suffix} ({final_pct:.2f}%)')
    ax.fill_between(x, y - y_err, y + y_err, color=line.get_color(), alpha=0.2)

def _plot_coincidence_marginal(ax_marg, layers, layer_triggers_map, ref_data, ref_pairs_df, ref_hits_len, step, layer_drows_map=None, row_tol=5):
    """
    Plots the marginal coincidence probability (P(Ln | L_ref)).
    """
    ref_layer = layers[-1]
    target_layers = layers[::-1] # Descending (Top -> Bottom)
    
    # 1. Base Mask: Hits in Ref Layer that are correlated (the condition)
    ref_corr_mask = _generate_mask_from_pairs(ref_hits_len, ref_pairs_df)
    
    # Ensure ref_triggers is a Series for .map() functionality
    ref_triggers_series = pd.Series(ref_data['TriggerID']) 
    
    cum_ref = np.cumsum(ref_corr_mask)
    
    # Check strict dRow map availability. Ignore if row_tol is None.
    has_strict = (layer_drows_map is not None) and (row_tol is not None)
    ref_drows_series = layer_drows_map.get(ref_layer) if has_strict else None

    # Tracks the cumulative condition (L4 & L3 & ...)
    current_mask = ref_corr_mask.copy()
    current_strict_mask = ref_corr_mask.copy() if has_strict else None
    
    for i in range(1, len(target_layers)):
        curr_lid = target_layers[i]
        
        # A. Standard Coincidence (Trigger Set Match)
        next_layer_set = layer_triggers_map.get(curr_lid, set())
        is_in_next = np.isin(ref_triggers_series, list(next_layer_set))
        
        current_mask = current_mask & is_in_next
        
        # B. Strict Coincidence (Optional dRow check)
        if has_strict and ref_drows_series is not None:
            tgt_drows_series = layer_drows_map.get(curr_lid)
            if tgt_drows_series is not None:
                # Map full timeline triggers to dRows
                r_d = ref_triggers_series.map(ref_drows_series)
                t_d = ref_triggers_series.map(tgt_drows_series)
                
                # Check tolerance
                diff = np.abs(r_d - t_d)
                # Ignore NaNs (non-matches)
                is_spatial_match = (diff <= row_tol)
                
                if current_strict_mask is not None:
                    current_strict_mask = current_strict_mask & is_in_next & is_spatial_match

        # Helper to compute prob and plot
        def plot_line(mask, linestyle):
            cum_chain = np.cumsum(mask)
            with np.errstate(divide='ignore', invalid='ignore'):
                prob = cum_chain / cum_ref
                prob = np.nan_to_num(prob, nan=0.0)
                prob = np.clip(prob, 0, 1)
                
            indices = np.arange(0, len(prob), step)
            if len(prob) > 0 and indices[-1] != len(prob) - 1:
                indices = np.append(indices, len(prob) - 1)
            
            x = indices + 1
            y = prob[indices]
            
            label_str = f"L{ref_layer}"
            for l in target_layers[1:i+1]: label_str += f"&{l}"
            
            final = y[-1] * 100 if len(y) > 0 else 0
            label = f"{label_str} ({final:.1f}%)" if linestyle == '-' else f"Strict ({final:.1f}%)"
            
            line, = ax_marg.plot(x, y, linewidth=2, linestyle=linestyle, label=label)
            return line.get_color()

        # Plot Standard
        col = plot_line(current_mask, '-')
        
        # Plot Strict (dotted, same color)
        if current_strict_mask is not None:
            cum_s = np.cumsum(current_strict_mask)
            with np.errstate(divide='ignore', invalid='ignore'):
                prob_s = np.nan_to_num(cum_s / cum_ref)
            
            indices = np.arange(0, len(prob_s), step)
            if len(prob_s)>0 and indices[-1]!=len(prob_s)-1: indices = np.append(indices, len(prob_s)-1)
            
            final_s = prob_s[indices][-1] * 100 if len(indices)>0 else 0
            ax_marg.plot(indices+1, prob_s[indices], linewidth=2, linestyle=':', color=col, 
                         label=f"Strict (dRow±{row_tol}) ({final_s:.1f}%)")

    ax_marg.set_ylabel("Prob")
    ax_marg.legend(loc='center right', fontsize='x-small')
    ax_marg.grid(True, alpha=0.3)
    plt.setp(ax_marg.get_xticklabels(), visible=False)

# --- Main Unified Function ---

def plot_convergence_analysis(data_raw, dx=50, dx2=None, step=100, row_tol=5, ratio_thresh=None):
    """
    Plots the convergence of correlated hit probability vs Total Hits processed.
    
    Args:
        dx2: If set (int or list), overrides dx to filter specific row separations (exact or range).
    """
    layers = np.unique(data_raw['Layer'])
    layers.sort()
    if len(layers) > 4: layers = layers[:4]
    
    layer_meta = {}
    modes = ['Total'] if ratio_thresh is None else ['Low', 'High']
    figs = {}
    axes_map = {}
    
    for mode in modes:
        fig, ax = plt.subplots(figsize=(12, 10))
        div = make_axes_locatable(ax)
        ax_marg = div.append_axes("top", size="35%", pad=0.3, sharex=ax)
        figs[mode] = fig
        axes_map[mode] = (ax, ax_marg)
        layer_meta[mode] = {}

    print(f"Processing Layers (Modes: {modes})...")
    
    for i, lid in enumerate(progress_bar(layers, description="Analyzing")):
        mask_layer = data_raw['Layer'] == lid
        l_data = {k: v[mask_layer] for k, v in data_raw.items()}
        
        # Pass dx2 here
        n_total, pairs_all = _extract_correlated_pairs(l_data, dx=dx, dx2=dx2)
        
        pairs_dict = {}
        if ratio_thresh is None:
            pairs_dict['Total'] = pairs_all
        else:
            if not pairs_all.empty:
                pairs_dict['Low'] = pairs_all[pairs_all['Ratio'] < ratio_thresh]
                pairs_dict['High'] = pairs_all[pairs_all['Ratio'] >= ratio_thresh]
            else:
                pairs_dict['Low'] = pairs_all # empty
                pairs_dict['High'] = pairs_all # empty
        
        for mode in modes:
            ax_main, _ = axes_map[mode]
            curr_pairs = pairs_dict[mode]
            
            mask = _generate_mask_from_pairs(n_total, curr_pairs)
            triggers = set(curr_pairs['TriggerID']) if not curr_pairs.empty else set()
            
            if not curr_pairs.empty:
                drows = curr_pairs.drop_duplicates('TriggerID').set_index('TriggerID')['dRow']
            else:
                drows = pd.Series(dtype=float)
            
            layer_meta[mode][lid] = {
                'triggers': triggers,
                'drows': drows,
                'pairs_df': curr_pairs,
                'total_hits': n_total,
                'ref_data': l_data
            }
            
            _calculate_and_plot_series(ax_main, mask, step, label_suffix=f"Layer {lid}")

    ref_layer = layers[-1]
    
    for mode in modes:
        fig = figs[mode]
        ax_main, ax_marg = axes_map[mode]
        
        ref_meta = layer_meta[mode][ref_layer]
        ref_data_raw = ref_meta['ref_data']
        ref_pairs = ref_meta['pairs_df']
        ref_total = ref_meta['total_hits']
        
        trig_map = {lid: layer_meta[mode][lid]['triggers'] for lid in layers}
        drow_map = {lid: layer_meta[mode][lid]['drows'] for lid in layers}
        
        _plot_coincidence_marginal(ax_marg, layers, trig_map, ref_data_raw, ref_pairs, ref_total, step, 
                                   layer_drows_map=drow_map, row_tol=row_tol)
        
        label_extra = f" (Ratio < {ratio_thresh})" if mode == 'Low' else \
                      f" (Ratio >= {ratio_thresh})" if mode == 'High' else ""
        
        # Adjust title based on dx/dx2
        dx_label = f"dx2={dx2}" if dx2 is not None else f"dx > {dx}"

        ax_main.set_xlabel(f"Total Hits Processed")
        ax_main.set_ylabel("Cumulative Fraction")
        ax_main.set_title(f"Convergence: Correlated Hits{label_extra} ({dx_label})", y=-0.15)
        ax_marg.set_title(f"Coincidence Probability (Given L{ref_layer} Corr{label_extra})")
        
        fig.suptitle(f"Correlation Convergence{label_extra}", fontsize=14)
        ax_main.legend(loc='center right')
        ax_main.grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    plt.show()

def plot_crosstalk_characteristics(data_raw, dx=50, dx2=None, ratio_thresh=0.2):
    """
    Plots statistics of correlated hits to characterize crosstalk (Seed vs Victim).
    Args:
        dx2: If set, overrides dx.
    """
    layers = np.unique(data_raw['Layer'])
    layers.sort()
    if len(layers) > 4: layers = layers[:4]
    
    colors = {'Low': 'blue', 'High': 'red'}
    alphas = {'Low': 0.5, 'High': 0.5}
    labels = {'Low': f'Ratio < {ratio_thresh}', 'High': f'Ratio >= {ratio_thresh}'}

    for i, lid in enumerate(progress_bar(layers, description="Analyzing Layers")):
        mask_layer = data_raw['Layer'] == lid
        l_data = {k: v[mask_layer] for k, v in data_raw.items()}
        
        # Pass dx2 here
        _, pairs_df = _extract_correlated_pairs(l_data, dx=dx, dx2=dx2)
        
        if pairs_df.empty:
            print(f"Layer {lid}: No correlations found.")
            continue
        
        # Calculate stats
        t1 = pairs_df['ToT_1'].values.astype(float)
        t2 = pairs_df['ToT_2'].values.astype(float)
        
        pairs_df['MaxToT'] = np.maximum(t1, t2)
        pairs_df['MinToT'] = np.minimum(t1, t2)
        
        # Calculate TS Diffs safely
        if 'TS_1' in pairs_df.columns:
            ts1 = pairs_df['TS_1'].values.astype(np.int64)
            ts2 = pairs_df['TS_2'].values.astype(np.int64)
            pairs_df['dTS'] = np.abs(ts1 - ts2)
            
        if 'TS2_1' in pairs_df.columns:
            ts2_1 = pairs_df['TS2_1'].values.astype(np.int64)
            ts2_2 = pairs_df['TS2_2'].values.astype(np.int64)
            pairs_df['dTS2'] = np.abs(ts2_1 - ts2_2)
        
        low_df = pairs_df[pairs_df['Ratio'] < ratio_thresh]
        high_df = pairs_df[pairs_df['Ratio'] >= ratio_thresh]
        
        # Helper to extract column
        def get_col(df, c): return df[c].values if (not df.empty and c in df.columns) else np.array([])
        
        layer_stats = {
            'Low': {
                'MaxToT': get_col(low_df, 'MaxToT'),
                'MinToT': get_col(low_df, 'MinToT'),
                'dRow': get_col(low_df, 'dRow'),
                'dTS': get_col(low_df, 'dTS'),
                'dTS2': get_col(low_df, 'dTS2')
            },
            'High': {
                'MaxToT': get_col(high_df, 'MaxToT'),
                'MinToT': get_col(high_df, 'MinToT'),
                'dRow': get_col(high_df, 'dRow'),
                'dTS': get_col(high_df, 'dTS'),
                'dTS2': get_col(high_df, 'dTS2')
            }
        }

        # Create Figure for this Layer (2x3 Grid)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        dx_label = f"dx2={dx2}" if dx2 is not None else f"dx > {dx}"
        fig.suptitle(f"Layer {lid} Crosstalk Characteristics ({dx_label}, Ratio Thresh={ratio_thresh})", fontsize=16)
        
        # 1. Scatter
        ax_scatter = axes[0]
        for grp in ['Low', 'High']:
            max_tot = layer_stats[grp]['MaxToT']
            min_tot = layer_stats[grp]['MinToT']
            if len(max_tot) > 0:
                n_pts = len(max_tot)
                idx = np.random.choice(n_pts, min(n_pts, 5000), replace=False)
                ax_scatter.scatter(max_tot[idx], min_tot[idx], 
                                   alpha=alphas[grp], s=5, label=labels[grp], c=colors[grp])
        ax_scatter.set_title("Pair Hit Size Correlation")
        ax_scatter.set_xlabel("Max ToT (Seed Candidate)")
        ax_scatter.set_ylabel("Min ToT (Victim Candidate)")
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend()

        # 2. Hist Max ToT
        ax_max = axes[1]
        bins_tot = np.linspace(0, 255, 256)
        for grp in ['Low', 'High']:
            vals = layer_stats[grp]['MaxToT']
            if len(vals) > 0:
                ax_max.hist(vals, bins=bins_tot, density=True, histtype='step', linewidth=2, 
                            label=labels[grp], color=colors[grp])
        ax_max.set_title("Distribution of Larger Hit (Seed?)")
        ax_max.set_xlabel("ToT")
        ax_max.set_ylabel("Density")
        ax_max.grid(True, alpha=0.3)
        ax_max.legend()

        # 3. Hist Min ToT
        ax_min = axes[2]
        for grp in ['Low', 'High']:
            vals = layer_stats[grp]['MinToT']
            if len(vals) > 0:
                ax_min.hist(vals, bins=bins_tot, density=True, histtype='step', linewidth=2, 
                            label=labels[grp], color=colors[grp])
        ax_min.set_title("Distribution of Smaller Hit (Victim?)")
        ax_min.set_xlabel("ToT")
        ax_min.set_ylabel("Density")
        ax_min.grid(True, alpha=0.3)
        ax_min.legend()

        # 4. Hist dRow
        ax_dist = axes[3]
        # Adjust bins for dx2 if present
        if dx2 is not None:
             if isinstance(dx2, (list, tuple, np.ndarray)):
                 d_bins = np.linspace(dx2[0], dx2[1] + 1, max(10, int(dx2[1]-dx2[0]) + 1))
             else:
                 d_bins = np.linspace(dx2-2, dx2+3, 6) # Small window around scalar
        else:
            d_bins = np.linspace(dx, 371, 372-dx)

        for grp in ['Low', 'High']:
            vals = layer_stats[grp]['dRow']
            if len(vals) > 0:
                ax_dist.hist(vals, bins=d_bins, density=True, histtype='step', linewidth=2, 
                             label=labels[grp], color=colors[grp])
        ax_dist.set_title("Row Separation Distribution")
        ax_dist.set_xlabel("Abs Row Diff")
        ax_dist.set_ylabel("Density")
        ax_dist.grid(True, alpha=0.3)
        ax_dist.legend()
        
        # 5. Hist dTS
        ax_ts = axes[4]
        for grp in ['Low', 'High']:
            vals = layer_stats[grp]['dTS']
            if len(vals) > 0:
                vmax = np.percentile(vals, 99) if len(vals) > 100 else vals.max()
                if vmax == 0: vmax = 10
                bins_ts = np.linspace(0, vmax, 50)
                ax_ts.hist(vals, bins=bins_ts, density=True, histtype='step', linewidth=2, 
                           label=labels[grp], color=colors[grp])
        ax_ts.set_title("Delta TS Distribution")
        ax_ts.set_xlabel("|TS1 - TS2|")
        ax_ts.set_ylabel("Density")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend()
        
        # 6. Hist dTS2
        ax_ts2 = axes[5]
        for grp in ['Low', 'High']:
            vals = layer_stats[grp]['dTS2']
            if len(vals) > 0:
                vmax = np.percentile(vals, 99) if len(vals) > 100 else vals.max()
                if vmax == 0: vmax = 10
                bins_ts2 = np.linspace(0, vmax, 50)
                ax_ts2.hist(vals, bins=bins_ts2, density=True, histtype='step', linewidth=2, 
                            label=labels[grp], color=colors[grp])
        ax_ts2.set_title("Delta TS2 Distribution")
        ax_ts2.set_xlabel("|TS2_1 - TS2_2|")
        ax_ts2.set_ylabel("Density")
        ax_ts2.grid(True, alpha=0.3)
        ax_ts2.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- Legacy wrappers ---
def plot_correlation_convergence(data_raw, dx=50, dx2=None, step=100, row_tol=5):
    plot_convergence_analysis(data_raw, dx, dx2, step, row_tol, ratio_thresh=None)

def plot_correlation_ratio_convergence(data_raw, dx=50, dx2=None, ratio_thresh=0.2, step=100, row_tol=5):
    plot_convergence_analysis(data_raw, dx, dx2, step, row_tol, ratio_thresh=ratio_thresh)


def plot_correlation_significance(correlated_data, uncorrelated_data, 
                                  dx=0, dx2=None, n_bins=256, ToTnoise_thresh=None, 
                                  vmin=-5, vmax=5, cmap='RdBu_r',
                                  shuffle_uncorrelated=False):
    """
    Plots 4 Z-score significance graphs:
      1. Row vs Row
      2. Row Separation vs ToT Ratio
      3. Row Sum vs ToT Ratio
      4. ToT (Hit A) vs ToT (Hit B) [Symmetric]

    Parameters:
    - dx (int): Minimum separation threshold (used if dx2 is None). Logic: dRow > dx.
    - dx2 (int, optional): Exact separation target. If provided, overrides dx. Logic: dRow == dx2.
    """

    # Helper to convert dict to numpy arrays
    def to_arrays(d):
        return {k: np.asarray(v) for k, v in d.items()}

    sig_data = to_arrays(correlated_data)
    bg_data  = to_arrays(uncorrelated_data)

    layers = np.unique(sig_data['Layer'])
    layers.sort()
    if len(layers) > 4: layers = layers[:4]

    # --- Determine constraint string for titles ---
    if dx2 is not None:
        const_str = f"dRow == {dx2}"
        # Adjust binning for row separation graph to focus on the specific value if needed
        # But usually keeping the full range is safer to see context, or we can just plot points.
        # We will keep standard binning but the data will be localized.
    else:
        const_str = f"dRow > {dx}"

    # --- Setup 4 Figures ---
    fig1, ax1 = plt.subplots(2, 2, figsize=(15, 14))
    fig1.suptitle(f"Graph 1: Row-Row Significance ({const_str})", fontsize=16)
    ax1 = ax1.flatten()

    fig2, ax2 = plt.subplots(2, 2, figsize=(15, 14))
    fig2.suptitle(f"Graph 2: Row Separation vs Ratio ({const_str})", fontsize=16)
    ax2 = ax2.flatten()

    fig3, ax3 = plt.subplots(2, 2, figsize=(15, 14))
    fig3.suptitle(f"Graph 3: Row Sum vs Ratio ({const_str})", fontsize=16)
    ax3 = ax3.flatten()
    
    fig4, ax4 = plt.subplots(2, 2, figsize=(15, 14))
    fig4.suptitle(f"Graph 4: ToT A vs ToT B Significance ({const_str})", fontsize=16)
    ax4 = ax4.flatten()

    # --- Bin Definitions ---
    bins_rows  = np.linspace(0, 372, 372 + 1)
    bins_sep   = np.linspace(0, 372, 372 + 1) # Full range to see where the peak lands
    bins_ratio = np.linspace(0, 1, n_bins + 1)
    bins_sum   = np.linspace(0, 744, 744 + 1)
    
    # Dynamic ToT bins
    max_tot = max(np.max(sig_data['ToT']), np.max(bg_data['ToT']))
    max_tot_plot = int(np.ceil(max_tot / 10.0)) * 10
    if max_tot_plot < 255: max_tot_plot = 255
    bins_tot = np.linspace(0, max_tot_plot, n_bins + 1)

    for i, lid in enumerate(progress_bar(layers, description="Processing Layers")):
        # =========================================================
        # 1. SIGNAL GENERATION (Correlated)
        # =========================================================
        mask_s = sig_data['Layer'] == lid
        s_row = sig_data['Row'][mask_s]
        s_col = sig_data['Column'][mask_s]
        s_tot = sig_data['ToT'][mask_s]
        s_ts  = sig_data['TS'][mask_s]

        # Sort by Column then TS
        sort_idx_s = np.lexsort((s_ts, s_col))
        s_row_s = s_row[sort_idx_s]
        s_col_s = s_col[sort_idx_s]
        s_tot_s = s_tot[sort_idx_s]
        s_ts_s  = s_ts[sort_idx_s]

        group_indicators = np.column_stack((s_col_s, s_ts_s))
        change_mask = np.any(group_indicators[1:] != group_indicators[:-1], axis=1)
        split_indices = np.nonzero(change_mask)[0] + 1
        
        group_starts = np.concatenate(([0], split_indices))
        group_ends   = np.concatenate((split_indices, [len(s_row_s)]))

        s_r1_list, s_r2_list, s_t1_list, s_t2_list = [], [], [], []

        for start, end in zip(group_starts, group_ends):
            n_hits = end - start
            if n_hits < 2: continue 

            r_chunk = s_row_s[start:end]
            t_chunk = s_tot_s[start:end]

            r1_g, r2_g = np.meshgrid(r_chunk, r_chunk)
            mask_tri = np.triu(np.ones((n_hits, n_hits), dtype=bool), k=1)
            
            dr_g = np.abs(r1_g - r2_g)
            
            # --- Apply dx or dx2 filter ---
            if dx2 is not None:
                mask_valid = mask_tri & (dr_g == dx2)
            else:
                mask_valid = mask_tri & (dr_g > dx)

            if not np.any(mask_valid): continue

            t1_g, t2_g = np.meshgrid(t_chunk, t_chunk)

            s_r1_list.append(r1_g[mask_valid])
            s_r2_list.append(r2_g[mask_valid])
            s_t1_list.append(t1_g[mask_valid])
            s_t2_list.append(t2_g[mask_valid])

        if not s_r1_list:
            for ax_list in [ax1, ax2, ax3, ax4]:
                ax_list[i].text(0.5, 0.5, "No Correlations", ha='center')
            continue

        sig_r1 = np.concatenate(s_r1_list)
        sig_r2 = np.concatenate(s_r2_list)
        sig_t1 = np.concatenate(s_t1_list)
        sig_t2 = np.concatenate(s_t2_list)

        # Signal ToT Filter
        if ToTnoise_thresh is not None:
            mask_keep = ~((sig_t1 < ToTnoise_thresh) & (sig_t2 < ToTnoise_thresh))
            sig_r1 = sig_r1[mask_keep]
            sig_r2 = sig_r2[mask_keep]
            sig_t1 = sig_t1[mask_keep]
            sig_t2 = sig_t2[mask_keep]

        if len(sig_r1) == 0: continue

        sig_drow = np.abs(sig_r1 - sig_r2)
        sig_sum  = sig_r1 + sig_r2
        with np.errstate(divide='ignore', invalid='ignore'):
            sig_ratio = np.nan_to_num(np.minimum(sig_t1, sig_t2) / np.maximum(sig_t1, sig_t2), nan=0.0)

        # =========================================================
        # 2. BACKGROUND GENERATION (Uncorrelated)
        # =========================================================
        mask_b = bg_data['Layer'] == lid
        b_row = bg_data['Row'][mask_b]
        b_col = bg_data['Column'][mask_b]
        b_tot = bg_data['ToT'][mask_b]
        
        if shuffle_uncorrelated:
            perm = np.random.permutation(len(b_row))
            b_row = b_row[perm]
            b_tot = b_tot[perm]
        
        sort_idx_b = np.argsort(b_col)
        b_row_s = b_row[sort_idx_b]
        b_col_s = b_col[sort_idx_b]
        b_tot_s = b_tot[sort_idx_b]

        _, b_starts = np.unique(b_col_s, return_index=True)
        b_ends = np.append(b_starts[1:], len(b_row_s))

        n_sig_pairs = len(sig_r1)
        n_bg_target = n_sig_pairs * 5 

        bg_r1_list, bg_r2_list, bg_t1_list, bg_t2_list = [], [], [], []

        for start, end in zip(b_starts, b_ends):
            n_hits = end - start
            if n_hits < 2: continue

            n_samples = int(n_bg_target * (n_hits / len(b_row))) + 10

            idx1 = np.random.randint(start, end, n_samples)
            idx2 = np.random.randint(start, end, n_samples)

            r1_val = b_row_s[idx1]
            r2_val = b_row_s[idx2]
            dr_val = np.abs(r1_val - r2_val)

            # --- Apply dx or dx2 filter (Background) ---
            if dx2 is not None:
                mask_bg_valid = (dr_val == dx2)
            else:
                mask_bg_valid = (dr_val > dx)
                
            if not np.any(mask_bg_valid): continue

            r1_val = r1_val[mask_bg_valid]
            r2_val = r2_val[mask_bg_valid]
            t1_val = b_tot_s[idx1][mask_bg_valid]
            t2_val = b_tot_s[idx2][mask_bg_valid]

            if ToTnoise_thresh is not None:
                mask_noise = (t1_val < ToTnoise_thresh) & (t2_val < ToTnoise_thresh)
                valid_bg = ~mask_noise
                if not np.any(valid_bg): continue
                r1_val = r1_val[valid_bg]
                r2_val = r2_val[valid_bg]
                t1_val = t1_val[valid_bg]
                t2_val = t2_val[valid_bg]

            bg_r1_list.append(r1_val)
            bg_r2_list.append(r2_val)
            bg_t1_list.append(t1_val)
            bg_t2_list.append(t2_val)

        if not bg_r1_list: continue

        bg_r1 = np.concatenate(bg_r1_list)
        bg_r2 = np.concatenate(bg_r2_list)
        bg_t1 = np.concatenate(bg_t1_list)
        bg_t2 = np.concatenate(bg_t2_list)
        
        bg_drow = np.abs(bg_r1 - bg_r2)
        bg_sum  = bg_r1 + bg_r2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            bg_ratio = np.nan_to_num(np.minimum(bg_t1, bg_t2) / np.maximum(bg_t1, bg_t2), nan=0.0)

        # =========================================================
        # 3. PLOTTING
        # =========================================================
        def plot_z_score(ax, sx, sy, bx, by, bins_x, bins_y, xlabel, ylabel, symmetric=False):
            if symmetric:
                s_x_p = np.concatenate([sx, sy])
                s_y_p = np.concatenate([sy, sx])
                b_x_p = np.concatenate([bx, by])
                b_y_p = np.concatenate([by, bx])
            else:
                s_x_p, s_y_p = sx, sy
                b_x_p, b_y_p = bx, by

            H_sig, _, _ = np.histogram2d(s_x_p, s_y_p, bins=[bins_x, bins_y])
            H_bg, _, _  = np.histogram2d(b_x_p, b_y_p, bins=[bins_x, bins_y])

            norm_factor = H_sig.sum() / (H_bg.sum() + 1e-9)
            H_bg_norm = H_bg * norm_factor

            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (H_sig - H_bg_norm) / np.sqrt(H_bg_norm + 1)

            X, Y = np.meshgrid(bins_x, bins_y)
            mesh = ax.pcolormesh(X, Y, Z.T, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Layer {lid} (N_sig={len(sx)})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return mesh

        # Plot 1: Row vs Row (Symmetric)
        mesh1 = plot_z_score(ax1[i], sig_r1, sig_r2, bg_r1, bg_r2, 
                             bins_rows, bins_rows, "Row 1", "Row 2", symmetric=True)
        # Plot 2: Sep vs Ratio
        mesh2 = plot_z_score(ax2[i], sig_drow, sig_ratio, bg_drow, bg_ratio, 
                             bins_sep, bins_ratio, "Row Separation", "ToT Ratio")
        # Plot 3: Sum vs Ratio
        mesh3 = plot_z_score(ax3[i], sig_sum, sig_ratio, bg_sum, bg_ratio, 
                             bins_sum, bins_ratio, "Row Sum", "ToT Ratio")
        # Plot 4: ToT vs ToT (Symmetric)
        mesh4 = plot_z_score(ax4[i], sig_t1, sig_t2, bg_t1, bg_t2, 
                             bins_tot, bins_tot, "ToT Hit A", "ToT Hit B", symmetric=True)

        if i == 1:
            for fig_ref, mesh_ref in zip([fig1, fig2, fig3, fig4], [mesh1, mesh2, mesh3, mesh4]):
                cbar = fig_ref.colorbar(mesh_ref, ax=fig_ref.axes, shrink=0.6, pad=0.05)
                cbar.set_label('Significance (Z-Score)')

    for j in range(i + 1, 4): 
        fig1.delaxes(ax1[j])
        fig2.delaxes(ax2[j])
        fig3.delaxes(ax3[j])
        fig4.delaxes(ax4[j])

    plt.show()
    
    
plot_correlation_significance(data_raw, dx=2, n_bins=256, sig_bins=0, 
                                 ToTnoise_thresh=0, vmin=-10, vmax=10, cmap='RdBu_r')



plot_convergence_analysis(data_raw, dx=15, step=10, row_tol=None, ratio_thresh=0.2)
plot_convergence_analysis(data_raw, dx=15, step=10, row_tol=None, ratio_thresh=None)


from matplotlib.colors import LogNorm, Normalize

def plot_GlobalLayerSignificance(data_raw, num_rows=372, dTS=25, log_scale=False, normalize_sections=False):
    df = pd.DataFrame(data_raw).sort_values('ext_TS')
    layers = [4, 3, 2, 1]
    num_layers = len(layers)

    # Create the large empty super-matrix
    total_dim = num_layers * num_rows
    global_sig = np.zeros((total_dim, total_dim))

    total_time = df['ext_TS'].max() - df['ext_TS'].min()
    if total_time <= 0: total_time = 1

    # Pre-calculate hit counts for all layers to save time
    layer_counts = {l: df[df['Layer'] == l]['Row'].value_counts().reindex(range(num_rows), fill_value=0).values 
                    for l in layers}

    for i, lA in enumerate(layers):
        for j, lB in enumerate(layers):
            # Include diagonal blocks (i == j) for self-correlation
            if i > j: continue 

            dist = abs(lA - lB)
            current_window = max(dTS * dist, 1)

            df_A = df[df['Layer'] == lA]
            df_B = df[df['Layer'] == lB]

            # Perform the efficient windowed join
            merged = pd.merge_asof(
                df_A[['ext_TS', 'Column', 'Row']], 
                df_B[['ext_TS', 'Column', 'Row']], 
                on='ext_TS', by='Column', direction='forward',
                tolerance=current_window, suffixes=('_A', '_B')
            ).dropna(subset=['Row_B'])

            obs_matrix = np.zeros((num_rows, num_rows))
            if not merged.empty:
                rA, rB = merged['Row_A'].values.astype(int), merged['Row_B'].values.astype(int)
                mask = (rA < num_rows) & (rB < num_rows)
                np.add.at(obs_matrix, (rA[mask], rB[mask]), 1)

            # Significance Calculation
            exp_matrix = np.outer(layer_counts[lA], layer_counts[lB]) * (current_window / total_time)
            with np.errstate(divide='ignore', invalid='ignore'):
                sig_sub_matrix = (obs_matrix - exp_matrix) / np.sqrt(exp_matrix)
                sig_sub_matrix = np.nan_to_num(sig_sub_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Optional: Normalize each section independently
            if normalize_sections:
                # Use 99.9th percentile to be robust against outlying noise bins
                robust_max = np.percentile(sig_sub_matrix, 99.999)
                if robust_max > 0:
                    sig_sub_matrix = sig_sub_matrix / robust_max

            # Insert into the global matrix at the correct block location
            global_sig[i*num_rows : (i+1)*num_rows, j*num_rows : (j+1)*num_rows] = sig_sub_matrix

    # Plotting logic
    fig, ax = plt.subplots(figsize=(14, 12))

    # Adjust max scale based on normalization
    m_max = 1.0 if normalize_sections else 100
    norm = LogNorm(vmin=0.01 if normalize_sections else 1, vmax=m_max) if log_scale else Normalize(vmin=0, vmax=m_max)

    im = ax.imshow(global_sig, cmap='gist_stern', norm=norm, origin='upper', extent=[0, total_dim, total_dim, 0])

    # Add dividers between layers
    for p in range(1, num_layers):
        ax.axhline(p * num_rows, color='white', lw=1, ls='--')
        ax.axvline(p * num_rows, color='white', lw=1, ls='--')

    # Label the blocks
    tick_locs = np.arange(num_rows / 2, total_dim, num_rows)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_yticks(tick_locs)
    ax.set_yticklabels([f"Layer {l}" for l in layers])

    title = "Global Significance Matrix (Normalized per Section)" if normalize_sections else "Global Significance Matrix"
    ax.set_title(title, fontsize=16)
    fig.colorbar(im, label='Significance (Relative)' if normalize_sections else 'Significance (Z-score)')

    plt.tight_layout()
    plt.show()
    
plot_GlobalLayerSignificance(data_raw, num_rows=372, dTS=25, log_scale=False, normalize_sections=True)
def compare_CorrelationSignificance(correlated_data, uncorrelated_data, num_rows=372, dTS=25):
    """
    Computes and compares the global significance matrices for two datasets
    to show the statistical effect of correlations.
    """

    def get_global_matrix(data):
        df = pd.DataFrame(data).sort_values('ext_TS')
        layers = [4, 3, 2, 1]
        n_layers = len(layers)
        total_dim = n_layers * num_rows
        matrix = np.zeros((total_dim, total_dim))

        total_time = df['ext_TS'].max() - df['ext_TS'].min() or 1
        layer_counts = {l: df[df['Layer'] == l]['Row'].value_counts().reindex(range(num_rows), fill_value=0).values 
                        for l in layers}

        for i, lA in enumerate(layers):
            for j, lB in enumerate(layers):
                if i > j: continue 

                dist = abs(lA - lB)
                window = max(dTS * dist, 1)

                df_A = df[df['Layer'] == lA]
                df_B = df[df['Layer'] == lB]

                merged = pd.merge_asof(
                    df_A[['ext_TS', 'Column', 'Row']], 
                    df_B[['ext_TS', 'Column', 'Row']], 
                    on='ext_TS', by='Column', direction='forward',
                    tolerance=window, suffixes=('_A', '_B')
                ).dropna(subset=['Row_B'])

                obs = np.zeros((num_rows, num_rows))
                if not merged.empty:
                    rA, rB = merged['Row_A'].values.astype(int), merged['Row_B'].values.astype(int)
                    mask = (rA < num_rows) & (rB < num_rows)
                    np.add.at(obs, (rA[mask], rB[mask]), 1)

                exp = np.outer(layer_counts[lA], layer_counts[lB]) * (window / total_time)
                with np.errstate(divide='ignore', invalid='ignore'):
                    sig = (obs - exp) / np.sqrt(exp)
                    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

                matrix[i*num_rows : (i+1)*num_rows, j*num_rows : (j+1)*num_rows] = sig
        return matrix

    # 1. Generate both matrices
    print("Processing correlated dataset...")
    sig_corr = get_global_matrix(correlated_data)
    print("Processing uncorrelated dataset...")
    sig_uncorr = get_global_matrix(uncorrelated_data)

    # 2. Calculate the Statistical Effect (Difference)
    diff_matrix = sig_corr - sig_uncorr

    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Use SymLogNorm for the difference to see both positive and negative shifts
    norm_diff = SymLogNorm(linthresh=1, linscale=1, vmin=-50, vmax=50, base=10)
    norm_orig = Normalize(vmin=0, vmax=100)

    # Plot Correlated (Signal)
    im1 = axes[0].imshow(sig_corr, cmap='gist_stern', norm=norm_orig, origin='upper')
    axes[0].set_title("Correlated Data Significance ($Z_{signal}$)", fontsize=14)

    # Plot Difference (Statistical Effect)
    im2 = axes[1].imshow(diff_matrix, cmap='RdBu_r', norm=norm_diff, origin='upper')
    axes[1].set_title("Statistical Effect ($\Delta Z = Z_{signal} - Z_{background}$)", fontsize=14)

    # Styling
    for ax in axes:
        for p in range(1, 4):
            ax.axhline(p * num_rows, color='black', lw=0.5, ls='--')
            ax.axvline(p * num_rows, color='black', lw=0.5, ls='--')
        ax.set_xticks(np.arange(num_rows/2, 4*num_rows, num_rows))
        ax.set_xticklabels(['L4', 'L3', 'L2', 'L1'])
        ax.set_yticks(np.arange(num_rows/2, 4*num_rows, num_rows))
        ax.set_yticklabels(['L4', 'L3', 'L2', 'L1'])

    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Z-score')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='$\Delta$ Significance')

    plt.tight_layout()
    plt.show()


compare_CorrelationSignificance(correlated_data, uncorrelated_data, num_rows=372, dTS=25)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compare_TrackQuality_ThreeSets(datasets, dTS=25, radial_tolerance=5):
    """
    Reconstructs tracks for multiple datasets (Signal, Noise, Mix) and overlays their quality metrics.
    """
    
    # --- 1. TRACK RECONSTRUCTION ENGINE ---
    def get_tracks(data, label):
        print(f"Processing {label}...")
        df = pd.DataFrame(data).sort_values('ext_TS')
        df['ext_TS'] = df['ext_TS'].astype(np.int64)
        layers = {l: df[df['Layer'] == l][['Row', 'Column', 'ext_TS', 'ToT']].copy() for l in [4, 3, 2, 1]}

        def step(df_curr, df_next, suffix_curr, suffix_next):
            s = df_curr.copy()
            s['bin'] = s['ext_TS'] // dTS
            e = df_next.copy()
            
            rename_map = {'Row': 'Rn', 'Column': 'Cn', 'ext_TS': 'Tn', 'ToT': f'ToT_{suffix_next}'}
            e = e.rename(columns=rename_map)
            e['bin'] = e['Tn'] // dTS
            
            # Efficient Merge on Time Bins
            m1 = pd.merge(s, e, on='bin')
            m2 = pd.merge(s, e.assign(bin=e['bin']-1), on='bin')
            cand = pd.concat([m1, m2], ignore_index=True)
            
            if cand.empty: return pd.DataFrame()
            
            dt = np.abs(cand['Tn'] - cand['ext_TS'])
            dist = np.sqrt((cand['Rn'] - cand['Row'])**2 + (cand['Cn'] - cand['Column'])**2)
            matches = cand[(dt <= dTS) & (dist <= radial_tolerance)].copy()
            
            # Preserve Start Tip Info if first step
            if 'Row_L4' not in matches.columns:
                matches['Row_L4'] = matches['Row']
                matches['Col_L4'] = matches['Column']
                matches['TS_L4'] = matches['ext_TS']
                matches['ToT_L4'] = matches['ToT'] if 'ToT' in matches.columns else matches[f'ToT_{suffix_curr}']

            # Update Tip
            matches[f'Row_{suffix_next}'] = matches['Rn']
            matches[f'Col_{suffix_next}'] = matches['Cn']
            matches[f'TS_{suffix_next}'] = matches['Tn']
            
            matches = matches.drop(columns=['Row', 'Column', 'ext_TS', 'bin', 'Rn', 'Cn', 'Tn', 'ToT'], errors='ignore')
            matches['Row'] = matches[f'Row_{suffix_next}']
            matches['Column'] = matches[f'Col_{suffix_next}']
            matches['ext_TS'] = matches[f'TS_{suffix_next}']
            return matches

        # Pipeline
        tr = step(layers[4], layers[3], 'L4', 'L3')
        tr = step(tr, layers[2], 'L3', 'L2')
        tr = step(tr, layers[1], 'L2', 'L1')
        
        if tr.empty: return pd.DataFrame()

        # Metrics
        tr['avg_dTS'] = (tr['TS_L1'] - tr['TS_L4']).abs() / 3.0
        tr['avg_ToT'] = (tr['ToT_L4'] + tr['ToT_L3'] + tr['ToT_L2'] + tr['ToT_L1']) / 4.0
        
        z = np.array([4, 3, 2, 1])
        chi2_list = []
        rows = tr[['Row_L4', 'Row_L3', 'Row_L2', 'Row_L1']].values
        cols = tr[['Col_L4', 'Col_L3', 'Col_L2', 'Col_L1']].values
        
        for i in range(len(tr)):
            slope_r, intercept_r, _, _, _ = linregress(z, rows[i])
            slope_c, intercept_c, _, _, _ = linregress(z, cols[i])
            chi2_r = np.sum((rows[i] - (slope_r * z + intercept_r))**2)
            chi2_c = np.sum((cols[i] - (slope_c * z + intercept_c))**2)
            chi2_list.append(chi2_r + chi2_c)
            
        tr['chi2'] = chi2_list
        return tr

    # --- 2. COLLECT DATA ---
    results = {}
    for label, data in datasets.items():
        results[label] = get_tracks(data, label)

    # --- 3. PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'Uncorrelated': 'navy', 'Correlated': 'crimson', 'Mix': 'gray'}
    styles = {'Uncorrelated': '-', 'Correlated': '--', 'Mix': ':'}

    def plot_hist(ax, col, title, xlabel, log_y=False, bin_range=None):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density (Normalized)")
        
        valid_vals = []
        for df in results.values():
            if not df.empty: valid_vals.extend(df[col].values)
        if not valid_vals: return
        
        # Use provided range or auto-calculate
        if bin_range is not None:
            bins = bin_range
        else:
            v_max = np.percentile(valid_vals, 99)
            v_min = np.min(valid_vals)
            bins = np.linspace(v_min, v_max, 50)

        for label, df in results.items():
            if df.empty: continue
            c = colors.get(label, 'black')
            s = styles.get(label, '-')
            weights = np.ones_like(df[col]) / len(df[col])
            ax.hist(df[col], bins=bins, histtype='step', lw=2, label=f"{label}",
                    color=c, ls=s, weights=weights)
            
        if log_y: ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot 1: ToT (Energy)
    # Adjust range 0-255 based on typical ToT max
    plot_hist(axes[0], 'avg_ToT', "Hit Energy Distribution", "Average ToT", bin_range=np.linspace(0, 255, 50))
    
    # Plot 2: Chi2 (Linearity)
    # Log Y is essential here as Correlated tracks often have huge Chi2 tails
    plot_hist(axes[1], 'chi2', "Track Linearity (Quality)", r"Spatial $\chi^2$", log_y=True, bin_range=np.linspace(0, 50, 50))
    
    # Plot 3: dTS (Timing)
    plot_hist(axes[2], 'avg_dTS', "Temporal Stability", "Avg Time Separation (ticks)", log_y=True, bin_range=np.linspace(0, 5, 20))

    plt.tight_layout()
    plt.show()

    # --- 4. STATISTICAL TABLE ---
    print("\nSTATISTICAL COMPARISON TABLE")
    print(f"{'Dataset':<15} | {'N Tracks':<10} | {'Mean ToT':<10} | {'Mean Chi2':<10} | {'Mean dTS':<10} | {'Corr(ToT, Chi2)':<15}")
    print("-" * 80)
    
    for label, df in results.items():
        if df.empty:
            print(f"{label:<15} | {'0':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<15}")
            continue
            
        n = len(df)
        m_tot = df['avg_ToT'].mean()
        m_chi = df['chi2'].mean()
        m_dts = df['avg_dTS'].mean()
        corr = df['avg_ToT'].corr(df['chi2'])
        
        print(f"{label:<15} | {n:<10} | {m_tot:<10.2f} | {m_chi:<10.2f} | {m_dts:<10.2f} | {corr:<15.3f}")

# --- EXECUTION ---
data_map = {
    'Uncorrelated': uncorrelated_data, # Signal
    'Correlated': correlated_data,     # Noise/Crosstalk
    'Mix': data_raw                    # Raw
}

compare_TrackQuality_ThreeSets(data_map, dTS=25, radial_tolerance=5)