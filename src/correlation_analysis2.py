import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import time
from joblib import Parallel, delayed
from correlation_analysis import _create_heatmap
from typing import Dict, List, Optional, Tuple
from itertools import combinations
from matplotlib.colors import LogNorm, Normalize
from utils import progress_bar


def _calculate_average_correlation(
    matrices_list: List[pd.DataFrame],
    is_cross_dataset: bool = False
) -> pd.DataFrame:
    """Averages a list of correlation matrices."""
    if not matrices_list:
        return pd.DataFrame()

    stack_method_kwargs = {'future_stack': True}
    stacked_matrices = [
        m.stack(**stack_method_kwargs) 
        for m in progress_bar(matrices_list, description="Stacking matrices")
    ]
    
    avg_corr_series = pd.concat(stacked_matrices).groupby(level=[0, 1]).mean()
    avg_correlation_matrix = avg_corr_series.unstack()
    
    if not is_cross_dataset:
        all_indices = avg_correlation_matrix.index.union(avg_correlation_matrix.columns).sort_values()
        avg_correlation_matrix = avg_correlation_matrix.reindex(index=all_indices, columns=all_indices)
        np.fill_diagonal(avg_correlation_matrix.values, np.nan)
        avg_correlation_matrix = (avg_correlation_matrix + avg_correlation_matrix.T) / 2  
        
    return avg_correlation_matrix

def _process_column(
    col_id: int, 
    df: pd.DataFrame, 
    group_by: str = 'TriggerID',
    tot_ratio_threshold: float = 1,
    use_covariance: bool = False  
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Processes a single column to find pairs and the correlation matrix.
    Reflects adjustments from layer_correlation.py (ToT ratio filtering, Cosine Sim, No Self-Corr).
    """
    df_col = df[df['Column'] == col_id].copy()
    
    if df_col.shape[0] < 2 or df_col['Row'].nunique() < 2:
        return None, None
    
    # --- Pair Generation Logic (Prioritized for ToT Filtering) ---
    df_col_indexed = df_col.reset_index().rename(columns={'index': 'HitID'})
    
    # Merge on the grouping column
    merged_pairs = df_col_indexed.merge(
        df_col_indexed, 
        on=group_by, 
        suffixes=('_A', '_B')
    )
    
    if group_by == 'TriggerTS':
        merged_pairs['TriggerTS_A'] = merged_pairs['TriggerTS']
        merged_pairs['TriggerTS_B'] = merged_pairs['TriggerTS']
    merged_pairs = merged_pairs[merged_pairs['HitID_A'] != merged_pairs['HitID_B']]
    merged_pairs = merged_pairs[merged_pairs['Row_A'] != merged_pairs['Row_B']]
    
    if merged_pairs.empty:
        return None, None
    tot_a = merged_pairs['ToT_A'].values.astype(float)
    tot_b = merged_pairs['ToT_B'].values.astype(float)
    
    max_tot = np.maximum(tot_a, tot_b)
    min_tot = np.minimum(tot_a, tot_b)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = min_tot / max_tot
        ratios[max_tot == 0] = 0 
        
    merged_pairs['ToT_Ratio'] = ratios
    
    # --- Apply Filter ---
    valid_pairs = merged_pairs[merged_pairs['ToT_Ratio'] <= tot_ratio_threshold].copy()
    
    if valid_pairs.empty:
        return None, None
    pair_counts = valid_pairs.groupby(['Row_A', 'Row_B']).size()
    
    # Reindex to ensure square matrix including 0s for rows that exist but had no valid pairs
    all_rows = df_col['Row'].unique()
    all_rows.sort()
    
    # Create square DataFrame filled with 0s initially
    co_occurrence_matrix = pair_counts.unstack(fill_value=0)
    co_occurrence_matrix = co_occurrence_matrix.reindex(index=all_rows, columns=all_rows, fill_value=0)

    # Normalization: Cosine Similarity
    # M_ij = Count(i, j) / sqrt(Count(i) * Count(j))
    row_hit_counts = df_col['Row'].value_counts()
    
    n_i = row_hit_counts.loc[co_occurrence_matrix.index].values
    n_j = row_hit_counts.loc[co_occurrence_matrix.columns].values
    
    denominator = np.sqrt(np.outer(n_i, n_j))
    
    # Avoid division by zero
    correlation_matrix = pd.DataFrame(
        np.divide(co_occurrence_matrix.values, denominator, where=denominator!=0),
        index=co_occurrence_matrix.index,
        columns=co_occurrence_matrix.columns
    )

    # --- Pairs DataFrame Construction ---
    cols_to_keep = {
        'Row_A': 'seed_row', 'ToT_A': 'seed_tot', 'TriggerTS_A': 'seed_ts',
        'Row_B': 'other_row', 'ToT_B': 'other_tot', 'TriggerTS_B': 'other_ts'
    }
    
    pairs_df_for_col = valid_pairs[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    
    # Memory Optimization (from original function)
    pairs_df_for_col['seed_row'] = pairs_df_for_col['seed_row'].astype(np.int16)
    pairs_df_for_col['other_row'] = pairs_df_for_col['other_row'].astype(np.int16)
    pairs_df_for_col['seed_tot'] = pairs_df_for_col['seed_tot'].astype(np.uint8)
    pairs_df_for_col['other_tot'] = pairs_df_for_col['other_tot'].astype(np.uint8)
    pairs_df_for_col['seed_ts'] = pairs_df_for_col['seed_ts'].astype(np.uint64)
    pairs_df_for_col['other_ts'] = pairs_df_for_col['other_ts'].astype(np.uint64)
    
    return correlation_matrix, pairs_df_for_col



def plot_analysis_for_pixel_pairs(
    data: Dict[str, pd.Series],
    pixel_pairs: List[Tuple[int, int]],
    columns_to_analyze: List[int]
):
    """
    Finds "correlated" (same TriggerID or neighbor) and "uncorrelated" (different
    TriggerID) hits for a specific list of pixel pairs.

    Generates one figure for each pair in pixel_pairs, containing:
    1. ToT Heatmap for "correlated" hits (A, B in same TriggerID).
    2. ToT Heatmap for "uncorrelated" hits (A, B in different TriggerIDs).
    3. Overlayed step distribution of ToT A / ToT B for both groups, with inset.
    4. Also plots an average correlation matrix for the selected columns.

    Args:
        data (Dict[str, pd.Series]): The input data dictionary.
        pixel_pairs (List[Tuple[int, int]]): A list of pixel pairs to analyze, e.g., [(0, 23), (56, 75)].
        columns_to_analyze (List[int]): List of columns to process.
    """
    # 1. --- Data Preparation: Find ALL hits ---
    print("Preparing data...")
    df = pd.DataFrame(data)
    
    # --- NEW: Filter df NOW and get ToT data for final plot ---
    # Filter df to only columns we care about
    df = df[df['Column'].isin(columns_to_analyze)]
    
    # Get all unique pixels from the target pairs
    unique_target_pixels = sorted(list(set(p for pair in pixel_pairs for p in pair)))    
    tot_data_by_pixel = {}
    for pix in unique_target_pixels:
        tot_data_by_pixel[pix] = df[df['Row'] == pix]['ToT'].dropna()

    all_correlated_hits_list = []
    all_uncorrelated_hits_list = []
    all_correlation_matrices = [] 

    # --- Find Group 1: Correlated Hits (Same TriggerID) ---
    print("Finding correlated (same TriggerID) hits...")
    for col_id in progress_bar(columns_to_analyze, description="Finding correlated hits"):
        df_col = df[df['Column'] == col_id] # <-- df is already filtered by column
        if df_col.shape[0] < 2 or df_col['Row'].nunique() < 2:
            continue
            
        # --- NEW: Calculate correlation matrix for this column ---
        hit_matrix = pd.crosstab(df_col['Row'], df_col['TriggerID'])
        correlation_matrix = hit_matrix.T.corr()
        if not correlation_matrix.empty and not correlation_matrix.isna().all().all():
            all_correlation_matrices.append(correlation_matrix)
        # --- END NEW ---
        
        # Gather all pairs of hits within the same trigger
        grouped = df_col.groupby('TriggerID')
        for trigger_id, group in grouped:
            if len(group) >= 2:
                for hit1, hit2 in combinations(group.to_dict('records'), 2):
                    all_correlated_hits_list.extend([
                        {'pix_a': hit1['Row'], 'tot_a': hit1['ToT'], 'pix_b': hit2['Row'], 'tot_b': hit2['ToT'], 'col': col_id, 'trigger_id': trigger_id},
                        {'pix_a': hit2['Row'], 'tot_a': hit2['ToT'], 'pix_b': hit1['Row'], 'tot_b': hit1['ToT'], 'col': col_id, 'trigger_id': trigger_id}
                    ])
    
    all_correlated_hits_df = pd.DataFrame(all_correlated_hits_list)

    # --- NEW: Calculate and plot average correlation matrix ---
    print("\nCalculating and plotting average correlation matrix...")
    if not all_correlation_matrices:
        print("⚠️  No correlation matrices were calculated (insufficient data in columns).")
    else:
        avg_corr_matrix = _calculate_average_correlation(all_correlation_matrices)
        
        plt.figure(figsize=(12, 10))
        # Determine tick skip rate for clarity
        num_ticks = 20 # A reasonable default
        tick_skip_rate = max(1, len(avg_corr_matrix.index) // num_ticks)
        
        ax = sns.heatmap(
            avg_corr_matrix, 
            cmap='viridis', 
            annot=False, 
            cbar_kws={'label': 'Average Pearson Correlation Coefficient'}, 
            xticklabels=tick_skip_rate, 
            yticklabels=tick_skip_rate
        )
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        col_str = ", ".join(map(str, columns_to_analyze))
        plt.title(f'Average Pixel Correlation for Columns [{col_str}]')
        plt.xlabel('Pixel Row')
        plt.ylabel('Pixel Row')
        plt.tight_layout()
        plt.show()
    # --- END NEW ---

    # --- Find Group 2: Uncorrelated Hits (Different TriggerID) ---
    print("\nFinding uncorrelated (different TriggerID) hits...")
    # Create a set for faster lookup
    target_pairs_set = set()
    for p_a, p_b in pixel_pairs:
        target_pairs_set.add(tuple(sorted((p_a, p_b))))

    for col_id in progress_bar(columns_to_analyze, description="Finding uncorrelated hits"):
        df_col = df[df['Column'] == col_id] # <-- df is already filtered by column
        
        for pix_a, pix_b in target_pairs_set:
            hits_a = df_col[df_col['Row'] == pix_a][['TriggerID', 'ToT']].rename(columns={'ToT': 'tot_a'})
            hits_b = df_col[df_col['Row'] == pix_b][['TriggerID', 'ToT']].rename(columns={'ToT': 'tot_b'})
            
            if hits_a.empty or hits_b.empty:
                continue
                
            # Perform cross join
            hits_a['key'] = 1
            hits_b['key'] = 1
            merged = pd.merge(hits_a, hits_b, on='key', suffixes=('_a', '_b'))

            uncorr_hits_col = merged[merged['TriggerID_a'] != merged['TriggerID_b']]
            if not uncorr_hits_col.empty:
                # --- NEW: Combined assign and append to remove intermediate variable ---
                all_uncorrelated_hits_list.append(
                    uncorr_hits_col.assign(
                        pix_a = pix_a,
                        pix_b = pix_b,
                        col = col_id
                    )[['pix_a', 'tot_a', 'pix_b', 'tot_b', 'col', 'TriggerID_a', 'TriggerID_b']]
                )

    if not all_uncorrelated_hits_list and not all_correlated_hits_list:
         print("❌ No correlated or uncorrelated hits found.")
         return
         
    all_uncorrelated_hits_df = pd.DataFrame()
    if all_uncorrelated_hits_list:
        all_uncorrelated_hits_df = pd.concat(all_uncorrelated_hits_list, ignore_index=True)

    # --- NEW: Split 'different-TriggerID' hits into 'neighbor' and 'truly-uncorrelated' ---
    print(f"\nFound {len(all_correlated_hits_df)} total same-TriggerID hit pairs (diff=0).")
    print(f"Found {len(all_uncorrelated_hits_df)} total different-TriggerID hit pairs (diff!=0).")
    
    neighbor_hits_df = pd.DataFrame()
    uncorrelated_gt1_df = pd.DataFrame()

    if not all_uncorrelated_hits_df.empty:
        print("Splitting different-TriggerID hits into neighbor (diff=1) and uncorrelated (diff>1)...")
        neighbor_hits_df = all_uncorrelated_hits_df[
            (all_uncorrelated_hits_df['TriggerID_a'] - all_uncorrelated_hits_df['TriggerID_b']).abs() == 1
        ].copy()
        
        uncorrelated_gt1_df = all_uncorrelated_hits_df[
            (all_uncorrelated_hits_df['TriggerID_a'] - all_uncorrelated_hits_df['TriggerID_b']).abs() > 1
        ].copy()
        
        print(f"Found {len(neighbor_hits_df)} neighbor-TriggerID hit pairs (diff=1).")
        print(f"Found {len(uncorrelated_gt1_df)} truly-uncorrelated hit pairs (diff>1).")
    else:
        print("No different-TriggerID hits found to split.")


    # 4. --- Plot Analysis for Each Target Pair ---
    print(f"\nGenerating plots for {len(pixel_pairs)} target pair(s)...")
    for pix_a, pix_b in progress_bar(pixel_pairs, description="Plotting pairs"):
        
        # --- Get Plot 1 Data (Correlated) ---
        
        # Group 1A: Same TriggerID (diff=0)
        pair_corr_0_df = all_correlated_hits_df[
            ((all_correlated_hits_df['pix_a'] == pix_a) & (all_correlated_hits_df['pix_b'] == pix_b)) |
            ((all_correlated_hits_df['pix_a'] == pix_b) & (all_correlated_hits_df['pix_b'] == pix_a))
        ]
        
        # Group 1B: Neighbor TriggerID (diff=1)
        pair_corr_1_df = neighbor_hits_df[
            ((neighbor_hits_df['pix_a'] == pix_a) & (neighbor_hits_df['pix_b'] == pix_b)) |
            ((neighbor_hits_df['pix_a'] == pix_b) & (neighbor_hits_df['pix_b'] == pix_a))
        ]

        # Standardize Group 1A
        plot_df_corr_0 = pd.DataFrame()
        if not pair_corr_0_df.empty:
            plot_df_corr_0 = pair_corr_0_df.copy()
            needs_swap_0 = plot_df_corr_0['pix_a'] != pix_a
            plot_df_corr_0.loc[needs_swap_0, ['tot_a', 'tot_b']] = plot_df_corr_0.loc[needs_swap_0, ['tot_b', 'tot_a']].values

        # Standardize Group 1B
        plot_df_corr_1 = pd.DataFrame()
        if not pair_corr_1_df.empty:
            plot_df_corr_1 = pair_corr_1_df.copy()
            needs_swap_1 = plot_df_corr_1['pix_a'] != pix_a
            plot_df_corr_1.loc[needs_swap_1, ['tot_a', 'tot_b']] = plot_df_corr_1.loc[needs_swap_1, ['tot_b', 'tot_a']].values

        # Combine standardized correlated dataframes (diff=0 and diff=1)
        plot_df_corr = pd.concat([plot_df_corr_0, plot_df_corr_1], ignore_index=True)
        
        # --- NEW: Calculate ToT A / ToT B ratio ---
        correlated_ratio_specific = pd.Series(dtype=float)
        if not plot_df_corr.empty:
            with np.errstate(divide='ignore', invalid='ignore'):
                correlated_ratio_specific = plot_df_corr['tot_a'] / plot_df_corr['tot_b']
            correlated_ratio_specific = correlated_ratio_specific.replace([np.inf, -np.inf], np.nan).dropna()
        # --- END NEW ---

        # --- Get Plot 2 Data (Uncorrelated, diff > 1) ---
        pair_uncorrelated_df = pd.DataFrame() # This is the final df for the plot
        uncorrelated_ratio_specific = pd.Series(dtype=float)
        
        pair_uncorr_gt1_df = uncorrelated_gt1_df[
            ((uncorrelated_gt1_df['pix_a'] == pix_a) & (uncorrelated_gt1_df['pix_b'] == pix_b)) |
            ((uncorrelated_gt1_df['pix_a'] == pix_b) & (uncorrelated_gt1_df['pix_b'] == pix_a))
        ]
        
        if not pair_uncorr_gt1_df.empty:
            # Standardize this one
            pair_uncorrelated_df = pair_uncorr_gt1_df.copy()
            needs_swap_uncorr = pair_uncorrelated_df['pix_a'] != pix_a
            pair_uncorrelated_df.loc[needs_swap_uncorr, ['tot_a', 'tot_b']] = pair_uncorrelated_df.loc[needs_swap_uncorr, ['tot_b', 'tot_a']].values
            
            # --- NEW: Calculate ToT A / ToT B ratio ---
            with np.errstate(divide='ignore', invalid='ignore'):
                uncorrelated_ratio_specific = pair_uncorrelated_df['tot_a'] / pair_uncorrelated_df['tot_b']
            uncorrelated_ratio_specific = uncorrelated_ratio_specific.replace([np.inf, -np.inf], np.nan).dropna()
            # --- END NEW ---


        if plot_df_corr.empty and pair_uncorrelated_df.empty:
            print(f"  -> Skipping pair ({pix_a}, {pix_b}): No correlated or uncorrelated hits found.")
            continue
            
        print(f"  -> Plotting for pair ({pix_a}, {pix_b}): {len(plot_df_corr)} correlated hits, {len(pair_uncorrelated_df)} uncorrelated hits.")

        fig, axes = plt.subplots(2, 3, figsize=(21, 12)) 
        fig.suptitle(f'Analysis for Pixel Pair ({pix_a}, {pix_b})', fontsize=16, y=0.97) # Adjusted y

        # Plot 1: Heatmap (Correlated)
        ax1 = axes[0, 0] # Changed to [0, 0]
        if not plot_df_corr.empty:
            try:
                h1, _, _, im1 = ax1.hist2d(
                    plot_df_corr['tot_a'], 
                    plot_df_corr['tot_b'], 
                    bins=256, # <-- CHANGED
                    range=[[-0.5, 255.5], [-0.5, 255.5]], # <-- CHANGED
                    cmap='plasma', 
                    norm=LogNorm()
                )
                fig.colorbar(im1, ax=ax1, label='Frequency (Log Scale)', fraction=0.046, pad=0.04)
            except ValueError as e:
                ax1.text(0.5, 0.5, f"Plotting error: {e}", ha='center', va='center')
        else:
             ax1.text(0.5, 0.5, "No correlated hits found", ha='center', va='center', fontsize=9)
        # --- NEW: Added hit count (N) to title ---
        ax1.set_title(f'Correlated Hits (Trigger Diff <= 1)\nN = {len(plot_df_corr)}')
        ax1.set_xlabel(f'Pixel {pix_a} ToT')
        ax1.set_ylabel(f'Pixel {pix_b} ToT')
        ax1.set_aspect('equal', adjustable='box')

        # Plot 2: Heatmap (Uncorrelated)
        ax2 = axes[0, 1] # Changed to [0, 1]
        if not pair_uncorrelated_df.empty: 
            try:
                h2, _, _, im2 = ax2.hist2d(
                    pair_uncorrelated_df['tot_a'],  # ToT of pixel A
                    pair_uncorrelated_df['tot_b'],  # ToT of pixel B
                    bins=256, # <-- CHANGED
                    range=[[-0.5, 255.5], [-0.5, 255.5]], # <-- CHANGED
                    cmap='magma', 
                    norm=LogNorm()
                )
                fig.colorbar(im2, ax=ax2, label='Frequency (Log Scale)', fraction=0.046, pad=0.04)
            except ValueError as e:
                ax2.text(0.5, 0.5, f"Plotting error: {e}", ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, "No uncorrelated hits found", ha='center', va='center', fontsize=9)
        # --- NEW: Added hit count (N) to title ---
        ax2.set_title(f'Uncorrelated Hits (Trigger Diff > 1)\nN = {len(pair_uncorrelated_df)}')
        ax2.set_xlabel(f'Pixel {pix_a} ToT') 
        ax2.set_ylabel(f'Pixel {pix_b} ToT') 
        ax2.set_aspect('equal', adjustable='box')

        # --- Plot 3: Step Plot (Overlay) ---
        ax3 = axes[0, 2] # Changed to [0, 2]
        
        main_bins = 256
        main_range = [0, 5] # Main plot range
        
        has_data_ax3 = False
        if not correlated_ratio_specific.empty:
            ax3.hist(correlated_ratio_specific, bins=main_bins, range=main_range, histtype='step', lw=2, label=f'Correlated (N={len(correlated_ratio_specific)})', density=True)
            has_data_ax3 = True
        if not uncorrelated_ratio_specific.empty: 
            ax3.hist(uncorrelated_ratio_specific, bins=main_bins, range=main_range, histtype='step', lw=2, label=f'Uncorrelated (N={len(uncorrelated_ratio_specific)})', density=True, linestyle='--') 
            has_data_ax3 = True
        
        ax3.set_title('Normalized ToT Ratio Distribution')
        ax3.set_xlabel(f'ToT (Pixel {pix_a}) / ToT (Pixel {pix_b})') # Updated label
        ax3.set_ylabel('Normalized Frequency')
        ax3.grid(True, linestyle='--', alpha=0.6)
        if has_data_ax3: 
            ax3.legend()

            # --- NEW: Add inset plot ---
            axins = ax3.inset_axes([0.35, 0.35, 0.6, 0.6])
            inset_bins = 64
            inset_range = [0, 1]
            
            if not correlated_ratio_specific.empty:
                axins.hist(correlated_ratio_specific, bins=inset_bins, range=inset_range, histtype='step', lw=1.5, density=True)
            if not uncorrelated_ratio_specific.empty:    
                axins.hist(uncorrelated_ratio_specific, bins=inset_bins, range=inset_range, histtype='step', lw=1.5, density=True, linestyle='--')
                
            axins.set_xlim(inset_range)
            axins.set_title('Zoom [0, 1]', fontsize=10)
            axins.tick_params(axis='both', which='major', labelsize=8)
            axins.grid(True, linestyle=':', alpha=0.7)
            ax3.indicate_inset_zoom(axins, edgecolor="black")
            # --- END NEW ---
        else:
            ax3.text(0.5, 0.5, "No ratio data found", ha='center', va='center', fontsize=9)


        # --- Plot 4: Scatter Plot (Correlated) ---
        ax4 = axes[1, 0] # New plot
        if not plot_df_corr.empty:
            # --- Get min/max ToT ---
            x_data = plot_df_corr[['tot_a', 'tot_b']].max(axis=1)
            y_data = plot_df_corr[['tot_a', 'tot_b']].min(axis=1)
            
            # --- NEW: Changed to hist2d (heatmap) ---
            try:
                # Use 256 bins on x-axis (0-255), and 15 bins on y-axis (0-15)
                h4, _, _, im4 = ax4.hist2d(
                    x_data, 
                    y_data, 
                    bins=(256, 15), # <-- CHANGED x-bins
                    range=[[-0.5, 255.5], [0, 15]], # <-- CHANGED x-range
                    cmap='viridis', 
                    norm=LogNorm()
                )
                fig.colorbar(im4, ax=ax4, label='Frequency (Log Scale)', fraction=0.046, pad=0.04)
            except ValueError as e:
                ax4.text(0.5, 0.5, f"Plotting error: {e}", ha='center', va='center')
            
            ax4.set_title('Correlated Hits (Heatmap - Zoomed Y)')
            ax4.set_xlabel('Max(ToT_A, ToT_B)')
            ax4.set_ylabel('Min(ToT_A, ToT_B)')
            ax4.set_ylim(0, 15) # Keep Y limit
            ax4.set_xlim(-0.5, 255.5) # Set X limit to full ToT range
        else:
            ax4.text(0.5, 0.5, "No correlated hits found", ha='center', va='center', fontsize=9)
            
        # --- Plot 5: Step Distribution (Absolute ToT Difference) ---
        ax5 = axes[1, 1] # New plot
        
        has_corr = False
        if not plot_df_corr.empty:
            # "binned data from the scatter plot" -> "reflected" -> abs()
            tot_diff_corr = (plot_df_corr['tot_a'] - plot_df_corr['tot_b']).abs()
            if not tot_diff_corr.empty:
                # --- NEW: Changed bins to 256 and range to 0-255.5 ---
                ax5.hist(tot_diff_corr, bins=256, range=[0, 255.5], histtype='step', lw=2, label=f'Correlated (N={len(tot_diff_corr)})', density=True)
                has_corr = True

        has_uncorr = False
        if not pair_uncorrelated_df.empty:
            # "uncorrelated data (after reflected in y=x)" -> abs()
            tot_diff_uncorr = (pair_uncorrelated_df['tot_a'] - pair_uncorrelated_df['tot_b']).abs()
            if not tot_diff_uncorr.empty:
                # --- NEW: Changed bins to 256 and range to 0-255.5 ---
                ax5.hist(tot_diff_uncorr, bins=256, range=[0, 255.5], histtype='step', lw=2, label=f'Uncorrelated (N={len(tot_diff_uncorr)})', density=True, linestyle='--')
                has_uncorr = True

        if has_corr or has_uncorr:
            ax5.set_title('Absolute ToT Difference |A-B|')
            ax5.set_xlabel(f'|ToT (Pixel {pix_a}) - ToT (Pixel {pix_b})|')
            ax5.set_ylabel('Normalized Frequency')
            ax5.grid(True, linestyle='--', alpha=0.6)
            ax5.legend()
            ax5.set_xlim(0, 255.5) # Match the range
        else:
            ax5.text(0.5, 0.5, "No data for diff plot", ha='center', va='center', fontsize=9)

        # --- Plot 6: ToT Spectrum for this Pair ---
        ax6 = axes[1, 2] # Use the 6th slot
        
        plot_bins_ax6 = 256
        plot_range_ax6 = [-0.5, 255.5]
        
        has_data_ax6 = False
        
        # Plot ToT for pix_a
        if pix_a in tot_data_by_pixel and not tot_data_by_pixel[pix_a].empty:
            ax6.hist(
                tot_data_by_pixel[pix_a], 
                bins=plot_bins_ax6, 
                range=plot_range_ax6, 
                histtype='step', 
                lw=2, 
                label=f'Pixel {pix_a} (N={len(tot_data_by_pixel[pix_a])})'
            )
            has_data_ax6 = True

        # Plot ToT for pix_b, *only if it's a different pixel*
        if pix_a != pix_b:
            if pix_b in tot_data_by_pixel and not tot_data_by_pixel[pix_b].empty:
                ax6.hist(
                    tot_data_by_pixel[pix_b], 
                    bins=plot_bins_ax6, 
                    range=plot_range_ax6, 
                    histtype='step', 
                    lw=2, 
                    label=f'Pixel {pix_b} (N={len(tot_data_by_pixel[pix_b])})'
                )
                has_data_ax6 = True
        
        if has_data_ax6:
            ax6.set_title('Overall ToT Spectrum for Pair')
            ax6.set_xlabel('ToT')
            ax6.set_ylabel('Frequency')
            ax6.legend()
            ax6.grid(True, linestyle='--', alpha=0.6)
            ax6.set_xlim(plot_range_ax6)
            ax6.set_yscale('log')
        else:
            ax6.text(0.5, 0.5, "No ToT data found for pixels", ha='center', va='center', fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjusted tight_layout
        plt.show()


    print("\n✅ Analysis complete.")


import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic_2d # <-- NEW IMPORT

def plot_pixel_pair_correlation_analysis(
    data: Dict[str, pd.Series],
    columns_to_analyze: Optional[List[int]] = None,
    filter_tot_low_threshold: Optional[float] = None,
    filter_tot_high_threshold: Optional[float] = None,
    filter_disp_gate: Optional[Tuple[float, float]] = None,
    min_abs_correlation: Optional[float] = None,
    filter_tot_zero: bool = False,
    uncorrelated_background_mode: bool = False,
    return_heatmap_data: bool = False,
    dataset_description: str = "",
    use_covariance: bool = False,
    log_y_disp_plots: bool = False # <-- ADDED NEW PARAMETER
):
    """
    First, plots the average pixel-hit-correlation matrix.
    Second, finds hit pairs and generates detailed analysis plots for them.
    [MEMORY OPTIMIZED & TIMED, with Fig 3 (counts) and Fig 4 (complex)]

    Args:
        data (Dict[str, pd.Series]): The input data dictionary.
        columns_to_analyze (Optional[List[int]], optional): Columns to analyze. If None, all are used.
        filter_tot_low_threshold (Optional[float], optional): If set, removes pairs where both
            seed_tot and other_tot are BELOW this value. Defaults to None.
        filter_tot_high_threshold (Optional[float], optional): If set, removes pairs where EITHER
            seed_tot OR other_tot is ABOVE this value. Defaults to None.
        filter_disp_gate (Optional[Tuple[float, float]], optional): If set, e.g., (-30, 30),
            removes all pairs where the displacement is BETWEEN these two values.
        min_abs_correlation (Optional[float], optional): If set (e.g., 0.01),
            removes all hit pairs from the analysis where the *absolute value*
            of the Pearson correlation for that pixel pair (row_A, row_B)
            is BELOW this value. This filter is applied *before* concatenation.
            Also plots a comparison heatmap of the filtered correlation matrix.
            Defaults to None.
        filter_tot_zero (bool, optional): If True, removes all pairs where
            seed_tot == 0 or other_tot == 0 before any other analysis.
        uncorrelated_background_mode (bool, optional): If True, shuffles TriggerIDs
            to generate an uncorrelated background dataset for analysis.
        return_heatmap_data (bool, optional): If True, plots are shown AND
            (h_data, x_edges, y_edges) from the ToT-filtered heatmap are returned.
        dataset_description (str, optional): A custom subtitle to add to all figures.
        use_covariance (bool, optional): If True, plot covariance instead of correlation.
        log_y_disp_plots (bool, optional): If True, uses a logarithmic scale for the
            Y-axis of displacement-related joint plots (e.g., Fig 3). Defaults to False.
    """
    # 1. --- Data Preparation and Time Calculation ---
    t_func_start = time.perf_counter()
    print("Preparing data...")
    df = pd.DataFrame(data)

    if 'TriggerTS' not in df.columns:
        print("❌ 'TriggerTS' column not found in data. Cannot calculate frequencies.")
        return

    ts_to_seconds = 25e-9
    min_ts = df['TriggerTS'].min()
    max_ts = df['TriggerTS'].max()
    total_duration_ts = max_ts - min_ts

    if total_duration_ts <= 0:
        print("⚠️ Warning: Could not determine valid time duration (total_duration_ts <= 0). Frequencies will be 0.")
        total_duration_sec = 0.0
    else:
        total_duration_sec = total_duration_ts * ts_to_seconds
        print(f"Total experiment duration: {total_duration_sec:.2f} seconds ({total_duration_ts} TS)")

    if columns_to_analyze is None:
        columns_to_analyze = sorted(df['Column'].unique())

    if uncorrelated_background_mode:
        print("\n--- RUNNING IN UNCORRELATED BACKGROUND MODE ---")
        print("Shuffling TriggerIDs to create uncorrelated background dataset...")
        df_to_process = df.copy(deep=False)
        df_to_process['TriggerID'] = np.random.permutation(df['TriggerID'].values)
        analysis_mode_title = "Uncorrelated (mock) data"
        print("Shuffling complete.")
    else:
        print("\n--- RUNNING IN CORRELATED (IN-TRIGGER) PAIR MODE ---")
        df_to_process = df
        analysis_mode_title = "Correlated In-Trigger"

    print(f"\nProcessing {len(columns_to_analyze)} columns in parallel (n_jobs=-1)...")

    t_parallel_start = time.perf_counter()
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_process_column)(
            col_id=col_id, 
            df=df_to_process, 
            group_by='TriggerID',          
            use_covariance=use_covariance  
        )
        for col_id in progress_bar(columns_to_analyze, description="Processing columns")
    )
    
    t_parallel_end = time.perf_counter()

    all_correlation_matrices = []
    all_pairs_dfs = []

    # --- Statistics for correlation filtering ---
    total_pairs_before_corr_filter = 0
    total_pairs_after_corr_filter = 0
    # ---

    print("\nAggregating and filtering parallel results...")
    for corr_matrix, pairs_df_for_col in progress_bar(results, description="Aggregating results"):

        if corr_matrix is not None:
            all_correlation_matrices.append(corr_matrix)

        if pairs_df_for_col is not None and not pairs_df_for_col.empty:

            total_pairs_before_corr_filter += len(pairs_df_for_col) # Add to stats

            # --- Apply Minimum Absolute Correlation Filter ---
            filtered_pairs_df = pairs_df_for_col # Default to original

            if min_abs_correlation is not None and not use_covariance:
                if corr_matrix is None:
                    print("⚠️ Warning: Cannot apply correlation filter for a column chunk, corr_matrix is None.")
                else:
                    try:
                        # 1. Get row indices
                        seed_rows = pairs_df_for_col['seed_row'].values
                        other_rows = pairs_df_for_col['other_row'].values

                        # 2. Convert matrix to indexable numpy array
                        matrix_to_index = corr_matrix
                        if hasattr(matrix_to_index, "toarray"):
                            matrix_to_index = matrix_to_index.toarray()
                        elif isinstance(matrix_to_index, pd.DataFrame):
                            matrix_to_index = matrix_to_index.to_numpy()

                        # 3. Check for out-of-bounds indices *before* indexing
                        matrix_shape = matrix_to_index.shape
                        max_valid_index_rows = matrix_shape[0] - 1
                        max_valid_index_cols = matrix_shape[1] - 1

                        if seed_rows.any() and other_rows.any():
                            max_seed = seed_rows.max()
                            max_other = other_rows.max()

                            if max_seed > max_valid_index_rows or max_other > max_valid_index_cols:
                                print(f"⚠️ Warning: Data-Matrix shape mismatch. Skipping filter for this chunk.")
                                print(f"    (Matrix shape: {matrix_shape}, Max requested index: ({max_seed}, {max_other}))")

                            else:
                                # 4. Indices are valid, proceed with filtering
                                pair_correlations_abs = np.abs(matrix_to_index[seed_rows, other_rows])

                                keep_mask = pair_correlations_abs >= min_abs_correlation

                                filtered_pairs_df = pairs_df_for_col[keep_mask]

                    except IndexError as ie:
                                print(f"⚠️ Warning: Unexpected IndexError during correlation filtering. Skipping filter for this chunk. Error: {ie}")
                    except Exception as e:
                                print(f"⚠️ Warning: Error during correlation filtering: {e} (Type: {type(e)}). Skipping filter for this chunk.")
                                print(f"    (Debug info: corr_matrix type was {type(corr_matrix)})")

            if not filtered_pairs_df.empty:
                all_pairs_dfs.append(filtered_pairs_df)

            total_pairs_after_corr_filter += len(filtered_pairs_df) # Add to stats

    del results

    # --- Print filter statistics ---
    if min_abs_correlation is not None:
        print("\n--- Minimum Correlation Filter Stats ---")
        print(f"  Applied threshold: abs(R) >= {min_abs_correlation}")
        if total_pairs_before_corr_filter > 0:
            removed_count = total_pairs_before_corr_filter - total_pairs_after_corr_filter
            percent_removed = (removed_count / total_pairs_before_corr_filter * 100)
            print(f"  Pairs before filter: {total_pairs_before_corr_filter}")
            print(f"  Pairs after filter:  {total_pairs_after_corr_filter} ({removed_count} removed, {percent_removed:.2f}%)")
        else:
            print("  No pairs found to filter.")
    # ---

    # --- Plot Average Pixel Hit Correlation Matrix (Figure 0) ---
    t_corr_avg_start = time.perf_counter()
    if not all_correlation_matrices:
        print("\nNo correlation matrices were generated to average.")
        t_corr_avg_end = time.perf_counter()
    else:
        print(f"\n--- Averaging {len(all_correlation_matrices)} column correlation matrices ---")
        avg_correlation_matrix = _calculate_average_correlation(all_correlation_matrices)
        del all_correlation_matrices
        t_corr_avg_end = time.perf_counter()

        print("Plotting the average correlation matrix (Linear vs. Log)...")

        # --- MODIFIED BLOCK: Create 1x2 or 2x2 plot grid ---
        if min_abs_correlation is not None:
            # Create 2x2 grid for Original and Filtered
            fig_corr, ((ax_lin, ax_log), (ax_lin_filt, ax_log_filt)) = plt.subplots(
                2, 2, figsize=(22, 20), sharex=True, sharey=True
            )
            print(f"Also plotting filtered matrix (threshold: {min_abs_correlation})...")

            # Create the filtered matrix for plotting
            # Handle both pandas and numpy inputs
            if isinstance(avg_correlation_matrix, pd.DataFrame):
                filtered_corr_matrix = avg_correlation_matrix.where(
                    np.abs(avg_correlation_matrix) >= min_abs_correlation
                )
            else:
                # Assuming numpy array
                filtered_corr_matrix = np.where(
                    np.abs(avg_correlation_matrix) >= min_abs_correlation,
                    avg_correlation_matrix,
                    np.nan
                )

            # Set titles for comparison
            title_lin = "Original (Linear z-Scale)"
            title_log = "Original (Log z-Scale)"

        else:
            # Original behavior: 1x2 grid
            fig_corr, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
            title_lin = "Linear z-Scale"
            title_log = "Log z-Scale"

        matrix_type_title = "Covariance" if use_covariance else "Correlation"

        fig_corr.suptitle(f'Average Pixel Hit {matrix_type_title}\n({analysis_mode_title} Pairs)\n{dataset_description}', fontsize=16)

        _create_heatmap(
            ax=ax_lin, matrix=avg_correlation_matrix, title=title_lin,
            xlabel='Pixel (Row ID)',
            ylabel='Pixel (Row ID)',
            analyze_by='column',
            use_log_scale=False, show_cbar=False, use_covariance=use_covariance
        )
        _create_heatmap(
            ax=ax_log, matrix=avg_correlation_matrix, title=title_log,
            xlabel='Pixel (Row ID)',
            ylabel='',
            analyze_by='column',
            use_log_scale=True, show_cbar=True, use_covariance=use_covariance
        )

        # --- NEW: Plot Filtered (Bottom row, only if applicable) ---
        if min_abs_correlation is not None:
            _create_heatmap(
                ax=ax_lin_filt, matrix=filtered_corr_matrix,
                title=f'Filtered |R| >= {min_abs_correlation} (Linear z-Scale)',
                xlabel='Pixel (Row ID)', ylabel='Pixel (Row ID)',
                analyze_by='column', use_log_scale=False, show_cbar=False
            )
            _create_heatmap(
                ax=ax_log_filt, matrix=filtered_corr_matrix,
                title=f'Filtered |R| >= {min_abs_correlation} (Log z-Scale)',
                xlabel='Pixel (Row ID)', ylabel='',
                analyze_by='column', use_log_scale=True, show_cbar=True
            )
        # --- END NEW ---

        fig_corr.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.show()

    if not all_pairs_dfs:
        print("❌ Insufficient data to perform analysis (no pairs found).")
        return

    print("\n--- Starting Pair Property Analysis ---")
    print("\nConcatenating pair data...")
    t_concat_start = time.perf_counter()
    cell_data = pd.concat(
        progress_bar(all_pairs_dfs, description="Concatenating pairs"),
        ignore_index=True
    )
    t_concat_end = time.perf_counter()
    del all_pairs_dfs
    print(f"Total pairs found: {len(cell_data)}")


    # 2. --- Detailed Analysis ---
    t_analysis_start = time.perf_counter()
    print(f"\nPlotting for '{analysis_mode_title} Pairs' ({len(cell_data)} total hit pairs)...")

    # --- ToT == 0 Analysis (on raw pair data) ---
    print("\n--- ToT == 0 Analysis (from all pairs) ---")
    total_pairs = len(cell_data)
    if total_pairs > 0:
        seed_zero_count = len(cell_data[cell_data['seed_tot'] == 0])
        other_zero_count = len(cell_data[cell_data['other_tot'] == 0])
        both_zero_count = len(cell_data[(cell_data['seed_tot'] == 0) & (cell_data['other_tot'] == 0)])
        either_zero_count = len(cell_data[(cell_data['seed_tot'] == 0) | (cell_data['other_tot'] == 0)])
        print(f"    Pairs with seed_tot == 0:    {seed_zero_count:10d} ({(seed_zero_count/total_pairs*100):.2f}%)")
        print(f"    Pairs with other_tot == 0:   {other_zero_count:10d} ({(other_zero_count/total_pairs*100):.2f}%)")
        print(f"    Pairs with BOTH == 0:        {both_zero_count:10d} ({(both_zero_count/total_pairs*100):.2f}%)")
        print(f"    Pairs with EITHER == 0:      {either_zero_count:10d} ({(either_zero_count/total_pairs*100):.2f}%)")
    else:
        print("    No pairs found to analyze.")
    print("---------------------------------------------------\n")

    if filter_tot_zero:
        original_count = len(cell_data)
        cell_data = cell_data[(cell_data['seed_tot'] > 0) & (cell_data['other_tot'] > 0)].copy()
        removed_count = original_count - len(cell_data)
        print(f"    Applying ToT==0 filter: Removed {removed_count} pairs.")
        print(f"    Pairs remaining for analysis: {len(cell_data)}\n")

    # --- Calculate derived quantities (AFTER ToT==0 filter) ---
    cell_data['displacement'] = cell_data['seed_row'].astype(np.int32) - cell_data['other_row'].astype(np.int32)
    cell_data['tot_diff'] = cell_data['seed_tot'].astype(np.int16) - cell_data['other_tot'].astype(np.int16)
    cell_data['tot_sum'] = cell_data['seed_tot'].astype(np.uint16) + cell_data['other_tot'].astype(np.uint16)
    min_tot = np.minimum(cell_data['seed_tot'].values, cell_data['other_tot'].values)
    max_tot = np.maximum(cell_data['seed_tot'].values, cell_data['other_tot'].values)
    cell_data['tot_ratio'] = np.where(max_tot > 0, min_tot / max_tot, np.nan)

    # --- NEW: Calculate TS diff ---
    # This assumes _process_column returned 'seed_ts' and 'other_ts'
    if 'seed_ts' in cell_data.columns and 'other_ts' in cell_data.columns:
        print("    Calculating 'trigger_ts_diff' (seed_ts - other_ts)...")
        cell_data['trigger_ts_diff'] = cell_data['seed_ts'].astype(np.int64) - cell_data['other_ts'].astype(np.int64)
    else:
        print("⚠️ Warning: 'seed_ts' or 'other_ts' columns not found in pair data.")
        print("    Cannot calculate 'trigger_ts_diff'. TS Diff plot will be skipped.")
    # --- END NEW ---

    # --- Data Filtering for Plots & Frequency Calculation ---
    label_suffix_tot, label_suffix_final = "", ""
    is_tot_filtered, is_disp_filtered = False, False
    freq_low, freq_high = 0.0, 0.0
    count_low, count_high = 0, 0

    if filter_tot_low_threshold is not None or filter_tot_high_threshold is not None:

        # Low ToT filter: Remove if BOTH are below threshold
        cond_low = (cell_data['seed_tot'] < filter_tot_low_threshold) & (cell_data['other_tot'] < filter_tot_low_threshold) if filter_tot_low_threshold is not None else pd.Series(False, index=cell_data.index)

        # High ToT filter: Remove if EITHER is above threshold
        cond_high = (cell_data['seed_tot'] > filter_tot_high_threshold) | (cell_data['other_tot'] > filter_tot_high_threshold) if filter_tot_high_threshold is not None else pd.Series(False, index=cell_data.index)

        count_low, count_high = cond_low.sum(), cond_high.sum()
        is_tot_filtered = True

        combined_tot_filter_cond = cond_low | cond_high
        data_tot_filtered = cell_data[~combined_tot_filter_cond]

        label_suffix_tot = " (ToT Filtered)"
        label_suffix_final = " (ToT Filtered"
        print(f"    Original pairs (after ToT==0 filter if any): {len(cell_data)}")
        if total_duration_sec > 0:
            freq_low = count_low / total_duration_sec
            freq_high = count_high / total_duration_sec
            if filter_tot_low_threshold is not None: print(f"    Filtered {count_low} pairs (Low-ToT < {filter_tot_low_threshold}) -> Frequency: {freq_low:.2f} Hz")
            if filter_tot_high_threshold is not None: print(f"    Filtered {count_high} pairs (High-ToT > {filter_tot_high_threshold}) -> Frequency: {freq_high:.2f} Hz")
        print(f"    Pairs after ToT filter: {len(data_tot_filtered)}")
    else:
        data_tot_filtered = cell_data

    if filter_disp_gate is not None:
        min_g, max_g = filter_disp_gate
        cond_disp_gate = (data_tot_filtered['displacement'] >= min_g) & (data_tot_filtered['displacement'] <= max_g)
        count_disp = cond_disp_gate.sum()
        is_disp_filtered = True
        label_suffix_final += " & Disp. Gated)"

        if total_duration_sec > 0:
            freq_disp = count_disp / total_duration_sec
            print(f"    Filtered {count_disp} pairs (Disp. Gate [{min_g}, {max_g}]) -> Frequency: {freq_disp:.2f} Hz")

        data_final_filtered = data_tot_filtered[~cond_disp_gate]
        print(f"    Pairs after Disp. filter: {len(data_final_filtered)}")
    else:
        data_final_filtered = data_tot_filtered
        if is_tot_filtered: label_suffix_final += ")"

    # --- Call Plotting Functions ---
    plot_params = {
        'analysis_mode_title': analysis_mode_title,
        'dataset_description': dataset_description,
        'filter_tot_zero': filter_tot_zero,
        'is_tot_filtered': is_tot_filtered,
        'is_disp_filtered': is_disp_filtered,
        'filter_tot_low_threshold': filter_tot_low_threshold,
        'filter_tot_high_threshold': filter_tot_high_threshold,
        'filter_disp_gate': filter_disp_gate,
        'label_suffix_tot': label_suffix_tot,
        'label_suffix_final': label_suffix_final,
        'freq_low': freq_low,
        'freq_high': freq_high,
        'total_duration_sec': total_duration_sec,
        'count_low': count_low,
        'count_high': count_high,
        # --- NEW PARAMETER PASSED TO PLOTTING FUNCTIONS ---
        'log_y_scale': log_y_disp_plots
        # --- END NEW ---
    }

    h_A, xedges_A, yedges_A = None, None, None

    # Note: Assuming _plot_figure_1_tot_ratio accepts 'log_y_scale' for the ratio plot
    _plot_figure_1_tot_ratio(cell_data, data_tot_filtered, data_final_filtered, **plot_params)

    h_A, xedges_A, yedges_A = _plot_figure_2_tot_heatmap(data_tot_filtered, data_final_filtered, **plot_params)

    # Note: Assuming _plot_figure_3_all_z_joints accepts 'log_y_scale' for the Disp vs Ratio plot
    _plot_figure_3_all_z_joints(cell_data, data_tot_filtered, data_final_filtered, **plot_params)

    _plot_figure_4_complex_joint(data_final_filtered, **plot_params)

    plt.show()

    # --- Stop Analysis Timer ---
    t_analysis_end = time.perf_counter()

    # --- Return logic (unchanged) ---
    if return_heatmap_data:
        print("Calculating ToT heatmap data for return...")
        if h_A is not None:
            print("Returning ToT-filtered heatmap data.")
            _print_timing_report(t_func_start, t_parallel_end, t_parallel_start,
                                 t_corr_avg_end, t_corr_avg_start, t_concat_end,
                                 t_concat_start, t_analysis_end, t_analysis_start)
            return h_A.T, xedges_A, yedges_A
        else:
            print("No data in ToT-filtered heatmap to return.")
            bins = np.arange(0, 256)
            _print_timing_report(t_func_start, t_parallel_end, t_parallel_start,
                                 t_corr_avg_end, t_corr_avg_start, t_concat_end,
                                 t_concat_start, t_analysis_end, t_analysis_start)
            return np.zeros((255, 255)), bins, bins

    print("\n✅ Analysis complete.")
    _print_timing_report(t_func_start, t_parallel_end, t_parallel_start,
                         t_corr_avg_end, t_corr_avg_start, t_concat_end,
                         t_concat_start, t_analysis_end, t_analysis_start)


def _plot_figure_1_tot_ratio(cell_data, data_tot_filtered, data_final_filtered,
                             analysis_mode_title, filter_tot_zero,
                             is_tot_filtered, is_disp_filtered, dataset_description,
                              **kwargs): # Added log_y_scale parameter
    """Plots the ToT Ratio (Fig 1) with an option for log y-scale."""
    fig_ratio, ax_tot_ratio = plt.subplots(figsize=(7, 6))

    ratio_all = cell_data['tot_ratio'].dropna()
    ratio_tot_filtered = data_tot_filtered['tot_ratio'].dropna()
    ratio_final_filtered = data_final_filtered['tot_ratio'].dropna()

    bins_lin = np.linspace(0, 1.1, 111)

    if not ratio_all.empty:
        label_all = "All Data" + (" (ToT > 0)" if filter_tot_zero else "")
        ax_tot_ratio.hist(ratio_all, bins=bins_lin, histtype='step', lw=2, label=label_all, zorder=1)

    if is_tot_filtered and not ratio_tot_filtered.empty:
        ax_tot_ratio.hist(ratio_tot_filtered, bins=bins_lin, histtype='step', lw=2, label='ToT Filtered', linestyle='--', zorder=2)

    if is_disp_filtered and not ratio_final_filtered.empty:
        ax_tot_ratio.hist(ratio_final_filtered, bins=bins_lin, histtype='step', lw=2, label='ToT + Disp. Filtered', linestyle=':', zorder=3)

    if not ratio_all.empty:
        ax_tot_ratio.legend()

    ax_tot_ratio.set_xscale('linear')
    ax_tot_ratio.set_xlim(0, 1)

    ax_tot_ratio.set_title('ToT Ratio (Min / Max) Distribution')
    ax_tot_ratio.set_xlabel('ToT Ratio (Min / Max)')
    ax_tot_ratio.set_ylabel('Frequency') # Updated y-axis label
    ax_tot_ratio.grid(True, which='both', linestyle='--', alpha=0.6)
    fig_ratio.suptitle(f'ToT Ratio for {analysis_mode_title}\n{dataset_description}', fontsize=14)

    # --- Inset Plot (Log X-axis zoom) ---
    # The inset is kept to show the low-ratio part on a log-x scale
    ax_inset = ax_tot_ratio.inset_axes([0.1, 0.4, 0.55, 0.55])

    # The min_val_log calculation uses a specific value (1/255) for data constraint, keep it.
    min_val_log = 1/255 if (filter_tot_zero or (not ratio_all.empty and ratio_all.min() > 1/255)) else 1e-3
    bins_log = np.logspace(np.log10(min_val_log), np.log10(1), 101)

    if not ratio_all.empty:
        ax_inset.hist(ratio_all, bins=bins_log, histtype='step', lw=1.5, zorder=1)
    if is_tot_filtered and not ratio_tot_filtered.empty:
        ax_inset.hist(ratio_tot_filtered, bins=bins_log, histtype='step', lw=1.5, linestyle='--', zorder=2)
    if is_disp_filtered and not ratio_final_filtered.empty:
        ax_inset.hist(ratio_final_filtered, bins=bins_log, histtype='step', lw=1.5, linestyle=':', zorder=3)

    ax_inset.set_xscale('log')
    ax_inset.set_title('Log-X Zoom', fontsize='small')
    ax_inset.set_xlim(min_val_log, 1.1)
    ax_inset.tick_params(axis='both', which='major', labelsize=8)
    ax_inset.grid(True, which='both', linestyle=':', alpha=0.5)

    fig_ratio.tight_layout(rect=[0, 0, 1, 0.93]) # Adjusted for new title line

    # Return the figure object (assuming this is part of a larger plotting system)
    return fig_ratio, ax_tot_ratio



def _plot_figure_2_tot_heatmap(data_tot_filtered, data_final_filtered,
                              analysis_mode_title, label_suffix_tot, label_suffix_final,
                              filter_tot_low_threshold, filter_tot_high_threshold,
                              total_duration_sec, is_tot_filtered, freq_low, freq_high,
                              count_low, count_high, dataset_description, **kwargs):
    """Plots the side-by-side ToT Heatmap (Fig 2)"""
    fig_tot, (ax_tot_corr_A, ax_tot_corr_B) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig_tot.suptitle(f'ToT Heatmap Comparison for {analysis_mode_title}\n{dataset_description}', fontsize=16)

    vmin, vmax = np.inf, -np.inf
    h_A, xedges_A, yedges_A = (None, None, None)
    if not data_tot_filtered.empty:
        h_A, xedges_A, yedges_A = np.histogram2d(data_tot_filtered['seed_tot'], data_tot_filtered['other_tot'], bins=255, range=[[0, 255], [0, 255]])
        if h_A.sum() > 0:
            vmin = min(vmin, h_A[h_A > 0].min())
            vmax = max(vmax, h_A.max())
    if not data_final_filtered.empty:
        h_B, _, _ = np.histogram2d(data_final_filtered['seed_tot'], data_final_filtered['other_tot'], bins=255, range=[[0, 255], [0, 255]])
        if h_B.sum() > 0:
            vmin = min(vmin, h_B[h_B > 0].min())
            vmax = max(vmax, h_B.max())

    norm = LogNorm(vmin=1, vmax=10) if vmin >= vmax and vmax > 0 else (Normalize(vmin=0, vmax=1) if vmin >= vmax else LogNorm(vmin=vmin, vmax=vmax))

    if h_A is not None:
        ax_tot_corr_A.pcolormesh(xedges_A, yedges_A, h_A.T, cmap='plasma', norm=norm)
    else:
        ax_tot_corr_A.text(0.5, 0.5, 'No data after ToT filtering', ha='center', va='center', transform=ax_tot_corr_A.transAxes)

    ax_tot_corr_A.set_title(f'ToT Pixel A vs. ToT B{label_suffix_tot}')
    ax_tot_corr_A.set_xlabel('Pixel A ToT')
    ax_tot_corr_A.set_ylabel('Pixel B ToT')
    ax_tot_corr_A.set_aspect('equal', adjustable='box')

    if not data_final_filtered.empty:
        _, _, _, im_main = ax_tot_corr_B.hist2d(
            data_final_filtered['seed_tot'], data_final_filtered['other_tot'],
            bins=255, range=[[0, 255], [0, 255]], cmap='plasma', norm=norm
        )
        fig_tot.colorbar(im_main, ax=ax_tot_corr_B, label='Frequency (Log Scale)', fraction=0.046, pad=0.04)
    else:
        ax_tot_corr_B.text(0.5, 0.5, 'No data after Disp. gate', ha='center', va='center', transform=ax_tot_corr_B.transAxes)

    ax_tot_corr_B.set_title(f'ToT Pixel A vs. ToT B{label_suffix_final}')
    ax_tot_corr_B.set_xlabel('Pixel A ToT')
    ax_tot_corr_B.set_ylabel('Pixel B ToT')
    ax_tot_corr_B.set_aspect('equal', adjustable='box')

    for ax in [ax_tot_corr_A, ax_tot_corr_B]:
        legend_patches = []
        if filter_tot_low_threshold is not None:
            rect_low = patches.Rectangle((0, 0), filter_tot_low_threshold, filter_tot_low_threshold,
                                         linewidth=1.5, edgecolor='r', facecolor='none', linestyle='--',
                                         label='Filtered Low-ToT')
            ax.add_patch(rect_low)
            legend_patches.append(rect_low)
        if filter_tot_high_threshold is not None:
            rect_high_size = 255 - filter_tot_high_threshold
            rect_high = patches.Rectangle((filter_tot_high_threshold, filter_tot_high_threshold),
                                         rect_high_size, rect_high_size,
                                         linewidth=1.5, edgecolor='r', facecolor='none', linestyle='--',
                                         label='Filtered High-ToT')
            ax.add_patch(rect_high)
            legend_patches.append(rect_high)
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper left', fontsize='small')

    if total_duration_sec > 0 and is_tot_filtered:
        text_props = dict(ha='center', va='center', color='white', fontsize=9,
                          bbox=dict(facecolor='black', alpha=0.4, pad=0.2, boxstyle='round,pad=0.2'))
        if filter_tot_low_threshold is not None and count_low > 0:
            ax_tot_corr_A.text(filter_tot_low_threshold * 0.5, filter_tot_low_threshold * 0.5, f"{freq_low:.1f} Hz", **text_props)
        if filter_tot_high_threshold is not None and count_high > 0:
            center_high = (filter_tot_high_threshold + 255) / 2
            ax_tot_corr_A.text(center_high, center_high, f"{freq_high:.1f} Hz", **text_props)

    fig_tot.tight_layout(rect=[0, 0, 1, 0.93]) # Adjusted for new title line
    return h_A, xedges_A, yedges_A


def _plot_figure_3_all_z_joints(cell_data, data_tot_filtered, data_final_filtered,
                              analysis_mode_title, label_suffix_final,
                              filter_tot_zero, is_tot_filtered, is_disp_filtered,
                              filter_disp_gate, dataset_description,
                              log_y_scale=False, **kwargs): # ADDED log_y_scale
    """
    Plots the Joint Plot (Disp vs. ToT Ratio) with 1D marginals for all Z-axis modes.
    Y-axis is ToT Ratio (Min/Max). Bins are log-distributed if log_y_scale is True.
    """

    # --- Define Bins for X (Displacement) ---
    disp_bins = 372 * 2 + 1
    disp_range = [-372.5, 372.5]
    disp_bin_edges = np.linspace(disp_range[0], disp_range[1], disp_bins + 1)

    # --- Define Bins for Y (ToT Ratio) ---
    # Linear Bins (Default)
    tot_ratio_bins_count = 105
    tot_ratio_range = [0, 1.05]
    tot_ratio_bin_edges_lin = np.linspace(tot_ratio_range[0], tot_ratio_range[1], tot_ratio_bins_count + 1)
    
    # Log Bins (If requested)
    # MODIFIED: Minimum set to 1/255 (smallest non-zero ratio for 8-bit data)
    min_log_val = 1/255 
    
    # Use the same number of bins (105) but distribute them logarithmically up to 1.0
    tot_ratio_bin_edges_log = np.logspace(np.log10(min_log_val), 0, tot_ratio_bins_count + 1)
    
    # --- Define Z Bins (ToT Difference) ---
    tot_diff_range = [-255.5, 255.5]
    tot_diff_bin_edges = np.linspace(tot_diff_range[0], tot_diff_range[1], 511 + 1)

    # --- Prep data for all plots (using tot_ratio for Y-axis) ---
    disp_all = cell_data['displacement'].dropna()
    ratio_all = cell_data['tot_ratio'].dropna()
    disp_tot_filtered = data_tot_filtered['displacement'].dropna()
    ratio_tot_filtered = data_tot_filtered['tot_ratio'].dropna()

    # --- Include all relevant columns for final data ---
    columns_for_fig3 = ['displacement', 'tot_ratio', 'tot_sum', 'tot_diff']
    if 'trigger_ts_diff' in data_final_filtered.columns:
        columns_for_fig3.append('trigger_ts_diff')

    valid_data_final = data_final_filtered[columns_for_fig3].dropna()

    if 'trigger_ts_diff' not in data_final_filtered.columns:
          print("    (Note: 'trigger_ts_diff' not available for plotting in Figure 3.)")

    if valid_data_final.empty:
        print("    ⚠️ Warning: No valid (non-NaN) data for Figure 3 joint plots.")
        fig_joint_simple = plt.figure(figsize=(8, 7))
        ax_empty_joint = fig_joint_simple.add_subplot(1, 1, 1)
        ax_empty_joint.text(0.5, 0.5, 'No valid data for joint plot',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_empty_joint.transAxes, color='red')
        fig_joint_simple.suptitle(f'Displacement vs. ToT Ratio (Min/Max){label_suffix_final}\n({analysis_mode_title})\n{dataset_description}', fontsize=14)
        return

    # --- Define Plot Configurations (Z-axis) ---
    plot_configs = [
        {
            'mode': 'counts',
            'cmap': 'magma',
            'cbar_label': 'Frequency (Log Scale)',
            'title_z': 'Z: Counts'
        },
        {
            'mode': 'tot_sum',
            'cmap': 'nipy_spectral',
            'cbar_label': 'Mean ToT Sum (ToT_A + ToT_B)',
            'title_z': 'Z: Mean ToT Sum'
        },
        {
            'mode': 'tot_diff',
            'cmap': 'coolwarm',
            'cbar_label': 'Mean ToT Difference (Seed - 2nd)',
            'title_z': 'Z: Mean ToT Difference'
        },
        {
            'mode': 'trigger_ts_diff',
            'cmap': 'Set1',
            'cbar_label': 'Mean TriggerTS Difference (seed - other) [TS]',
            'title_z': 'Z: Mean TriggerTS Diff'
        }
    ]

    # --- Loop and create each figure ---
    for config in plot_configs:
        z_axis_mode = config['mode']

        # --- Conditional Bin Setup ---
        if log_y_scale:
            # Use log-spaced bins for 2D histogram/statistic calculation
            y_bins_for_calc = tot_ratio_bin_edges_log
            y_range_for_calc = [min_log_val, 1.0] 
            y_plot_range = [min_log_val, tot_ratio_range[1]] # Plot up to 1.05
            y_plot_scale = 'log'
            y_label_suffix = ' [Log Scale]'
        else:
            # Use linear bins
            y_bins_for_calc = tot_ratio_bin_edges_lin
            y_range_for_calc = tot_ratio_range
            y_plot_range = tot_ratio_range
            y_plot_scale = 'linear'
            y_label_suffix = ''

        # --- Create Figure and Axes ---
        fig_joint = plt.figure(figsize=(8, 7))
        fig_joint.suptitle(f'Displacement vs. ToT Ratio (Min/Max) ({config["title_z"]}){label_suffix_final}\n({analysis_mode_title})\n{dataset_description}', fontsize=14)
        gs_joint = fig_joint.add_gridspec(5, 5, hspace=0, wspace=0, top=0.88)
        ax_joint_main = fig_joint.add_subplot(gs_joint[1:, :-1])
        ax_joint_marg_x = fig_joint.add_subplot(gs_joint[0, :-1], sharex=ax_joint_main)
        ax_joint_marg_y = fig_joint.add_subplot(gs_joint[1:, -1], sharey=ax_joint_main)

        # --- Plot Main Heatmap ---
        cbar_label = config['cbar_label']
        
        if z_axis_mode == 'counts':
            _, x_edge, y_edge, im_disp = ax_joint_main.hist2d(
                valid_data_final['displacement'], valid_data_final['tot_ratio'],
                bins=(disp_bin_edges, y_bins_for_calc), range=[disp_range, y_range_for_calc], 
                cmap=config['cmap'], norm=LogNorm()
            )
        else:
            stat_data = valid_data_final[z_axis_mode]
            mean_stat, x_edge, y_edge, _ = binned_statistic_2d(
                valid_data_final['displacement'], valid_data_final['tot_ratio'],
                stat_data, statistic='mean',
                bins=(disp_bin_edges, y_bins_for_calc) 
            )
            
            vmin, vmax = None, None
            if z_axis_mode == 'tot_sum':
                vmin, vmax = 0, 511
            elif z_axis_mode in ['tot_diff', 'trigger_ts_diff']:
                abs_max = np.nanmax(np.abs(mean_stat))
                if abs_max == 0 or not np.isfinite(abs_max): abs_max = 1.0
                vmin, vmax = -abs_max, abs_max
            
            im_disp = ax_joint_main.pcolormesh(
                x_edge, y_edge, mean_stat.T, 
                cmap=config['cmap'], vmin=vmin, vmax=vmax,
                norm=None if z_axis_mode != 'counts' else LogNorm()
            )

        cbar = fig_joint.colorbar(im_disp, ax=ax_joint_marg_y, pad=0.1, aspect=30)
        cbar.set_label(cbar_label, fontsize=10)

        # --- Plot 1D Marginals ---
        label_all_marg = "All" + (" (ToT>0)" if filter_tot_zero else "")
        # X Marginal (Disp)
        if not disp_all.empty:
            ax_joint_marg_x.hist(disp_all, bins=disp_bins, range=disp_range, histtype='step', label=label_all_marg, zorder=1)
        if is_tot_filtered and not disp_tot_filtered.empty:
            ax_joint_marg_x.hist(disp_tot_filtered, bins=disp_bins, range=disp_range, histtype='step', label='ToT Filtered', linestyle='--', zorder=2)
        if is_disp_filtered and not valid_data_final.empty:
            ax_joint_marg_x.hist(valid_data_final['displacement'], bins=disp_bins, range=disp_range, histtype='step', label='ToT+Disp. Filtered', linestyle=':', zorder=3)
        ax_joint_marg_x.legend(loc='upper left', fontsize='small')
        ax_joint_marg_x.set_yscale('log')
        ax_joint_marg_x.set_ylabel('Hit Sum')

        # Y Marginal (ToT Ratio) - Uses the chosen bins
        if not ratio_all.empty:
            ax_joint_marg_y.hist(ratio_all, bins=y_bins_for_calc, range=y_range_for_calc, histtype='step', orientation='horizontal', zorder=1)
        if is_tot_filtered and not ratio_tot_filtered.empty:
            ax_joint_marg_y.hist(ratio_tot_filtered, bins=y_bins_for_calc, range=y_range_for_calc, histtype='step', orientation='horizontal', linestyle='--', zorder=2)
        if is_disp_filtered and not valid_data_final.empty:
            ax_joint_marg_y.hist(valid_data_final['tot_ratio'], bins=y_bins_for_calc, range=y_range_for_calc, histtype='step', orientation='horizontal', linestyle=':', zorder=3)
            
        # --- Apply Plot Scale and Limits ---
        ax_joint_main.set_yscale(y_plot_scale) 
        ax_joint_main.set_ylim(y_plot_range) 

        ax_joint_main.set_ylabel(f'ToT Ratio (Min/Max){y_label_suffix}')
        ax_joint_marg_y.set_xscale('log')
        ax_joint_marg_y.set_xlabel('Hit Sum (Log Scale)')

        # --- Finalize Plot ---
        plt.setp(ax_joint_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_joint_marg_y.get_yticklabels(), visible=False)
        ax_joint_main.set_xlabel('Pixel Displacement (signed)')
        ax_joint_main.grid(True, linestyle='--', alpha=0.6)

        if filter_disp_gate is not None:
            min_g, max_g = filter_disp_gate
            ax_joint_main.axvline(min_g, color='r', linestyle='--', lw=1.5)
            ax_joint_main.axvline(max_g, color='r', linestyle='--', lw=1.5,
                                  label=f'Disp. Gate [{min_g}, {max_g}]')
            ax_joint_main.legend(loc='upper left', fontsize='small')

        ax_joint_main.set_xlim(disp_range)

        fig_joint.subplots_adjust(left=0.1, right=0.83, top=0.88, bottom=0.1)


def _plot_figure_4_complex_joint(data_final_filtered, analysis_mode_title,
                                 label_suffix_final, filter_disp_gate,
                                 dataset_description, log_y_scale=False, **kwargs): # ADDED log_y_scale
    """
    Plots the 4-way 2D marginal plot (Fig 4) with ToT Ratio on the Y-axis.
    Supports logarithmic binning on the Y-axis via log_y_scale.
    """
    fig_joint_complex = plt.figure(figsize=(12, 12))
    fig_joint_complex.suptitle(f'Full Correlation View (Disp. vs. ToT Ratio){label_suffix_final}\n({analysis_mode_title})\n{dataset_description}', fontsize=14)

    # --- Define Bins ---
    
    # 1. Displacement (X-axis Main, Top, Bottom)
    disp_bins = 372 * 2 + 1
    disp_range = [-372.5, 372.5]
    disp_bin_edges = np.linspace(disp_range[0], disp_range[1], disp_bins + 1)

    # 2. ToT (X-axis Right/Left, Y-axis Top/Bottom)
    tot_bins = 256
    tot_range = [-0.5, 255.5]
    tot_bin_edges = np.linspace(tot_range[0], tot_range[1], tot_bins + 1)

    # 3. ToT Ratio (Y-axis Main, Left, Right) - CONDITIONAL BINNING
    tot_ratio_bins_count = 105
    tot_ratio_range_lin = [0, 1.05]
    
    if log_y_scale:
        # MODIFIED: Minimum set to 1/255
        min_log_val = 1/255
        # Logarithmic bins
        tot_ratio_bin_edges = np.logspace(np.log10(min_log_val), 0, tot_ratio_bins_count + 1)
        # Set plot limits and labels
        ratio_plot_ylim = [min_log_val, 1.05]
        ratio_plot_scale = 'log'
        ratio_label_suffix = ' [Log Scale]'
    else:
        # Linear bins
        tot_ratio_bin_edges = np.linspace(tot_ratio_range_lin[0], tot_ratio_range_lin[1], tot_ratio_bins_count + 1)
        # Set plot limits and labels
        ratio_plot_ylim = tot_ratio_range_lin
        ratio_plot_scale = 'linear'
        ratio_label_suffix = ''

    # --- Filter Data ---
    valid_data_final = data_final_filtered[['displacement', 'tot_ratio', 'tot_sum', 'seed_tot', 'other_tot']].dropna()

    if valid_data_final.empty:
        print("    ⚠️ Warning: No valid (non-NaN) data for Figure 4 joint plot.")
        ax_empty_joint = fig_joint_complex.add_subplot(1, 1, 1)
        ax_empty_joint.text(0.5, 0.5, 'No valid data for joint plot',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_empty_joint.transAxes, color='red')
    else:
        # --- Setup Grid ---
        gs_joint = fig_joint_complex.add_gridspec(3, 4, hspace=0, wspace=0.05, top=0.88,
                                           width_ratios=[1.2, 4, 1.2, 0.2],
                                           height_ratios=[1.2, 4, 1.2])

        ax_joint_main = fig_joint_complex.add_subplot(gs_joint[1, 1])
        ax_marg_top = fig_joint_complex.add_subplot(gs_joint[0, 1], sharex=ax_joint_main)
        ax_marg_right = fig_joint_complex.add_subplot(gs_joint[1, 2], sharey=ax_joint_main)
        ax_marg_left = fig_joint_complex.add_subplot(gs_joint[1, 0], sharey=ax_joint_main)
        ax_marg_bottom = fig_joint_complex.add_subplot(gs_joint[2, 1], sharex=ax_joint_main)

        ax_cbar = fig_joint_complex.add_subplot(gs_joint[1, 3])
        ax_cbar_empty_top = fig_joint_complex.add_subplot(gs_joint[0, 3])
        ax_cbar_empty_top.axis('off')
        ax_cbar_empty_bottom = fig_joint_complex.add_subplot(gs_joint[2, 3])
        ax_cbar_empty_bottom.axis('off')

        # --- Calculate Histograms (Using tot_ratio_bin_edges) ---
        
        # Main: Disp vs Ratio
        h_main, _, _ = np.histogram2d(valid_data_final['displacement'], valid_data_final['tot_ratio'], bins=(disp_bin_edges, tot_ratio_bin_edges))
        
        # Top: Disp vs ToT_B (Unchanged, linear ToT Y-axis)
        h_top, _, _ = np.histogram2d(valid_data_final['displacement'], valid_data_final['other_tot'], bins=(disp_bin_edges, tot_bin_edges))
        
        # Right: ToT_B vs Ratio
        h_right, _, _ = np.histogram2d(valid_data_final['other_tot'], valid_data_final['tot_ratio'], bins=(tot_bin_edges, tot_ratio_bin_edges))
        
        # Left: ToT_A vs Ratio
        h_left, _, _ = np.histogram2d(valid_data_final['seed_tot'], valid_data_final['tot_ratio'], bins=(tot_bin_edges, tot_ratio_bin_edges))
        
        # Bottom: Disp vs ToT_A (Unchanged, linear ToT Y-axis)
        h_bottom, _, _ = np.histogram2d(valid_data_final['displacement'], valid_data_final['seed_tot'], bins=(disp_bin_edges, tot_bin_edges))

        # --- Normalization ---
        vmin_global, vmax_global = np.inf, -np.inf
        all_hists = [h_main, h_top, h_right, h_left, h_bottom]
        for h in all_hists:
            if h.sum() > 0:
                h_min_nz = h[h > 0].min()
                if h_min_nz > 0:
                    vmin_global = min(vmin_global, h_min_nz)
                vmax_global = max(vmax_global, h.max())

        if vmin_global >= vmax_global:
            norm_shared = LogNorm(vmin=1, vmax=10) if vmax_global > 0 else Normalize(vmin=0, vmax=1)
        else:
            norm_shared = LogNorm(vmin=vmin_global, vmax=vmax_global)

        cmap_shared = 'magma'

        # --- Plotting ---

        # Main Plot: Disp (X) vs ToT Ratio (Y - potentially Log Bins)
        im_main = ax_joint_main.pcolormesh(
            disp_bin_edges, tot_ratio_bin_edges, h_main.T,
            cmap=cmap_shared, norm=norm_shared
        )

        # Top Marginal: Disp (X) vs ToT B (Y)
        ax_marg_top.pcolormesh(
            disp_bin_edges, tot_bin_edges, h_top.T,
            cmap=cmap_shared, norm=norm_shared
        )
        ax_marg_top.set_ylabel('ToT Pixel B')
        ax_marg_top.grid(True, linestyle='--', alpha=0.6)
        ax_marg_top.set_yscale('log')
        ax_marg_top.set_ylim(0.5, 255.5)

        # Right Marginal: ToT B (X) vs ToT Ratio (Y - potentially Log Bins)
        ax_marg_right.pcolormesh(
            tot_bin_edges, tot_ratio_bin_edges, h_right.T,
            cmap=cmap_shared, norm=norm_shared
        )
        ax_marg_right.set_xlabel('ToT Pixel B')
        ax_marg_right.grid(True, linestyle='--', alpha=0.6)
        ax_marg_right.set_xscale('log')
        ax_marg_right.set_xlim(0.5, 255.5)

        # Left Marginal: ToT A (X) vs ToT Ratio (Y - potentially Log Bins)
        ax_marg_left.pcolormesh(
            tot_bin_edges, tot_ratio_bin_edges, h_left.T,
            cmap=cmap_shared, norm=norm_shared
        )
        ax_marg_left.set_xlabel('ToT Pixel A')
        ax_marg_left.set_ylabel(f'ToT Ratio (Min/Max){ratio_label_suffix}') # Updated Label
        ax_marg_left.grid(True, linestyle='--', alpha=0.6)
        ax_marg_left.set_xscale('log')
        ax_marg_left.set_xlim(0.5, 255.5)

        # Bottom Marginal: Disp (X) vs ToT A (Y)
        ax_marg_bottom.pcolormesh(
            disp_bin_edges, tot_bin_edges, h_bottom.T,
            cmap=cmap_shared, norm=norm_shared
        )
        ax_marg_bottom.set_xlabel('Pixel Displacement (signed)')
        ax_marg_bottom.set_ylabel('ToT Pixel A')
        ax_marg_bottom.grid(True, linestyle='--', alpha=0.6)
        ax_marg_bottom.set_yscale('log')
        ax_marg_bottom.set_ylim(0.5, 255.5)

        # --- Cleanup & Formatting ---
        plt.setp(ax_joint_main.get_xticklabels(), visible=False)
        plt.setp(ax_joint_main.get_yticklabels(), visible=False)
        plt.setp(ax_marg_top.get_xticklabels(), visible=False)
        plt.setp(ax_marg_right.get_yticklabels(), visible=False)

        ax_joint_main.grid(True, linestyle='--', alpha=0.6)

        if filter_disp_gate is not None:
            min_g, max_g = filter_disp_gate
            ax_joint_main.axvline(min_g, color='r', linestyle='--', lw=1.5)
            ax_joint_main.axvline(max_g, color='r', linestyle='--', lw=1.5,
                                  label=f'Disp. Gate [{min_g}, {max_g}]')
            ax_joint_main.legend(loc='upper left', fontsize='small')

        # --- Apply Scale and Limits to Main Plot (and shared Y axes) ---
        ax_joint_main.set_xlim(disp_range)
        ax_joint_main.set_yscale(ratio_plot_scale) # Log or Linear
        ax_joint_main.set_ylim(ratio_plot_ylim)    # Range

        fig_joint_complex.colorbar(im_main, cax=ax_cbar, label='Counts (magma, log)')

    # Use subplots_adjust instead of tight_layout to avoid warning
    fig_joint_complex.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1)

def _print_timing_report(t_func_start, t_parallel_end, t_parallel_start,
                         t_corr_avg_end, t_corr_avg_start, t_concat_end,
                         t_concat_start, t_analysis_end, t_analysis_start):
    """Prints a formatted summary of execution times."""
    t_func_end = time.perf_counter()
    print("\n--- ⏱️ Execution Time Report ---")
    print(f"  Parallel column processing:   {t_parallel_end - t_parallel_start:.2f} s")
    print(f"  Correlation matrix averaging: {t_corr_avg_end - t_corr_avg_start:.2f} s")
    print(f"  Pair data concatenation:    {t_concat_end - t_concat_start:.2f} s")
    print(f"  Analysis & plotting:        {t_analysis_end - t_analysis_start:.2f} s")
    print(f"  ---------------------------------")
    print(f"  Total function time:        {t_func_end - t_func_start:.2f} s")
    
    
    
def extract_high_correlation_pairs(
    correlation_matrix: pd.DataFrame, 
    hit_matrix: pd.DataFrame, 
    threshold: float = 0.01, 
    dx: int = 1
) -> pd.DataFrame:
    """
    Extracts pairs of pixels (Rows) with a correlation coefficient higher than the specified threshold,
    along with the total number of correlated hits for that pair.
    
    Args:
        correlation_matrix (pd.DataFrame): The square correlation matrix.
        hit_matrix (pd.DataFrame): The square matrix containing total hit counts (must match shape of correlation_matrix).
        threshold (float): The correlation cutoff value.
        dx (int): Minimum displacement (diagonal offset) to consider.

    Returns:
        pd.DataFrame: DataFrame containing 'Row_A', 'Row_B', 'Correlation', 'Hits', and 'Displacement'.
    """
    if correlation_matrix is None or correlation_matrix.empty:
        return pd.DataFrame(columns=['Row_A', 'Row_B', 'Correlation', 'Hits', 'Displacement'])
    
    # Generate Upper Triangle Mask
    mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=dx)
    
    # Find indices where correlation is above threshold
    # Note: We use correlation_matrix.values to get numpy array for faster indexing
    row_idx, col_idx = np.where(mask & (correlation_matrix.values > threshold))
    
    # Extract values using the indices
    corr_values = correlation_matrix.values[row_idx, col_idx]
    
    # Extract Hit counts using the SAME indices
    # We assume hit_matrix is aligned with correlation_matrix (output from calculate_aggregated_correlation)
    if hit_matrix is not None and not hit_matrix.empty:
        hit_values = hit_matrix.values[row_idx, col_idx]
    else:
        hit_values = np.zeros_like(corr_values)
    
    row_labels = correlation_matrix.index[row_idx]
    col_labels = correlation_matrix.columns[col_idx]
    
    results = pd.DataFrame({
        'Row_A': row_labels,
        'Row_B': col_labels,
        'Correlation': corr_values,
        'Hits': hit_values,        # <--- Added Hit Count
        'Displacement': col_labels - row_labels 
    })
    
    # Sort by highest correlation first
    return results.sort_values(by='Correlation', ascending=False).reset_index(drop=True)