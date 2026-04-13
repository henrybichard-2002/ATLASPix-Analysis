import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import sys
from utils import progress_bar

def remove_crosstalk(
    data_raw: Dict[str, np.ndarray],
    ToTthreshMin: int = 20,
    dx: int = 30,
    ToTrat: float = 0.11,
    n: int = 4,
    mode: str = 'filter'
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Removes crosstalk hits from the raw data based on specified criteria.

    The dataset is a dictionary of NumPy arrays. The function iteratively
    identifies and filters crosstalk hits over 'n' repetitions.

    Args:
        data_raw: The input data, as a dictionary of NumPy arrays.
        ToTthreshMin: ToT threshold. Pairs with both ToT <= this are ignored.
        dx: Row separation. Pairs with row separation <= this are ignored.
        ToTrat: ToT ratio. Pairs where (smaller ToT / larger ToT) <= this
                are considered crosstalk.
        n: The number of filtering repetitions to perform.
        mode: The filtering mode.
              'filter': Removes the smaller ToT hit.
              'add': Removes the smaller ToT hit and adds its ToT value
                     to the larger ToT hit in the pair.

    Returns:
        A tuple of two dictionaries (data_filtered, data_crosstalk):
        - data_filtered: Data with crosstalk hits removed/modified.
        - data_crosstalk: Data containing *only* the removed crosstalk hits.
    """
    
    # --- 1. Initial Setup ---
    
    # Handle empty input data
    if not data_raw['PackageID'].size:
        # Return an empty dataset and an empty crosstalk dataset
        empty_crosstalk = {
            col: np.array([], dtype=dt) for col, dt in data_raw.items()
        }
        return data_raw, empty_crosstalk

    # Validate mode
    if mode not in ['filter', 'add']:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'filter' or 'add'.")

    # Convert the dictionary of arrays to a Pandas DataFrame
    try:
        df = pd.DataFrame(data_raw)
    except ValueError as e:
        print(f"Error creating DataFrame. Check array lengths: {e}")
        # Print array lengths for debugging
        for key, arr in data_raw.items():
            print(f"  {key}: {len(arr)}")
        raise

    # Add a unique 'original_index' to track each hit
    df['original_index'] = np.arange(len(df))

    # --- 2. Step 1: Pre-filter hits with ToT == 0 ---
    
    # Find hits with ToT == 0
    zero_tot_mask = (df['ToT'] == 0)
    crosstalk_zero_df = df[zero_tot_mask]
    
    # Keep only hits with ToT > 0 for the main processing
    df = df[~zero_tot_mask]

    # This list will store all DataFrames of hits identified as crosstalk
    crosstalk_dfs_list = [crosstalk_zero_df]

    # --- 3. Steps 2-5: Iterative Filtering ---
    
    for i in progress_bar(range(n), description="Processing crosstalk", total=n):
        
        # If no data is left to process, stop iterating
        if df.empty:
            if i > 0: # Only print if not the first iteration
                 sys.stdout.write("\nFinished early: no data left to process.\n")
            break

        # --- Step 2: Find all pairs ---
        # Self-merge on the group keys. This is a vectorized way
        # to get all pairs within the same TriggerID, Layer, and Column.
        merged = pd.merge(df, df, on=['TriggerID', 'Layer', 'Column'], suffixes=('_1', '_2'))

        # Keep only unique pairs (A, B) where A != B
        # We use original_index to ensure we don't pair a hit with itself
        # and to get unique (A, B) pairs instead of (A, B) and (B, A).
        pairs = merged[merged['original_index_1'] < merged['original_index_2']]

        # If no pairs are found, no more crosstalk can be identified
        if pairs.empty:
            if i > 0: # Only print if not the first iteration
                sys.stdout.write("\nFinished early: no new pairs found.\n")
            break

        # --- Step 3: Ignore pairs based on conditions i) and ii) ---
        row_sep = np.abs(pairs['Row_1'] - pairs['Row_2'])
        
        # Condition i)
        cond_i = (pairs['ToT_1'] <= ToTthreshMin) & (pairs['ToT_2'] <= ToTthreshMin)
        # Condition ii)
        cond_ii = (row_sep <= dx)
        
        # Keep only the pairs that we *don't* ignore
        valid_pairs = pairs[~(cond_i | cond_ii)]

        if valid_pairs.empty:
            if i > 0:
                sys.stdout.write("\nFinished early: no valid pairs found after filtering.\n")
            break

        # --- Step 4: Filter remaining pairs by ToT ratio ---
        tot_1 = valid_pairs['ToT_1']
        tot_2 = valid_pairs['ToT_2']

        larger_tot = np.maximum(tot_1, tot_2)
        smaller_tot = np.minimum(tot_1, tot_2)

        # Calculate ratio. Use np.divide for safe division (handles larger_tot=0)
        # We already filtered ToT=0, so larger_tot should not be 0, but this is safer.
        ratio = np.divide(smaller_tot, larger_tot, 
                          out=np.full_like(smaller_tot, np.nan, dtype=np.float64), 
                          where=larger_tot!=0)

        # Find pairs that satisfy the crosstalk ratio condition
        crosstalk_mask = (ratio <= ToTrat)
        crosstalk_pairs = valid_pairs[crosstalk_mask]

        if crosstalk_pairs.empty:
            if i > 0:
                sys.stdout.write("\nFinished early: no new crosstalk identified.\n")
            break
            
        # --- Identify hits to remove/modify ---
        
        # Find original_index of the "smaller" hit in each crosstalk pair
        indices_1_smaller = crosstalk_pairs.loc[crosstalk_pairs['ToT_1'] < crosstalk_pairs['ToT_2'], 'original_index_1']
        indices_2_smaller = crosstalk_pairs.loc[crosstalk_pairs['ToT_2'] < crosstalk_pairs['ToT_1'], 'original_index_2']

        # Get a unique set of all "smaller" hits to be removed
        all_smaller_indices = pd.concat([indices_1_smaller, indices_2_smaller]).unique()

        # If no hits are identified, we are done
        if len(all_smaller_indices) == 0:
            if i > 0:
                sys.stdout.write("\nFinished early: no crosstalk hits to remove.\n")
            break

        # Store these crosstalk hits before removing them
        crosstalk_hits_this_round_df = df[df['original_index'].isin(all_smaller_indices)]
        crosstalk_dfs_list.append(crosstalk_hits_this_round_df)
        
        # --- Apply 'filter' or 'add' logic ---
        
        if mode == 'add':
            # We need to add the ToT of the smaller hit to the larger hit
            
            # 1. Find pairs where 1 is smaller
            pairs_1_smaller = crosstalk_pairs[crosstalk_pairs['ToT_1'] < crosstalk_pairs['ToT_2']]
            # 2. Find pairs where 2 is smaller
            pairs_2_smaller = crosstalk_pairs[crosstalk_pairs['ToT_2'] < crosstalk_pairs['ToT_1']]

            # For each "larger" hit, sum all "smaller" ToTs paired with it
            adds_for_index_2 = pairs_1_smaller.groupby('original_index_2')['ToT_1'].sum()
            adds_for_index_1 = pairs_2_smaller.groupby('original_index_1')['ToT_2'].sum()

            # Combine the additions
            total_additions = adds_for_index_1.add(adds_for_index_2, fill_value=0)

            # Use original_index to efficiently add ToT values
            df = df.set_index('original_index')
            df['ToT'] = df['ToT'].add(total_additions, fill_value=0)
            
            # Remove the smaller hits *after* adding their values
            df = df[~df.index.isin(all_smaller_indices)]
            df = df.reset_index()

        else: # mode == 'filter'
            # Just remove the smaller hits
            df = df[~df['original_index'].isin(all_smaller_indices)]

    # --- 4. Final Conversion ---
    
    # Combine all identified crosstalk hits
    if crosstalk_dfs_list:
        all_crosstalk_df = pd.concat(crosstalk_dfs_list).drop_duplicates(subset=['original_index'])
        # Sort by index to maintain original order
        all_crosstalk_df = all_crosstalk_df.sort_values('original_index')
    else:
        # Create an empty DF with the correct columns
        all_crosstalk_df = pd.DataFrame(columns=df.columns)

    # Drop the temporary 'original_index' column
    df_filtered = df.drop(columns=['original_index'])
    df_crosstalk = all_crosstalk_df.drop(columns=['original_index'])

    # Convert DataFrames back to the original dictionary format
    # This loop also ensures the original data types are preserved
    data_filtered = {}
    data_crosstalk = {}
    
    for col_name, original_array in data_raw.items():
        if col_name in df_filtered:
            data_filtered[col_name] = df_filtered[col_name].to_numpy(dtype=original_array.dtype)
        if col_name in df_crosstalk:
            data_crosstalk[col_name] = df_crosstalk[col_name].to_numpy(dtype=original_array.dtype)

    return data_filtered, data_crosstalk



def remove_crosstalk2(data_raw, corr_pairs, trigger_col='TriggerID', 
                       ratio_ToTthresh1=0.2, ratio_ToTthresh2=0.8, 
                       noise_ToTthresh=10):
    """
    Identifies and separates crosstalk hits based on ToT ratios and spatial correlation.
    
    Optimized for speed by avoiding full Cartesian products of hits.
    
    Parameters:
    -----------
    data_raw : dict or pd.DataFrame
        The raw data containing 'Layer', 'Column', 'Row', 'ToT', etc.
    corr_pairs : pd.DataFrame
        Table defining coupled rows with columns ['Row_A', 'Row_B', ...].
    trigger_col : str
        The column name to use for grouping events ('TriggerID' or 'TriggerTS').
    ratio_ToTthresh1 : float
        Lower threshold for ToT ratio.
    ratio_ToTthresh2 : float
        Upper threshold for ToT ratio.
    noise_ToTthresh : float
        Threshold for absolute ToT value to classify as noise (Type 2).

    Returns:
    --------
    data_clean : pd.DataFrame
        Dataset with crosstalk hits removed.
    crosstalk_data : pd.DataFrame
        Dataset containing only the filtered hits, with an added 'crosstalk_type' column.
    """

    df = pd.DataFrame(data_raw) if isinstance(data_raw, dict) else data_raw.copy()
    df['orig_id'] = df.index
    links_fwd = corr_pairs[['Row_A', 'Row_B']].rename(columns={'Row_A': 'Row', 'Row_B': 'TargetRow'})
    links_rev = corr_pairs[['Row_B', 'Row_A']].rename(columns={'Row_B': 'Row', 'Row_A': 'TargetRow'})
    neighbor_map = pd.concat([links_fwd, links_rev], ignore_index=True).drop_duplicates()
    merge_keys = ['Layer', 'Column', trigger_col]
    cols_needed = merge_keys + ['Row', 'ToT', 'orig_id']
    df_slim = df[cols_needed]
    df_with_expectations = df_slim.merge(neighbor_map, on='Row', how='inner')
    if df_with_expectations.empty:
        # No hits occurred in rows that have correlations
        return df.drop(columns=['orig_id']), pd.DataFrame(columns=df.columns.tolist() + ['crosstalk_type'])
    candidates = df_with_expectations.merge(
        df_slim,
        left_on=merge_keys + ['TargetRow'],
        right_on=merge_keys + ['Row'],
        suffixes=('_1', '_2')
    )
    candidates = candidates[candidates['orig_id_1'] < candidates['orig_id_2']]
    if candidates.empty:
        return df.drop(columns=['orig_id']), pd.DataFrame(columns=df.columns.tolist() + ['crosstalk_type'])
    tot_1 = candidates['ToT_1'].values
    tot_2 = candidates['ToT_2'].values
    id_1 = candidates['orig_id_1'].values
    id_2 = candidates['orig_id_2'].values  
    min_tot = np.minimum(tot_1, tot_2)
    max_tot = np.maximum(tot_1, tot_2)
    is_1_min = tot_1 < tot_2
    id_min = np.where(is_1_min, id_1, id_2)
    id_max = np.where(is_1_min, id_2, id_1)
    ratio = np.divide(min_tot, max_tot, out=np.zeros_like(min_tot, dtype=float), where=max_tot!=0)
    ids_to_flag = []
    types_to_flag = []
    mask_noise = (max_tot < noise_ToTthresh)
    mask_t1 = ratio < ratio_ToTthresh1
    if np.any(mask_t1):
        ids_to_flag.append(id_min[mask_t1])
        types_to_flag.append(np.full(np.sum(mask_t1), 1))

    # Type 2: Ratio >= Thresh1 AND Both < Noise -> Filter Both
    mask_t2 = (ratio >= ratio_ToTthresh1) & mask_noise
    if np.any(mask_t2):
        ids_to_flag.extend([id_min[mask_t2], id_max[mask_t2]])
        count = np.sum(mask_t2)
        types_to_flag.extend([np.full(count, 2), np.full(count, 2)])

    # Type 3: Thresh1 <= Ratio < Thresh2 (and not noise) -> Filter Lower ToT
    mask_t3 = (~mask_t1) & (~mask_noise) & (ratio < ratio_ToTthresh2)
    if np.any(mask_t3):
        ids_to_flag.append(id_min[mask_t3])
        types_to_flag.append(np.full(np.sum(mask_t3), 3))

    # Type 4: Ratio >= Thresh2 (and not noise) -> Filter Both
    mask_t4 = (~mask_t1) & (~mask_noise) & (ratio >= ratio_ToTthresh2)
    if np.any(mask_t4):
        ids_to_flag.extend([id_min[mask_t4], id_max[mask_t4]])
        count = np.sum(mask_t4)
        types_to_flag.extend([np.full(count, 4), np.full(count, 4)])

    # 6. Aggregate Results
    if not ids_to_flag:
        return df.drop(columns=['orig_id']), pd.DataFrame(columns=df.columns.tolist() + ['crosstalk_type'])

    # Concatenate all flagged hits
    flat_ids = np.concatenate(ids_to_flag)
    flat_types = np.concatenate(types_to_flag)
    flag_df = pd.DataFrame({'id': flat_ids, 'type': flat_types})
    flag_df = flag_df.sort_values('type', ascending=False).drop_duplicates('id')
    is_crosstalk = df['orig_id'].isin(flag_df['id'])
    type_map = flag_df.set_index('id')['type']
    crosstalk_data = df[is_crosstalk].copy()
    data_clean = df[~is_crosstalk].copy()
    crosstalk_data['crosstalk_type'] = crosstalk_data['orig_id'].map(type_map)
    crosstalk_data = crosstalk_data.drop(columns=['orig_id'])
    data_clean = data_clean.drop(columns=['orig_id'])
    return data_clean, crosstalk_data

def split_crosstalk_types(crosstalk_data):
    """
    Splits the crosstalk dataset into a dictionary of DataFrames, 
    one for each crosstalk type found.
    
    Parameters:
    -----------
    crosstalk_data : pd.DataFrame
        The DataFrame returned by identify_crosstalk (must contain 'crosstalk_type').
        
    Returns:
    --------
    dict
        Dictionary where keys are the crosstalk type (e.g., 1, 2, 3, 4) and values 
        are the corresponding filtered DataFrames.
    """
    if 'crosstalk_type' not in crosstalk_data.columns:
        raise ValueError("Dataset does not contain 'crosstalk_type' column.")
    unique_types = sorted(crosstalk_data['crosstalk_type'].unique())
    return {t: crosstalk_data[crosstalk_data['crosstalk_type'] == t].copy() for t in unique_types}


def remove_crosstalk_types(data_raw, corr_pairs, trigger_col='TriggerID', 
                           ratio_ToTthresh=0.2, noise_ToTthresh=20, dx=1,
                           n_chunks=20):
    """
    Identifies and separates crosstalk hits based on ToT ratios and spatial correlation.
    Integrates a progress bar and chunked processing for speed/memory optimization.
    
    Parameters:
    -----------
    data_raw : dict or pd.DataFrame
        Raw data containing 'Layer', 'Column', 'Row', 'ToT', etc.
    corr_pairs : pd.DataFrame
        Table defining coupled rows with columns ['Row_A', 'Row_B'].
    trigger_col : str
        The column name to use for grouping events.
    ratio_ToTthresh : float
        Threshold for ToT ratio (Ratio < thresh -> Type 1; Ratio >= thresh -> Type 2).
    noise_ToTthresh : float
        Threshold for absolute ToT value to classify as noise.
    dx : int
        Minimum row separation required for crosstalk candidates.
    n_chunks : int
        Number of batches to split the processing into for the progress bar.

    Returns:
    --------
    data_clean, crosstalk_type1, crosstalk_type2 : pd.DataFrame
    """
    
    # --- 1. Global Setup ---
    df = pd.DataFrame(data_raw) if isinstance(data_raw, dict) else data_raw.copy()
    
    # Create a unique identifier for tracking rows (using original index)
    if 'orig_id' not in df.columns:
        df['orig_id'] = df.index

    # Prepare Neighbor Map (spatial expectations)
    # We do this once as it applies to all chunks
    links_fwd = corr_pairs[['Row_A', 'Row_B']].rename(columns={'Row_A': 'Row', 'Row_B': 'TargetRow'})
    links_rev = corr_pairs[['Row_B', 'Row_A']].rename(columns={'Row_B': 'Row', 'Row_A': 'TargetRow'})
    neighbor_map = pd.concat([links_fwd, links_rev], ignore_index=True).drop_duplicates()

    # --- 2. Chunking Logic ---
    # We split unique Triggers into chunks. This prevents the 'merge' step from 
    # creating massive intermediate tables if the dataset is large.
    unique_triggers = df[trigger_col].unique()
    
    # Adjust chunks if dataset is small
    actual_chunks = min(n_chunks, len(unique_triggers)) if len(unique_triggers) > 0 else 1
    trigger_batches = np.array_split(unique_triggers, actual_chunks)
    
    # Containers for results
    remove_ids_list = []
    type1_dfs = []
    type2_dfs = []

    # --- 3. Processing Loop ---
    # We iterate through batches of events
    for batch_triggers in progress_bar(trigger_batches, description="Processing Crosstalk", total=len(trigger_batches)):
        
        # Fast filtering using isin
        # (Optimization note: if df is sorted by trigger_col, slicing is faster, 
        # but isin is robust for unsorted data)
        mask_batch = df[trigger_col].isin(batch_triggers)
        df_chunk = df[mask_batch]
        
        if df_chunk.empty:
            continue

        # Call the optimized kernel for this chunk
        rem_ids, t1, t2 = _process_chunk_kernel(
            df_chunk, neighbor_map, trigger_col, 
            ratio_ToTthresh, noise_ToTthresh, dx
        )
        
        if len(rem_ids) > 0:
            remove_ids_list.append(rem_ids)
        if not t1.empty:
            type1_dfs.append(t1)
        if not t2.empty:
            type2_dfs.append(t2)

    # --- 4. Aggregation & Cleanup ---
    print("Aggregating results...")
    
    # Combine crosstalk datasets
    cols_out = df.columns.tolist() + ['noise_correlation']
    
    crosstalk_type1 = pd.concat(type1_dfs, ignore_index=True) if type1_dfs else pd.DataFrame(columns=cols_out)
    crosstalk_type2 = pd.concat(type2_dfs, ignore_index=True) if type2_dfs else pd.DataFrame(columns=cols_out)

    # Construct Clean Dataset
    # Instead of dropping row-by-row, we use the collected IDs to filter once
    if remove_ids_list:
        all_remove_ids = np.concatenate(remove_ids_list)
        # Using a set for faster lookup is often faster for very large lists, 
        # but numpy isin is quite optimized
        is_crosstalk = df['orig_id'].isin(all_remove_ids)
        data_clean = df[~is_crosstalk].copy()
    else:
        data_clean = df.copy()

    # Drop the helper column
    data_clean.drop(columns=['orig_id'], inplace=True, errors='ignore')
    crosstalk_type1.drop(columns=['orig_id'], inplace=True, errors='ignore')
    crosstalk_type2.drop(columns=['orig_id'], inplace=True, errors='ignore')

    return data_clean, crosstalk_type1, crosstalk_type2


def _process_chunk_kernel(df, neighbor_map, trigger_col, ratio_thresh, noise_thresh, dx):
    """
    Internal optimized kernel to process a subset of data.
    Uses Vectorized NumPy operations for speed.
    """
    cols_merge = ['Layer', 'Column', trigger_col]
    cols_slim = cols_merge + ['Row', 'ToT', 'orig_id']
    
    df_slim = df[cols_slim]
    
    # 1. Filter rows that have potential neighbors (Inner Join)
    df_expect = df_slim.merge(neighbor_map, on='Row', how='inner')
    
    if df_expect.empty:
        empty = pd.DataFrame(columns=df.columns.tolist() + ['noise_correlation'])
        return np.array([]), empty, empty

    # 2. Find Candidates (Self-Join on Merge Keys + TargetRow matches Row)
    # This is usually the bottleneck. Doing it on a chunk reduces memory pressure.
    candidates = df_expect.merge(
        df_slim,
        left_on=cols_merge + ['TargetRow'],
        right_on=cols_merge + ['Row'],
        suffixes=('_1', '_2')
    )
    
    # 3. Apply Filters (Vectorized)
    # Enforce ID ordering to avoid double counting pairs
    # Enforce DX row separation
    # Using .values for speed (avoids index alignment overhead)
    
    id1 = candidates['orig_id_1'].values
    id2 = candidates['orig_id_2'].values
    r1 = candidates['Row_1'].values
    r2 = candidates['Row_2'].values
    
    # Combined mask: ID1 < ID2 AND abs(Row1 - Row2) >= dx
    valid_mask = (id1 < id2) & (np.abs(r1 - r2) >= dx)
    
    candidates = candidates[valid_mask]
    
    if candidates.empty:
        empty = pd.DataFrame(columns=df.columns.tolist() + ['noise_correlation'])
        return np.array([]), empty, empty

    # 4. Calculate Ratios & Logic
    tot_1 = candidates['ToT_1'].values
    tot_2 = candidates['ToT_2'].values
    id_1 = candidates['orig_id_1'].values  # Refresh after filter
    id_2 = candidates['orig_id_2'].values
    
    min_tot = np.minimum(tot_1, tot_2)
    max_tot = np.maximum(tot_1, tot_2)
    
    # Safe Division
    ratio = np.divide(min_tot, max_tot, out=np.zeros_like(min_tot, dtype=float), where=max_tot!=0)

    # 5. Classify
    is_1_min = tot_1 < tot_2
    id_min = np.where(is_1_min, id_1, id_2)
    
    # Noise Flag: Both hits <= noise threshold
    is_noise_pair = (tot_1 <= noise_thresh) & (tot_2 <= noise_thresh)
    
    # Types
    is_type1 = ratio < ratio_thresh
    is_type2 = ratio >= ratio_thresh
    
    # --- Collection Lists ---
    ids_to_remove = []
    
    # Helper to build result DF
    def extract_rows(mask, ids_pair_flat, noise_flag_val):
        if not np.any(mask):
            return pd.DataFrame()
        
        # Unique IDs involved in this specific crosstalk type
        unique_ids = np.unique(ids_pair_flat)
        
        # Get subset from original chunk
        # Note: We must map the noise_flag back. 
        # If an ID is in multiple pairs, we just take the first occurrence's flag logic 
        # or prioritize signal (1). Simpler here is just taking the subset.
        
        subset = df[df['orig_id'].isin(unique_ids)].copy()
        subset['noise_correlation'] = noise_flag_val
        return subset

    # === TYPE 1 ===
    # T1 Noise: Remove Both
    m_t1_noise = is_type1 & is_noise_pair
    if np.any(m_t1_noise):
        ids_to_remove.append(np.concatenate([id_1[m_t1_noise], id_2[m_t1_noise]]))
    
    # T1 Signal: Remove Smaller
    m_t1_sig = is_type1 & (~is_noise_pair)
    if np.any(m_t1_sig):
        ids_to_remove.append(id_min[m_t1_sig])
        
    # Compile Type 1 DF
    # We combine noise and signal hits for the output dataset
    t1_ids_noise = np.concatenate([id_1[m_t1_noise], id_2[m_t1_noise]]) if np.any(m_t1_noise) else []
    t1_ids_sig   = np.concatenate([id_1[m_t1_sig], id_2[m_t1_sig]]) if np.any(m_t1_sig) else []
    
    df_t1_noise = extract_rows(m_t1_noise, t1_ids_noise, 0)
    df_t1_sig   = extract_rows(m_t1_sig,   t1_ids_sig,   1)
    
    # === TYPE 2 ===
    # T2 Noise: Remove Both
    m_t2_noise = is_type2 & is_noise_pair
    if np.any(m_t2_noise):
        ids_to_remove.append(np.concatenate([id_1[m_t2_noise], id_2[m_t2_noise]]))
        
    # T2 Signal: Remove NOTHING (Keep both)
    m_t2_sig = is_type2 & (~is_noise_pair)
    # No append to ids_to_remove
    
    # Compile Type 2 DF
    t2_ids_noise = np.concatenate([id_1[m_t2_noise], id_2[m_t2_noise]]) if np.any(m_t2_noise) else []
    t2_ids_sig   = np.concatenate([id_1[m_t2_sig], id_2[m_t2_sig]]) if np.any(m_t2_sig) else []
    
    df_t2_noise = extract_rows(m_t2_noise, t2_ids_noise, 0)
    df_t2_sig   = extract_rows(m_t2_sig,   t2_ids_sig,   1)

    # Combine internal results
    ids_out = np.concatenate(ids_to_remove) if ids_to_remove else np.array([])
    
    df_t1_out = pd.concat([df_t1_noise, df_t1_sig], ignore_index=True)
    df_t2_out = pd.concat([df_t2_noise, df_t2_sig], ignore_index=True)
    
    return ids_out, df_t1_out, df_t2_out


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm, Normalize
from scipy.signal import find_peaks, peak_widths


def _process_group(
    df_col: pd.DataFrame, 
    group_by: str, 
    tot_ratio_threshold: float,
    noise_tot_threshold: float
) -> Optional[pd.DataFrame]:
    """
    Processes a pre-filtered group (Layer/Column) to return both Correlation and Hit Counts.
    Returns a DataFrame indexed by (Row_A, Row_B) with columns ['correlation', 'hits'].
    """
    # Need at least 2 hits and 2 different rows
    if df_col.shape[0] < 2 or df_col['Row'].nunique() < 2:
        return None
    
    # --- Pair generation logic ---
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
    
    # Filter out self-matches and self-correlations
    merged_pairs = merged_pairs[
        (merged_pairs['HitID_A'] != merged_pairs['HitID_B']) & 
        (merged_pairs['Row_A'] != merged_pairs['Row_B'])
    ]
    
    if merged_pairs.empty:
        return None

    # Calculate ToT Ratio and Arrays
    tot_a = merged_pairs['ToT_A'].values.astype(float)
    tot_b = merged_pairs['ToT_B'].values.astype(float)
    
    max_tot = np.maximum(tot_a, tot_b)
    min_tot = np.minimum(tot_a, tot_b)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = min_tot / max_tot
        ratios[max_tot == 0] = 0 
        
    merged_pairs['ToT_Ratio'] = ratios
    
    # --- Apply Filters ---
    # 1. Ratio Filter: Must be within similarity threshold
    ratio_condition = merged_pairs['ToT_Ratio'] <= tot_ratio_threshold
    
    # 2. Noise Filter: At least one hit must be > noise_tot_threshold
    #    (Prevents two weak noise pixels from counting as a correlation)
    noise_condition = (tot_a > noise_tot_threshold) | (tot_b > noise_tot_threshold)
    
    valid_pairs = merged_pairs[ratio_condition & noise_condition]
    
    if valid_pairs.empty:
        return None

    # --- Vectorized Matrix Construction ---
    # Count co-occurrences (Hits)
    pair_counts = valid_pairs.groupby(['Row_A', 'Row_B']).size()
    
    # --- Normalization (Cosine Similarity) ---
    row_hit_counts = df_col['Row'].value_counts()
    
    # Align indices
    n_i = row_hit_counts.loc[pair_counts.index.get_level_values(0)].values
    n_j = row_hit_counts.loc[pair_counts.index.get_level_values(1)].values
    
    denominator = np.sqrt(n_i * n_j)
    normalized_values = np.divide(pair_counts.values, denominator, where=denominator!=0)
    
    # Return DataFrame with both metrics
    return pd.DataFrame({
        'correlation': normalized_values,
        'hits': pair_counts.values
    }, index=pair_counts.index)


def calculate_aggregated_correlation(
    data_raw: dict, 
    group_by: str = 'TriggerID', 
    tot_ratio_threshold: float = 0.2,
    noise_tot_threshold: float = 20.0,
    target_layer: Optional[int] = None,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates weighted averaged correlation matrix and total hit count matrix across groups.
    
    Parameters:
    -----------
    noise_tot_threshold : float
        Minimum ToT required for at least one hit in a pair for it to count.
    target_layer : int, optional
        If provided (e.g., 1, 2, 3, 4), only analyzes that layer.
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame(data_raw) if isinstance(data_raw, dict) else data_raw

    if group_by not in df.columns:
        raise ValueError(f"Grouping column '{group_by}' not found.")

    # --- Layer Filtering ---
    if target_layer is not None:
        if 'Layer' not in df.columns:
             raise ValueError("Column 'Layer' missing from data, cannot filter by layer.")
        df = df[df['Layer'] == target_layer]
        if df.empty:
            print(f"Warning: No data found for Layer {target_layer}")
            return pd.DataFrame(), pd.DataFrame()

    # 2. Prepare Tasks
    groups = [g for _, g in df.groupby(['Layer', 'Column'])]
    
    if not groups:
        return pd.DataFrame(), pd.DataFrame()

    # 3. Parallel Execution
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_group)(
            df_col=group, 
            group_by=group_by, 
            tot_ratio_threshold=tot_ratio_threshold,
            noise_tot_threshold=noise_tot_threshold
        ) 
        for group in groups
    )
    
    valid_dfs = [res for res in results if res is not None]
    
    if not valid_dfs:
        return pd.DataFrame(), pd.DataFrame()

    # 4. Aggregation (Weighted Average Logic)
    print("Aggregating results...")
    combined = pd.concat(valid_dfs)
    
    # Pre-calculate the weighted component (Correlation * Hits)
    combined['weighted_component'] = combined['correlation'] * combined['hits']

    # Group by pixel pair (Row_A, Row_B)
    agg_df = combined.groupby(level=[0, 1]).agg({
        'weighted_component': 'sum', # Sum of (Corr * Hits)
        'hits': 'sum'                # Total Hits
    })
    
    # Calculate weighted mean: Sum(Corr * Hits) / Sum(Hits)
    # Using .values to avoid pandas/numpy stack overflow on Windows
    numer = agg_df['weighted_component'].values
    denom = agg_df['hits'].values
    
    weighted_corr_values = np.divide(
        numer, 
        denom, 
        where=denom!=0
    )
    
    agg_df['weighted_correlation'] = weighted_corr_values
    
    # Unstack to create matrices
    corr_matrix = agg_df['weighted_correlation'].unstack()
    hit_matrix = agg_df['hits'].unstack()
    
    # 5. Symmetrization
    all_idx = corr_matrix.index.union(corr_matrix.columns).sort_values()
    
    def symmetrize(mat):
        mat = mat.reindex(index=all_idx, columns=all_idx)
        np.fill_diagonal(mat.values, np.nan)
        return (mat.fillna(0) + mat.T.fillna(0)) / 2

    avg_corr = symmetrize(corr_matrix)
    total_hits = symmetrize(hit_matrix)
    
    return avg_corr, total_hits


# --- PLOTTING FUNCTIONS ---

def plot_correlation_matrix(correlation_matrix: pd.DataFrame, title='Global Correlation Matrix'):
    """Plots standard heatmap."""
    if correlation_matrix is None or correlation_matrix.empty:
        print("No correlation data to plot.")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        cmap='viridis',
        square=True,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.show()
import matplotlib.colors as colors
from scipy.stats import binned_statistic_2d
def plot_corr_disp(df, mode='heatmap', log_z=False, bins_y=2**8):
    """
    Plots Correlation analysis in two modes.
    
    Mode 'heatmap':
        X-Axis: Displacement (or Mean Location)
        Y-Axis: Correlation Coefficient
        Z-Axis (Color): Count of pairs (Density)
        
    Mode 'scatter':
        X-Axis: Displacement (or Mean Location)
        Left Y-Axis: Correlation Coefficient (Blue Dots)
        Right Y-Axis (Red Line): Count of pairs summing to that X value.
    
    Parameters:
    - df: DataFrame with 'Row_A', 'Row_B', 'Displacement', 'Correlation'.
    - mode: 'heatmap' or 'scatter'.
    - log_z: (Heatmap only) If True, uses logarithmic color scaling for the counts.
    - bins_y: (Heatmap only) Number of horizontal bins for the Correlation axis.
    """
    
    # 1. Prepare Data
    mean_loc = (df['Row_A'] + df['Row_B']) / 2
    
    # Define metrics: (Data Array, Label, Column Name)
    metrics = [
        (df['Displacement'], 'Displacement', 'Displacement'),
        (mean_loc, 'Mean Pixel Location', 'MeanLoc')
    ]
    
    # Ensure MeanLoc exists
    df['MeanLoc'] = mean_loc

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    for ax, (x_data, x_label, x_col) in zip(axes, metrics):
        
        # Prepare X-axis (Round MeanLoc to 0.5 for cleaner discrete steps)
        if x_col == 'MeanLoc':
             current_x = df[x_col].apply(lambda x: np.floor(x * 2) / 2)
             step = 0.5
        else:
             current_x = df[x_col]
             step = 1.0

        if mode == 'heatmap':
            # --- HISTOGRAM STRATEGY ---
            
            # 1. Define Bins
            # X-Axis: Linear steps based on data range
            min_x, max_x = current_x.min(), current_x.max()
            x_bins = np.arange(min_x - (step/2), max_x + (step*1.5), step)
            
            # Y-Axis: Linear steps from 0 to Max Correlation
            max_corr = df['Correlation'].max()
            # Ensure we go slightly above max to include it
            y_bins = np.linspace(0, max_corr * 1.05, bins_y + 1)
            
            # 2. Compute 2D Histogram (Counts)
            # H returns shape (len(x_bins)-1, len(y_bins)-1)
            H, x_edges, y_edges = np.histogram2d(
                current_x, 
                df['Correlation'], 
                bins=[x_bins, y_bins]
            )
            
            # 3. Plotting
            # Transpose H because pcolormesh/imshow expects (rows=y, cols=x)
            Z = H.T
            
            # Mask zeros so they appear white (cleaner look)
            Z_masked = np.ma.masked_where(Z == 0, Z)
            
            # Color Norm
            norm = None
            if log_z:
                norm = colors.LogNorm(vmin=1, vmax=np.max(Z))
            
            cmap = plt.cm.plasma
            cmap.set_bad('white') # Zero counts are white
            
            # Use pcolormesh for histograms as it handles edges automatically
            mesh = ax.pcolormesh(x_edges, y_edges, Z_masked, cmap=cmap, norm=norm, shading='flat')
            
            # Formatting
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label('Count of Pairs' + (' (Log)' if log_z else ''))
            
            ax.set_title(f'Density: Correlation vs {x_label}')
            ax.set_ylabel('Correlation Coefficient')

        elif mode == 'scatter':
            # --- SCATTER STRATEGY ---
            
            # 1. Primary Axis: Scatter (Correlation)
            ax.scatter(current_x, df['Correlation'], alpha=0.5, s=10, c='tab:blue', edgecolors='none', label='Correlation')
            ax.set_ylabel('Correlation Coefficient', color='tab:blue')
            ax.tick_params(axis='y', labelcolor='tab:blue')
            
            # 2. Secondary Axis: Line (Counts of pairs at this X)
            ax2 = ax.twinx()
            
            # Group by the X value to get the count of pairs
            counts = df.groupby(current_x)['Correlation'].count().sort_index()
            
            # Plot Red Line
            ax2.plot(counts.index, counts.values, color='red', linewidth=2, alpha=0.8, label='Pair Count')
            
            ax2.set_ylabel('Count of Pairs', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f'Correlation vs {x_label} (with Pair Count)')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

        # Shared Formatting
        ax.set_xlabel(x_label)
        # Set X limits to match the bins
        if mode == 'heatmap':
             ax.set_xlim(x_bins[0], x_bins[-1])
        else:
             ax.set_xlim(current_x.min() - step, current_x.max() + step)
        
    plt.tight_layout()
    plt.show()
def plot_displacement_analysis(correlation_matrix: pd.DataFrame, 
                               hit_matrix: pd.DataFrame = None,
                               title_prefix: str = 'Displacement Analysis', 
                               log_y: bool = True,
                               prominence: float = 0.001,
                               marginal_metric: str = 'hits'):
    """
    Plots the displacement vs diagonal analysis with marginals.
    """
    if correlation_matrix is None or correlation_matrix.empty:
        print("No correlation data to plot.")
        return

    # --- 1. Data Preparation ---
    rows = correlation_matrix.index.values.astype(np.int64)
    cols = correlation_matrix.columns.values.astype(np.int64)
    
    R_grid, C_grid = np.meshgrid(rows, cols, indexing='ij')
    
    # Displacement & Diagonal
    displacement_matrix = C_grid - R_grid
    diagonal_matrix = C_grid + R_grid
    
    flat_disp = displacement_matrix.flatten()
    flat_diag = diagonal_matrix.flatten()
    flat_corr = correlation_matrix.values.flatten()

    # Determine what to plot in Marginals
    if marginal_metric == 'hits' and hit_matrix is not None:
        hit_matrix_aligned = hit_matrix.reindex(index=correlation_matrix.index, 
                                                columns=correlation_matrix.columns).fillna(0)
        flat_marginal_metric = hit_matrix_aligned.values.flatten()
        marginal_label = "Total Hits"
    else:
        if marginal_metric == 'hits' and hit_matrix is None:
            print("Warning: 'hits' requested but hit_matrix not provided. Falling back to correlation.")
        flat_marginal_metric = np.nan_to_num(flat_corr, nan=0.0)
        marginal_label = "Sum Correlation"
    
    plot_df = pd.DataFrame({
        'Displacement': flat_disp,
        'Diagonal': flat_diag,
        'Correlation': flat_corr,
        'MarginalMetric': flat_marginal_metric
    })

    # --- 2. Calculate Marginals ---
    non_self_df = plot_df[plot_df['Displacement'] != 0]
    sum_disp = non_self_df.groupby('Displacement')['MarginalMetric'].sum().sort_index()
    sum_diag = non_self_df.groupby('Diagonal')['MarginalMetric'].sum().sort_index()

    # --- 3. Figure Layout ---
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.3, 0.05], height_ratios=[0.4, 1], 
                          wspace=0.1, hspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])            
    ax_rot = fig.add_subplot(gs[1, 0], sharex=ax_top) 
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_rot) 
    ax_cbar = fig.add_subplot(gs[1, 2])            

    # --- 4. Plot Config ---
    max_corr_val = np.nanmax(correlation_matrix.values)
    
    # Calculate Min Positive Correlation (Dynamic)
    positive_vals = correlation_matrix.values[correlation_matrix.values > 0]
    if len(positive_vals) > 0:
        min_corr_val = np.nanmin(positive_vals)
    else:
        min_corr_val = 1e-4 

    norm = LogNorm(vmin=min_corr_val, vmax=max_corr_val) if log_y else Normalize(vmin=0, vmax=max_corr_val)

    # --- 5. CENTER TOP: X-Marginal (Displacement) ---
    ax_top.plot(sum_disp.index, sum_disp.values, color='#1f77b4')
    
    # Set scale BEFORE annotation logic to ensure limits are handled in the correct domain
    if log_y: ax_top.set_yscale('log')

    def annotate_peaks(ax, x, y):
        curr_prominence = prominence
        if marginal_label == "Total Hits":
             curr_prominence = max(prominence, np.max(y) * 0.01)

        peaks, _ = find_peaks(y, prominence=curr_prominence)
        
        # --- Fix for Overlap ---
        # We only modify the TOP limit. We leave the bottom limit (ymin) alone 
        # so Matplotlib can autoscale the data visible.
        if len(peaks) > 0 or np.max(y) > 0:
            data_max = np.max(y)
            if log_y:
                # In log scale, multiplying by 50 adds roughly 1.7 decades of headroom
                ax.set_ylim(top=data_max * 50) 
            else:
                ax.set_ylim(top=data_max * 1.4) 

        if len(peaks) == 0: return

        widths, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
        x_resolution = np.mean(np.diff(x)) if len(x) > 1 else 1.0
        
        placed_labels = [] 
        range_val = x[-1] - x[0]
        
        def get_offset(curr_pos, placed, axis_range):
            base = 25
            limit = axis_range * 0.1
            extra = 0
            for prev in placed:
                if abs(curr_pos - prev) < limit:
                    extra += 30
            return base + extra

        for i, peak_idx in enumerate(peaks):
            px = x[peak_idx]
            py = y[peak_idx]
            
            ax.plot(px, py, "x", color="red", markersize=8)
            ax.hlines(y=width_heights[i], xmin=x[0] + left_ips[i]*x_resolution, 
                      xmax=x[0] + right_ips[i]*x_resolution, color='red', linestyle='--', alpha=0.5)
            
            off = get_offset(px, placed_labels, range_val)
            val_fmt = f"{py:.1e}" if py > 1000 else f"{py:.1f}"
            label_text = f"D:{int(px)}\nSum:{val_fmt}"
            
            ax.annotate(label_text, xy=(px, py), xytext=(0, off),
                        textcoords='offset points', ha='center', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.7),
                        arrowprops=dict(arrowstyle="-", color="red", alpha=0.3))
            placed_labels.append(px)

    annotate_peaks(ax_top, sum_disp.index.values, sum_disp.values)

    ax_top.set_title(f"{title_prefix}\n({marginal_label} vs Displacement)", pad=20)
    ax_top.set_ylabel(f'{marginal_label}')
    ax_top.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # --- 6. CENTER BOTTOM: Heatmap ---
    row_corners = np.arange(len(rows) + 1)
    col_corners = np.arange(len(cols) + 1)
    RC, CC = np.meshgrid(row_corners, col_corners, indexing='ij')
    
    X_mesh = CC - RC
    Y_mesh = CC + RC
    
    mesh = ax_rot.pcolormesh(
        X_mesh, Y_mesh, correlation_matrix.values, 
        cmap='viridis', norm=norm, shading='flat', edgecolors='face'
    )
    
    ax_rot.set_xlabel('Pixel Displacement (RowA - RowB)')
    ax_rot.set_ylabel('Matrix Diagonal (RowA + RowB)')
    max_disp = np.max(np.abs(flat_disp))
    ax_rot.set_xlim(-max_disp, max_disp)
    ax_rot.invert_yaxis()

    # --- 7. RIGHT BOTTOM: Y-Marginal (Diagonal) ---
    ax_right.plot(sum_diag.values, sum_diag.index, color='#1f77b4')
    ax_right.set_xlabel(f'{marginal_label}')
    ax_right.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    if log_y: ax_right.set_xscale('log')

    # --- 8. Colorbar ---
    cbar = plt.colorbar(mesh, cax=ax_cbar, orientation='vertical')
    cbar.set_label(f"Correlation Coefficient {'(Log Scale)' if log_y else ''}")

    plt.show()
    
def plot_crosstalk_spectra(data_raw: pd.DataFrame,
                           data_clean: pd.DataFrame, 
                           crosstalk_type1: pd.DataFrame, 
                           crosstalk_type2: pd.DataFrame, 
                           log_yscale: bool = False):
    """
    Plots ToT spectra for Raw, Clean, and Removed partitions.
    Uses robust value-based matching (instead of index-based) to handle reset indices.
    
    Parameters:
    -----------
    data_raw : pd.DataFrame
        The original raw dataset.
    data_clean : pd.DataFrame
        The clean dataset.
    crosstalk_type1 : pd.DataFrame
        Type 1 crosstalk dataset.
    crosstalk_type2 : pd.DataFrame
        Type 2 crosstalk dataset.
    log_yscale : bool
        If True, sets the y-axis to logarithmic scale.
    """
    # 1. Identify all unique layers
    layers = sorted(data_raw['Layer'].unique())
    
    colors = {
        'Raw': '#bfbfbf',     # Light Grey
        'Clean': 'black',
        'Type 1': '#d62728',  # Red
        'Type 2': '#2ca02c',  # Green
        'Noise': '#7f7f7f',   # Dark Grey
        'Unknown': 'magenta'  # Debug color for mismatches
    }

    # Dynamic binning
    max_tot = data_raw['ToT'].max()
    upper_limit = max(256, int(np.ceil(max_tot))) if pd.notna(max_tot) else 256
    bins = np.arange(0, upper_limit + 2) - 0.5
    
    # 2. Identify Matching Keys
    # We need to link hits by value if indices are lost.
    possible_keys = ['Column', 'Row', 'ToT', 'TriggerID', 'TriggerTS']
    join_keys = [k for k in possible_keys if k in data_raw.columns]
    
    print(f"Linking hits using keys: {join_keys}")

    for layer in layers:
        plt.figure(figsize=(10, 6))
        
        # --- Slice Data (Layer-wise) ---
        l_raw = data_raw[data_raw['Layer'] == layer].copy()
        l_clean = data_clean[data_clean['Layer'] == layer].copy()
        l_t1 = crosstalk_type1[crosstalk_type1['Layer'] == layer].copy()
        l_t2 = crosstalk_type2[crosstalk_type2['Layer'] == layer].copy()
        
        # --- Prepare Lookup Table from Crosstalk Data ---
        # This table maps unique hits to their Crosstalk Type and Noise Status
        l_t1['__c_type'] = 1
        l_t2['__c_type'] = 2
        
        # Concatenate and keep only relevant columns for matching
        cols_lookup = join_keys + ['noise_correlation', '__c_type']
        lookup = pd.concat([l_t1[cols_lookup], l_t2[cols_lookup]], ignore_index=True)
        
        # Handle potential duplicates in crosstalk log (e.g. if a hit is in multiple pairs)
        # We prioritize the entry that provides the most info.
        if not lookup.empty:
            lookup = lookup.drop_duplicates(subset=join_keys)

        # --- Identify REMOVED vs KEPT Hits ---
        # We rely on indices for Raw vs Clean (assuming Clean is a direct subset of Raw)
        idx_raw = set(l_raw.index)
        idx_clean = set(l_clean.index)
        idx_removed = list(idx_raw - idx_clean)
        
        # Create DataFrames for merge
        df_removed = l_raw.loc[idx_removed]
        df_kept = l_clean # This is just l_clean
        
        # --- Match Removed Hits to Crosstalk Logs ---
        # Left Join: Raw(Removed) -> Lookup
        if not df_removed.empty and not lookup.empty:
            merged_rem = df_removed.merge(lookup, on=join_keys, how='left', indicator=True)
            
            # Categories
            mask_found = merged_rem['_merge'] == 'both'
            mask_noise = mask_found & (merged_rem['noise_correlation'] == 0)
            # Type 1 Signal (Removed)
            mask_t1 = mask_found & (merged_rem['noise_correlation'] == 1) & (merged_rem['__c_type'] == 1)
            # Type 2 Signal (Removed - Should be rare/none)
            mask_t2 = mask_found & (merged_rem['noise_correlation'] == 1) & (merged_rem['__c_type'] == 2)
            # Unknown (Removed but not in logs)
            mask_unknown = ~mask_found
            
            tot_rem_noise = merged_rem.loc[mask_noise, 'ToT']
            tot_rem_t1 = merged_rem.loc[mask_t1, 'ToT']
            tot_rem_t2 = merged_rem.loc[mask_t2, 'ToT']
            tot_rem_unknown = merged_rem.loc[mask_unknown, 'ToT']
        else:
            # Fallbacks
            tot_rem_noise = pd.Series(dtype=float)
            tot_rem_t1 = pd.Series(dtype=float)
            tot_rem_t2 = pd.Series(dtype=float)
            tot_rem_unknown = df_removed['ToT'] # All unknown if lookup empty

        # --- Match Kept Hits to Crosstalk Logs (Ghost/Dashed) ---
        if not df_kept.empty and not lookup.empty:
            merged_kept = df_kept.merge(lookup, on=join_keys, how='inner')
            
            mask_t1_kept = (merged_kept['noise_correlation'] == 1) & (merged_kept['__c_type'] == 1)
            mask_t2_kept = (merged_kept['noise_correlation'] == 1) & (merged_kept['__c_type'] == 2)
            
            tot_kept_t1 = merged_kept.loc[mask_t1_kept, 'ToT']
            tot_kept_t2 = merged_kept.loc[mask_t2_kept, 'ToT']
        else:
            tot_kept_t1 = pd.Series(dtype=float)
            tot_kept_t2 = pd.Series(dtype=float)

        # --- Verification ---
        n_raw = len(l_raw)
        n_clean = len(l_clean)
        n_rem_noise = len(tot_rem_noise)
        n_rem_t1 = len(tot_rem_t1)
        n_rem_t2 = len(tot_rem_t2)
        n_rem_unknown = len(tot_rem_unknown)
        
        sum_check = n_clean + n_rem_noise + n_rem_t1 + n_rem_t2 + n_rem_unknown
        
        print(f"Layer {layer}: Raw({n_raw}) == Clean({n_clean}) + Rem_Noise({n_rem_noise}) + "
              f"Rem_T1({n_rem_t1}) + Rem_T2({n_rem_t2}) + Unclassified({n_rem_unknown}). "
              f"Sum={sum_check}")
        
        if n_raw != sum_check:
             diff = n_raw - sum_check
             print(f"  WARNING: Sum mismatch {diff}.")

        # --- Plotting ---
        
        # 1. Raw Data (Background)
        plt.hist(l_raw['ToT'], bins=bins, histtype='stepfilled',
                 color=colors['Raw'], alpha=0.3, label=f'Raw ({n_raw})')
        
        # 2. Clean Data
        plt.hist(l_clean['ToT'], bins=bins, histtype='step',
                 linewidth=2, color=colors['Clean'], label=f'Clean ({n_clean})')

        # 3. Removed Components (Solid)
        if not tot_rem_t1.empty:
            plt.hist(tot_rem_t1, bins=bins, histtype='step', linewidth=1.5, 
                     color=colors['Type 1'], label=f'Removed T1 ({n_rem_t1})')
            
        if not tot_rem_t2.empty:
            plt.hist(tot_rem_t2, bins=bins, histtype='step', linewidth=1.5, 
                     color=colors['Type 2'], label=f'Removed T2 ({n_rem_t2})')
            
        if not tot_rem_noise.empty:
            plt.hist(tot_rem_noise, bins=bins, histtype='step', linewidth=1.5, 
                     color=colors['Noise'], label=f'Removed Noise ({n_rem_noise})')
            
        if not tot_rem_unknown.empty:
            plt.hist(tot_rem_unknown, bins=bins, histtype='step', linewidth=1.5, 
                     color=colors['Unknown'], linestyle=':', label=f'Unclassified ({n_rem_unknown})')

        # 4. Kept Components (Dashed)
        if not tot_kept_t1.empty:
            plt.hist(tot_kept_t1, bins=bins, histtype='step', linewidth=1.5, 
                     color=colors['Type 1'], linestyle='--', label=f'Kept T1 ({len(tot_kept_t1)})')
            
        if not tot_kept_t2.empty:
            plt.hist(tot_kept_t2, bins=bins, histtype='step', linewidth=1.5, 
                     color=colors['Type 2'], linestyle='--', label=f'Kept T2 ({len(tot_kept_t2)})')

        plt.title(f'ToT Spectrum Partition - Layer {layer}')
        plt.xlabel('Time over Threshold (ToT)')
        plt.ylabel('Count')
        plt.legend(loc='upper right', fontsize='small', ncol=1)
        plt.xlim(0, upper_limit + 1)
        plt.grid(alpha=0.2)
        
        if log_yscale:
            plt.yscale('log')
        
        plt.tight_layout()
        plt.show()