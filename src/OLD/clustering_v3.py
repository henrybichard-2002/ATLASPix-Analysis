import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from utils import progress_bar
from multiprocessing import Pool, cpu_count
from functools import partial

def _process_group(group, epsilon, crosstalk_tot_delta):
    """
    Processes a single package group for clustering and crosstalk detection.
    This helper function is designed to be called in parallel.
    """
    # Make a copy to avoid SettingWithCopyWarning later on
    group = group.copy()
    
    # A package with only one hit is noise by definition
    if len(group) == 1:
        group['HitType'] = 0
        group['cluster_label'] = -1
        return group

    # Use DBSCAN for spatial clustering. min_samples=2 treats single hits as noise.
    coords = group[['Row', 'Column']].values
    db = DBSCAN(eps=epsilon, min_samples=2).fit(coords)
    
    group['cluster_label'] = db.labels_
    # Initially, mark noise as 0 and all clusters as 1. Clusters will be re-classified.
    group['HitType'] = np.where(group['cluster_label'] == -1, 0, 1)

    valid_clusters = group[group['cluster_label'] != -1]
    cluster_labels = valid_clusters['cluster_label'].unique()

    # If there are any valid clusters, classify them based on their shape
    if len(cluster_labels) > 0:
        # Calculate dimensional stats for all clusters at once for performance
        cluster_dims = valid_clusters.groupby('cluster_label').agg(
            row_min=('Row', 'min'),
            row_max=('Row', 'max'),
            col_min=('Column', 'min'),
            col_max=('Column', 'max'),
            col_nunique=('Column', 'nunique')
        )
        cluster_dims['row_span'] = cluster_dims['row_max'] - cluster_dims['row_min'] + 1
        cluster_dims['col_span'] = cluster_dims['col_max'] - cluster_dims['col_min'] + 1

        # Determine HitType based on dimensions, creating a map from cluster_label to hittype
        hittype_map = pd.Series(index=cluster_dims.index, dtype=int)

        # Condition for 'cluster' (HitType = 1): blob-like shape
        is_cluster = abs(cluster_dims['row_span'] - cluster_dims['col_span']) <= 1
        hittype_map.loc[is_cluster] = 1

        # Condition for 'delta' (HitType = 3): wide across columns
        is_delta = (cluster_dims['col_nunique'] >= 3) & (~is_cluster)
        hittype_map.loc[is_delta] = 3

        # Condition for 'track' (HitType = 2): narrow in columns
        is_track = (cluster_dims['col_nunique'] <= 2) & (~is_cluster)
        hittype_map.loc[is_track] = 2
        
        # Map the calculated hittypes back to the main group dataframe.
        # Noise hits (cluster_label = -1) won't be in the map, so we use fillna
        # to keep their original HitType (0).
        group['HitType'] = group['cluster_label'].map(hittype_map).fillna(group['HitType']).astype(int)

    # Identify crosstalk if there are at least two valid clusters
    if len(cluster_labels) >= 2:
        # Calculate stats for all valid clusters needed for crosstalk detection
        cluster_stats = valid_clusters.groupby('cluster_label').agg(
            avg_tot=('ToT', 'mean'),
            max_tot=('ToT', 'max'),
            column_min=('Column', 'min'),
            column_max=('Column', 'max')
        )
        
        # Identify the source cluster as the one containing the hit with the maximum ToT in the package
        if not cluster_stats.empty:
            source_cluster_label = cluster_stats['max_tot'].idxmax()
            source_cluster = cluster_stats.loc[source_cluster_label]
            
            # Define the spatial "hot zone" around the source cluster (its columns +/- 1)
            # Cast to int to prevent TypeError, as pandas may upcast integer columns to float during aggregation
            source_hot_columns = set(range(int(source_cluster['column_min']) - 1, int(source_cluster['column_max']) + 2))

            # Potential candidates are all clusters except the source
            candidate_clusters = cluster_stats.drop(source_cluster_label)

            if not candidate_clusters.empty:
                # First filter: Candidate's avg ToT must be lower than source's avg ToT by at least crosstalk_tot_delta
                is_low_enough_tot = candidate_clusters['avg_tot'] < (source_cluster['avg_tot'] - crosstalk_tot_delta)
                
                # Second filter: The candidate cluster must be spatially close to the source cluster
                def is_spatially_close(row):
                    # Cast to int to prevent the same TypeError within the .apply() method
                    candidate_cols = set(range(int(row['column_min']), int(row['column_max']) + 1))
                    return not candidate_cols.isdisjoint(source_hot_columns)

                # Apply both filters to find the crosstalk clusters
                crosstalk_candidates = candidate_clusters[is_low_enough_tot]
                if not crosstalk_candidates.empty:
                    is_close_mask = crosstalk_candidates.apply(is_spatially_close, axis=1)
                    crosstalk_labels = crosstalk_candidates[is_close_mask].index

                    # If any crosstalk clusters were found, update their HitType
                    if len(crosstalk_labels) > 0:
                        group.loc[group['cluster_label'].isin(crosstalk_labels), 'HitType'] = -1
    
    return group

def clustering_3(data, epsilon, crosstalk_tot_delta=40):
    """
    Performs clustering on hit data based on package ID, spatial proximity, and Time-over-Threshold.
    This version is optimized for speed using multiprocessing.

    Args:
        data (dict): A dictionary of numpy arrays representing the hit data.
        epsilon (float): The maximum distance between hits to be considered in the same cluster.
        crosstalk_tot_delta (int, optional): The minimum difference between the average ToT of a
                                             source cluster and a candidate cluster for the
                                             candidate to be considered crosstalk. Defaults to 40.

    Returns:
        dict: A dictionary of numpy arrays with added 'HitType' and 'ClusterID' columns.
              HitType:
                - -1: Crosstalk hit
                -  0: Noise hit (a cluster with only one hit)
                -  1: 'Cluster' hit (roughly same row/column span)
                -  2: 'Track' hit (spans 1-2 columns)
                -  3: 'Delta' hit (spans 3+ columns)
              ClusterID: A unique identifier for each cluster, or -1 for noise.
    """
    df = pd.DataFrame(data)
    if df.empty:
        data['HitType'] = np.array([], dtype=int)
        data['ClusterID'] = np.array([], dtype=int)
        return data

    package_groups = [group for _, group in df.groupby('PackageID')]

    if not package_groups:
        data['HitType'] = np.array([], dtype=int)
        data['ClusterID'] = np.array([], dtype=int)
        return data

    # Create a partial function with fixed arguments for the worker processes
    worker_func = partial(_process_group, epsilon=epsilon, crosstalk_tot_delta=crosstalk_tot_delta)
    
    # Use a pool of workers to process groups in parallel
    # Uses all available CPU cores for maximum speed
    with Pool(processes=cpu_count()) as pool:
        # Use pool.imap to get an iterator that we can wrap with the progress bar
        processed_groups = list(progress_bar(
            pool.imap(worker_func, package_groups),
            description="Clustering Hits",
            total=len(package_groups)
        ))

    # Concatenate results and restore original order
    final_df = pd.concat(processed_groups).sort_index()

    # Create a globally unique ClusterID
    # Initialize ClusterID column, defaulting to -1 for noise hits
    final_df['ClusterID'] = -1
    
    # Identify the rows that belong to a valid cluster (i.e., not noise)
    is_clustered_mask = final_df['cluster_label'] != -1
    
    # Only proceed if there are any clustered hits
    if is_clustered_mask.any():
        # Create a composite key from PackageID and local cluster_label to uniquely identify each cluster.
        # Then, use pandas.factorize to assign a unique integer ID to each unique cluster.
        cluster_ids = pd.factorize(final_df.loc[is_clustered_mask, ['PackageID', 'cluster_label']].apply(tuple, axis=1))[0]
        
        # Assign the new global IDs to the ClusterID column for the clustered hits
        final_df.loc[is_clustered_mask, 'ClusterID'] = cluster_ids
        
    # Drop the temporary cluster_label column as it's no longer needed in the final output
    final_df.drop(columns=['cluster_label'], inplace=True)
    
    # Ensure final column types are integers
    final_df['HitType'] = final_df['HitType'].astype(int)
    final_df['ClusterID'] = final_df['ClusterID'].astype(int)

    return {col: final_df[col].to_numpy() for col in final_df.columns}




from math import ceil

def _process_group_v4(group, epsilon, crosstalk_rules, cluster_min_hits):
    """
    Worker function for clustering_4, optimized for speed.
    1. Pre-filters for crosstalk using a vectorized pandas approach.
    2. Clusters non-crosstalk hits using DBSCAN (min_samples=cluster_min_hits).
    3. Clusters crosstalk hits separately using DBSCAN (min_samples=cluster_min_hits).
    4. Classifies hits as crosstalk (-1), noise (0), or cluster (1).
    """
    group = group.copy()
    
    # Initialize HitType: 1 is a temporary value for potential cluster hits
    group['HitType'] = 1
    
    # --- Vectorized Crosstalk Pre-filtering ---
    col_counts = group['Column'].value_counts()
    multi_hit_cols = col_counts[col_counts > 1].index
    
    if len(multi_hit_cols) > 0:
        # Work only with hits in these potentially problematic columns
        multi_hit_df = group[group['Column'].isin(multi_hit_cols)].copy()
        
        # Create all pairs of hits within the same column by merging the df with itself
        pairs = pd.merge(multi_hit_df.reset_index(), multi_hit_df.reset_index(), on='Column', suffixes=('_1', '_2'))
        
        # Filter out self-pairs and duplicate pairs
        pairs = pairs[pairs['index_1'] < pairs['index_2']]

        if not pairs.empty:
            row_pairs_sorted = np.sort(pairs[['Row_1', 'Row_2']].values, axis=1)
            pairs['row_pair_key'] = list(map(tuple, row_pairs_sorted))
            
            pairs['min_before_peak'] = pairs['row_pair_key'].map(crosstalk_rules)
            pairs.dropna(subset=['min_before_peak'], inplace=True)

            if not pairs.empty:
                is_1_higher = pairs['ToT_1'] > pairs['ToT_2']
                cond_2_is_xtalk = is_1_higher & (pairs['ToT_2'] < (pairs['ToT_1'] - pairs['min_before_peak']))
                cond_1_is_xtalk = ~is_1_higher & (pairs['ToT_1'] < (pairs['ToT_2'] - pairs['min_before_peak']))
                crosstalk_indices_1 = pairs.loc[cond_1_is_xtalk, 'index_1']
                crosstalk_indices_2 = pairs.loc[cond_2_is_xtalk, 'index_2']
                all_crosstalk_indices = pd.concat([crosstalk_indices_1, crosstalk_indices_2]).unique()
                group.loc[all_crosstalk_indices, 'HitType'] = -1

    # Initialize cluster_label for all hits
    group['cluster_label'] = -1
    
    # --- Clustering on Non-Crosstalk Hits ---
    non_crosstalk_hits_mask = group['HitType'] != -1
    non_crosstalk_hits = group[non_crosstalk_hits_mask].copy()

    # Only run DBSCAN if there are enough hits to potentially form a cluster
    if len(non_crosstalk_hits) < cluster_min_hits:
        # Not enough hits to form a cluster, all are noise
        group.loc[non_crosstalk_hits.index, 'HitType'] = 0
    else:
        # Run DBSCAN
        coords = non_crosstalk_hits[['Row', 'Column']].values
        db = DBSCAN(eps=epsilon, min_samples=cluster_min_hits).fit(coords)
        non_crosstalk_hits['cluster_label'] = db.labels_

        # Hits labeled -1 by DBSCAN are noise
        noise_mask = non_crosstalk_hits['cluster_label'] == -1
        group.loc[non_crosstalk_hits[noise_mask].index, 'HitType'] = 0
        
        # Valid clusters
        valid_clusters_mask = non_crosstalk_hits['cluster_label'] != -1
        if valid_clusters_mask.any():
            non_crosstalk_hits.loc[valid_clusters_mask, 'HitType'] = 1
            # Update the main group dataframe with the new HitTypes and the cluster labels
            group.update(non_crosstalk_hits[['HitType', 'cluster_label']])


    # --- Clustering on Crosstalk Hits ---
    crosstalk_hits_mask = group['HitType'] == -1
    crosstalk_hits = group[crosstalk_hits_mask].copy()
    
    # Only run DBSCAN if there are enough hits to potentially form a cluster
    if len(crosstalk_hits) >= cluster_min_hits:
        coords_xtalk = crosstalk_hits[['Row', 'Column']].values
        db_xtalk = DBSCAN(eps=epsilon, min_samples=cluster_min_hits).fit(coords_xtalk)
        crosstalk_hits['cluster_label'] = db_xtalk.labels_
        
        valid_xtalk_clusters_mask = crosstalk_hits['cluster_label'] != -1
        if valid_xtalk_clusters_mask.any():
            # Make cluster labels negative and offset by 2 to avoid -1 (noise)
            crosstalk_hits.loc[valid_xtalk_clusters_mask, 'cluster_label'] = \
                - (crosstalk_hits.loc[valid_xtalk_clusters_mask, 'cluster_label'] + 2)
        
        # Update the main group df with these new crosstalk cluster labels
        group.loc[crosstalk_hits.index, 'cluster_label'] = crosstalk_hits['cluster_label']
    
    # Note: Crosstalk hits with len < cluster_min_hits will remain
    # HitType = -1 and cluster_label = -1. They are re-classified as
    # noise (HitType = 0) in the final step of clustering_4.

    return group


def clustering_4(data, crosstalk_df, epsilon, cluster_min_hits=2):
    """
    Performs clustering with crosstalk pre-filtering based on pixel-pair rules.
    This version is optimized for speed using a vectorized crosstalk check
    and clusters both valid hits and crosstalk hits.

    Args:
        data (dict): Dictionary of numpy arrays for hit data.
        crosstalk_df (pd.DataFrame): DataFrame with pixel-pair rules.
                                      Must contain ['pixel_1', 'pixel_2', 'min_before_peak'].
        epsilon (float): The maximum distance between hits to be in the same cluster.
        cluster_min_hits (int, optional): The minimum number of hits required to
                                          form a cluster. Defaults to 2.

    Returns:
        dict: A dictionary of numpy arrays with 'HitType' and 'ClusterID' columns.
              HitType:
                - -1: Crosstalk cluster (part of a crosstalk cluster of cluster_min_hits or more)
                -  0: Noise hit (non-crosstalk hits < cluster_min_hits, or unclustered crosstalk)
                -  1: Cluster hit (part of a valid cluster of cluster_min_hits or more)
              ClusterID: A unique identifier for each cluster.
                -  >= 0: For a valid cluster (HitType = 1).
                -   -1: For noise (HitType = 0).
                -  <= -2: For a crosstalk cluster (HitType = -1).
    """
    # --- Preparation ---
    df = pd.DataFrame(data)
    if df.empty:
        data['HitType'] = np.array([], dtype=int)
        data['ClusterID'] = np.array([], dtype=int)
        return data

    # Create a fast lookup dictionary for crosstalk rules from the input DataFrame
    ct_rules_df = crosstalk_df[crosstalk_df['min_before_peak'] > 0].copy()
    crosstalk_rules = {}
    for _, row in ct_rules_df.iterrows():
        # Use a canonical key (sorted tuple) for the pixel pair for consistent lookups
        p1, p2 = tuple(sorted((row['pixel_1'], row['pixel_2'])))
        crosstalk_rules[(p1, p2)] = row['min_before_peak']
        
    # --- Parallel Processing ---
    package_groups = [group for _, group in df.groupby('PackageID')]
    if not package_groups:
        data['HitType'] = np.array([], dtype=int)
        data['ClusterID'] = np.array([], dtype=int)
        return data

    # *** CHANGED: Pass cluster_min_hits to the worker function ***
    worker_func = partial(_process_group_v4, 
                          epsilon=epsilon, 
                          crosstalk_rules=crosstalk_rules, 
                          cluster_min_hits=cluster_min_hits)
    
    num_processes = cpu_count()
    num_tasks = len(package_groups)
    # Calculate a chunksize to balance overhead and load distribution for multiprocessing.
    chunksize = 1
    if num_processes > 1:
        chunksize = max(1, ceil(num_tasks / (num_processes * 4)))

    with Pool(processes=num_processes) as pool:
        # The progress_bar is integrated here as requested
        processed_groups = list(progress_bar(
            pool.imap(worker_func, package_groups, chunksize=chunksize),
            description="Clustering Packages v4",
            total=num_tasks
        ))

    # --- Finalization ---
    final_df = pd.concat(processed_groups).sort_index()

    # Create a globally unique ClusterID
    final_df['ClusterID'] = -1 # Default for noise and singletons
    
    # Mask for positive clusters (HitType = 1, cluster_label >= 0)
    pos_cluster_mask = final_df['cluster_label'] >= 0
    
    # Mask for negative clusters (HitType = -1, cluster_label <= -2)
    neg_cluster_mask = final_df['cluster_label'] < -1

    # Process positive clusters (assigns ClusterID >= 0)
    if pos_cluster_mask.any():
        pos_ids = pd.factorize(final_df.loc[pos_cluster_mask, ['PackageID', 'cluster_label']].apply(tuple, axis=1))[0]
        final_df.loc[pos_cluster_mask, 'ClusterID'] = pos_ids
    
    # Process negative (crosstalk) clusters (assigns ClusterID <= -2)
    if neg_cluster_mask.any():
        neg_ids = pd.factorize(final_df.loc[neg_cluster_mask, ['PackageID', 'cluster_label']].apply(tuple, axis=1))[0]
        # Map IDs 0, 1, 2... to -2, -3, -4... to avoid -1 (noise)
        final_df.loc[neg_cluster_mask, 'ClusterID'] = - (neg_ids + 2)
        
    final_df.drop(columns=['cluster_label'], inplace=True)
    
    # *** Move unclustered crosstalk (HitType -1, ClusterID -1) to noise (HitType 0) ***
    final_df.loc[(final_df['ClusterID'] == -1) & (final_df['HitType'] == -1), 'HitType'] = 0
    
    final_df['HitType'] = final_df['HitType'].astype(int)
    final_df['ClusterID'] = final_df['ClusterID'].astype(int)

    return {col: final_df[col].to_numpy() for col in final_df.columns}


def sort_clustered_data(clustered_data, sort_by_id=True):
    """
    Sorts the clustered data into three groups.
    Can be toggled to sort by HitType (default) or ClusterID.

    Args:
        clustered_data (dict): The input data dictionary.
        sort_by_id (bool): 
            - If False (default): Sorts by HitType (1=clusters, -1=coupling, 0=noise).
            - If True: Sorts by ClusterID (>=0=clusters, <=-2=coupling, -1=noise).

    Returns:
        dict: The sorted datasets.
    """
    
    masks = {}
    if sort_by_id:
        # --- New functionality: Sort by ClusterID ---
        masks = {
            'clusters': clustered_data['ClusterID'] >= 0,
            'coupling': clustered_data['ClusterID'] <= -2,
            'noise':    clustered_data['ClusterID'] == -1,
        }
    else:
        # --- Original functionality: Sort by HitType ---
        EVENT_TYPE_MAP = {'CLUSTER': 1, 'COUPLING': -1, 'NOISE': 0}
        masks = {
            'clusters': clustered_data['HitType'] == EVENT_TYPE_MAP['CLUSTER'],
            'coupling': clustered_data['HitType'] == EVENT_TYPE_MAP['COUPLING'],
            'noise':    clustered_data['HitType'] == EVENT_TYPE_MAP['NOISE'],
        }
    
    # This part remains the same, it just applies the masks
    sorted_datasets = {name: {} for name in masks.keys()}
    for name, mask in masks.items():
        for key, value_array in clustered_data.items():
            sorted_datasets[name][key] = value_array[mask]
            
    return sorted_datasets