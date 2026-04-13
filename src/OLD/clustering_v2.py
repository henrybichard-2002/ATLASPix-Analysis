import numpy as np
from sklearn.cluster import DBSCAN
from utils import progress_bar

from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def clustering_2(data, dt, spatial_eps, min_hits_cluster=3, min_hits_track=5, track_max_col_std=1.5, delta_min_span=5, ToT_threshold=50, crosstalk_dt=5, track_merge_max_gap_row=3):
    """
    Performs temporal/spatial clustering and classifies clusters from a dictionary of arrays.

    This function first groups hits into clusters based on their proximity in time (dt)
    and space (spatial_eps). It includes a step to merge broken track segments.
    Hits not belonging to any cluster are labeled as noise.
    It then classifies each found cluster into one of the following categories based on
    the EVENT_TYPE_MAP = {'CLUSTER': 0, 'TRACK': 1, 'DELTA': 2, 'COUPLING': 3, 'NOISE': -1}:
    - 1 ('TRACK'): A cluster that is generally straight and column-wise with high Time-over-Threshold (ToT).
    - 2 ('DELTA'): A cluster that is not straight, possibly diagonal, and spans a large area.
    - 3 ('COUPLING'): A track-like cluster with low ToT that is temporally and spatially close to a high-ToT track.
    - 0 ('CLUSTER'): A small, localized cluster that doesn't fit the other categories.
    - -1 ('NOISE'): Hits not assigned to any cluster.

    Args:
        data (dict): A dictionary of NumPy arrays with keys 'ext_TS', 'Layer', 'Column', 'Row', and 'ToT'.
        dt (float): The maximum time difference (in timestamp units) for hits to be in the same temporal group.
        spatial_eps (float): The maximum spatial distance between two samples for one to be considered as in the neighborhood of the other.
        min_hits_cluster (int): The number of samples in a neighborhood for a point to be considered as a core point. This is the minimum size of a cluster.
        min_hits_track (int): The minimum number of hits required for a cluster to be considered a 'track'.
        track_max_col_std (float): The maximum standard deviation of column values for a cluster to be classified as a 'track'.
                                   This measures the straightness in the column direction.
        delta_min_span (int): The minimum row or column span required for a cluster to be considered a 'delta'.
        ToT_threshold (float): The Time-over-Threshold value to distinguish between low and high ToT tracks for crosstalk analysis.
        crosstalk_dt (float): The maximum time difference between a low-ToT and high-ToT track to be considered coupling.
        track_merge_max_gap_row (int): The maximum gap in rows to merge two track-like clusters in the same column.

    Returns:
        dict: A new dictionary containing the original data arrays plus two new keys:
              'ClusterID' (np.ndarray): An ID for each cluster. -1 indicates noise.
              'ClusterType' (np.ndarray): An integer type for each cluster (-1, 0, 1, 2, 3).
    """
    if 'ToT' not in data:
        raise ValueError("Input data dictionary must contain a 'ToT' key for crosstalk classification.")

    if data['ext_TS'].size == 0:
        output_data = data.copy()
        output_data['ClusterID'] = np.array([], dtype=int)
        output_data['ClusterType'] = np.array([], dtype=int)
        return output_data

    # --- Part 1: Spatio-Temporal Clustering (Optimized) ---
    
    num_hits = data['ext_TS'].size
    cluster_ids = np.full(num_hits, -1, dtype=int)
    cluster_types = np.full(num_hits, -1, dtype=int) # Default to NOISE

    final_cluster_id_counter = 0
    
    unique_layers = np.unique(data['Layer'])

    for layer_id in progress_bar(unique_layers, description="Clustering Layers"):
        layer_mask = (data['Layer'] == layer_id)
        
        if np.sum(layer_mask) < min_hits_cluster:
            continue

        original_indices_layer = np.where(layer_mask)[0]
        layer_coords = np.column_stack((data['Column'][layer_mask], data['Row'][layer_mask]))
        
        if layer_coords.shape[0] < 2:
            continue

        layer_ts = data['ext_TS'][layer_mask]

        tree = KDTree(layer_coords)
        pairs = tree.query_pairs(r=spatial_eps)
        pairs = np.array(list(pairs))

        if pairs.size == 0:
            continue

        idx1, idx2 = pairs[:, 0], pairs[:, 1]
        time_diff = np.abs(layer_ts[idx1] - layer_ts[idx2])
        time_mask = time_diff <= dt
        valid_pairs = pairs[time_mask]

        if valid_pairs.size == 0:
            continue

        n_hits_in_layer = len(layer_coords)
        row_indices = np.concatenate([valid_pairs[:, 0], valid_pairs[:, 1]])
        col_indices = np.concatenate([valid_pairs[:, 1], valid_pairs[:, 0]])
        adj_matrix = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_hits_in_layer, n_hits_in_layer))

        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count >= min_hits_cluster:
                component_member_mask = (labels == label)
                indices_to_update = original_indices_layer[component_member_mask]
                cluster_ids[indices_to_update] = final_cluster_id_counter
                final_cluster_id_counter += 1

    # --- Part 2: Initial Cluster Classification ---
    
    unique_cluster_ids = np.unique(cluster_ids[cluster_ids != -1])
    TEMP_TRACK_SEGMENT = 10 # Temporary type for potential track parts

    for cluster_id in progress_bar(unique_cluster_ids, description="Classifying clusters"):
        cluster_mask = (cluster_ids == cluster_id)
        n_hits = np.sum(cluster_mask)
        cols, rows = data['Column'][cluster_mask], data['Row'][cluster_mask]
        
        col_std = np.std(cols)
        
        if col_std < track_max_col_std:
             cluster_types[cluster_mask] = TEMP_TRACK_SEGMENT # Potential track
             continue

        row_span = np.max(rows) - np.min(rows) if n_hits > 0 else 0
        col_span = np.max(cols) - np.min(cols) if n_hits > 0 else 0
        
        if max(row_span, col_span) >= delta_min_span:
            cluster_types[cluster_mask] = 2 # DELTA
        else:
            cluster_types[cluster_mask] = 0 # CLUSTER

    # --- Part 2.5: Merge Broken Track Segments ---
    segment_mask = (cluster_types == TEMP_TRACK_SEGMENT)
    segment_ids = np.unique(cluster_ids[segment_mask])

    if len(segment_ids) > 1:
        props = {}
        for cid in progress_bar(segment_ids, description="Analyzing track segments"):
            mask = (cluster_ids == cid)
            props[cid] = { 'rows': data['Row'][mask], 'ts': data['ext_TS'][mask], 'col': np.mean(data['Column'][mask]) }

        id_map = {cid: i for i, cid in enumerate(segment_ids)}
        pairs_to_check = np.array([(id1, id2) for i, id1 in enumerate(segment_ids) for id2 in segment_ids[i+1:]])
        
        if pairs_to_check.size > 0:
            p1 = [props[pair[0]] for pair in pairs_to_check]
            p2 = [props[pair[1]] for pair in pairs_to_check]

            col_ok = [abs(p1[i]['col'] - p2[i]['col']) < 1.0 for i in range(len(p1))]
            row_gap_ok = [min(abs(p1[i]['rows'].max() - p2[i]['rows'].min()), abs(p2[i]['rows'].max() - p1[i]['rows'].min())) <= track_merge_max_gap_row for i in range(len(p1))]
            ts_gap_ok = [min(abs(p1[i]['ts'].max() - p2[i]['ts'].min()), abs(p2[i]['ts'].max() - p1[i]['ts'].min())) <= dt for i in range(len(p1))]

            merge_mask = np.logical_and.reduce([col_ok, row_gap_ok, ts_gap_ok])
            valid_pairs = pairs_to_check[merge_mask]

            if valid_pairs.size > 0:
                adj_matrix_seg = csr_matrix((np.ones(len(valid_pairs)*2), 
                                            (np.concatenate([valid_pairs[:,0], valid_pairs[:,1]]), 
                                             np.concatenate([valid_pairs[:,1], valid_pairs[:,0]]))),
                                            shape=(num_hits, num_hits))
                n_merged, labels_merged = connected_components(csgraph=adj_matrix_seg, directed=False)
                
                merged_ids = labels_merged[segment_ids]
                unique_merged, counts = np.unique(merged_ids, return_counts=True)

                for merged_label in unique_merged[counts > 1]:
                    original_ids = segment_ids[merged_ids == merged_label]
                    hits_to_merge_mask = np.isin(cluster_ids, original_ids)
                    cluster_ids[hits_to_merge_mask] = final_cluster_id_counter
                    final_cluster_id_counter += 1

    # --- Part 3: Final Classification & Coupling Refinement ---
    unique_cluster_ids = np.unique(cluster_ids[cluster_ids != -1])
    for cluster_id in progress_bar(unique_cluster_ids, description="Final Classification"):
        mask = (cluster_ids == cluster_id)
        if cluster_types[mask][0] == TEMP_TRACK_SEGMENT or np.all(cluster_types[mask] == -1):
            n_hits = np.sum(mask)
            col_std = np.std(data['Column'][mask])
            if n_hits >= min_hits_track and col_std < track_max_col_std:
                 cluster_types[mask] = 1 # TRACK
            else:
                 cluster_types[mask] = 0 # CLUSTER

    track_mask = (cluster_types == 1)
    track_cluster_ids = np.unique(cluster_ids[track_mask])
    
    track_properties = {}
    for cid in progress_bar(track_cluster_ids, description="Analyzing tracks"):
        mask = (cluster_ids == cid)
        track_properties[cid] = {'mean_ToT': np.mean(data['ToT'][mask]),
                                 'mean_TS': np.mean(data['ext_TS'][mask]),
                                 'mean_col': np.mean(data['Column'][mask]),
                                 'id': cid}

    high_ToT = [p for p in track_properties.values() if p['mean_ToT'] >= ToT_threshold]
    low_ToT = [p for p in track_properties.values() if p['mean_ToT'] < ToT_threshold]

    for low_track in progress_bar(low_ToT, description="Refining crosstalk"):
        for high_track in high_ToT:
            if abs(low_track['mean_TS'] - high_track['mean_TS']) <= crosstalk_dt and \
               abs(low_track['mean_col'] - high_track['mean_col']) < 1.0:
                cluster_types[cluster_ids == low_track['id']] = 3 # COUPLING
                break
        
    output_data = data.copy()
    output_data['ClusterID'] = cluster_ids
    output_data['ClusterType'] = cluster_types
    return output_data

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import sys

def clustering_3(data, epsilon, crosstalk_tot_threshold=50):
    """
    Performs clustering on hit data based on package ID, spatial proximity, and Time-over-Threshold.

    This function follows these steps:
    1. Sorts data by PackageID.
    2. Groups hits that share the same PackageID.
    3. Within each group, it forms spatial clusters where neighboring hits are closer than epsilon.
    4. Single-hit clusters are labeled as noise.
    5. In packages with two or more clusters, it identifies crosstalk, defined as a low-ToT
       cluster sharing a column with another, higher-ToT cluster.

    Args:
        data (dict): A dictionary of numpy arrays, where each key represents a data field
                     (e.g., 'PackageID', 'Column', 'Row', 'ToT'). The arrays should all
                     have the same length.
        epsilon (float): The maximum distance (in pixels) between two hits for them to be
                         considered as in the same neighborhood during clustering.
        crosstalk_tot_threshold (int, optional): The maximum ToT for a cluster to be
                                                considered a potential crosstalk candidate.
                                                Defaults to 50.

    Returns:
        dict: A dictionary of numpy arrays in the same format as the input, with an added
              'HitType' column indicating the classification of each hit.
              'HitType' values:
                - -1: Crosstalk hit
                -  0: Noise hit (a cluster with only one hit)
                -  1: Clustered hit
    """
    # Convert input dictionary to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Sort by PackageID as requested, though groupby makes this not strictly necessary for logic
    df.sort_values('PackageID', inplace=True)

    processed_groups = []
    
    # Use a unique index for each group to handle potential empty groups from groupby
    # This also helps in showing a more accurate total for the progress bar
    package_groups = [group for _, group in df.groupby('PackageID')]

    # Group into groups that share the same packageID
    for group in progress_bar(package_groups, description="Clustering Packages"):
        # Make a copy to avoid SettingWithCopyWarning
        group = group.copy()
        
        # A package with only one hit is noise by definition
        if len(group) == 1:
            group['HitType'] = 0
            processed_groups.append(group)
            continue

        # Use DBSCAN for spatial clustering on 'Row' and 'Column' coordinates.
        # min_samples=2 ensures that any "cluster" of 1 is classified as noise.
        coords = group[['Row', 'Column']].values
        db = DBSCAN(eps=epsilon, min_samples=2).fit(coords)
        
        # Add cluster labels to the group. Noise points are labeled -1 by DBSCAN.
        group['cluster_label'] = db.labels_
        
        # Assign initial HitType: 1 for clustered hits, 0 for noise
        group['HitType'] = np.where(group['cluster_label'] == -1, 0, 1)

        # Identify crosstalk if there are at least two valid clusters
        valid_clusters = group[group['cluster_label'] != -1]
        cluster_labels = valid_clusters['cluster_label'].unique()

        if len(cluster_labels) >= 2:
            # Optimized crosstalk detection
            # Calculate max ToT and check for single-column constraint for each cluster
            cluster_stats = valid_clusters.groupby('cluster_label').agg(
                max_tot=('ToT', 'max'),
                column_min=('Column', 'min'),
                column_max=('Column', 'max')
            )
            
            # Filter for clusters that are entirely within a single column
            single_col_clusters = cluster_stats[cluster_stats['column_min'] == cluster_stats['column_max']].copy()
            
            if not single_col_clusters.empty:
                single_col_clusters.rename(columns={'column_min': 'column'}, inplace=True)
                
                # Check how many clusters are in each column
                single_col_clusters['clusters_in_col'] = single_col_clusters.groupby('column')['column'].transform('count')
                
                # Consider only columns with 2+ clusters for potential crosstalk
                potential_cols = single_col_clusters[single_col_clusters['clusters_in_col'] >= 2]
                
                if not potential_cols.empty:
                    # Identify low ToT clusters
                    potential_cols['is_low_tot'] = potential_cols['max_tot'] < crosstalk_tot_threshold
                    
                    # For each column, check if it contains BOTH a low ToT and a high ToT cluster
                    potential_cols['has_low_tot_in_col'] = potential_cols.groupby('column')['is_low_tot'].transform('any')
                    potential_cols['has_high_tot_in_col'] = (~potential_cols['is_low_tot']).groupby(potential_cols['column']).transform('any')
                    
                    # A cluster is crosstalk if it is a low_tot cluster and its column also has a high_tot cluster
                    is_crosstalk = (
                        potential_cols['is_low_tot'] & 
                        potential_cols['has_high_tot_in_col']
                    )
                    
                    crosstalk_labels = potential_cols[is_crosstalk].index
                    
                    # Update HitType for the crosstalk hits
                    group.loc[group['cluster_label'].isin(crosstalk_labels), 'HitType'] = -1
        
        # The temporary cluster_label column is not needed in the final output
        processed_groups.append(group.drop(columns=['cluster_label']))

    if not processed_groups:
         # Handle case where there's no data, return original dict with empty HitType
        data['HitType'] = np.array([], dtype=int)
        return data

    # Concatenate all processed groups back into a single DataFrame
    # Sort by the original index to maintain the input order
    final_df = pd.concat(processed_groups).sort_index()
    
    # Ensure HitType is an integer type
    final_df['HitType'] = final_df['HitType'].astype(int)

    # Convert the resulting DataFrame back to the required dictionary of numpy arrays format
    return {col: final_df[col].to_numpy() for col in final_df.columns}


def sort_clustered_data(clustered_data):
    EVENT_TYPE_MAP = {'CLUSTER': 0, 'TRACK': 1, 'DELTA': 2, 'COUPLING': 3, 'NOISE': -1}
    masks = {
        'clusters': clustered_data['ClusterType'] == EVENT_TYPE_MAP['CLUSTER'],
        'tracks': clustered_data['ClusterType'] == EVENT_TYPE_MAP['TRACK'],
        'deltas': clustered_data['ClusterType'] == EVENT_TYPE_MAP['DELTA'],
        'coupling': clustered_data['ClusterType'] == EVENT_TYPE_MAP['COUPLING'],
        'noise': clustered_data['ClusterType'] == EVENT_TYPE_MAP['NOISE']
    }
    sorted_datasets = {name: {} for name in masks.keys()}
    for name, mask in masks.items():
        for key, value_array in clustered_data.items():
            sorted_datasets[name][key] = value_array[mask]
    return sorted_datasets
