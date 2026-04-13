import numpy as np
from sklearn.cluster import DBSCAN
import warnings
from scipy.spatial.distance import pdist, squareform
from utils import progress_bar


def _filter_clustered_data(data):
    # Handle cases where clustering information is absent or data is empty
    if 'ClusterID' not in data or data['ClusterID'].size == 0:
        return None, None
    unclustered_mask = data['ClusterID'] == -1
    clustered_mask = ~unclustered_mask
    unclustered_data = {key: value[unclustered_mask] for key, value in data.items()}
    if not np.any(clustered_mask):
        return None, unclustered_data
    clustered_data = {key: value[clustered_mask] for key, value in data.items()}
    return clustered_data, unclustered_data

def _calculate_max_length(points):
    if points.shape[0] > 1:
        dist_matrix = squareform(pdist(points))
        max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        point1_idx, point2_idx = max_dist_idx
        point1 = points[point1_idx]
        point2 = points[point2_idx]
        start_column, start_row = point1[0], point1[1]
        end_column, end_row = point2[0], point2[1]
        max_length = dist_matrix[max_dist_idx]
    else:
        start_column, start_row = points[0][0], points[0][1]
        end_column, end_row = points[0][0], points[0][1]
        max_length = 0.0
    return max_length, start_column, start_row, end_column, end_row

def _calculate_timescale(ts, ts_modulus=1023):
    ts_values = np.sort(np.unique(ts))
    if len(ts_values) > 1:
        diffs = np.diff(ts_values)
        wrapped_diff = (ts_values[0] + ts_modulus) - ts_values[-1]
        largest_gap = np.max(np.append(diffs, wrapped_diff))
        return ts_modulus - largest_gap
    return 0

def _calculate_track_properties(x, y, n_hits, max_length):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    reduced_chi_square = 0.0
    n_missing_hits = 0
    rms_deviation = 0.0
    reduced_rms_deviation = 0.0

    if n_hits > 1:
        if x_min == x_max:
            actual_coords = set(zip(x, y))
            expected_rows = np.arange(y_min, y_max + 1)
            expected_coords = set([(x[0], r) for r in expected_rows])
            n_missing_hits = len(expected_coords - actual_coords)
        elif y_min == y_max:
            actual_coords = set(zip(x, y))
            expected_cols = np.arange(x_min, x_max + 1)
            expected_coords = set([(c, y[0]) for c in expected_cols])
            n_missing_hits = len(expected_coords - actual_coords)
        else:
            slope, intercept = np.polyfit(x, y, 1)
            distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
            rms_deviation = np.sqrt(np.mean(distances**2))
            reduced_rms_deviation = rms_deviation/n_hits
            if n_hits > 2:
                y_expected = slope * x + intercept
                sum_squared_residuals = np.sum((y - y_expected)**2)
                degrees_of_freedom = n_hits - 2
                reduced_chi_square = sum_squared_residuals / degrees_of_freedom
            
            actual_coords = set(zip(x, y))
            num_points = max(2, int(max_length * 2))
            x_line = np.linspace(x_min, x_max, num=num_points)
            y_line = slope * x_line + intercept
            expected_coords = set(zip(np.round(x_line).astype(int), np.round(y_line).astype(int)))
            missing_coords = expected_coords - actual_coords
            n_missing_hits = len(missing_coords)

    return rms_deviation, reduced_rms_deviation, n_missing_hits

def cluster_events_numpy(data, dt, spatial_eps, spatial_min_samples):
    """
    Performs temporal and spatial clustering on data stored in NumPy arrays.
    """
    if data['ext_TS'].size == 0:
        data['ClusterID'] = np.array([], dtype=int)
        return data

    print("Starting temporal clustering...")
    sort_indices = np.argsort(data['ext_TS'])
    sorted_ext_ts = data['ext_TS'][sort_indices]
    ts_diff = np.diff(sorted_ext_ts, prepend=sorted_ext_ts[0])
    is_new_temporal_cluster = (ts_diff >= dt) | (np.arange(len(sorted_ext_ts)) == 0)
    temporal_cluster_ids = np.cumsum(is_new_temporal_cluster) - 1
    print("Temporal clustering finished.")
    
    print("Starting spatial clustering...")
    data['ClusterID'] = np.full_like(data['ext_TS'], -1, dtype=int)
    final_cluster_id_counter = 0

    sorted_data = {key: value[sort_indices] for key, value in data.items()}
    sorted_data['TemporalClusterID'] = temporal_cluster_ids

    unique_temporal_clusters = np.unique(sorted_data['TemporalClusterID'])

    for temp_id in progress_bar(unique_temporal_clusters, description="Processing temporal clusters"):
        temp_mask = sorted_data['TemporalClusterID'] == temp_id
        layers_in_temp_cluster = np.unique(sorted_data['Layer'][temp_mask])
        
        for layer_id in layers_in_temp_cluster:
            layer_mask = sorted_data['Layer'] == layer_id
            group_mask_sorted = temp_mask & layer_mask
            num_hits_in_group = np.sum(group_mask_sorted)

            if num_hits_in_group < spatial_min_samples:
                continue

            X_spatial = np.column_stack((
                sorted_data['Column'][group_mask_sorted],
                sorted_data['Row'][group_mask_sorted]
            ))

            dbscan_spatial = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples)
            spatial_labels = dbscan_spatial.fit_predict(X_spatial)

            unique_spatial_labels = np.unique(spatial_labels[spatial_labels != -1])
            original_indices_group = sort_indices[group_mask_sorted]

            for label in unique_spatial_labels:
                cluster_member_mask = spatial_labels == label
                indices_to_update = original_indices_group[cluster_member_mask]
                data['ClusterID'][indices_to_update] = final_cluster_id_counter
                final_cluster_id_counter += 1
    
    print("Spatial clustering finished.")
    return data

def analyze_cluster_tracks_numpy(data, angle = 86.5):
    """
    Analyzes cluster tracks from data stored in a dictionary of NumPy arrays.
    """
    clustered_data, n_unclustered_hits = _filter_clustered_data(data)
    if clustered_data is None:
        print("No multi-hit clusters found to analyze.")
        return {}, n_unclustered_hits

    analysis_results = []
    unique_cluster_ids = np.unique(clustered_data['ClusterID'])

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Polyfit may be poorly conditioned', np.RankWarning)

        for cluster_id in progress_bar(unique_cluster_ids, description="Analyzing cluster tracks"):
            cluster_mask = clustered_data['ClusterID'] == cluster_id
            n_hits = np.sum(cluster_mask)
            
            x = clustered_data['Column'][cluster_mask]
            y = clustered_data['Row'][cluster_mask]
            ts = clustered_data['TS'][cluster_mask]
            ext_ts = clustered_data['ext_TS'][cluster_mask]

            points = np.column_stack((x, y))
            max_length, start_column, start_row, end_column, end_row = _calculate_max_length(points)
            timescale = _calculate_timescale(ts)
            rms_deviation, reduced_rms_deviation, n_missing_hits = _calculate_track_properties(x, y, n_hits, max_length)
            
            
            
            total_hits = n_hits + n_missing_hits
            completeness_ratio = np.float32(n_hits) / np.float32(total_hits) if total_hits > 0 else 0
            min_act_depth = abs((np.float64(max_length-1)*50)/np.tan(angle))
            max_act_depth = abs((np.float64(max_length)*50)/np.tan(angle))
            if n_hits > 1:
                sort_indices = np.argsort(ext_ts)
                sorted_ext_ts = ext_ts[sort_indices]
                ext_ts_diffs = np.diff(sorted_ext_ts)
                ext_ts_diff_mean = np.mean(ext_ts_diffs, dtype = np.float64)
                ext_ts_diff_std = np.std(ext_ts_diffs, dtype = np.float64)
            else:
                ext_ts_diff_mean = 0.0
                ext_ts_diff_std = 0.0
            
            analysis_results.append({
                'cluster_id': cluster_id,
                'n_hits': n_hits,
                'timescale': timescale,
                'reduced_timescale': np.float32(timescale)/np.float32(n_hits),
                'max_length': max_length,
                
                'rms_deviation': np.float64(rms_deviation),
                'reduced_rms_deviation': np.float64(reduced_rms_deviation),
                'n_missing_hits': n_missing_hits,
                'completeness_ratio': np.float32(completeness_ratio),
                'ext_ts_diff_mean': np.float64(ext_ts_diff_mean),
                'ext_ts_diff_std': np.float64(ext_ts_diff_std),
                'start_column': start_column,
                'end_column': end_column,
                'start_row': start_row,
                'end_row': end_row,
                'min_active_depth':min_act_depth,
                'max_active_depth':max_act_depth
            })

    if not analysis_results:
        return {}, n_unclustered_hits
        
    return {key: np.array([d[key] for d in analysis_results]) for key in analysis_results[0]}, n_unclustered_hits

def calculate_timing_uniformity(data, sort_by='space'):
    """
    Calculates timing uniformity within clusters.
    """
    clustered_data, _ = _filter_clustered_data(data)
    if clustered_data is None:
        return {}

    results = []
    unique_cluster_ids = np.unique(clustered_data['ClusterID'])

    for cluster_id in progress_bar(unique_cluster_ids, description="Calculating timing uniformity"):
        cluster_mask = clustered_data['ClusterID'] == cluster_id
        n_hits = np.sum(cluster_mask)
        if n_hits <= 1:
            continue

        cluster_hits = {key: value[cluster_mask] for key, value in clustered_data.items()}
        
        cluster_cols = cluster_hits['Column'].astype(np.int64)
        cluster_rows = cluster_hits['Row'].astype(np.int64)
        cluster_ts = cluster_hits['TS'].astype(np.int64)
        cluster_ext_ts = cluster_hits['ext_TS'].astype(np.int64)

        if sort_by == 'space':
            sort_indices = np.lexsort((cluster_rows, cluster_cols))
        elif sort_by == 'time':
            sort_indices = np.argsort(cluster_ext_ts)
        else:
            raise ValueError("sort_by must be either 'space' or 'time'")

        first_hit_idx = sort_indices[0]
        first_hit_col = cluster_cols[first_hit_idx]
        first_hit_row = cluster_rows[first_hit_idx]
        first_hit_ts = cluster_ts[first_hit_idx]
        first_hit_ext_ts = cluster_ext_ts[first_hit_idx]

        for i in range(n_hits):
            displacement = cluster_rows[i] - first_hit_row
            dTS = cluster_ts[i] - first_hit_ts
            dTS = (dTS + 511) % 1023 - 511
            d_ext_TS = cluster_ext_ts[i] - first_hit_ext_ts
            
            results.append({
                'cluster_id': cluster_id,
                'displacement': displacement,
                'dTS': dTS,
                'd_ext_TS': d_ext_TS,
                'ToT': cluster_hits['ToT'][i]
            })

    if not results:
        return {}

    return {key: np.array([d[key] for d in results]) for key in results[0]}