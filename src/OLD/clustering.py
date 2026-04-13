import pandas as pd


def find_clusters(df: pd.DataFrame, max_dT: int, ts_modulus: int = 1024) -> pd.DataFrame:
    if not all(col in df.columns for col in ['PackageID', 'TS']):
        raise ValueError("Input DataFrame must contain 'PackageID' and 'TS' columns.")

    # Sort by time to find consecutive hits
    df_sorted = df.sort_values(by=['TS']).reset_index(drop=True)

    # Calculate time difference between consecutive hits
    time_diff = df_sorted['TS'].diff()
    time_diff_corrected = time_diff % ts_modulus

    # Calculate PackageID difference between consecutive hits
    package_id_diff = df_sorted['PackageID'].diff().abs()

    # A new cluster starts if the time difference is too large,
    # or if the PackageID is not consecutive (diff > 1)
    time_jump = time_diff_corrected > max_dT
    package_jump = package_id_diff > 1

    cluster_start = (time_jump | package_jump).fillna(True)
    cluster_ids = cluster_start.cumsum()

    df_with_clusters = df_sorted.assign(cluster_id=cluster_ids)
    return df_with_clusters

def summarize_clusters(df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    if 'cluster_id' not in df_with_clusters.columns:
        raise ValueError("Input DataFrame must contain a 'cluster_id' column.")
    aggregations = {
        'PackageID': 'first',
        'TS': ['min', 'max'],
        'Layer': ['nunique', 'count']
    }

    cluster_summary = df_with_clusters.groupby('cluster_id').agg(aggregations)

    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    cluster_summary = cluster_summary.rename(columns={
        'PackageID_first': 'PackageID',
        'TS_min': 'cluster_TS_start',
        'TS_max': 'cluster_TS_end',
        'Layer_nunique': 'n_unique_layers',
        'Layer_count': 'n_hits'
    })

    cluster_summary['cluster_duration'] = (
        cluster_summary['cluster_TS_end'] - cluster_summary['cluster_TS_start']
    )
    
    return cluster_summary

import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import warnings


def cluster_events(df, dt, spatial_eps, spatial_min_samples):
    if df.empty:
        df['ClusterID'] = pd.Series(dtype='int')
        return df

    # --- Stage 1: Temporal Clustering ---
    # Sort data to process hits in order of occurrence
    df_sorted = df.sort_values(['PackageID', 'TS']).reset_index(drop=True)

    # Calculate differences with the previous hit in the sorted list
    ts_diff = df_sorted['TS'].diff()
    package_diff = df_sorted['PackageID'].diff()

    # A new temporal cluster starts if the time gap is too large,
    # or if the package is not the same or the next consecutive one.
    # The first hit always starts a new cluster (as its diff is NaN).
    is_new_temporal_cluster = (ts_diff >= dt) | \
                              (~package_diff.isin([0, 1])) | \
                              (ts_diff.isna())
    
    # Use cumsum() to assign a unique ID to each temporal cluster
    df_sorted['TemporalClusterID'] = is_new_temporal_cluster.cumsum() - 1

    # --- Stage 2: Spatial Clustering ---
    # This will be performed on each temporal cluster, separated by layer.
    
    df_sorted['ClusterID'] = -1  # Final cluster ID, default to noise (-1)
    final_cluster_id_counter = 0

    # Group by the temporal clusters we just found
    grouped_by_temporal = df_sorted.groupby('TemporalClusterID')

    for temp_id, temporal_cluster_df in grouped_by_temporal:
        # Within each temporal cluster, further group by layer
        grouped_by_layer = temporal_cluster_df.groupby('Layer')
        
        for layer_id, layer_df in grouped_by_layer:
            # If there aren't enough points to meet the minimum for a cluster, skip
            if len(layer_df) < spatial_min_samples:
                continue

            # Prepare data for spatial DBSCAN (Column and Row)
            X_spatial = layer_df[['Column', 'Row']].values

            # Initialize and run spatial DBSCAN
            dbscan_spatial = DBSCAN(eps=spatial_eps, min_samples=spatial_min_samples)
            spatial_labels = dbscan_spatial.fit_predict(X_spatial)

            # Map the local spatial cluster labels (0, 1, -1, etc.) to global unique ClusterIDs
            
            # Create a Series of the new labels, using the original index
            # to ensure we update the correct rows in the main DataFrame.
            spatial_labels_series = pd.Series(spatial_labels, index=layer_df.index)
            
            # Find the unique non-noise labels assigned by DBSCAN
            unique_spatial_labels = np.unique(spatial_labels[spatial_labels != -1])

            for label in unique_spatial_labels:
                # Find the original indices of all hits belonging to this spatial cluster
                indices_to_update = spatial_labels_series[spatial_labels_series == label].index
                
                # Assign the next available global cluster ID
                df_sorted.loc[indices_to_update, 'ClusterID'] = final_cluster_id_counter
                final_cluster_id_counter += 1

    # Return the final DataFrame, dropping the intermediate temporal cluster ID
    return df_sorted.drop(columns=['TemporalClusterID'])

def analyze_cluster_tracks(df_clustered: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required_cols = ['ClusterID', 'TS', 'Column', 'Row']
    if not all(col in df_clustered.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain {required_cols} columns.")

    if df_clustered.empty:
        return pd.DataFrame(), 0

    n_unclustered_hits = len(df_clustered[df_clustered['ClusterID'] == -1])
    df_to_analyze = df_clustered[df_clustered['ClusterID'] != -1]

    if df_to_analyze.empty:
        print("No multi-hit clusters found to analyze.")
        return pd.DataFrame(), n_unclustered_hits
    
    analysis_results = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Polyfit may be poorly conditioned', np.RankWarning)
        
        for cluster_id, cluster_data in df_to_analyze.groupby('ClusterID'):
            n_hits = len(cluster_data)
            
            x = cluster_data['Column'].values
            y = cluster_data['Row'].values

            ts_modulus = 1023
            ts_values = np.sort(cluster_data['TS'].unique())

            if len(ts_values) > 1:
                diffs = np.diff(ts_values)
                wrapped_diff = (ts_values[0] + ts_modulus) - ts_values[-1]
                largest_gap = np.max(np.append(diffs, wrapped_diff))
                timescale = ts_modulus - largest_gap
            else:
                timescale = 0

            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            reduced_chi_square = 0.0

            if x_min == x_max:
                absolute_length = y_max - y_min
                rms_deviation = 0.0
                start_row, end_row = y_min, y_max
                
                actual_coords = set(zip(x, y))
                constant_x = x[0]
                expected_rows = np.arange(y_min, y_max + 1)
                expected_coords = set([(constant_x, r) for r in expected_rows])
                n_missing_hits = len(expected_coords - actual_coords)
            
            elif y_min == y_max:
                absolute_length = x_max - x_min
                rms_deviation = 0.0
                start_row, end_row = y_min, y_max
                
                actual_coords = set(zip(x, y))
                constant_y = y[0]
                expected_cols = np.arange(x_min, x_max + 1)
                expected_coords = set([(c, constant_y) for c in expected_cols])
                n_missing_hits = len(expected_coords - actual_coords)

            else:
                slope, intercept = np.polyfit(x, y, 1)

                y_start_fit = slope * x_min + intercept
                y_end_fit = slope * x_max + intercept
                absolute_length = np.sqrt((x_max - x_min)**2 + (y_end_fit - y_start_fit)**2)
                start_row, end_row = y_start_fit, y_end_fit

                distances = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
                rms_deviation = np.sqrt(np.mean(distances**2))
                if n_hits > 2:
                    y_expected = slope * x + intercept
                    sum_squared_residuals = np.sum((y - y_expected)**2)
                    degrees_of_freedom = n_hits - 2
                    reduced_chi_square = sum_squared_residuals / degrees_of_freedom
                actual_coords = set(zip(x, y))
                num_points = max(2, int(absolute_length * 2))
                x_line = np.linspace(x_min, x_max, num=num_points)
                y_line = slope * x_line + intercept
                expected_coords = set(zip(np.round(x_line).astype(int), np.round(y_line).astype(int)))
                missing_coords = expected_coords - actual_coords
                n_missing_hits = len(missing_coords)
            
            analysis_results.append({
                'cluster_id': cluster_id,
                'n_hits': n_hits,
                'timescale': timescale,
                'absolute_length': absolute_length,
                'rms_deviation': rms_deviation,
                # --- NEW: Add the calculated value to the results ---
                'reduced_chi_square': reduced_chi_square,
                'n_missing_hits': n_missing_hits,
                'rat_missing_hits': n_missing_hits / n_hits,
                'start_column': x_min,
                'end_column': x_max,
                'start_row': start_row,
                'end_row': end_row
            })

    summary_df = pd.DataFrame(analysis_results)
    return summary_df, n_unclustered_hits
