import pandas as pd
import numpy as np
import sys

def numpy_to_dataframe(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Converts a dictionary of NumPy arrays to a pandas DataFrame and casts all columns to int64."""
    df = pd.DataFrame(data)
    return df.astype(np.int64)

def layer_split(data: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    """Splits the data dictionary by layer."""
    unique_layers = np.unique(data['Layer'])
    data_layers = []
    for layer in unique_layers:
        layer_mask = data['Layer'] == layer
        layer_data = {key: value[layer_mask] for key, value in data.items()}
        data_layers.append(layer_data)
    return data_layers

def progress_bar(iterable, description="", total=None):
    """Displays a progress bar in the terminal."""
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = 0

    for i, item in enumerate(iterable):
        yield item
        progress = (i + 1) / total
        bar_length = 20
        block = int(round(bar_length * progress))
        text = f"\r{description}: [{'#' * block + '-' * (bar_length - block)}] {progress:.1%}"
        sys.stdout.write(text)
        sys.stdout.flush()
    sys.stdout.write('\n')
    

def _calculate_row_cog(cols_in_row, tots_in_row):
    total_tot = np.sum(tots_in_row)
    if total_tot > 0:
        weighted_sum = np.sum(cols_in_row.astype(np.float64) * tots_in_row)
        return weighted_sum / total_tot
    return None

import numpy as np
from typing import Dict, Union, Sequence

def filter_by_tot(data: Dict[str, np.ndarray],
                        tot_range: Sequence[Union[int, float]],
                        description: str = "") -> Dict[str, np.ndarray]:
    """
    Filters a dataset by keeping entries where 'ToT' is within a given range.
    Returns:
        Dict[str, np.ndarray]: A new dictionary containing only the filtered data.
    """
    if not isinstance(tot_range, (list, tuple, np.ndarray)) or len(tot_range) < 2:
        raise ValueError("tot_range must be a sequence (e.g., list) with at least two elements: [min, max].")
    initial_count = len(data['ToT'])
    min_thresh, max_thresh = tot_range[0], tot_range[-1]
    mask = (data['ToT'] >= min_thresh) & (data['ToT'] <= max_thresh)
    filtered_data = {key: value[mask] for key, value in data.items()}
    final_count = len(filtered_data['ToT'])
    if description:
        print(f"{description}: Kept {final_count} of {initial_count} entries "
              f"({final_count / initial_count:.2%}) using {min_thresh} <= ToT <= {max_thresh}")
    return filtered_data


def filter_data_by_row(data_raw):
    if 'Row' not in data_raw:
        raise ValueError("Input dictionary must contain a 'Row' key.")
    rows = data_raw['Row']
    mask_A = (rows >= 0) & (rows <= 123)
    mask_B = (rows >= 124) & (rows <= 247)
    mask_C = ~((rows >= 0) & (rows <= 247)) # Inverted mask for all other rows
    group_A = {}
    group_B = {}
    group_C = {}
    for key, arr in data_raw.items():
        group_A[key] = arr[mask_A]
        group_B[key] = arr[mask_B]
        group_C[key] = arr[mask_C]
    return group_A, group_B, group_C


import numpy as np

def filter_clusters_by_size(data, n_min, n_max, cluster_col='ClusterID'):
    if cluster_col not in data:
        raise ValueError(f"Cluster column '{cluster_col}' not found in data keys.")
    all_cluster_ids = data[cluster_col]
    unique_clusters, counts = np.unique(all_cluster_ids, return_counts=True)
    size_mask = (counts >= n_min) & (counts <= n_max)
    valid_cluster_ids = unique_clusters[size_mask]
    if valid_cluster_ids.size == 0:
        print(f"Warning: No clusters found with size between {n_min} and {n_max}.")
        # Return a dictionary with the same keys but empty arrays
        empty_data = {key: np.array([], dtype=value.dtype) for key, value in data.items()}
        empty_data['n_hits'] = np.array([], dtype=int) # Also return empty n_hits
        return empty_data
    filter_mask = np.isin(all_cluster_ids, valid_cluster_ids)
    filtered_data = {key: value[filter_mask] for key, value in data.items()}
    id_to_count_map = dict(zip(unique_clusters, counts))
    filtered_ids = filtered_data[cluster_col]
    n_hits_array = np.vectorize(id_to_count_map.get)(filtered_ids)
    filtered_data['n_hits'] = n_hits_array
    return filtered_data

def save_correlation_matrices(data_dict: Dict[int, pd.DataFrame], filename: str):
    if not filename.endswith('.npz'):
        filename += '.npz'
    to_save = {}
    for key, df in data_dict.items():
        to_save[f'data_{key}'] = df.to_numpy()
        to_save[f'index_{key}'] = df.index.to_numpy()
        to_save[f'columns_{key}'] = df.columns.to_numpy()
    try:
        np.savez_compressed(filename, **to_save)
        print(f"Successfully saved correlation matrices to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

def load_correlation_matrices(filename: str) -> Dict[int, pd.DataFrame]:
    loaded_data = {}
    try:
        with np.load(filename) as data:
            # Find all the unique keys (column IDs)
            keys = sorted(list(set([int(k.split('_')[1]) for k in data.files])))
            for key in keys:
                matrix_data = data[f'data_{key}']
                index = data[f'index_{key}']
                columns = data[f'columns_{key}']
                df = pd.DataFrame(matrix_data, index=index, columns=columns)
                loaded_data[key] = df
        print(f"Successfully loaded correlation matrices from {filename}")
        return loaded_data
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return {}
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}