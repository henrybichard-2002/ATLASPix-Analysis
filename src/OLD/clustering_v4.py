import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from utils import progress_bar

def cluster_hits(data_raw, dx, dt, min_length, max_length, max_total_time):
    """
    Clusters hits into tracks using DBSCAN with spatial and temporal constraints.
    
    Args:
        data_raw (dict): Dictionary of numpy arrays containing hit data.
        dx (float): Maximum epsilon distance (Row, Column).
        dt (float): Maximum time difference (ext_TS).
        min_length (float): Minimum Euclidean distance between start/end of track.
        max_length (float): Maximum Euclidean distance between start/end of track.
        max_total_time (int): Maximum duration (ext_TS difference) of the track.
        
    Returns:
        tuple: (tracks_dict, noise_dict)
            tracks_dict: Dictionary containing hits belonging to valid tracks (includes TrackID).
            noise_dict: Dictionary containing hits that are noise or short/long tracks.
    """
    
    # Define logical execution phases for the progress bar
    phases = ['Preprocessing', 'DBSCAN', 'Filtering', 'Finalizing']
    # Initialize generator
    pbar = progress_bar(phases, description="Clustering", total=len(phases))
    
    # --- PHASE 1: Preprocessing ---
    next(pbar) # Yields 'Preprocessing'

    # 1. Convert Input to DataFrame for efficient indexing and manipulation
    df = pd.DataFrame(data_raw)
    
    # 2. Feature Scaling
    # DBSCAN uses a spherical epsilon. We have an anisotropic constraint:
    # Spatial limit = dx, Temporal limit = dt.
    # We scale the time dimension such that a difference of 'dt' becomes 'dx'.
    time_scale_factor = dx / dt
    
    # Create a feature matrix. We convert to float32 to save memory but ensure precision
    # columns: Row, Column, Scaled_Time
    features = df[['Row', 'Column', 'ext_TS']].astype(np.float32).values
    features[:, 2] *= time_scale_factor 
    
    # --- PHASE 2: DBSCAN ---
    next(pbar) # Updates bar (25%), Yields 'DBSCAN'

    # 3. Run DBSCAN (Parallelized)
    # eps=dx: Because time is scaled, this enforces spatial < dx AND temporal < dt
    model = DBSCAN(eps=dx, min_samples=2, metric='euclidean', n_jobs=-1)
    labels = model.fit_predict(features)
    
    df['TrackID'] = labels

    # --- PHASE 3: Filtering ---
    next(pbar) # Updates bar (50%), Yields 'Filtering'

    # 4. Vectorized Length Calculation & Filtering
    
    # Separate initial noise (-1) from potential candidates
    candidate_mask = df['TrackID'] != -1
    
    if not candidate_mask.any():
        # Complete the progress bar before returning early
        try: list(pbar) 
        except StopIteration: pass
        return {}, data_raw

    candidates = df[candidate_mask].copy()
    
    # Group by TrackID and find the index of the earliest and latest hit
    grouped = candidates.groupby('TrackID')['ext_TS']
    start_indices = grouped.idxmin()
    end_indices = grouped.idxmax()
    
    # Extract coordinates for start and end points using the indices
    coords = df[['Row', 'Column']].values
    timestamps = df['ext_TS'].values

    start_coords = coords[start_indices]
    end_coords = coords[end_indices]
    
    start_times = timestamps[start_indices]
    end_times = timestamps[end_indices]

    # Calculate Euclidean distance (vectorized)
    dists = np.linalg.norm(start_coords - end_coords, axis=1)
    
    # Calculate Duration (vectorized)
    durations = end_times - start_times

    # Get TrackIDs that satisfy the length condition
    valid_length_mask = (dists >= min_length) & (dists <= max_length) & (durations <= max_total_time)
    valid_track_ids = start_indices.index[valid_length_mask]
    
    # --- PHASE 4: Finalizing ---
    next(pbar) # Updates bar (75%), Yields 'Finalizing'

    # 5. Split Data into Final Datasets
    final_track_mask = df['TrackID'].isin(valid_track_ids)
    df_tracks = df[final_track_mask].copy()
    df_noise = df[~final_track_mask].drop(columns=['TrackID'], errors='ignore')

    # 6. Convert back to Dictionary of Arrays
    def df_to_dict(dataframe):
        out_dict = {}
        for col in dataframe.columns:
            out_dict[col] = dataframe[col].values
        return out_dict

    result = (df_to_dict(df_tracks), df_to_dict(df_noise))

    # Finish progress bar (Updates to 100%)
    try:
        next(pbar)
    except StopIteration:
        pass

    return result