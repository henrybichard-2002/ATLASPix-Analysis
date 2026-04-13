# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:36:03 2026

@author: henry
"""

import os
import numpy as np
from config import DATA_DIR, TOT_THRESHOLDS, MINOR_FREQ_HZ
from data_loading_optimized import load_data_numpy
from Beam_freq_pipeline import BunchClassifierPipeline
from Tracking2_pipeline import tracking_fast, separate_competing_tracks_fast

def run_full_pipeline(target_filename, n_lines=None):
    print(f"Starting pipeline for {target_filename}")
    file_path = os.path.join(DATA_DIR, target_filename)
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    # Stage 0: Data Loading
    print("Stage 0: Loading Raw Data")
    data_raw = load_data_numpy(file_path, n_lines=n_lines)
    if data_raw is None:
        print("Data loading failed.")
        return

    # Stage 1: Beam Frequency & Hit Labeling
    print("Stage 1: Beam Frequency Classification")
    bunch_pipeline = BunchClassifierPipeline(
        major_bunch_thresholds=TOT_THRESHOLDS,
        minor_freq_hz=MINOR_FREQ_HZ
    )
    labeled_hits = bunch_pipeline.process(data_raw)

    # Stage 2: Clustering
    print("Stage 2: Clustering and Crosstalk Separation")
    from Clustering_pipeline import process_clusters
    final_clusters = process_clusters(labeled_hits)
    
    # Stage 3: Tracking
    print("Stage 3: Track Reconstruction")
    raw_tracks = tracking_fast(final_clusters)
    clean_tracks, dirty_tracks = separate_competing_tracks_fast(raw_tracks, final_clusters)

    print("Pipeline execution complete.")
    return clean_tracks, dirty_tracks

if __name__ == "__main__":
    test_file = "202204061308_udp_beamonall_6Gev_kit_0_decode.dat"
    run_full_pipeline(test_file, n_lines=5000000)