# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:14:44 2025

@author: henry
"""
import os
from data_loading_optimized import load_data_numpy
from utils import numpy_to_dataframe, layer_split

# Hardcoded arguments
file_path = r"C:\Users\henry\ATLASpix-analysis\data\202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
file_path2 = r"C:\Users\henry\ATLASpix-analysis\data\202204061308_udp_beamonall_6Gev_kit_0_decode.dat"
#gamma = 86.5

n_lines = 50000000 # Load all lines


filename = os.path.splitext(os.path.basename(file_path))[0]

# Load data
data_raw = load_data_numpy(file_path, n_lines=n_lines)
if data_raw is None:
    print("File not found")
    exit()
data_raw_df = numpy_to_dataframe(data_raw)
data_layer_split = layer_split(data_raw)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_raw_L4 = data_layer_split[-1]
data_raw_L4_df = numpy_to_dataframe(data_raw_L4)
    
from crosstalk_mit import remove_crosstalk2, calculate_aggregated_correlation, plot_correlation_matrix, plot_displacement_analysis, plot_crosstalk_spectra
from correlation_analysis2 import extract_high_correlation_pairs, plot_pixel_pair_correlation_analysis

avg_correlations, hit_correlations = calculate_aggregated_correlation(data_raw, group_by= 'TriggerID', tot_ratio_threshold = 1,
                                                                      target_layer = None)

plot_correlation_matrix(avg_correlations, title = 'Global correlation matrix - Data Raw')
plot_displacement_analysis(avg_correlations, hit_correlations, 
                               title_prefix = 'Data Raw Correlation', 
                               log_y= True,
                               prominence = 0.5)

corr_pairs_disp_0 = extract_high_correlation_pairs(avg_correlations, hit_correlations, threshold = 0.052, dx = 0)
corr_pairs_disp_16 = extract_high_correlation_pairs(avg_correlations, hit_correlations, threshold = 0.052, dx = 16)

from crosstalk_mit import plot_corr_disp
plot_corr_disp(corr_pairs_disp_0, mode='scatter', log_z = False)
plot_corr_disp(corr_pairs_disp_0, mode='heatmap', log_z = True)

#corr_pairs_disp_60 = extract_high_correlation_pairs(correlations, threshold = 0.01, dx = 60)

#data_clean, crosstalk_data = remove_crosstalk2(data_raw, corr_pairs_disp_16, trigger_col='TriggerID', 
                       #ratio_ToTthresh1=0.2, ratio_ToTthresh2=0.9, 
                       #noise_ToTthresh=15)
from crosstalk_mit import remove_crosstalk_types             
data_clean, crosstalk_type1, crosstalk_type2 = remove_crosstalk_types(data_raw, corr_pairs_disp_16, trigger_col= 'TriggerID',
                       ratio_ToTthresh=0.2, noise_ToTthresh=20, dx=40)

avg_clean_corr, clean_total_hits = calculate_aggregated_correlation(data_clean, group_by= 'TriggerID', tot_ratio_threshold = 1)


plot_correlation_matrix(avg_clean_corr, title = 'Global correlation matrix - Clean Data')
plot_displacement_analysis(avg_clean_corr, clean_total_hits,
                               title_prefix = 'Data CLEAN Correlation', 
                               log_y= True,
                               prominence = 0.5)

plot_crosstalk_spectra(data_raw_df, data_clean, crosstalk_type1, crosstalk_type2, log_yscale=True)

_ = plot_pixel_pair_correlation_analysis(
    data = data_raw_L4,
    columns_to_analyze = None,
    filter_tot_low_threshold = None,
    filter_tot_high_threshold = None,
    filter_disp_gate = [-1,1],
    filter_tot_zero = True,
    uncorrelated_background_mode = False, 
    return_heatmap_data = True,
    dataset_description = 'Data [Raw], ToT cut = 15, Displ. cuts dx = 16',
    log_y_disp_plots=True
)

cleaned_data_layer_split = layer_split(data_clean)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_cleaned_L4 = cleaned_data_layer_split[-1]


_ = plot_pixel_pair_correlation_analysis(
    data = data_cleaned_L4,
    columns_to_analyze = None,
    filter_tot_low_threshold = None,
    filter_tot_high_threshold = None,
    filter_disp_gate = [-1,1],
    filter_tot_zero = True,
    uncorrelated_background_mode = False, 
    return_heatmap_data = True,
    dataset_description = 'Data [Cleaned], ToT cut = 15, Displ. cuts dx = 16',
    log_y_disp_plots=True
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def plot_crosstalktype_displacement(data, crosstalk_type = 4):
    """
    Plots the frequency distribution of the absolute Row Displacement (|Delta Row|)
    between hit pairs (hits sharing the same TriggerID) for Crosstalk Type 4.
    
    All layers are overlaid on a single step graph. 
    Identifies multiple peaks per layer, but labels a specific x-location 
    only once (at the highest occurrence) to avoid clutter.
    """
    # Filter for Type 4
    df_t4 = data[data['crosstalk_type'] == crosstalk_type].copy()
    
    if df_t4.empty:
        print("No Type 4 crosstalk data found.")
        return

    unique_layers = np.sort(df_t4['Layer'].unique())
    
    if len(unique_layers) == 0:
        return

    # Setup Single Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Type 4 Crosstalk: Absolute Row Displacement Distribution", fontsize=16)

    # Define bins: One bin per integer step from 0 to 372
    bins = np.arange(-0.5, 372.5, 1)

    # Dictionary to store the highest peak at each x-location found across all layers
    # Key: int(x_location), Value: (y_value, color_of_line)
    peak_candidates = {}

    for i, layer in enumerate(unique_layers):
        # Create explicit copy
        layer_data = df_t4[df_t4['Layer'] == layer].copy()

        # Group by TriggerID (and PackageID)
        group_cols = ['TriggerID']
        if 'PackageID' in layer_data.columns:
            group_cols = ['PackageID', 'TriggerID']
            
        # Filter for pairs (count == 2)
        layer_data['group_count'] = layer_data.groupby(group_cols)['TriggerID'].transform('count')
        pairs_df = layer_data[layer_data['group_count'] == 2].copy()
        
        if pairs_df.empty:
            continue
            
        # Sort to ensure consistent ordering within pairs
        pairs_df = pairs_df.sort_values(by=group_cols)
        
        # Extract rows
        rows = pairs_df['Row'].values.astype(np.int32)
        
        # Reshape to (N_pairs, 2)
        rows_pairs = rows.reshape(-1, 2)
        
        # Calculate Absolute Row Displacement
        abs_d_row = np.abs(rows_pairs[:, 0] - rows_pairs[:, 1])
        
        # Calculate Histogram stats for labeling
        counts, edges = np.histogram(abs_d_row, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Plot Step Histogram
        lines = ax.step(
            centers, 
            counts, 
            where='mid', 
            linewidth=1.5, 
            label=f'Layer {layer}',
            alpha=0.8
        )
        
        line_color = lines[0].get_color()
        

        peaks_indices, _ = find_peaks(counts, height=20, distance = 20)

        # Iterate over all found peaks for this layer
        for peak_idx in peaks_indices:
            peak_val = counts[peak_idx]
            peak_loc = int(centers[peak_idx])
            
            # Store logic: If this x-location hasn't been labeled yet, 
            # OR if the peak at this x-location on this layer is taller than previous layers
            if peak_loc not in peak_candidates or peak_val > peak_candidates[peak_loc][0]:
                peak_candidates[peak_loc] = (peak_val, line_color)

    # Draw Peak Labels (iterating through the de-duplicated dictionary)
    for x_loc, (y_val, color) in peak_candidates.items():
        ax.annotate(
            f'{x_loc}', 
            xy=(x_loc, y_val), 
            xytext=(0, 5), 
            textcoords='offset points', 
            ha='center', 
            va='bottom',
            color=color,
            fontweight='bold',
            fontsize=9
        )

    # Formatting
    ax.set_xlabel(r"Absolute Row Displacement ($|\Delta$ Row|)")
    ax.set_ylabel("Frequency (Count)")
    ax.legend(title="Layer")
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()