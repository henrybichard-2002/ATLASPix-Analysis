# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:14:44 2025

@author: henry
"""
import os
from data_loading_optimized import load_data_numpy
from clustering_optimized import cluster_events_numpy, analyze_cluster_tracks_numpy, calculate_timing_uniformity
from utils import numpy_to_dataframe, layer_split

# Hardcoded arguments
file_path = r"C:\Users\henry\ATLASpix-analysis\data\202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
file_path2 = r"C:\Users\henry\ATLASpix-analysis\data\202204061308_udp_beamonall_6Gev_kit_0_decode.dat"
#gamma = 86.5

n_lines = 50000000 # Load all lines


filename = os.path.splitext(os.path.basename(file_path2))[0]

# Load data
data_raw = load_data_numpy(file_path2, n_lines=n_lines)
if data_raw is None:
    print("File not found")
    exit()
    
data_layer_split = layer_split(data_raw)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_raw_L4 = data_layer_split[-1]
data_raw_L4_df = numpy_to_dataframe(data_raw_L4)


from crosstalk_mit import remove_crosstalk
data_crossfiltered, data_crosstalk = remove_crosstalk(
    data_raw,
    ToTthreshMin = 20,
    dx = 30,
    ToTrat = 0.2,
    n = 4,
    mode = 'filter')


data_layer_split_crossfiltered = layer_split(data_crossfiltered)
if len(data_layer_split_crossfiltered) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_crossfiltered_L4 = data_layer_split_crossfiltered[-1]
data_crossfiltered_L4_df = numpy_to_dataframe(data_crossfiltered_L4)


from Noise2 import process_hits_with_threshold
data_in, data_out = process_hits_with_threshold(

    dataset=data_raw,
    major_bunch_bin_s= 1.0,
    major_bunch_threshold_count = 4000,
    minor_bunch_freq_hz = 12.5,
    minor_bunch_duration_s = 0.07,
    minor_freq_scan_range_hz = 0.01,
    minor_freq_scan_step_hz = 0.0001,
    minor_phase_scan_step_s = 0.001,
    minor_phase_scan_window_s = 0.1,
    trigger_ts_unit_seconds = 25e-9,
    normalize_tot_heatmaps = False, 
    heatmap_cmap = 'plasma',
    bunch_stat_tot_threshold = 10.0,
    dist_log_y = True,
    dist_num_bins_in = 100,
    dist_num_bins_out = 50,
    kde_bw_bins = 0.5
)
from Noise2 import compare_tot_vs_hits_plots_by_layer
compare_tot_vs_hits_plots_by_layer(data_in, data_out, timestamp_key='TriggerTS', tot_threshold= 10)
#compare_tot_vs_hits_plots_by_layer(data_in, data_out, timestamp_key='ext_TS')

from plotting_optimized import plot_layer_heatmaps
plot_layer_heatmaps(data_in, 'Column', 'Row')
plot_layer_heatmaps(data_out, 'Column', 'Row')


data_in_layer_split = layer_split(data_in)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
    
data_in_L4 = data_in_layer_split[-1] 
data_in_L3 = data_in_layer_split[-2] 
data_in_L2 = data_in_layer_split[-3] 
data_in_L1 = data_in_layer_split[-4] 

data_out_layer_split = layer_split(data_out)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_out_L4 = data_out_layer_split[-1] 
data_out_L3 = data_out_layer_split[-2] 
data_out_L2 = data_out_layer_split[-3] 
data_out_L1 = data_out_layer_split[-4] 


from correlation_analysis2 import plot_pixel_pair_correlation_analysis

# Assuming 'data_raw_L4' is already loaded in your environment

# --- Task 1 ---
print("\n" + "="*80)
print("   RUNNING TASK 1: Correlated Pairs (ToT > 0 only)")
print("   Description: 'Data L4 [Raw], No ToT cuts, No Displ. cuts,'")
print("="*80 + "\n")

correlation_pix_map_l4_raw = plot_pixel_pair_correlation_analysis(
    data = data_raw_L4,
    columns_to_analyze = None,
    filter_tot_low_threshold = None,
    filter_tot_high_threshold = None,
    filter_disp_gate = None,
    filter_tot_zero = True,
    uncorrelated_background_mode = False, 
    return_heatmap_data = True,
    dataset_description = 'Data L4 [Raw], No ToT cuts, No Displ. cuts,'
)
print("\n   > Task 1 complete.\n")

# --- Task 2 ---
print("\n" + "="*80)
print("   RUNNING TASK 2: Uncorrelated Background (ToT > 0 only)")
print("   Description: 'Data L4 [Raw], Uncorrelated background mode,'")
print("="*80 + "\n")

uncorrelation_pix_map_l4_raw = plot_pixel_pair_correlation_analysis(
    data = data_raw_L4,
    columns_to_analyze = None,
    filter_tot_low_threshold = None,
    filter_tot_high_threshold = None,
    filter_disp_gate = None,
    filter_tot_zero = True,
    uncorrelated_background_mode = True, 
    return_heatmap_data = True,
    dataset_description = 'Data L4 [Raw], Uncorrelated background mode,'
)
print("\n   > Task 2 complete.\n")

# --- Task 3 ---
print("\n" + "="*80)
print("   RUNNING TASK 3: Correlated Pairs (ToT & Displacement Filtered)")
print("   Description: 'Data L4 [Raw], ToT < 20 cut - ToT > 245 cut, [-10,10] Displ. cuts,'")
print("="*80 + "\n")

correlation_pix_map_l4_filtered = plot_pixel_pair_correlation_analysis( 
    data = data_raw_L4,
    columns_to_analyze = None,
    filter_tot_low_threshold = 20,
    filter_tot_high_threshold = 245,
    filter_disp_gate = [-10,10],
    min_abs_correlation = 0,
    filter_tot_zero = True,
    uncorrelated_background_mode = False, 
    return_heatmap_data = True,
    dataset_description = 'Data L4 [Raw], 20<ToT[A,B]>245 cut, [-10,10] Displ. cuts,'
)
print("\n   > Task 3 complete.\n")

# --- Task 4 ---
print("\n" + "="*80)
print("   RUNNING TASK 4: Uncorrelated Background (ToT & Displacement Filtered)")
print("   Description: 'Data L4 [Raw], ToT < 20 cut - ToT > 245 cut, [-10,10] Displ. cuts, uncorrelated background mode'")
print("="*80 + "\n")

uncorrelation_pix_map_l4_filtered = plot_pixel_pair_correlation_analysis( # Renamed variable
    data = data_raw_L4,
    columns_to_analyze = None,
    filter_tot_low_threshold = 20,
    filter_tot_high_threshold = 245,
    filter_disp_gate = [-10,10],
    filter_tot_zero = True,
    min_abs_correlation = 0.02,
    uncorrelated_background_mode = True, 
    return_heatmap_data = True,
    dataset_description = 'Data L4 [Raw], 20<ToT[A,B]>245 cut, [-10,10] Displ. cuts, \n uncorrelated background mode'
)
print("\n   > Task 4 complete.\n")

correlation_pix_map_l4_filtered = plot_pixel_pair_correlation_analysis( 
    data = data_raw_L4,
    columns_to_analyze = None,
    filter_tot_low_threshold = 20,
    filter_tot_high_threshold = 245,
    filter_disp_gate = [-10,10],
    min_abs_correlation = 0.05,
    filter_tot_zero = True,
    uncorrelated_background_mode = False, 
    return_heatmap_data = True,
    dataset_description = 'Data L4 [Raw],  20<ToT[A,B]>245 cut, [-10,10] Displ. cuts, \n > 0.05 abs correlation thresh'
)
'''
from correlation_analysis2 import plot_correlation_by_value
plot_correlation_by_value(data_in, low_corr_threshold = 0, mid_corr_min = 0, mid_corr_max = 0.5, filter_tot_low_threshold = 15)
plot_correlation_by_value(data_out, low_corr_threshold = 0, mid_corr_min = 0, mid_corr_max = 0.5, filter_tot_low_threshold = 15)
'''