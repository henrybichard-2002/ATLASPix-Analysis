# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 16:14:44 2025

@author: henry
"""
import matplotlib.pyplot as plt
import os
import numpy as np
from data_loading_optimized import load_data_numpy
from clustering_optimized import cluster_events_numpy, analyze_cluster_tracks_numpy, calculate_timing_uniformity

from utils import numpy_to_dataframe, layer_split, filter_by_tot, progress_bar

# Hardcoded arguments
file_path = r"C:\Users\henry\ATLASpix-analysis\data\202204061308_udp_beamonall_6Gev_kit_0_decode.dat"
#gamma = 86.5

n_lines = 30000000 # Load all lines
tot_thresh = [10,250]
dt = 12
eps = 1.5
min_hits = 1

import os
filename = os.path.splitext(os.path.basename(file_path))[0]

# Load data
data_raw = load_data_numpy(file_path, n_lines=n_lines)
if data_raw is None:
    print("File not found")
    exit()
    
data_layer_split = layer_split(data_raw)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_raw_L4 = data_layer_split[-1]
data_raw_L4_df = numpy_to_dataframe(data_raw_L4)

from utils import filter_data_by_row

Adata_raw_L4, Bdata_raw_L4, Cdata_raw_L4 = filter_data_by_row(data_raw_L4)

plot_column_hist_numpy(Adata_raw_L4, column = 'ToT', density = False)
plot_column_hist_numpy(Bdata_raw_L4, column = 'ToT', density = False)
plot_column_hist_numpy(Cdata_raw_L4, column = 'ToT', density = False)


def plot_multiple_column_histograms(datasets, labels, column, colors=None, bins=50, 
                                    density=False, title=None, xlabel=None, ylabel=None,
                                    yscale = None):

    if len(datasets) != len(labels):
        raise ValueError("The number of datasets must match the number of labels.")
    if colors and len(datasets) != len(colors):
        raise ValueError("The number of datasets must match the number of colors.")

    plt.figure(figsize=(10, 6))
    
    # Use provided colors or default if None
    color_cycle = colors if colors else plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (data, label) in enumerate(zip(datasets, labels)):
        if column not in data:
            print(f"Warning: Column '{column}' not found in dataset labeled '{label}'. Skipping.")
            continue
        plt.hist(data[column], bins=bins, density=density, alpha=0.7, label=label,
                 histtype='step', color=color_cycle[i])

    plt.title(title if title else f'Histogram of {column}')
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel(ylabel if ylabel else ('Density' if density else 'Count'))
    
    if yscale == 'log':
        plt.yscale('log')
        
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

plot_multiple_column_histograms([Adata_raw_L4, Bdata_raw_L4, Cdata_raw_L4], 
                                 labels=["SectionA", "SectionB", "Sectionc"],
                                 colors=['green', 'red', 'blue'],
                                 density=False,
                                 column='ToT',
                                 xlabel='ToT [TS]',
                                 ylabel='Counts',
                                 title='Comparing ToT Spectra for A,B,C row sections',
                                 bins=256,
                                 yscale='log')


'''
# Clustering
Adata_clustered_All_L4 = cluster_events_numpy(
    data=Adata_raw_L4,
    dt=dt,
    spatial_eps=eps,
    spatial_min_samples=min_hits
)
Adata_clustered_All_L4_df = numpy_to_dataframe(Adata_clustered_All_L4)
Bdata_clustered_All_L4 = cluster_events_numpy(
    data=Bdata_raw_L4,
    dt=dt,
    spatial_eps=eps,
    spatial_min_samples=min_hits
)
Bdata_clustered_All_L4_df = numpy_to_dataframe(Bdata_clustered_All_L4)
Cdata_clustered_All_L4 = cluster_events_numpy(
    data=Cdata_raw_L4,
    dt=dt,
    spatial_eps=eps,
    spatial_min_samples=min_hits
)
Cdata_clustered_All_L4_df = numpy_to_dataframe(Cdata_clustered_All_L4)
'''
data_clustered_All_L4 = cluster_events_numpy(
    data=data_raw_L4,
    dt=dt,
    spatial_eps=eps,
    spatial_min_samples=min_hits
)
data_clustered_All_L4_df = numpy_to_dataframe(data_clustered_All_L4)

from utils import filter_clusters_by_size
n_min = 15
n_max = 40
tot_thresh = [0,10]
data_clustered_All_L4_Tracks = filter_clusters_by_size(data_clustered_All_L4_df, n_min = n_min, n_max=n_max)

data_clustered_All_L4_Tracks_loToT = filter_by_tot(data_clustered_All_L4_Tracks, tot_range= tot_thresh)
data_Raw_L4_Hits_loToT = filter_by_tot(data_clustered_All_L4, tot_range=[0,10])
data_Raw_L4_Hits_meanToT = filter_by_tot(data_clustered_All_L4, tot_range=[10,252])


plot_HeatHitmap(data_clustered_All_L4_Tracks, 
                xcol = 'Column', ycol='Row', 
                title= f"Cluster size: {n_min}-{n_max} hits/cluster, NOT ToT gated \n (LogScale)")
plot_HeatHitmap(data_clustered_All_L4_Tracks_loToT, 
                xcol = 'Column', ycol='Row', 
                title = f"Cluster size: {n_min}-{n_max} hits/cluster, ToT gated: {tot_thresh}[TS] \n (LogScale)")
plot_HeatHitmap(data_clustered_All_L4_Tracks_loToT, 
                xcol = 'Column', ycol='Row', 
                title = f"Cluster size: {n_min}-{n_max} hits/cluster, ToT gated: {tot_thresh}[TS]",
                log_z=False)

data_clustered_All_L4_Tracks_loToT_df = numpy_to_dataframe(data_clustered_All_L4_Tracks_loToT )

from correlation_analysis import plot_pixel_correlation, plot_cross_dataset_pixel_correlation

plot_pixel_correlation(data_raw_L4, 
                       analyze_by='row', 
                       use_log_scale=True, 
                       title='Row-wise Pixel Correlation for Raw Data (Log Scale)')

plot_pixel_correlation(data_raw_L4, 
                       analyze_by='column', 
                       use_log_scale=False, 
                       title='Column-wise Pixel Correlation for Raw Data (Linear Scale)')

plot_pixel_correlation(data_Raw_L4_Hits_loToT, 
                       analyze_by='column', 
                       title='Column-wise Pixel Correlation for Low ToT Hits Data')

plot_pixel_correlation(data_Raw_L4_Hits_meanToT, 
                       analyze_by='column', 
                       title='Column-wise Pixel Correlation for Mean ToT Hits Data')

column_correlation_matrices = plot_cross_dataset_pixel_correlation(data_raw_L4, 
                                     data_raw_L4, 
                                     analyze_by='column', 
                                     title='Column-wise Cross-Dataset Pixel Correlation')
from utils import save_correlation_matrices, load_correlation_matrices

save_correlation_matrices(column_correlation_matrices, f"column_correlation_matrices_{filename}")

from correlation_analysis import plot_correlation_for_manual_areas
plot_correlation_for_manual_areas(
    data_raw_L4,
    #columns_to_analyze = [0,1,2,3],
    manual_areas = [
        (0, 12, 186, 198),    
        (0, 18, 246, 264),    
        (10, 62, 198, 250),   
        (16, 104, 266, 354),  
        (104, 124, 352, 372), 
        (62, 104, 142, 184),  
        (104, 142, 104, 142),
        (0,372,0,372)
    ],
    num_ticks = 12
)



from matplotlib.colors import LogNorm

def plot_tot_heatmaps(data, view_by='column', tot_window=None, log_z=False):
    # --- 1. Input Validation ---
    if view_by not in ['column', 'row']:
        raise ValueError("view_by must be either 'column' or 'row'.")

    # --- 2. Filter data by ToT window ---
    filtered_data = {k: v.copy() for k, v in data.items()}
    if tot_window is not None:
        try:
            min_tot, max_tot = tot_window
            print(f"Filtering data for ToT values between {min_tot} and {max_tot}...")
            mask = (filtered_data['ToT'] >= min_tot) & (filtered_data['ToT'] <= max_tot)
            for key in filtered_data:
                filtered_data[key] = filtered_data[key][mask]
        except (ValueError, IndexError, TypeError):
            print("Warning: Invalid tot_window format. Should be (min, max). Ignoring.")
            tot_window = None

    # --- 3. Set up axes and bins based on view ---
    if view_by == 'column':
        primary_key, secondary_key = 'Column', 'Row'
        secondary_bins = np.arange(373)  # Rows 0-371
        secondary_label = 'Pixel Row'
    else:  # view_by == 'row'
        primary_key, secondary_key = 'Row', 'Column'
        max_col = np.max(filtered_data['Column']) if filtered_data['Column'].size > 0 else 0
        secondary_bins = np.arange(max_col + 2)
        secondary_label = 'Pixel Column'
        
    tot_bins = np.arange(tot_window[0], tot_window[1] + 2) if tot_window else np.arange(256)
    
    # --- 4. Generate Heatmaps ---
    unique_elements = np.sort(np.unique(filtered_data[primary_key]))
    if len(unique_elements) == 0:
        print(f"No {primary_key}s found in the filtered data.")
        return

    all_heatmaps = []
    print(f"Found {len(unique_elements)} unique {primary_key}s. Generating heatmaps...")
    
    iterable = list(enumerate(unique_elements))
    for i, element in progress_bar(iterable, description=f"Processing {primary_key}s", total=len(unique_elements)):
        mask = (filtered_data[primary_key] == element)
        
        secondary_axis_data = filtered_data[secondary_key][mask]
        tots_in_element = filtered_data['ToT'][mask]

        heatmap, _, _ = np.histogram2d(
            secondary_axis_data, tots_in_element, bins=[secondary_bins, tot_bins]
        )
        all_heatmaps.append(heatmap)

        # Plot for the first three elements
        if i < 3:
            plt.figure(figsize=(10, 6))
            norm = LogNorm() if log_z else None
            plt.pcolormesh(secondary_bins, tot_bins, heatmap.T, norm=norm, shading='auto')
            plt.title(f'ToT Spectrum for {primary_key.capitalize()} {element}')
            plt.xlabel(secondary_label)
            plt.ylabel('ToT Value')
            cbar = plt.colorbar()
            cbar.set_label('Counts' + (' (log scale)' if log_z else ''))
            plt.tight_layout()
            plt.show()

    # --- 5. Average and plot final heatmap ---
    if not all_heatmaps:
        print("No heatmaps generated to average.")
        return
        
    print(f"Averaging heatmaps across all {primary_key}s...")
    average_heatmap = np.mean(np.stack(all_heatmaps, axis=0), axis=0)

    plt.figure(figsize=(10, 6))
    norm = LogNorm() if log_z else None
    plt.pcolormesh(secondary_bins, tot_bins, average_heatmap.T, norm=norm, shading='auto')
    plt.title(f'Average ToT Spectrum Across All {primary_key.capitalize()}s')
    plt.xlabel(secondary_label)
    plt.ylabel('ToT Value')
    cbar = plt.colorbar()
    cbar.set_label('Average Counts' + (' (log scale)' if log_z else ''))
    plt.tight_layout()
    plt.show()

plot_tot_heatmaps(data_raw, view_by='column',  log_z=True)
    