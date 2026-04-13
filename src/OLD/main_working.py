import os
import numpy as np
from data_loading_optimized import load_data_numpy
from clustering_optimized import cluster_events_numpy, analyze_cluster_tracks_numpy, calculate_timing_uniformity
from plotting_optimized import (
    plot_column_hist_numpy, plot_multi_column_hist_numpy,
    plot_Hitmap_numpy, plot_HeatHitmap,
    plot_histograms_with_fits, plot_timing_uniformity
)
from utils import numpy_to_dataframe, layer_split, filter_by_tot

# Hardcoded arguments
file_path = r"C:\Users\henry\ATLASpix-analysis\data\202204071613_udp_beamonall_angle6_6Gev_kitHV30_kit_5_decode.dat"
gamma = 86.5
n_lines = 150000 # Load all lines
tot_thresh = [35,250]
dt = 12
eps = 2.5
min_hits = 2

import os
filename = os.path.splitext(os.path.basename(file_path))[0]

# Load data
data_raw = load_data_numpy(file_path, n_lines=n_lines)
if data_raw is None:
    exit()
    
data_layer_split = layer_split(data_raw)
if len(data_layer_split) < 4:
    print("Error: Not enough layers in the data.")
    exit()
data_raw_L4 = data_layer_split[-1]

#ToT filtering
data_L4_filt = filter_by_tot(data_raw_L4, tot_thresh, description=True)
plot_multi_column_hist_numpy([data_raw_L4, data_L4_filt], labels = ['All hits', 'ToT gated hits'],
                             column_name="ToT",logy=True, density=False,
                    title="L4 ToT (gated)", xlabel="ToT [TS]", ylabel="Hits")


# Initial data exploration plots
hist_plots = [
    {"col": "ToT", "bins": np.max(data_raw_L4["ToT"]) // 2, "title": "ToT spectrum", "xlabel": "Time over Threshold (ns)"},
    {"col": "PackageID", "title": "Hits/Package", "xlabel": "Package ID"},
    {"col": "Column", "bins": np.max(data_raw_L4["Column"]), "title": "Hits/Column", "xlabel": "Column"},
    {"col": "Row", "bins": np.max(data_raw_L4["Row"]), "title": "Hits/Row", "xlabel": "Row"},
]
for plot_spec in hist_plots:
    plot_column_hist_numpy(data_raw_L4, plot_spec["col"], bins=plot_spec.get("bins"), logy=False, density=False,
                        title=plot_spec["title"], xlabel=plot_spec["xlabel"], ylabel="Hits")

plot_HeatHitmap(data_raw_L4, "Column", "Row", title="Hit Heatmap")
plot_Hitmap_numpy(data_raw_L4, "Column", "Row", title="Hit Map (KIT Layer 4)", xlabel="Column", ylabel="Row")

# Clustering
data_clustered_All_L4 = cluster_events_numpy(
    data=data_raw_L4,
    dt=dt,
    spatial_eps=eps,
    spatial_min_samples=min_hits
)
data_clustered_Gated_L4 = cluster_events_numpy(
    data=data_L4_filt,
    dt=dt,
    spatial_eps=eps,
    spatial_min_samples=min_hits
)

import pandas as pd
row_row_crosstalk = pd.read_csv("crosstalk_df_(202204071613_udp_beamonall_angle6_6Gev_kitHV30_kit_5_decode).csv")

from plotting_optimized import plot_cluster_characteristics
from clustering_v3 import clustering_3, clustering_4, sort_clustered_data
data4_clustered_raw_L4 = clustering_4(
    data=data_raw_L4,
    epsilon = 3,
    crosstalk_df = row_row_crosstalk 
    
)

srtd_data4 = sort_clustered_data(data4_clustered_raw_L4)


df_coupling_data4 = numpy_to_dataframe(srtd_data4['coupling'])
df_noise_data4 = numpy_to_dataframe(srtd_data4['noise'])
df_cluster_data4 = numpy_to_dataframe(srtd_data4['clusters'])

plot_cluster_characteristics(srtd_data4, log_y = True)


plot_HeatHitmap(srtd_data4['clusters'], "Column", "Row", title="Cluster HeatMap (Layer 4) No ToT Gate")



tracks_data4_analysis, _  = analyze_cluster_tracks_numpy(srtd_data4['clusters'], angle = gamma)
tracks_data3_analysis_df = numpy_to_dataframe(tracks_data3_analysis)

from single_track import plot_cluster
plot_cluster(srtd_data3['track'], cluster_ids = [52222,8962,5451,2636])


from utils import load_correlation_matrices
column_correlation_matrices_ldd = load_correlation_matrices(f"column_correlation_matrices_{filename}.npz")


# Analyze and plot clustered data
from clustering_optimized import _filter_clustered_data
clustered_data_ungated, unclustered_data_ungated = _filter_clustered_data(data_clustered_All_L4)
clustered_data, unclustered_data = _filter_clustered_data(data_clustered_Gated_L4)

unclustered_count = len(unclustered_data['PackageID'])
if clustered_data is not None:
    plot_HeatHitmap(clustered_data_ungated, "Column", "Row", title="Cluster HeatMap (Layer 4) No ToT Gate")
    plot_HeatHitmap(clustered_data, "Column", "Row", title="Cluster HeatMap (Layer 4) [35,250] ToT Gate")
    
    #plot_HeatHitmap(unclustered_data_ungated, "Column", "Row", title="Unclustered HeatMap (Layer 4)")
    
    from plotting_optimized import plot_heatmap_ratio
    plot_heatmap_ratio(clustered_data_ungated, clustered_data, label1 = 'Tracks (ungated)', 
                       label2 = 'Tracks (ToT gated)', log_c=False)
    
    
    
    print(f"--- Clustered Data (dt={dt}, spatial_eps={eps}, min_hits={min_hits}) ---")
    print(f"Found {len(clustered_data['ClusterID'])} clustered hits.")
       
    clustered_data_df = numpy_to_dataframe(clustered_data)
    cluster_analysis_ungated, _ = analyze_cluster_tracks_numpy(clustered_data_ungated, angle=gamma)
    cluster_analysis_gated, _ = analyze_cluster_tracks_numpy(clustered_data, angle=gamma)
    
    cluster_analysis_ungated_df = numpy_to_dataframe(cluster_analysis_ungated)
    cluster_analysis_gated_df = numpy_to_dataframe(cluster_analysis_gated)
    
    fit_plots = [
        [["n_hits", "timescale"], "Distr. of Number of Hits and Timescale per Track"],
        [["n_hits", "n_missing_hits"], "Distr. of Number of Hits and Missing Hits per Track"],
        [["ext_ts_diff_mean", "ext_ts_diff_std"], "Distr. of Mean dT and Std dT per Track"]
    ]
    
    for cols, title in fit_plots:
        plot_histograms_with_fits(
            datasets = [cluster_analysis_ungated, cluster_analysis_gated],
            labels =[ "Ungated","Gated"],
            columns=cols,
            logy=True,
            title=title,
            bins = 20
        )

# Timing uniformity analysis
timing_data = calculate_timing_uniformity(data_clustered_All_L4, sort_by='space')
if timing_data:
    timing_plots = [
        {'y_key': 'dTS', 'title': 'Timing Uniformity'},
        {'y_key': 'd_ext_TS', 'title': 'Timing Uniformity'},
        {'y_key': 'ToT', 'title': 'ToT function'},
    ]
    for plot_spec in timing_plots:
        plot_timing_uniformity(timing_data, x_key='displacement', y_key=plot_spec['y_key'],
                               title=plot_spec['title'], log_z=True, bins=None)
from grazing_track_analysis import grazing_track_analysis
grazing_track_analysis(data_clustered_All_L4, n_max_hits = 6)
print("Script End")