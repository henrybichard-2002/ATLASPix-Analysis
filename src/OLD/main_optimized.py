import os

import numpy as np
from data_loading_optimized import load_data_numpy
from clustering_optimized import cluster_events_numpy, analyze_cluster_tracks_numpy, calculate_timing_uniformity
from plotting_optimized import (
    plot_column_hist_numpy, plot_Hitmap_numpy, plot_HeatHitmap, 
    plot_histograms_with_fits_subplots, plot_timing_uniformity
)
from correlation_analysis import plot_pixel_correlation
from grazing_track_analysis import grazing_track_analysis
from utils import numpy_to_dataframe, layer_split

def get_user_input():
    file_path = input(r"Enter the path to the data file [C:\Users\henry\ATLASpix-analysis\data\202204071613_udp_beamonall_angle6_6Gev_kitHV30_kit_5_decode.dat]: ") or r"C:\Users\henry\ATLASpix-analysis\data\202204071613_udp_beamonall_angle6_6Gev_kitHV30_kit_5_decode.dat"
    lines_str = input("Enter the number of lines to read (or leave blank for all) [all]: ")
    lines = int(lines_str) if lines_str else None
    dt = int(input("Enter the max time difference for temporal clustering [36]: ") or 36)
    eps = int(input("Enter the max spatial distance for spatial clustering [5]: ") or 5)
    min_hits = int(input("Enter the minimum number of hits to form a spatial cluster [2]: ") or 2)
    return file_path, lines, dt, eps, min_hits

def main():
    file_path, n_lines, dt, eps, min_hits = get_user_input()

    # Load data
    data_raw = load_data_numpy(file_path, n_lines=n_lines)
    if data_raw is None:
        return

    # Layer splitting
    data_layer_split = layer_split(data_raw)
    # For this example, we focus on Layer 4, but this could be parameterized
    if len(data_layer_split) < 4:
        print("Error: Not enough layers in the data.")
        return
    data_raw_L4 = data_layer_split[3]

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

    # Plot pixel correlation
    plot_pixel_correlation(data_raw_L4)

    # Clustering
    data_clustered_All_L4 = cluster_events_numpy(
        data=data_raw_L4,
        dt=dt,
        spatial_eps=eps,
        spatial_min_samples=min_hits
    )

    # Grazing track analysis
    grazing_track_analysis(data_clustered_All_L4)

    # Analyze and plot clustered data
    from clustering_optimized import _filter_clustered_data
    clustered_data, unclustered_count = _filter_clustered_data(data_clustered_All_L4)
    if clustered_data is not None:
        plot_Hitmap_numpy(clustered_data, "Column", "Row", title="Cluster Hit Map (Layer 4)")
        print(f"--- Clustered Data (dt={dt}, spatial_eps={eps}, min_hits={min_hits}) ---")
        print(f"Found {len(clustered_data['ClusterID'])} clustered hits.")

        cluster_analysis, _ = analyze_cluster_tracks_numpy(data_clustered_All_L4)
        if cluster_analysis:
            print("\n" + "="*40 + "\n")
            print("--- Cluster Track Analysis ---")
            print(f"Found {unclustered_count} unclustered hits (noise).")
            print("Analysis of multi-hit clusters:")
            print(numpy_to_dataframe(cluster_analysis))

            fit_plots = [
                ["rms_deviation", "reduced_chi_square"],
                ["n_hits", "timescale"],
                ["n_hits", "n_missing_hits"],
                ["ext_ts_diff_mean", "ext_ts_diff_std"],
            ]
            for cols in fit_plots:
                plot_histograms_with_fits_subplots(cluster_analysis, columns=cols, logy=True)

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

if __name__ == '__main__':
    main()