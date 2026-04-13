mport os
from data_loading import load_data
from clustering import find_clusters, summarize_clusters
from plotting import plot_column_hist, plot_Hitmap, plot_scatter_multiple, plot_time_difference_hist, plot_2d_hist
from filtering import filter_tot

data_file = r"C:\Users\henry\ATLASpix-analysis\data\202204071711_udp_beamonall_angle6_6Gev_kitHV6_kit_10_decode.dat"
# Read the file into a DataFrame
df = load_data(data_file)
print(df)

plot_column_hist(df, "ToT", bins=max(df["ToT"])//2, logy=False, density=False,
                    title="ToT spectrum", xlabel="Time over Threshold (ns)", ylabel="Hits")
plot_column_hist(df, "PackageID", logy=False, density=False,
                    title="Hits/Package", xlabel="Package ID", ylabel="Hits")
plot_column_hist(df, "Column", bins=max(df["Column"]), logy=False, density=False,
                    title="Hits/Column", xlabel="Column", ylabel="Hits")
plot_column_hist(df, "Row", bins=max(df["Row"]), logy=False, density=False,
                    title="Hits/Row", xlabel="Row", ylabel="Hits")
plot_Hitmap(df, "Column", "Row",
                title="Hit Map (all Layers)", xlabel="Column", ylabel="Row")

             
             
from clustering import analyze_cluster_tracks, cluster_events

DT = 36          # Max time difference for temporal clustering
SPATIAL_EPS = 5.0   # Max spatial distance (rows/cols) for spatial clustering
MIN_HITS = 2        # Minimum number of hits to form a spatial cluster

df_clustered = cluster_events(
    df=df,
    dt=DT,
    spatial_eps=SPATIAL_EPS,
    spatial_min_samples=MIN_HITS
)
df_clustered_minHit = df_clustered.loc[df_clustered['ClusterID'] != -1]

print(f"--- Clustered Data (dt={DT}, spatial_eps={SPATIAL_EPS}, min_hits={MIN_HITS}) ---")
print(df_clustered.sort_values(by=['ClusterID', 'TS']))
print("\n" + "="*40 + "\n")
print("--- Hits from Cluster 0 ---")
cluster_0 = df_clustered[df_clustered['ClusterID'] == 0]
print(cluster_0)

print("\n" + "="*40 + "\n")
print("--- Cluster Track Analysis ---")
df_analysis, unclustered_count = analyze_cluster_tracks(df_clustered)

print(f"Found {unclustered_count} unclustered hits (noise).")
print("Analysis of multi-hit clusters:")
if df_analysis.empty:
    print("No clusters to analyze.")
else:
    print(df_analysis)
    
from plotting import plot_analysis_distributions, plot_analysis_correlations
plot_analysis_distributions(df_analysis)
plot_analysis_correlations(df_analysis)

dfs = {Layer: group.sort_values(by='Layer') for Layer, group in df_clustered.groupby('Layer')}
df_L4_clustered = dfs.get(4)

df_L4_analysis, L4_unclustered_count = analyze_cluster_tracks(df_L4_clustered )

df_L4_clustered_minHit = df_L4_clustered.loc[df_L4_clustered['ClusterID'] != -1]