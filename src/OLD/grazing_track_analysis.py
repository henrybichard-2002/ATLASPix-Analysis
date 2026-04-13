import numpy as np
import matplotlib.pyplot as plt

def _setup_axes(title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    return fig, ax

def _apply_tight_layout(fig):
    fig.tight_layout()

def _group_clusters_by_hits(clustered_data, n_max_hits):
    cluster_ids, counts = np.unique(clustered_data['ClusterID'], return_counts=True)
    groups = {}
    for n_hits in range(1, n_max_hits + 1):
        clusters_with_n_hits = cluster_ids[counts == n_hits]
        if len(clusters_with_n_hits) > 0:
            groups[n_hits] = clusters_with_n_hits
    return groups

def _plot_tot_spectrum(ax, data, cluster_ids, label):
    mask = np.isin(data['ClusterID'].astype(np.int64), cluster_ids.astype(np.int64))
    tot_values = data['ToT'][mask]
    if len(tot_values) > 0:
        ax.hist(tot_values, bins=np.arange(0, np.max(data['ToT']) + 2, 1), 
                 histtype='step', label=label)

def _plot_total_tot_spectrum(ax, data, cluster_ids, label):
    total_tot_values = []
    for cluster_id in cluster_ids:
        cluster_mask = data['ClusterID'].astype(np.int64) == np.int64(cluster_id)
        total_tot = np.sum(data['ToT'][cluster_mask])
        total_tot_values.append(total_tot)
    
    if total_tot_values:
        max_total_tot = np.max(total_tot_values) if total_tot_values else 100
        ax.hist(total_tot_values, bins=np.arange(0, max_total_tot + 2, 1), 
                 histtype='step', label=label)


def _plot_spatial_deviation(ax, data, cluster_ids, label):
    spatial_deviations = []
    for cluster_id in cluster_ids:
        # Using .astype is good practice but ensure both sides match for comparison
        cluster_mask = data['ClusterID'] == cluster_id
        
        # Check if the cluster has more than one hit
        if np.sum(cluster_mask) > 1:
            cluster_tots = data['ToT'][cluster_mask]
            cluster_cols = data['Column'][cluster_mask]
            
            signed_cluster_cols = cluster_cols.astype(np.int32)
            seed_hit_index = np.argmax(cluster_tots)
            seed_hit_col = signed_cluster_cols[seed_hit_index]
            
            # The subtraction is now safe and can result in negative numbers
            deviations = signed_cluster_cols - seed_hit_col
            spatial_deviations.extend(deviations)
    
    if spatial_deviations:
        # Calculate histogram bins based on the max deviation
        max_dev = np.max(np.abs(spatial_deviations))
        # Create bins centered on integers: from -max_dev-0.5 to max_dev+1.5
        bins = np.arange(-max_dev - 0.5, max_dev + 1.5, 1)
        
        ax.hist(spatial_deviations, bins=bins, histtype='step', label=label)
        
from utils import _calculate_row_cog  
      
def _plot_cog_deviation(ax, clustered_data, cluster_ids, label):

    deviations_for_group = []
    for cluster_id in cluster_ids:
        # Isolate all data for the current cluster
        cluster_mask = clustered_data['ClusterID'] == cluster_id
        cluster_tots = clustered_data['ToT'][cluster_mask]
        cluster_cols = clustered_data['Column'][cluster_mask]
        cluster_rows = clustered_data['Row'][cluster_mask]

        # Find the seed hit (highest ToT) and its column
        seed_hit_index_in_cluster = np.argmax(cluster_tots)
        seed_col = cluster_cols[seed_hit_index_in_cluster]

        # Find unique rows and calculate CoG for each
        for row in np.unique(cluster_rows):
            row_mask = (cluster_rows == row)
            cols_in_row = cluster_cols[row_mask]
            tots_in_row = cluster_tots[row_mask]
            
            # Calculate CoG for this specific row using the helper
            cog_pos = _calculate_row_cog(cols_in_row, tots_in_row)

            if cog_pos is not None:
                deviation = cog_pos - seed_col
                deviations_for_group.append(deviation)
    
    if deviations_for_group:
        # Plot histogram of deviations for this group onto the provided axes
        ax.hist(deviations_for_group, bins=np.arange(-2.5, 3, 0.5),
                histtype='step', label=label, density=True)


def grazing_track_analysis(clustered_data, n_max_hits=5):
    # Check for all required data fields
    required_fields = ['ClusterID', 'ToT', 'Column', 'Row']
    if not all(key in clustered_data for key in required_fields):
        print(f"Error: clustered_data must contain {required_fields} keys.")
        return

    cluster_groups = _group_clusters_by_hits(clustered_data, n_max_hits)

    # Setup plots
    fig1, ax1 = _setup_axes(f'ToT Spectrum for Clusters with up to {n_max_hits} Hits', 'Time over Threshold (ToT)', 'Counts')
    fig2, ax2 = _setup_axes(f'Total ToT Spectrum for Clusters with up to {n_max_hits} Hits', 'Total Time over Threshold (ToT)', 'Counts')
    fig3, ax3 = _setup_axes(f'Spatial Deviation for Clusters with up to {n_max_hits} Hits', 'Column Deviation from Seed Hit', 'Counts')
    fig4, ax4 = _setup_axes(f'CoG Deviation for Clusters with up to {n_max_hits} Hits', 'CoG Position - Seed Column', 'Counts')
    
    for n_hits, cluster_ids in cluster_groups.items():
        label = f'{n_hits} hits/cluster'
        _plot_tot_spectrum(ax1, clustered_data, cluster_ids, label)
        _plot_total_tot_spectrum(ax2, clustered_data, cluster_ids, label)
        _plot_spatial_deviation(ax3, clustered_data, cluster_ids, label)
        # --- CORRECTED CALL ---
        # Call the new helper to plot on the fourth axes (ax4)
        _plot_cog_deviation(ax4, clustered_data, cluster_ids, label)
    
    # Finalize and show plots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend()
        #ax.set_yscale('log')
    
    _apply_tight_layout(fig1)
    _apply_tight_layout(fig2)
    _apply_tight_layout(fig3)
    _apply_tight_layout(fig4)

    plt.show()