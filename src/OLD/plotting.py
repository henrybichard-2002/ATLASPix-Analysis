import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

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
    plt.show()

def _get_bins(data, bins, data_range=None):
    if data_range is None:
        data_min, data_max = np.min(data), np.max(data)
    else:
        data_min, data_max = data_range

    data_range_int = int(data_max - data_min) + 1

    if bins is None:
        return data_range_int
    elif isinstance(bins, int) and bins > data_range_int:
        return data_range_int
    else:
        return bins

def plot_column_hist(
    df, column, bins=None, range=None, density=True, logy=False,
    title=None, xlabel=None, ylabel=None
):
    data = df[column].dropna().values

    bins = _get_bins(data, bins, data_range=range)

    fig, ax = _setup_axes(
        title=title or f"Frequency density of {column}",
        xlabel=xlabel or column,
        ylabel=ylabel or ("Density" if density else "Counts")
    )

    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=density)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax.step(bin_centers, counts, where="mid", lw=1.5)
    if logy:
        ax.set_yscale("log")

    _apply_tight_layout(fig)

def plot_time_difference_hist(
    time_diffs, bins=None, range=None, density=True, logy=False,
    title=None, xlabel=None, ylabel=None
):
    """Plots a histogram of time differences."""
    data = time_diffs.dropna().values

    bins = _get_bins(data, bins, data_range=range)

    fig, ax = _setup_axes(
        title=title or "Distribution of Time Differences in Clusters",
        xlabel=xlabel or "Time Difference (ns)",
        ylabel=ylabel or ("Density" if density else "Counts")
    )

    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=density)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    ax.step(bin_centers, counts, where="mid", lw=1.5)
    if logy:
        ax.set_yscale("log")

    _apply_tight_layout(fig)

def plot_2d_hist(
    df, xcol, ycol, bins=None, log_norm=True, cmap="viridis",
    title=None, xlabel=None, ylabel=None
):
    x = df[xcol].dropna().values
    y = df[ycol].dropna().values

    if bins is None or isinstance(bins, int):
        bins_x = _get_bins(x, bins)
        bins_y = _get_bins(y, bins)
        bins = [bins_x, bins_y]
    elif isinstance(bins, (list, tuple)) and len(bins) == 2:
        bins_x = _get_bins(x, bins[0])
        bins_y = _get_bins(y, bins[1])
        bins = [bins_x, bins_y]

    fig, ax = _setup_axes(
        title=title or f"2D Frequency density: {xcol} vs {ycol}",
        xlabel=xlabel or xcol,
        ylabel=ylabel or ycol
    )

    h = ax.hist2d(
        x, y,
        bins=bins,
        cmap=cmap,
        norm=LogNorm() if log_norm else None
    )

    plt.colorbar(h[3], ax=ax, label="Counts")
    _apply_tight_layout(fig)
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_Hitmap(
    df, xcol, ycol,
    title=None, xlabel=None, ylabel=None,
    inset_xlim=(90, 100), inset_ylim=(175, 200),
    inset_bounds=[0.7, 0.7, 0.25, 0.25]
):
    """
    Plots a hitmap with an inset and perfectly aligned marginal histograms.
    """
    x = df[xcol].dropna().values
    y = df[ycol].dropna().values

    # --- Create the main figure and axes ---
    fig, ax_main = plt.subplots(figsize=(8, 8))

    # --- Create a divider for the main axes to attach new axes ---
    divider = make_axes_locatable(ax_main)

    # Create the new axes for the histograms, perfectly aligned
    ax_histx = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_main)
    ax_histy = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_main)

    # Hide unnecessary labels for a cleaner look
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # --- Plot the main hitmap ---
    ax_main.plot(x, y, marker='s', linestyle='none', label='Main Data', markersize=1)
    ax_main.set_xlabel(xlabel or xcol)
    ax_main.set_ylabel(ylabel or ycol)
    ax_main.set_title(title or f"{xcol} vs {ycol}")
    ax_main.grid(True, linestyle='--', alpha=0.5)

    # --- Calculate and plot the marginal histograms ---
    x_counts = pd.Series(x).value_counts()
    y_counts = pd.Series(y).value_counts()

    # Top histogram (x-projection)
    ax_histx.bar(x_counts.index, x_counts.values, width=1.0, color='C0')
    ax_histx.set_ylabel("Counts")

    # Right histogram (y-projection)
    ax_histy.barh(y_counts.index, y_counts.values, height=1.0, color='C0')
    ax_histy.set_xlabel("Counts")

    # --- Add the inset plot ---
    ax_inset = ax_main.inset_axes(inset_bounds)
    ax_inset.plot(x, y, marker='s', linestyle='none', color='red', markersize=5)
    ax_inset.set_xlim(inset_xlim)
    ax_inset.set_ylim(inset_ylim)

    rect_x_min, rect_x_max = inset_xlim
    rect_y_min, rect_y_max = inset_ylim
    ax_main.add_patch(plt.Rectangle(
        (rect_x_min, rect_y_min), rect_x_max - rect_x_min, rect_y_max - rect_y_min,
        fill=False, edgecolor='blue', lw=2, linestyle='--', label='Zoomed Region'
    ))

    ax_inset.set_title("Zoomed In", fontsize=10)
    ax_inset.tick_params(labelleft=False, labelbottom=False)
    ax_inset.patch.set_edgecolor('gray')
    ax_inset.patch.set_linewidth(1.5)
    ax_main.indicate_inset_zoom(ax_inset, edgecolor="gray", linewidth=1.5)

    ax_main.legend()
    fig.tight_layout()
    plt.show()

def plot_scatter_multiple(
    dfs, xcol, ycol, s=5, alpha=0.6, title=None, xlabel=None, ylabel=None, cmap="Set1"
):
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(dfs)))

    for (label, df), color in zip(dfs.items(), colors):
        x = df[xcol].dropna().values
        y = df[ycol].dropna().values
        ax.scatter(x, y, s=s, alpha=alpha, label=f"Layer {label}", color=color)

    ax.set_title(title or f"{xcol} vs {ycol} by group", fontsize=14)
    ax.set_xlabel(xlabel or xcol, fontsize=12)
    ax.set_ylabel(ylabel or ycol, fontsize=12)
    ax.legend(title="Groups", markerscale=2, fontsize=10)

    fig.tight_layout()
    plt.show()
    


def plot_layer_hitmaps(
    df: pd.DataFrame,
    fig_title: str = "Hitmaps by Layer with Cluster Tracks and ToT"
):

    # --- Input Validation ---
    required_cols = ['Layer', 'Column', 'Row', 'ToT', 'ClusterID', 'TS']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain: {required_cols}")

    # --- Figure Setup ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(fig_title, fontsize=16)
    axes = axes.flatten()  # Flatten the 2x2 array for easy iteration

    # Get the first 4 unique layers to plot
    unique_layers = sorted(df['Layer'].unique())
    layers_to_plot = unique_layers[:4]
    
    if len(unique_layers) > 4:
        print(f"Warning: Found {len(unique_layers)} layers. Plotting the first 4: {layers_to_plot}")

    # Normalize color scale across all data for consistent coloring
    if not df.empty:
        vmin = df['ToT'].min()
        vmax = df['ToT'].max()
    else:
        vmin, vmax = 0, 1

    # --- Iterate and Plot for each Layer ---
    for i, layer_id in enumerate(layers_to_plot):
        ax = axes[i]
        layer_df = df[df['Layer'] == layer_id]

        if layer_df.empty:
            ax.set_title(f'Layer {layer_id} (No Data)')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot the hits (scatter plot) with ToT as color
        scatter = ax.scatter(
            layer_df['Column'],
            layer_df['Row'],
            c=layer_df['ToT'],
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            zorder=2  # Ensure points are drawn on top of lines
        )

        # Plot the lines connecting hits within the same cluster
        # Group by ClusterID, but only consider actual clusters (ID >= 0)
        clusters = layer_df[layer_df['ClusterID'] != -1].groupby('ClusterID')
        
        for cluster_id, cluster_data in clusters:
            if len(cluster_data) > 1:
                # Sort by timestamp to draw lines in chronological order
                cluster_data = cluster_data.sort_values('TS')
                ax.plot(
                    cluster_data['Column'],
                    cluster_data['Row'],
                    color='gray',
                    alpha=0.7,
                    marker = 's',
                    markersize = 5,
                    linewidth=1.0,
                    zorder=1
                )

        ax.set_title(f'Layer {layer_id}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.grid(True, linestyle='--', alpha=0.5)
        # Inverting y-axis can be conventional for some detectors
        # ax.invert_yaxis()

    # --- Clean up unused subplots ---
    for i in range(len(layers_to_plot), len(axes)):
        axes[i].axis('off')

    # --- Add a single color bar for the whole figure ---
    fig.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout to make space for colorbar
    if not df.empty and not df.ToT.empty:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('ToT (Time-over-Threshold)')
    
    plt.show()

    
    
def plot_analysis_distributions(df_analysis):
    """
    Plots distributions of key track analysis metrics.

    Args:
        df_analysis (pd.DataFrame): The summary DataFrame from analyze_cluster_tracks.
    """
    if df_analysis.empty:
        print("Analysis DataFrame is empty. No distributions to plot.")
        return

    # Set plot style
    sns.set_style("whitegrid")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distributions of Cluster Track Properties', fontsize=16)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()

    # Columns to plot
    cols_to_plot = [
        'n_hits', 'timescale', 'absolute_length',
        'rms_deviation', 'n_missing_hits'
    ]
    
    # Titles for plots
    plot_titles = [
        'Number of Hits per Cluster', 'Cluster Timescale (TS units)', 'Track Absolute Length (pixels)',
        'RMS Deviation from Fit (pixels)', 'Number of Missing Hits'
    ]

    for i, col in enumerate(cols_to_plot):
        sns.histplot(data=df_analysis, x=col, ax=axes[i], kde=True)
        axes[i].set_title(plot_titles[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for i in range(len(cols_to_plot), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

def plot_analysis_correlations(df_analysis):
    """
    Plots correlation scatter plots for key track analysis metrics.

    Args:
        df_analysis (pd.DataFrame): The summary DataFrame from analyze_cluster_tracks.
    """
    if df_analysis.empty:
        print("Analysis DataFrame is empty. No correlations to plot.")
        return
        
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Correlations Between Cluster Track Properties', fontsize=16)
    axes = axes.flatten()

    # Plot 1: Absolute Length vs. Number of Hits
    sns.scatterplot(data=df_analysis, x='n_hits', y='absolute_length', ax=axes[0], alpha=0.7)
    axes[0].set_title('Absolute Length vs. Number of Hits')
    
    # Plot 2: RMS Deviation vs. Absolute Length
    sns.scatterplot(data=df_analysis, x='absolute_length', y='rms_deviation', ax=axes[1], alpha=0.7)
    axes[1].set_title('RMS Deviation vs. Absolute Length')

    # Plot 3: RMS Deviation vs. Number of Hits
    sns.scatterplot(data=df_analysis, x='n_hits', y='rms_deviation', ax=axes[2], alpha=0.7)
    axes[2].set_title('RMS Deviation vs. Number of Hits')

    # Plot 4: Timescale vs. Absolute Length
    sns.scatterplot(data=df_analysis, x='absolute_length', y='timescale', ax=axes[3], alpha=0.7)
    axes[3].set_title('Timescale vs. Absolute Length')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

