import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple
from matplotlib.colors import SymLogNorm

# --- Helper Functions ---

def _validate_and_setup(
    data_dicts: List[Dict[str, pd.Series]],
    analyze_by: str
) -> Tuple[List[pd.DataFrame], str, str]:
    """Validates input data and sets up analysis keys."""
    required_keys = ['Column', 'Row', 'TriggerID']
    for i, data in enumerate(data_dicts):
        if not all(key in data for key in required_keys):
            raise KeyError(f"Data dictionary {i+1} must contain all keys: {required_keys}")

    if analyze_by not in ['column', 'row']:
        raise ValueError("analyze_by must be either 'column' or 'row'")

    dataframes = [pd.DataFrame(data) for data in data_dicts]
    grouping_key = 'Column' if analyze_by == 'column' else 'Row'
    pixel_key = 'Row' if analyze_by == 'column' else 'Column'

    return dataframes, grouping_key, pixel_key

def _create_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    analyze_by: str,
    use_log_scale: bool = False,
    show_cbar: bool = True,
    use_covariance: bool = False 
):
    """Generates and displays a single correlation heatmap onto a given ax."""
    region_boundaries = [123.5, 247.5] # Assuming this is specific to your setup

    norm = None
    heatmap_kwargs = {} # Dictionary for heatmap arguments
    
    if use_covariance:
        cbar_label = 'Covariance'
        if use_log_scale:
            # Symmetrical log scale, but vmin/vmax are data-driven
            norm = SymLogNorm(linthresh=0.001) 
        # For linear scale, let seaborn determine vmin/vmax
    else:
        # --- Pearson Correlation Mode ---
        cbar_label = 'Pearson Correlation'
        if use_log_scale:
            norm = SymLogNorm(linthresh=0.01, vmin=-1.0, vmax=1.0)
        else:
            # Clamp linear scale from -1 to 1
            heatmap_kwargs['vmin'] = -1.0
            heatmap_kwargs['vmax'] = 1.0

    # Draw the heatmap onto the provided ax
    sns.heatmap(
        matrix,
        ax=ax, 
        cmap='viridis',
        annot=False,
        cbar=show_cbar,
        cbar_kws={'label': cbar_label} if show_cbar else None,
        norm=norm,
        **heatmap_kwargs # Apply vmin/vmax only if needed
    )
    
    if analyze_by == 'column':
        for boundary in region_boundaries:
            ax.axhline(y=boundary, color='white', linestyle='--', linewidth=2)
            ax.axvline(x=boundary, color='white', linestyle='--', linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    
def _calculate_average_correlation(
    matrices_list: List[pd.DataFrame],
    is_cross_dataset: bool = False
) -> pd.DataFrame:
    """Averages a list of correlation matrices."""
    # Add future_stack=True to adopt the new implementation and silence the warning.
    # The new implementation does not support the 'dropna' argument.
    stack_method_kwargs = {'future_stack': True}
    
    avg_corr_series = pd.concat(
        [m.stack(**stack_method_kwargs) for m in matrices_list]
    ).groupby(level=[0, 1]).mean()
    
    avg_correlation_matrix = avg_corr_series.unstack()

    if not is_cross_dataset:
        np.fill_diagonal(avg_correlation_matrix.values, 1)
        
    return avg_correlation_matrix

# --- Main Analysis Functions ---

def plot_pixel_correlation(
    data: Dict[str, pd.Series],
    analyze_by: str = 'column',
    ids_to_analyze: Optional[List[int]] = None,
    use_log_scale: bool = False,
    title: str = 'Pixel Hit Correlation Matrix'
):
    """
    Analyzes and visualizes hit correlation between pixels within the same column or row.
    """
    [df], grouping_key, pixel_key = _validate_and_setup([data], analyze_by)

    if ids_to_analyze is None:
        ids_to_analyze = sorted(df[grouping_key].unique())

    print(f"--- Pixel Correlation Analysis (analyzing by {analyze_by}) ---")
    print(f"Analyzing {len(ids_to_analyze)} {grouping_key.lower()}s...")

    all_correlation_matrices = []
    plot_counter = 0
    max_example_plots = 3

    for id_val in ids_to_analyze:
        print(f"\nProcessing {grouping_key}: {id_val}")
        df_group = df[df[grouping_key] == id_val]

        if df_group.shape[0] < 2 or df_group[pixel_key].nunique() < 2:
            print(f"  Skipping {grouping_key} {id_val}: Not enough hits or unique pixels.")
            continue

        hit_matrix = pd.crosstab(df_group[pixel_key], df_group['TriggerID'])
        correlation_matrix = hit_matrix.T.corr()

        if correlation_matrix.empty or correlation_matrix.isna().all().all():
            print(f"  Skipping {grouping_key} {id_val}: Could not compute a valid correlation matrix.")
            continue

        all_correlation_matrices.append(correlation_matrix)

        if plot_counter < max_example_plots:
            print(f"  Plotting example for {grouping_key} {id_val}...")
            _create_heatmap(
                matrix=correlation_matrix,
                title=title,
                xlabel=f'Pixel ({pixel_key} ID)',
                ylabel=f'Pixel ({pixel_key} ID)',
                analyze_by=analyze_by,
                use_log_scale=False  # Example plots are always linear
            )
            plot_counter += 1

    if not all_correlation_matrices:
        print("\nNo correlation matrices were generated to average.")
        return

    print(f"\n--- Averaging {len(all_correlation_matrices)} {grouping_key.lower()} correlation matrices ---")
    avg_correlation_matrix = _calculate_average_correlation(all_correlation_matrices)

    print("Plotting the average correlation matrix...")
    _create_heatmap(
        matrix=avg_correlation_matrix,
        title=f'Average Pixel Hit Correlation Across All {grouping_key}s',
        xlabel=f'Pixel ({pixel_key} ID)',
        ylabel=f'Pixel ({pixel_key} ID)',
        analyze_by=analyze_by,
        use_log_scale=use_log_scale # Log scale applied only to the average plot
    )

import pandas as pd

import matplotlib.gridspec as gridspec 
from typing import Dict, List, Optional



def plot_cross_dataset_pixel_correlation(
    data1: Dict[str, pd.Series],
    data2: Dict[str, pd.Series],
    analyze_by: str = 'column',
    ids_to_analyze: Optional[List[int]] = None,
    use_log_scale: bool = False,
    title: str = 'Cross-Dataset Pixel Correlation'
) -> Dict[int, pd.DataFrame]:
    """
    Analyzes and plots hit correlation between two datasets, including a
    community grid visualization, and returns the correlation matrices.

    Returns:
        A dictionary where each key is a column/sensor ID and each value
        is the corresponding full pixel-to-pixel correlation matrix.
    """
    [df1, df2], grouping_key, pixel_key = _validate_and_setup([data1, data2], analyze_by)
    
    # --- MODIFICATION: Dictionary to store matrices ---
    column_correlation_matrices = {}

    if ids_to_analyze is None:
        ids1 = set(df1[grouping_key].dropna().unique())
        ids2 = set(df2[grouping_key].dropna().unique())
        ids_to_analyze = sorted(list(ids1.intersection(ids2)))

    print(f"--- Cross-Dataset Pixel Correlation (by {analyze_by}) ---")
    print(f"Analyzing {len(ids_to_analyze)} common {grouping_key.lower()}s...")

    all_cross_corr_matrices = []
    all_full_corr_matrices = []
    
    plot_counter = 0
    max_example_plots = 3
    full_pixel_grid = range(372)

    for id_val in ids_to_analyze:
        print(f"\nProcessing {grouping_key}: {id_val}")
        df1_group = df1[df1[grouping_key] == id_val]
        df2_group = df2[df2[grouping_key] == id_val]

        if df1_group.empty or df2_group.empty:
            print(f"  Skipping {grouping_key} {id_val}: Data not present in both datasets.")
            continue

        pixels1 = sorted(df1_group[pixel_key].unique())
        pixels2 = sorted(df2_group[pixel_key].unique())
        
        df_group_combined = pd.concat([df1_group, df2_group])
        
        if df_group_combined['TriggerID'].nunique() < 2:
            print(f"  Skipping {grouping_key} {id_val}: Not enough common trigger events.")
            continue

        hit_matrix_combined = pd.crosstab(df_group_combined[pixel_key], df_group_combined['TriggerID'])
        full_corr_matrix = hit_matrix_combined.T.corr()
        
        # --- MODIFICATION: Save the matrix to the dictionary ---
        column_correlation_matrices[id_val] = full_corr_matrix
        
        cross_corr_matrix = full_corr_matrix.loc[pixels2, pixels1]

        if cross_corr_matrix.empty or cross_corr_matrix.isna().all().all():
            print(f"  Skipping {grouping_key} {id_val}: Could not compute valid cross-correlation.")
            continue
        
        square_cross_matrix = cross_corr_matrix.reindex(index=full_pixel_grid, columns=full_pixel_grid)
        all_cross_corr_matrices.append(square_cross_matrix)
        
        square_full_matrix = full_corr_matrix.reindex(index=full_pixel_grid, columns=full_pixel_grid)
        all_full_corr_matrices.append(square_full_matrix)

        if plot_counter < max_example_plots:
            print(f"  Plotting example heatmap for {grouping_key} {id_val}...")
            _create_heatmap(
                matrix=square_cross_matrix,
                title=f"{title}\n(Example for {grouping_key} {id_val})",
                xlabel=f'Pixel ({pixel_key} ID) in Dataset 1',
                ylabel=f'Pixel ({pixel_key} ID) in Dataset 2',
                analyze_by=analyze_by,
                use_log_scale=False
            )
            plot_counter += 1

    if not all_cross_corr_matrices:
        print("\nNo cross-correlation matrices were generated to average.")
        return column_correlation_matrices

    # --- Plot Average Cross-Correlation Heatmap ---
    print(f"\n--- Averaging {len(all_cross_corr_matrices)} {grouping_key.lower()} cross-correlation matrices ---")
    avg_correlation_matrix = _calculate_average_correlation(all_cross_corr_matrices, is_cross_dataset=True)
    print("Plotting the average cross-correlation matrix...")
    _create_heatmap(
        matrix=avg_correlation_matrix,
        title=f"Average {title}",
        xlabel=f'Pixel ({pixel_key} ID) in Dataset 1',
        ylabel=f'Pixel ({pixel_key} ID) in Dataset 2',
        analyze_by=analyze_by,
        use_log_scale=use_log_scale
    )
    
    # --- Regional Sum Plots (original functionality) ---
    print("Skipping regional sum plots in this example.")


    # --- MODIFICATION: Generate and Plot the Community Grid Map ---
    if not all_full_corr_matrices:
        print("\nNo full correlation matrices were generated for the community map.")
        return column_correlation_matrices
        
    print(f"\n--- Averaging {len(all_full_corr_matrices)} full correlation matrices for Community Map ---")
    avg_full_correlation_matrix = _calculate_average_correlation(all_full_corr_matrices)
    
    np.fill_diagonal(avg_full_correlation_matrix.values, np.nan) # Ignore self-correlation
    
    return column_correlation_matrices

#######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple
from scipy.stats import gaussian_kde
from utils import progress_bar # Assuming utils is in the same package
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # For inset plot
from matplotlib.patches import Patch # For custom legends
import matplotlib.cm as cm # For colormaps
import matplotlib.colors as mcolors # For color normalization

# --- Helper Functions (Assumed to exist) ---

def _validate_and_setup_for_tot(
    data_dicts: List[Dict[str, pd.Series]],
    analyze_by: str
) -> Tuple[List[pd.DataFrame], str, str]:
    """
    Validates input data, ensuring required keys are present and have the
    correct integer data types, then sets up analysis keys.
    """
    required_keys = ['Column', 'Row', 'TriggerID', 'ToT']
    for i, data in enumerate(data_dicts):
        if not all(key in data for key in required_keys):
            raise KeyError(
                f"Data dictionary {i+1} must contain all keys: {required_keys}"
            )

    if analyze_by not in ['column', 'row']:
        raise ValueError("analyze_by must be either 'column' or 'row'")

    dataframes = []
    for data in data_dicts:
        df = pd.DataFrame(data)
        try:
            df['Column'] = pd.to_numeric(df['Column'], errors='raise').astype(int)
            df['Row'] = pd.to_numeric(df['Row'], errors='raise').astype(int)
            df['ToT'] = pd.to_numeric(df['ToT'], errors='raise').astype(np.int64)
            df['TriggerID'] = pd.to_numeric(df['TriggerID'], errors='raise').astype(np.int64)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Could not convert required columns to numeric types. Check input data. Error: {e}")
        dataframes.append(df)

    grouping_key = 'Column' if analyze_by == 'column' else 'Row'
    pixel_key = 'Row' if analyze_by == 'column' else 'Column'

    return dataframes, grouping_key, pixel_key

def _calculate_average_correlation_np(
    matrices_list: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Averages a list of correlation matrices using NumPy for performance.
    Assumes all matrices have been re-indexed to the same shape.
    """
    if not matrices_list:
        return pd.DataFrame()

    np_arrays = [m.to_numpy(dtype=np.float32) for m in matrices_list]
    stacked_arrays = np.stack(np_arrays, axis=0)
    
    with np.errstate(all='ignore'): # Suppress "mean of empty slice" warning
        avg_matrix_np = np.nanmean(stacked_arrays, axis=0)
    
    index_cols = matrices_list[0].index
    avg_correlation_matrix = pd.DataFrame(avg_matrix_np, index=index_cols, columns=index_cols)
    
    np.fill_diagonal(avg_correlation_matrix.values, 1.0)
    return avg_correlation_matrix.fillna(0)


# --- Main Analysis Function (Serial Version with Plot Updates) ---

def find_correlated_pairs_and_analyze_tot_ratio(
    data: Dict[str, pd.Series],
    analyze_by: str = 'column',
    dx: int = 1,
    correlation_threshold: float = 0.01,
    ids_to_analyze: Optional[List[int]] = None,
    num_examples_to_plot: int = 3,
    asymmetric_tot_cut: Optional[int] = None 
) -> pd.DataFrame:
    """
    Finds correlated pixel pairs and analyzes their ToT distributions.

    For the top correlated pairs, plots:
    1. A figure with:
       - (Left) A 2D heatmap of ToT(p1) vs ToT(p2).
       - (Right) A histogram of max(ToT)/min(ToT) with a log-scale inset of p1/p2 from [0, 1].
    2. If 'asymmetric_tot_cut' is set, a 2x2 grid with:
       - (Top) Overlaid histograms (raw counts) for each integer cut.
       - (Bottom) Overlaid normalized KDE fits for each integer cut.
       - (Left Col) Red Data: ToT(p1) vs. ToT(p2) <= cut
       - (Right Col) Blue Data: ToT(p2) vs. ToT(p1) <= cut

    Args:
        data: Dictionary of pd.Series with 'Column', 'Row', 'TriggerID', 'ToT'.
        analyze_by: 'column' or 'row'.
        dx: Distance from the main diagonal to exclude.
        correlation_threshold: Minimum Pearson correlation to consider.
        ids_to_analyze: Optional list of group IDs to process.
        num_examples_to_plot: How many top pairs to plot.
        asymmetric_tot_cut: If set, plots 2D ToT maps with an asymmetric cut.

    Returns:
        A pandas DataFrame with correlated pairs, their correlation, peak ToT
        ratios (for both p1/p2 and large/small), and co-occurrence count.
    """
    
    # --- 1. Validation and Setup ---
    try:
        [df], grouping_key, pixel_key = _validate_and_setup_for_tot(
            [data], analyze_by
        )
    except (KeyError, ValueError, TypeError) as e:
        print(f"Input data error: {e}")
        return pd.DataFrame()
    except NameError:
        print("Error: Helper function '_validate_and_setup_for_tot' not found.")
        return pd.DataFrame()

    if ids_to_analyze:
        df = df[df[grouping_key].isin(ids_to_analyze)]
    
    print("--- Correlated Pair & ToT Ratio Analysis ---")
    print(f"Analyzing by {analyze_by} (Grouping: {grouping_key}, Pixel: {pixel_key})")
    print(f"Finding pairs with correlation > {correlation_threshold}, excluding diagonal +/- {dx} pixels.")

    # --- 2. Calculate Average Correlation Matrix ---
    all_pixels = sorted(df[pixel_key].unique())
    
    def _get_corr_matrix(group: pd.DataFrame) -> Optional[pd.DataFrame]:
        if group.shape[0] < 2 or group[pixel_key].nunique() < 2:
            return None
        hit_matrix = pd.crosstab(group[pixel_key], group['TriggerID'])
        if hit_matrix.shape[1] < 2: return None
        correlation_matrix = hit_matrix.T.corr()
        if correlation_matrix.empty or correlation_matrix.isna().all().all():
            return None
        return correlation_matrix.reindex(index=all_pixels, columns=all_pixels)

    grouped = df.groupby(grouping_key)
    desc = f"Calculating correlation for {len(grouped)} {grouping_key}s"
    
    all_correlation_matrices = [
        matrix for matrix in (
            _get_corr_matrix(group) for _, group in progress_bar(grouped, description=desc)
        ) if matrix is not None
    ]

    if not all_correlation_matrices:
        print("\nNo valid correlation matrices were generated. Aborting.")
        return pd.DataFrame()

    print(f"Averaging {len(all_correlation_matrices)} correlation matrices using NumPy...")
    try:
        avg_correlation_matrix = _calculate_average_correlation_np(all_correlation_matrices)
    except NameError:
         print("Error: Helper function '_calculate_average_correlation_np' not found.")
         return pd.DataFrame()


    # --- 3. Find High-Correlation Pairs ---
    print("Finding correlated pairs above threshold...")
    corr_stack = avg_correlation_matrix.stack()
    corr_stack.index.names = ['pixel_1', 'pixel_2']
    corr_df = corr_stack.reset_index(name='correlation')
    
    corr_df = corr_df[corr_df['pixel_1'] < corr_df['pixel_2']].copy()
    corr_df = corr_df[np.abs(corr_df['pixel_1'] - corr_df['pixel_2']) > dx]
    top_pairs_df = corr_df[corr_df['correlation'] > correlation_threshold].sort_values(
        by='correlation', ascending=False
    )

    if top_pairs_df.empty:
        print(f"No correlated pairs found above threshold={correlation_threshold} and outside the dx={dx} diagonal band.")
        return pd.DataFrame()
        
    print(f"\nFound {len(top_pairs_df)} correlated pairs above the threshold.")

    # --- 4. Analyze ToT Ratios ---
    print("\n--- Analyzing ToT Ratios for Found Pairs ---")
    
    def _get_kde_peak(data_series: pd.Series) -> float:
        """Helper to find the peak (mode) of a distribution using KDE."""
        if data_series.empty or data_series.nunique() < 2:
            return np.nan
        try:
            kde = gaussian_kde(data_series)
            x_grid = np.linspace(data_series.min(), data_series.max(), 500)
            kde_values = kde.evaluate(x_grid)
            peak_index = np.argmax(kde_values)
            return float(x_grid[peak_index])
        except (np.linalg.LinAlgError, ValueError):
            return float(data_series.mean()) # Fallback to mean

    results_list = []
    plot_counter = 0

    desc_pairs = f"Analyzing {len(top_pairs_df)} pairs"
    for _, row in progress_bar(top_pairs_df.iterrows(), description=desc_pairs, total=len(top_pairs_df)):
        p1, p2, correlation = row['pixel_1'], row['pixel_2'], row['correlation']
        
        hits_p1 = df[df[pixel_key] == p1]
        hits_p2 = df[df[pixel_key] == p2]
        
        merged_hits = pd.merge(
            hits_p1, hits_p2, 
            on=['TriggerID', grouping_key], 
            suffixes=('_p1', '_p2')
        )

        peak_ratio_p1_p2 = np.nan
        peak_ratio_large_small = np.nan
        ratio_p1_p2 = pd.Series(dtype=float)
        ratio_large_small = pd.Series(dtype=float)
        tot_p1 = pd.Series(dtype=float)
        tot_p2 = pd.Series(dtype=float)

        if not merged_hits.empty and len(merged_hits) > 1:
            tot_p1 = (merged_hits['ToT_p1'] & 0xFF).astype(float)
            tot_p2 = (merged_hits['ToT_p2'] & 0xFF).astype(float)
            
            # --- Calculate Ratios, handling division by zero ---
            tot_p2_safe = tot_p2.replace(0, np.nan)
            ratio_p1_p2 = (tot_p1 / tot_p2_safe).dropna()
            
            tot_large = np.maximum(tot_p1, tot_p2)
            tot_small = np.minimum(tot_p1, tot_p2)
            tot_small_safe = tot_small.replace(0, np.nan)
            ratio_large_small = (tot_large / tot_small_safe).dropna()

            # --- Calculate peaks ---
            if not ratio_p1_p2.empty:
                peak_ratio_p1_p2 = _get_kde_peak(ratio_p1_p2)
            if not ratio_large_small.empty:
                peak_ratio_large_small = _get_kde_peak(ratio_large_small)
            
            # --- 5. Plot Distributions for TOP EXAMPLES ONLY ---
            if plot_counter < num_examples_to_plot and not ratio_large_small.empty:
                print(f"\nPlotting example for Pair ({p1}, {p2}) (Correlation: {correlation:.4f})")
                
                # --- PLOT 1: 2D Heatmap vs. Max/Min Ratio ---
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                
                # --- Graph 1 (Left): 2D ToT Heatmap ---
                hist_bins = (64, 64)
                hist_range = ((0, 255), (0, 255))
                sns.histplot(x=tot_p1, y=tot_p2, bins=hist_bins, binrange=hist_range, 
                             cmap='viridis', cbar=True, cbar_kws={'label': 'Counts'}, ax=ax1)
                ax1.set_title(f'ToT({p1}) vs. ToT({p2})')
                ax1.set_xlabel(f'ToT({p1})')
                ax1.set_ylabel(f'ToT({p2})')
                ax1.set_aspect('equal') 

                # --- Graph 2 (Right): max(ToT) / min(ToT) ---
                sns.histplot(ratio_large_small, kde=True, bins=100, ax=ax2)
                ax2.set_title(f'Ratio max(ToT) / min(ToT)')
                ax2.set_xlabel(f'max(ToT) / min(ToT)')
                ax2.set_ylabel('Frequency')
                if not np.isnan(peak_ratio_large_small):
                    ax2.axvline(peak_ratio_large_small, color='red', ls='--', label=f'Peak: {peak_ratio_large_small:.2f}')
                ax2.legend()
                ax2.grid(axis='y', linestyle=':', alpha=0.7)

                # --- MODIFIED Inset Plot for [0, 1] range ---
                ax_inset = inset_axes(ax2, width="45%", height="45%", loc='upper right')
                inset_data = ratio_p1_p2[
                    (ratio_p1_p2 >= 0) & (ratio_p1_p2 <= 1)
                ]
                # Use 64 bins and log scale
                ax_inset.hist(inset_data, bins=64, range=(0, 1), color='darkgray', edgecolor='black')
                ax_inset.set_yscale('log') # Set y-axis to log scale
                ax_inset.set_title(f'Zoom ToT({p1})/ToT({p2}) [0.0, 1.0]')
                ax_inset.set_xlabel('Ratio', fontsize=10)
                ax_inset.set_ylabel('Freq. (log)', fontsize=10)
                ax_inset.set_xlim(0, 1)
                ax_inset.tick_params(axis='x', labelsize=8)
                ax_inset.tick_params(axis='y', labelsize=8)

                fig.suptitle(f'ToT Analysis for {pixel_key} Pair ({p1}, {p2}) | Corr: {correlation:.4f}', fontsize=16)
                try:
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
                except UserWarning:
                    pass
                plt.show()
                
                # --- PLOT 2: Asymmetric ToT Cuts (Histograms & KDEs) ---
                if asymmetric_tot_cut is not None:
                    
                    fig_asym, ((ax_hist1, ax_hist2), (ax_kde1, ax_kde2)) = plt.subplots(
                        2, 2, figsize=(16, 10), sharex=True, sharey='row'
                    )
                    
                    # --- Setup Colormap ---
                    cmap = cm.get_cmap('viridis', asymmetric_tot_cut + 1)
                    norm = mcolors.Normalize(vmin=0, vmax=asymmetric_tot_cut)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])

                    x_grid = np.linspace(0, 255, 200) # Grid for KDE plots

                    # --- Left Column (Red Data) ---
                    # X = ToT(p1), Y = ToT(p2)
                    mask_1 = (tot_p2 <= asymmetric_tot_cut)
                    df_1 = pd.DataFrame({
                        'x': tot_p1[mask_1], 
                        'y': tot_p2[mask_1].round().astype(int)
                    })
                    
                    for i in range(asymmetric_tot_cut + 1):
                        data_slice = df_1[df_1['y'] == i]['x']
                        if len(data_slice) > 1: # Need at least 2 points for KDE
                            color = cmap(norm(i))
                            # Top-Left: Histogram (Counts)
                            sns.histplot(data_slice, ax=ax_hist1, stat='count', # <-- CHANGED
                                         element='step', fill=False, color=color,
                                         bins=64, binrange=(0,256))
                            # Bottom-Left: KDE
                            try:
                                kde = gaussian_kde(data_slice)
                                ax_kde1.plot(x_grid, kde(x_grid), color=color)
                            except (np.linalg.LinAlgError, ValueError):
                                pass # Skip if KDE fails
                    
                    ax_hist1.set_title(f'Histograms: ToT({p1}) dist. for each ToT({p2})')
                    ax_hist1.set_ylabel('Counts')
                    ax_kde1.set_title(f'KDEs: ToT({p1}) dist. for each ToT({p2})')
                    ax_kde1.set_xlabel(f'ToT({p1})')
                    ax_kde1.set_ylabel('Density')
                    ax_kde1.set_xlim(0, 256)


                    # --- Right Column (Blue Data) ---
                    # X = ToT(p2), Y = ToT(p1)
                    mask_2 = (tot_p1 <= asymmetric_tot_cut)
                    df_2 = pd.DataFrame({
                        'x': tot_p2[mask_2], # X is ToT(p2)
                        'y': tot_p1[mask_2].round().astype(int) # Y is ToT(p1)
                    })

                    for i in range(asymmetric_tot_cut + 1):
                        data_slice = df_2[df_2['y'] == i]['x']
                        if len(data_slice) > 1: # Need at least 2 points for KDE
                            color = cmap(norm(i))
                            # Top-Right: Histogram (Counts)
                            sns.histplot(data_slice, ax=ax_hist2, stat='count', # <-- CHANGED
                                         element='step', fill=False, color=color,
                                         bins=64, binrange=(0,256))
                            # Bottom-Right: KDE
                            try:
                                kde = gaussian_kde(data_slice)
                                ax_kde2.plot(x_grid, kde(x_grid), color=color)
                            except (np.linalg.LinAlgError, ValueError):
                                pass # Skip if KDE fails

                    ax_hist2.set_title(f'Histograms: ToT({p2}) dist. for each ToT({p1})')
                    ax_kde2.set_title(f'KDEs: ToT({p2}) dist. for each ToT({p1})')
                    ax_kde2.set_xlabel(f'ToT({p2})')
                    ax_kde2.set_xlim(0, 256)

                    # --- Add Colorbar Legend ---
                    # *** ERROR FIX: Pass axes.ravel().tolist() to ax ***
                    fig_asym.colorbar(sm, ax=fig_asym.get_axes(), orientation='vertical', 
                                      label=f'Cut Axis Integer Value (0 to {asymmetric_tot_cut})')
                    fig_asym.suptitle(f'Asymmetric ToT Distributions for Pair ({p1}, {p2}) | Cut = {asymmetric_tot_cut}')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.show()

                plot_counter += 1

        # Append results regardless of plotting
        results_list.append({
            'pixel_pair': f"({p1}, {p2})",
            'pixel_1': p1,
            'pixel_2': p2,
            'correlation': correlation,
            'peak_ratio_p1_p2': peak_ratio_p1_p2,
            'peak_ratio_large_small': peak_ratio_large_small,
            'num_co_occurrences': len(merged_hits)
        })

    # --- 6. Format and Return Output Spreadsheet ---
    print("\n--- Summary Spreadsheet ---")
    output_df = pd.DataFrame(results_list).set_index('pixel_pair')
    print(output_df.to_string(float_format="%.4f"))
    return output_df

# --- Heatmap Plotting Function (Unchanged) ---
def plot_tot_heatmap(results_df: pd.DataFrame, value_to_plot: str = 'peak_tot_diff'):
    """
    Plots a specified value (e.g., peak_tot_diff) as a 2D heatmap of pixel pairs.

    The x and y axes represent the pixel IDs of a correlated pair, and
    the color value represents the specified value for that pair.

    Args:
        results_df: The DataFrame output by find_correlated_pairs_and_analyze_tot.
        value_to_plot: The column name to use for the z-axis of the heatmap.
                       Options: 'peak_tot_diff', 'min_before_peak'.
    """
    if results_df.empty:
        print("Input DataFrame is empty. Nothing to plot for heatmap.")
        return
        
    required_cols = ['pixel_1', 'pixel_2', value_to_plot]
    if not all(col in results_df.columns for col in required_cols):
        print(f"Input DataFrame is missing one or more required columns: {required_cols}")
        print(f"Available columns: {results_df.columns.tolist()}")
        return

    plot_df = results_df.dropna(subset=required_cols)
    
    if plot_df.empty:
        print(f"No data available to plot for '{value_to_plot}' after removing NaNs.")
        return

    all_pixels_in_results = pd.concat([plot_df['pixel_1'], plot_df['pixel_2']])
    if all_pixels_in_results.empty:
        print("No valid pixel pairs with ToT differences to plot.")
        return
    max_pixel = int(all_pixels_in_results.max())
    
    heatmap_matrix = pd.DataFrame(np.nan, index=range(max_pixel + 1), columns=range(max_pixel + 1))

    for _, row in plot_df.iterrows():
        p1 = int(row['pixel_1'])
        p2 = int(row['pixel_2'])
        value = row[value_to_plot]
        heatmap_matrix.loc[p1, p2] = value
        heatmap_matrix.loc[p2, p1] = value

    print(f"\nDisplaying '{value_to_plot}' heatmap for {len(plot_df)} correlated pairs...")
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        heatmap_matrix,
        cmap='inferno',
        vmin=0,         
        vmax=255,       
        cbar_kws={'label': value_to_plot.replace('_', ' ').title()}
    )
    ax.set_facecolor('black') 
    
    plt.title(f"Heatmap of {value_to_plot.replace('_', ' ').title()} Between Correlated Pixel Pairs")
    plt.xlabel('Pixel ID')
    plt.ylabel('Pixel ID')
    plt.show()

import pandas as pd
from matplotlib.patches import Rectangle 
from matplotlib.colors import LogNorm
from itertools import combinations
from typing import Dict, List, Optional



def plot_correlation_for_manual_areas(
    data: Dict[str, pd.Series],
    columns_to_analyze: Optional[List[int]] = None,
    manual_areas: Optional[List[Tuple[int, int, int, int]]] = None,
    num_ticks: int = 8
):
    """
    Calculates a correlation matrix, plots it, and generates detailed analysis
    plots for a user-provided list of manually defined areas.

    Args:
        data (Dict[str, pd.Series]): The input data dictionary.
        columns_to_analyze (Optional[List[int]], optional): Columns to analyze. If None, all are used.
        manual_areas (Optional[List[Tuple[int, int, int, int]]], optional): 
            A list of areas to analyze. Each area is a tuple of
            *inclusive* pixel labels in the format:
            (row_min, row_max, col_min, col_max).
            Example: [(100, 150, 200, 250)]
        num_ticks (int, optional): The approximate number of tick labels to display on the main plot.
    """
    # 1. --- Data Preparation and Matrix Calculation ---
    print("Preparing data and calculating correlations...")
    df = pd.DataFrame(data)
    was_run_on_all_columns = columns_to_analyze is None
    if was_run_on_all_columns:
        columns_to_analyze = sorted(df['Column'].unique())
    all_correlation_matrices = []
    all_pixel_pairs = []
    for col_id in columns_to_analyze:
        df_col = df[df['Column'] == col_id]
        if df_col.shape[0] < 2 or df_col['Row'].nunique() < 2: continue
        hit_matrix = pd.crosstab(df_col['Row'], df_col['TriggerID'])
        correlation_matrix = hit_matrix.T.corr()
        if not correlation_matrix.empty and not correlation_matrix.isna().all().all():
            all_correlation_matrices.append(correlation_matrix)
        grouped = df_col.groupby('TriggerID')
        for _, group in grouped:
            if len(group) >= 2:
                for hit1, hit2 in combinations(group.to_dict('records'), 2):
                    all_pixel_pairs.extend([
                        {'seed_row': hit1['Row'], 'seed_tot': hit1['ToT'], 'other_row': hit2['Row'], 'other_tot': hit2['ToT']},
                        {'seed_row': hit2['Row'], 'seed_tot': hit2['ToT'], 'other_row': hit1['Row'], 'other_tot': hit1['ToT']}])
    if not all_correlation_matrices:
        print("❌ Insufficient data to perform analysis.")
        return
    avg_corr_matrix = _calculate_average_correlation(all_correlation_matrices)

    # 2. --- Automatic Area Detection REMOVED ---

    # 3. --- Plot Main Correlation Matrix (with Manual Area Highlights) ---
    print("\nPlotting main correlation matrix...")
    plt.figure(figsize=(12, 10))
    tick_skip_rate = max(1, len(avg_corr_matrix.index) // num_ticks)
    ax = sns.heatmap(avg_corr_matrix, cmap='viridis', annot=False, cbar_kws={'label': 'Average Pearson Correlation Coefficient'}, xticklabels=tick_skip_rate, yticklabels=tick_skip_rate)
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    title_text = 'Average Pixel Correlation for All Columns' if was_run_on_all_columns else f'Average Pixel Correlation for Columns {columns_to_analyze}'
    plt.title(title_text)
    plt.xlabel('Pixel Row'); plt.ylabel('Pixel Row')
    
    # Draw rectangles for the user-defined areas
    if manual_areas:
        print(f"Highlighting {len(manual_areas)} user-defined area(s)...")
        pixel_rows = avg_corr_matrix.index
        pixel_cols = avg_corr_matrix.columns
        
        for (row_min_label, row_max_label, col_min_label, col_max_label) in manual_areas:
            try:
                # Get integer indices from pixel labels for drawing
                y_start_idx = pixel_rows.get_indexer([row_min_label], method='nearest')[0]
                y_stop_idx = pixel_rows.get_indexer([row_max_label], method='nearest')[0]
                x_start_idx = pixel_cols.get_indexer([col_min_label], method='nearest')[0]
                x_stop_idx = pixel_cols.get_indexer([col_max_label], method='nearest')[0]
                
                # Calculate width/height based on *inclusive* indices
                width = (x_stop_idx - x_start_idx) + 1
                height = (y_stop_idx - y_start_idx) + 1
                
                rect = Rectangle(
                    (x_start_idx, y_start_idx),  # (x, y) bottom-left corner
                    width,
                    height,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)
            except Exception as e:
                print(f"⚠️ Warning: Could not draw rectangle for area ({row_min_label}, {row_max_label}, {col_min_label}, {col_max_label}). Error: {e}")

    plt.show()

    # 4. --- Detailed Analysis for Each Manual Area ---
    if not manual_areas:
        print("\nNo 'manual_areas' provided. Skipping detailed analysis.")
        print("✅ Analysis complete.")
        return

    print(f"\nStarting detailed analysis for {len(manual_areas)} manual area(s)...")
    pairs_df = pd.DataFrame(all_pixel_pairs)
    
    for i, (row_min_i, row_max_i, row_min_j, row_max_j) in enumerate(manual_areas):
        
        # Bounds are already inclusive pixel labels
        cell_title = f"Seed pixels [{row_min_i}-{row_max_i}] vs victim pixels [{row_min_j}-{row_max_j}]"

        # Filter pairs_df for data within this area's bounding box (inclusive)
        cell_data = pairs_df[
            (pairs_df['seed_row'] >= row_min_i) & (pairs_df['seed_row'] <= row_max_i) &
            (pairs_df['other_row'] >= row_min_j) & (pairs_df['other_row'] <= row_max_j)
        ]
        
        if cell_data.empty: 
            print(f"  -> Skipping area {i} ({cell_title}): No raw pairs found in bounding box.")
            continue
        
        # Get the correlation snippet using the *inclusive* labels
        try:
            corr_snippet = avg_corr_matrix.loc[row_min_i:row_max_i, row_min_j:row_max_j]
        except KeyError:
             print(f"  -> Skipping area {i}: Invalid pixel labels in ({cell_title}). Check your bounds.")
             continue
        
        print(f"  -> Plotting for {cell_title} with {len(cell_data)} pairs...")

        fig = plt.figure(figsize=(14, 12)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1])
        fig.suptitle(f'Comprehensive Analysis for {cell_title} (Area {i})', fontsize=16)

        # Top-Left: Correlation Snippet
        ax_corr_snippet = fig.add_subplot(gs[0, 0])
        sns.heatmap(corr_snippet, ax=ax_corr_snippet, cmap='viridis', cbar=True, annot=False)
        ax_corr_snippet.set_title('Pixel-Pixel Correlation Snippet'); ax_corr_snippet.set_xlabel('Pixel Row'); ax_corr_snippet.set_ylabel('Pixel Row')
        # Make snippet square-ish if bounds are different
        if corr_snippet.shape[0] != corr_snippet.shape[1]:
            ax_corr_snippet.set_aspect('equal', adjustable='box') 

        # Top-Right: ToT Correlation
        ax_tot_corr = fig.add_subplot(gs[0, 1])
        h_main, _, _, im_main = ax_tot_corr.hist2d(cell_data['seed_tot'], cell_data['other_tot'], bins=255, range=[[0, 255], [0, 255]], cmap='plasma', norm=LogNorm())
        ax_tot_corr.set_title('Seed ToT vs. victim Pixel ToT'); ax_tot_corr.set_xlabel('Seed Pixel ToT'); ax_tot_corr.set_ylabel('2nd Pixel ToT')
        ax_tot_corr.set_aspect('equal', adjustable='box')
        fig.colorbar(im_main, ax=ax_tot_corr, label='Frequency (Log Scale)', fraction=0.046, pad=0.04)

        # Bottom: Joint Plot
        tot_diff = cell_data['seed_tot'] - cell_data['other_tot']; displacement = cell_data['seed_row'] - cell_data['other_row']
        
        # Calculate displacement range from *inclusive* bounds
        min_disp = row_min_i - row_max_j
        max_disp = row_max_i - row_min_j
        
        disp_range = [min_disp - 0.5, max_disp + 0.5]; 
        disp_bins = int(np.ceil(max_disp - min_disp)) + 1
        if disp_bins <= 0: disp_bins = 1 # Safety check
            
        gs_joint = gs[1, :].subgridspec(5, 5, hspace=0, wspace=0)
        ax_joint_main = fig.add_subplot(gs_joint[1:, :-1]); ax_joint_marg_x = fig.add_subplot(gs_joint[0, :-1], sharex=ax_joint_main); ax_joint_marg_y = fig.add_subplot(gs_joint[1:, -1], sharey=ax_joint_main)
        h, _, _, im_disp = ax_joint_main.hist2d(displacement, tot_diff, bins=(disp_bins, 511), range=[disp_range, [-255.5, 255.5]], cmap='magma', norm=LogNorm())
        ax_joint_marg_x.hist(displacement, bins=disp_bins, range=disp_range, histtype='step')
        ax_joint_marg_y.hist(tot_diff, bins=511, range=[-255.5, 255.5], histtype='step', orientation='horizontal')
        plt.setp(ax_joint_marg_x.get_xticklabels(), visible=False); plt.setp(ax_joint_marg_y.get_yticklabels(), visible=False)
        ax_joint_main.set_xlabel('Pixel Displacement (signed)'); ax_joint_main.set_ylabel('ToT Difference (Seed - 2nd)')
        ax_joint_main.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
    
    print("\n✅ Analysis complete.")