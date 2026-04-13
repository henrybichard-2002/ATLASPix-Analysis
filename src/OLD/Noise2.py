import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Union, List, Optional
import sys
from matplotlib.colors import LogNorm, Normalize
import matplotlib.gridspec as gridspec
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

Dataset = Dict[str, np.ndarray]

from utils import progress_bar

def find_major_bunch_edges(
    all_timestamps_s: np.ndarray,
    bin_width_s: float,
    threshold_count: int
) -> Tuple[List[Tuple[float, float]], Tuple]:
    """
    Finds the start and end times of major bunches.
    """
    print("Finding major bunch edges...")
    if len(all_timestamps_s) == 0:
        return [], (np.array([]), np.array([]), np.array([]))
        
    t_min = all_timestamps_s[0]
    t_max = all_timestamps_s[-1]
    
    bins = int((t_max - t_min) / bin_width_s)
    if bins <= 0:
        print("Not enough data to create bins.")
        return [], (np.array([]), np.array([]), np.array([]))

    counts, bin_edges = np.histogram(
        all_timestamps_s, 
        bins=bins, 
        range=(t_min, t_max)
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find where the counts are above the threshold
    above_threshold = counts >= threshold_count

    # Find the edges (transitions from False to True, and True to False)
    edges = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
    
    start_indices = np.where(edges == 1)[0]
    end_indices = np.where(edges == -1)[0]
    
    bunch_edges_s = []
    if len(start_indices) > 0 and len(start_indices) == len(end_indices):
        for start_idx, end_idx in zip(start_indices, end_indices):
            start_s = bin_edges[start_idx]
            end_s = bin_edges[end_idx]
            bunch_edges_s.append((start_s, end_s))
            
    print(f"Found {len(bunch_edges_s)} major bunches.")
    return bunch_edges_s, (bin_centers, counts, bin_edges)


def optimize_minor_bunch_parameters(
    relative_timestamps_s: np.ndarray,
    base_freq_hz: float,
    freq_scan_range_hz: float,
    freq_step_hz: float,
    phase_step_s: float,
    phase_scan_window_s: float,
    duration_s: float,
    bunch_index: int,
    total_bunches: int
) -> Tuple[float, float, float, int]:
    """
    Performs a highly efficient grid search to find the optimal
    frequency and phase for the minor bunches *for a single major bunch*.
    """    
    if len(relative_timestamps_s) == 0:
        # print("No hits found in this major bunch to optimize.")
        return base_freq_hz, 0.0, (1.0 / base_freq_hz), 0
    
    total_hits = len(relative_timestamps_s)

    freq_scan = np.arange(
        base_freq_hz - freq_scan_range_hz,
        base_freq_hz + freq_scan_range_hz + freq_step_hz,
        freq_step_hz
    )
    
    best_score = -total_hits
    best_freq = base_freq_hz
    best_phase = 0.0
    best_period = 1.0 / base_freq_hz
    
    coarse_bin_s = 0.01 

    # --- New, Faster Optimization ---
    
    prog_desc = f"Optimizing Bunch {bunch_index+1}/{total_bunches}"
    for i, freq_hz in enumerate(progress_bar(freq_scan, description=prog_desc)):
        if freq_hz <= 0: continue
            
        period_s = 1.0 / freq_hz
        
        # 1. Fold all hits and SORT them. This is the key.
        time_in_cycle = np.sort(relative_timestamps_s % period_s)
        
        # 2. Find coarse phase (same as before)
        bins = int(period_s / coarse_bin_s)
        if bins <= 0: continue
        counts, bin_edges = np.histogram(time_in_cycle, bins=bins, range=(0, period_s))
        
        coarse_phase_center_s = 0.0
        if len(counts) > 0:
            peak_bin_index = np.argmax(counts)
            coarse_phase_center_s = (bin_edges[peak_bin_index] + bin_edges[peak_bin_index+1]) / 2.0
        
        # 3. Create fine phase scan window
        phase_scan_start = coarse_phase_center_s - phase_scan_window_s
        phase_scan_end = coarse_phase_center_s + phase_scan_window_s
        phase_scan = np.arange(phase_scan_start, phase_scan_end, phase_step_s)
        
        # 4. Iterate over phases and find scores *efficiently*
        for phase_s in phase_scan:
            start_phase = (phase_s % period_s)
            end_phase = ((phase_s + duration_s) % period_s)
            
            hits_inside = 0
            if start_phase < end_phase:
                i_start = np.searchsorted(time_in_cycle, start_phase, 'left')
                i_end = np.searchsorted(time_in_cycle, end_phase, 'right')
                hits_inside = i_end - i_start
            else:
                i_start_1 = np.searchsorted(time_in_cycle, start_phase, 'left')
                i_end_1 = np.searchsorted(time_in_cycle, period_s, 'right')
                i_start_2 = np.searchsorted(time_in_cycle, 0, 'left')
                i_end_2 = np.searchsorted(time_in_cycle, end_phase, 'right')
                hits_inside = (i_end_1 - i_start_1) + (i_end_2 - i_start_2)

            # current_score = 2 * hits_inside - total_hits
            current_score = hits_inside - (total_hits - hits_inside)
            
            if current_score > best_score:
                best_score = current_score
                best_freq = freq_hz
                best_phase = start_phase
                best_period = period_s

    return best_freq, best_phase, best_period, best_score


def plot_tot_vs_phase_heatmap_by_layer(
    all_times: np.ndarray,
    all_tots: np.ndarray,
    all_layers: np.ndarray,
    base_period_s: float,
    minor_bunch_duration_s: float,
    freq_data: Dict, 
    normalize_y: bool = False,
    cmap: str = 'viridis'
) -> None:
    """
    Plots ToT vs. Phase-Folded Time as a 2x2 grid of joint plots
    (heatmap + marginals), with one subplot per layer.
    """
    print("Generating ToT vs. Phase (by Layer) heatmap plot with marginals...")
    if len(all_times) == 0:
        print("No hits to plot for ToT vs. Phase.")
        return

    fig = plt.figure(figsize=(17, 15))
    
    outer_grid = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
    
    x_bins = np.arange(0, base_period_s + 0.001, 0.001)
    y_bins = np.arange(-0.5, 256.5, 1) # 257 edges for 256 bins
    
    h_list = [] # To store pcolormesh handles for colorbar
    cbar_label = "Hit Count" # Default
    
    for i, layer in enumerate([1, 2, 3, 4]):
        
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 2, 
            subplot_spec=outer_grid[i], 
            wspace=0.05, hspace=0.05, 
            width_ratios=[4, 1], height_ratios=[1, 4]
        )
        
        ax_main = fig.add_subplot(inner_grid[1, 0])
        ax_histx = fig.add_subplot(inner_grid[0, 0], sharex=ax_main)
        ax_histy = fig.add_subplot(inner_grid[1, 1], sharey=ax_main)
        
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)
        ax_histx.set_yticks([])
        ax_histy.set_xticks([])

        layer_mask = (all_layers == layer)
        
        if not np.any(layer_mask):
            ax_main.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax_main.transAxes)
            ax_histx.set_title(f"Layer {layer}")
            continue
            
        x_data_raw = all_times[layer_mask]
        y_data_raw = all_tots[layer_mask]

        # Filter out non-finite values (NaN, Inf)
        finite_mask = np.isfinite(x_data_raw) & np.isfinite(y_data_raw)
        x_data = x_data_raw[finite_mask]
        y_data = y_data_raw[finite_mask]

        if len(x_data) == 0: # Check if filtering removed everything
            ax_main.text(0.5, 0.5, "No Finite Data", ha='center', va='center', transform=ax_main.transAxes)
            ax_histx.set_title(f"Layer {layer}")
            continue
        
        H, x_edges, y_edges = np.histogram2d(
            x_data, y_data, bins=[x_bins, y_bins]
        )
        
        plot_data = H
        norm_options = {}
        cbar_label = "Hit Count"

        if normalize_y:
            H_sum_over_y = H.sum(axis=1)
            H_sum_safe = np.where(H_sum_over_y == 0, 1.0, H_sum_over_y)
            H_norm = H / H_sum_safe[:, np.newaxis]
            
            plot_data = np.nan_to_num(H_norm) 
            actual_vmax = plot_data.max()
            if actual_vmax == 0:
                actual_vmax = 1.0 
            norm_options = {'norm': Normalize(vmin=0.0, vmax=actual_vmax)}
            cbar_label = "Normalized Count (in Y)"
        else:
            norm_options = {'norm': LogNorm(vmin=1)} 
        
        h = ax_main.pcolormesh(
            x_edges, y_edges, plot_data.T,
            cmap=cmap, **norm_options
        )
        h_list.append(h)
        
        ax_main.axvline(
            minor_bunch_duration_s, color='r', linestyle='--', linewidth=2,
            label=f"Cut ({minor_bunch_duration_s * 1000:.1f} ms)"
        )
        
        ax_histx.hist(x_data, bins=x_bins, histtype='step', color='k')
        ax_histy.hist(y_data, bins=y_bins, histtype='step', color='k', orientation='horizontal')
        
        try:
            freq_in = freq_data.get(layer, {}).get('inside', 0.0)
            freq_out = freq_data.get(layer, {}).get('outside', 0.0)
            freq_text = (
                f"Freq (In):  {freq_in:7.2f} Hz\n"
                f"Freq (Out): {freq_out:7.2f} Hz"
            )
        except Exception:
            freq_text = "Freq: N/A"
            
        ax_main.text(
            0.03, 0.97, freq_text, 
            transform=ax_main.transAxes, 
            ha='left', va='top', 
            fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8)
        )
        
        ax_histx.set_title(f"Layer {layer}") 
        
        ax_main.set_xlabel("Time in Minor Cycle (s)")
        ax_main.set_ylabel("ToT")
        ax_main.legend(loc='upper right')
        
        ax_main.set_xlim(0, base_period_s)
        ax_main.set_ylim(y_bins[0], y_bins[-1])

        # --- ADJUSTMENT HERE ---
        # 1. Set scale to 'symlog'
        #    We use linthresh=1.5 so that the bins for ToT=0 ([-0.5, 0.5])
        #    and ToT=1 ([0.5, 1.5]) are both in the linear-scaled region.
        ax_main.set_yscale('symlog', linthresh=1.5)
        ax_histy.set_yscale('symlog', linthresh=1.5)
        
        # 2. Set user-friendly ticks, as symlog defaults are not intuitive
        y_ticks = [0, 1, 2, 5, 10, 20, 50, 100, 200, 255]
        ax_main.set_yticks(y_ticks)
        ax_main.set_yticklabels([str(t) for t in y_ticks])
        # --- END ADJUSTMENT ---

    fig.suptitle("ToT vs. Phase-Folded Time (by Layer)", fontsize=16)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.92)

    if h_list:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7]) 
        fig.colorbar(h_list[0], cax=cbar_ax, label=cbar_label)
        
    print("Displaying ToT vs. Phase (by Layer) plot...")
    plt.show()


def _print_hit_frequencies(
    bunch_edges_s: List[Tuple[float, float]],
    all_optimized_params: List[Dict],
    minor_bunch_duration_s: float,
    all_layers: np.ndarray,
    all_inside_masks: np.ndarray
) -> Dict:
    """
    Calculates and prints the average hit frequency (Hz) for
    inside vs. outside the minor bunches, broken down by layer.
    
    Returns:
        Dict: A dictionary with frequency data, e.g.,
              {1: {'inside': 120.5, 'outside': 10.1}, ...}
    """
    print("\n--- Average Hit Frequencies ---")
    
    total_duration_inside_s = 0.0
    total_duration_outside_s = 0.0
    
    # Calculate the total effective "on" time for inside/outside
    for i, (start_s, end_s) in enumerate(bunch_edges_s):
        try:
            params = all_optimized_params[i]
            period = params['period']
            bunch_duration = end_s - start_s
        except (IndexError, KeyError):
            continue # Skip if params are missing

        if period > 0:
            fraction_inside = minor_bunch_duration_s / period
            # Clamp fraction to [0, 1] just in case
            fraction_inside = max(0.0, min(1.0, fraction_inside))
            fraction_outside = 1.0 - fraction_inside
            
            total_duration_inside_s += bunch_duration * fraction_inside
            total_duration_outside_s += bunch_duration * fraction_outside
            
    if len(all_layers) == 0:
        print("No hits found to calculate frequencies.")
        return {}

    print(f"Total 'Inside' Duration (Est.): {total_duration_inside_s:.2f} s")
    print(f"Total 'Outside' Duration (Est.): {total_duration_outside_s:.2f} s")
    
    outside_mask_global = ~all_inside_masks
    
    freq_data_out = {} 
    
    for layer in sorted(np.unique(all_layers)):
        layer_mask = (all_layers == layer)
        
        count_inside = np.sum(layer_mask & all_inside_masks)
        count_outside = np.sum(layer_mask & outside_mask_global)
        
        freq_inside = count_inside / total_duration_inside_s if total_duration_inside_s > 0 else 0
        freq_outside = count_outside / total_duration_outside_s if total_duration_outside_s > 0 else 0
        
        print(f"Layer {layer}:")
        print(f"  Inside:  {count_inside:10d} hits / {total_duration_inside_s:6.2f}s = {freq_inside:10.2f} Hz")
        print(f"  Outside: {count_outside:10d} hits / {total_duration_outside_s:6.2f}s = {freq_outside:10.2f} Hz")
        
        freq_data_out[layer] = {'inside': freq_inside, 'outside': freq_outside}
        
    return freq_data_out 

def _build_output_dicts(
    dataset: Dataset,
    all_inside_indices: List[np.ndarray],
    all_outside_indices: List[np.ndarray],
    all_timestamps_s: np.ndarray
) -> Tuple[Dataset, Dataset]:
    """
    Internal helper to construct the final dictionaries from the
    aggregated index lists.
    """
    print("\nBuilding final output dictionaries...")
    
    if all_inside_indices:
        final_inside_indices = np.concatenate(all_inside_indices)
    else:
        final_inside_indices = np.array([], dtype=int)
        
    if all_outside_indices:
        final_outside_indices = np.concatenate(all_outside_indices)
    else:
        final_outside_indices = np.array([], dtype=int)
    
    final_mask_inside = np.zeros(all_timestamps_s.shape, dtype=bool)
    final_mask_outside = np.zeros(all_timestamps_s.shape, dtype=bool)
    
    final_mask_inside[final_inside_indices] = True
    final_mask_outside[final_outside_indices] = True
    
    hits_inside_minor_bunches = {}
    hits_outside_minor_bunches = {}

    for key, full_array in dataset.items():
        if not isinstance(full_array, np.ndarray):
            print(f"Warning: Skipping key '{key}' as its value is not a numpy array.")
            continue
            
        try:
            hits_inside_minor_bunches[key] = full_array[final_mask_inside]
            hits_outside_minor_bunches[key] = full_array[final_mask_outside]
        except Exception as e:
            print(f"An unexpected error occurred while filtering key '{key}': {e}")
            hits_inside_minor_bunches[key] = np.array([], dtype=full_array.dtype)
            hits_outside_minor_bunches[key] = np.array([], dtype=full_array.dtype)
            
    return hits_inside_minor_bunches, hits_outside_minor_bunches

def _plot_bunch_stat_distribution(
    data_inside: np.ndarray,
    layers_inside: np.ndarray,
    hits_inside: np.ndarray,
    data_outside: np.ndarray,
    layers_outside: np.ndarray,
    hits_outside: np.ndarray,
    title_in: str,
    title_out: str,
    xlabel: str,
    num_bins_in: int = 100,
    num_bins_out: int = 50,
    log_y: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    kde_bw_bins: float = 3.0
) -> None:
    """
    REFACTORED HELPER: Plots the distribution of a given statistic
    per minor bunch/gap, separated by layer.
    KDE smoothing is applied *only* to non-zero, finite data.
    """
    if len(data_inside) == 0 and len(data_outside) == 0:
        print(f"No minor bunch data to plot for {xlabel}.")
        return

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple'}

    # --- Left Plot: Inside Bunches ---
    if len(data_inside) > 0:
        # --- Standard binning logic ---
        if xlim:
            bins_in = np.linspace(xlim[0], xlim[1], num_bins_in + 1)
        else:
            finite_data_in = data_inside[np.isfinite(data_inside)]
            if len(finite_data_in) == 0:
                bins_in = np.linspace(0, 1, num_bins_in + 1) # Default
            else:
                v_min_in = np.min(finite_data_in)
                v_max_in = np.max(finite_data_in)
                if v_min_in == v_max_in: # Handle case of all same values
                   v_min_in -= 0.5
                   v_max_in += 0.5
                bins_in = np.linspace(v_min_in, v_max_in, num_bins_in + 1)
        
        bin_width_in = bins_in[1] - bins_in[0]
        
        for layer in [1, 2, 3, 4]:
            layer_mask = (layers_inside == layer)
            layer_data = data_inside[layer_mask]
            
            if len(layer_data) > 0:
                total_hits = np.sum(hits_inside[layer_mask])
                # --- Faint histogram (plots ALL data) ---
                ax_in.hist(layer_data, bins=bins_in,
                           histtype='step', linewidth=0.5, alpha=0.4,
                           color=colors.get(layer), density=True, label='_nolegend_')
                
                # --- MODIFIED: Create data for KDE, excluding zeros/infs ---
                kde_data = layer_data[(layer_data > 0) & np.isfinite(layer_data)]
                
                # --- Bold KDE Plot (plots only non-zero data) ---
                if len(kde_data) > 1:
                    try:
                        # --- Calculate bw_factor from kde_data stats ---
                        target_bw = kde_bw_bins * bin_width_in
                        data_std = np.std(kde_data)
                        data_n = len(kde_data)
                        scotts_bw = data_std * data_n**(-1./5.)
                        
                        bw_factor = 1.0 # Default
                        if scotts_bw > 0:
                            bw_factor = target_bw / scotts_bw
                        
                        kde = gaussian_kde(kde_data, bw_method=bw_factor) # Use kde_data
                        x_kde = np.linspace(bins_in[0], bins_in[-1], 200)
                        ax_in.plot(x_kde, kde(x_kde), color=colors.get(layer), 
                                   linewidth=2.5, label=f'L{layer} (Total Hits={total_hits:.0f})')
                    except np.linalg.LinAlgError:
                         ax_in.axvline(kde_data[0], color=colors.get(layer), 
                                   linewidth=2.5, linestyle='--',
                                   label=f'L{layer} (Single Value, Hits={total_hits:.0f})')


    ax_in.set_title(title_in)
    ax_in.set_xlabel(xlabel)
    ax_in.set_ylabel("Density")
    ax_in.legend()
    ax_in.grid(True, linestyle='--')
    if xlim:
        ax_in.set_xlim(xlim)
    elif 'bins_in' in locals() and len(bins_in) > 1:
        ax_in.set_xlim(bins_in[0], bins_in[-1]) 

    # --- Right Plot: Outside Bunches ---
    if len(data_outside) > 0:
        # --- Standard binning logic ---
        if xlim:
            bins_out = np.linspace(xlim[0], xlim[1], num_bins_out + 1)
        else:
            finite_data_out = data_outside[np.isfinite(data_outside)]
            if len(finite_data_out) == 0:
                bins_out = np.linspace(0, 1, num_bins_out + 1) # Default
            else:
                v_min_out = np.min(finite_data_out)
                v_max_out = np.max(finite_data_out)
                if v_min_out == v_max_out:
                   v_min_out -= 0.5
                   v_max_out += 0.5
                bins_out = np.linspace(v_min_out, v_max_out, num_bins_out + 1)
        
        bin_width_out = bins_out[1] - bins_out[0]
        
        for layer in [1, 2, 3, 4]:
            layer_mask = (layers_outside == layer)
            layer_data = data_outside[layer_mask]
            
            if len(layer_data) > 0:
                total_hits = np.sum(hits_outside[layer_mask])
                # --- Faint histogram (plots ALL data) ---
                ax_out.hist(layer_data, bins=bins_out,
                            histtype='step', linewidth=0.5, alpha=0.4,
                            color=colors.get(layer), density=True, label='_nolegend_')
                
                # --- MODIFIED: Create data for KDE, excluding zeros/infs ---
                kde_data = layer_data[(layer_data > 0) & np.isfinite(layer_data)]

                # --- Bold KDE Plot (plots only non-zero data) ---
                if len(kde_data) > 1:
                    try:
                        # --- Calculate bw_factor from kde_data stats ---
                        target_bw = kde_bw_bins * bin_width_out
                        data_std = np.std(kde_data)
                        data_n = len(kde_data)
                        scotts_bw = data_std * data_n**(-1./5.)
                        
                        bw_factor = 1.0 # Default
                        if scotts_bw > 0:
                            bw_factor = target_bw / scotts_bw

                        kde = gaussian_kde(kde_data, bw_method=bw_factor) # Use kde_data
                        x_kde = np.linspace(bins_out[0], bins_out[-1], 200)
                        ax_out.plot(x_kde, kde(x_kde), color=colors.get(layer), 
                                    linewidth=2.5, label=f'L{layer} (Total Hits={total_hits:.0f})')
                    except np.linalg.LinAlgError:
                         ax_out.axvline(kde_data[0], color=colors.get(layer), 
                                   linewidth=2.5, linestyle='--',
                                   label=f'L{layer} (Single Value, Hits={total_hits:.0f})')


    ax_out.set_title(title_out)
    ax_out.set_xlabel(xlabel)
    ax_out.legend()
    ax_out.grid(True, linestyle='--')
    if xlim:
        ax_out.set_xlim(xlim)
    elif 'bins_out' in locals() and len(bins_out) > 1:
        ax_out.set_xlim(bins_out[0], bins_out[-1])
    
    if log_y:
        ax_in.set_yscale('log')
        ax_out.set_yscale('log')
        # Set a reasonable bottom limit for log scale
        ax_in.set_ylim(bottom=1e-5) 
        ax_out.set_ylim(bottom=1e-5)
    
    fig.suptitle(f"Per-Minor-Bunch {xlabel} Distributions by Layer", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Displaying minor bunch {xlabel} plot...")
    plt.show()

def _plot_bunch_frequency_distribution(
    freq_inside: np.ndarray,
    layers_inside: np.ndarray,
    hits_inside: np.ndarray, 
    freq_outside: np.ndarray,
    layers_outside: np.ndarray,
    hits_outside: np.ndarray, 
    dist_num_bins_in: int = 100,
    dist_num_bins_out: int = 50,
    log_y: bool = False,
    kde_bw_bins: float = 3.0  # <-- MODIFIED
) -> None:
    """
    REFACTORED: Plots the distribution of hit frequencies per *minor* bunch/gap.
    """
    print("Generating minor bunch frequency distribution plot...")
    _plot_bunch_stat_distribution(
        data_inside=freq_inside,
        layers_inside=layers_inside,
        hits_inside=hits_inside,
        data_outside=freq_outside,
        layers_outside=layers_outside,
        hits_outside=hits_outside,
        title_in="Hit Frequency 'Inside' Minor Bunches (per Minor Bunch)",
        title_out="Hit Frequency 'Outside' Minor Bunches (per Minor Gap)",
        xlabel="Hit Frequency (Hz)",
        num_bins_in=dist_num_bins_in,
        num_bins_out=dist_num_bins_out,
        log_y=log_y,
        xlim=None,
        kde_bw_bins=kde_bw_bins  # <-- Pass through
    )

def _plot_bunch_tot_ratio_distribution(
    ratio_inside: np.ndarray,
    layers_inside: np.ndarray,
    hits_inside: np.ndarray, 
    ratio_outside: np.ndarray,
    layers_outside: np.ndarray,
    hits_outside: np.ndarray, 
    tot_threshold: float,
    dist_num_bins_in: int = 100,
    dist_num_bins_out: int = 50,
    log_y: bool = False,
    kde_bw_bins: float = 3.0
) -> None:
    """
    REFACTORED: Plots the distribution of high/low ToT ratios per *minor* bunch/gap.
    """
    print("Generating minor bunch ToT ratio distribution plot...")
    
    # Group Infs and Zeros
    ratio_inside_clean = ratio_inside.copy()
    ratio_outside_clean = ratio_outside.copy()
    ratio_inside_clean[np.isinf(ratio_inside_clean)] = 0.0
    ratio_outside_clean[np.isinf(ratio_outside_clean)] = 0.0
    
    xlim_ratio = (-0.001, 3.5)

    _plot_bunch_stat_distribution(
        data_inside=ratio_inside_clean,
        layers_inside=layers_inside,
        hits_inside=hits_inside,
        data_outside=ratio_outside_clean,
        layers_outside=layers_outside,
        hits_outside=hits_outside,
        title_in=f"'Inside' Minor Bunches (ToT <= {tot_threshold}) / (ToT > {tot_threshold})",
        title_out=f"'Outside' Minor Gaps (ToT <= {tot_threshold}) / (ToT > {tot_threshold})",
        xlabel="Low/High ToT Ratio (Infs grouped with Zeros)",
        num_bins_in=dist_num_bins_in,
        num_bins_out=dist_num_bins_out,
        log_y=log_y,
        xlim=xlim_ratio,
        kde_bw_bins=kde_bw_bins
    )
    
def _plot_diagnostic_charts(
    hist_data: Tuple,
    threshold_count: int,
    bunch_edges_s: List[Tuple[float, float]],
    all_phase_folded_times: np.ndarray,
    all_tots: np.ndarray,       
    all_layers: np.ndarray,       
    all_inside_masks: np.ndarray, 
    base_period_s: float,
    minor_bunch_duration_s: float,
    all_timestamps_s: np.ndarray,
    all_optimized_params: List[Dict],
    plot_window_s_major: float,
    plot_window_s_minor: float,
    normalize_tot_heatmaps: bool, 
    heatmap_cmap: str,
    bunch_stat_tot_threshold: float,
    all_bunch_freq_inside: np.ndarray,
    all_bunch_freq_outside: np.ndarray,
    all_bunch_layers_inside: np.ndarray,
    all_bunch_layers_outside: np.ndarray,
    all_bunch_ratio_inside: np.ndarray,
    all_bunch_ratio_outside: np.ndarray,
    all_bunch_hits_inside: np.ndarray,
    all_bunch_hits_outside: np.ndarray,
    dist_num_bins_in: int,
    dist_num_bins_out: int,
    dist_log_y: bool,
    kde_bw_bins: float = 3.0  # <-- MODIFIED
) -> None:
    """
    Internal helper function to generate and display all diagnostic plots.
    This consolidates all plotting logic.
    """
    
    print("\n--- Generating Diagnostic Plots ---")
    
    # ... (Plots 1-6 are unchanged) ...
    # [Plots 1-6 code omitted for brevity, it is unchanged from your last version]
    
    # --- Plot 1: Major Bunch Finder ---
    print("Generating major bunch finder plot...")
    bin_centers, counts, _ = hist_data
    if len(bin_centers) > 0:
        plt.figure(figsize=(15, 7))
        plt.plot(bin_centers, counts, drawstyle='steps-mid', 
                 label="Hit Rate (Counts per Bin)")
        plt.axhline(threshold_count, color='r', linestyle='--', linewidth=2,
                    label=f"Threshold ({threshold_count} counts)")
        for start_s, end_s in bunch_edges_s:
            plt.axvspan(
                start_s, end_s, color='g', alpha=0.3, 
                label='Detected Bunch' if 'Detected Bunch' not in plt.gca().get_legend_handles_labels()[1] else ""
            )
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Hit Count per Bin", fontsize=12)
        plt.title("Major Bunch Finder", fontsize=14)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        print("Displaying major bunch finder plot...")
        plt.show()
    
    # --- Plot 2: Phase-Folded Distribution ---
    print("Generating aggregate phase-folded distribution plot...")
    if len(all_phase_folded_times) > 0:
        precision_us = 100 # 100 microsecond bins
        rounded_times_us = (all_phase_folded_times * 1_000_000).astype(int)
        rounded_times_us = (rounded_times_us // precision_us) * precision_us
        unique_times_us, counts = np.unique(rounded_times_us, return_counts=True)
        unique_times_s = unique_times_us / 1_000_000.0

        plt.figure(figsize=(15, 7))
        plt.plot(unique_times_s, counts, drawstyle='steps-mid', 
                 label=f"Hit Rate (Counts per {precision_us}us Bin)")
        plt.axvline(
            minor_bunch_duration_s, color='r', linestyle='--', linewidth=2,
            label=f"Minor Bunch End ({minor_bunch_duration_s * 1000:.1f} ms)"
        )
        plt.xlabel("Time in Minor Cycle (seconds)", fontsize=12)
        plt.ylabel("Hit Count", fontsize=12)
        plt.title("Aggregate Phase Folded Hit Distribution (All Bunches)", fontsize=14)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(0, base_period_s)
        plt.tight_layout()
        print("Displaying phase-folded plot...")
        plt.show()

    # --- Plot 3: Major Bunch Verification ---
    print("Generating major bunch verification plot...")
    if len(all_timestamps_s) > 0 and bunch_edges_s:
        plt.figure(figsize=(15, 7))
        plot_start_s = bunch_edges_s[0][0]
        plot_end_s = plot_start_s + plot_window_s_major
        window_mask = (all_timestamps_s >= plot_start_s) & (all_timestamps_s < plot_end_s)
        timestamps_in_window = all_timestamps_s[window_mask]

        if len(timestamps_in_window) > 0:
            bins = int(plot_window_s_major / 0.1) # 0.1s bins
            if bins > 0:
                counts, bin_edges = np.histogram(
                    timestamps_in_window, bins=bins, range=(plot_start_s, plot_end_s)
                )
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.plot(bin_centers, counts, drawstyle='steps-mid', label="Hit Rate")

        for start_s, end_s in bunch_edges_s:
            if start_s < plot_end_s and end_s > plot_start_s:
                plt.axvspan(
                    max(start_s, plot_start_s), min(end_s, plot_end_s), 
                    color='g', alpha=0.3, 
                    label='Detected Bunch' if 'Detected Bunch' not in plt.gca().get_legend_handles_labels()[1] else ""
                )
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Hit Count per 0.1s Bin", fontsize=12)
        plt.title("Major Bunch Cut Verification", fontsize=14)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(plot_start_s, plot_end_s)
        plt.tight_layout()
        print("Displaying major bunch plot...")
        plt.show()

    # --- Plot 4: Minor Bunch Verification ---
    print("Generating per-bunch minor bunch verification plot...")
    num_bunches_to_plot = min(3, len(bunch_edges_s))
    if num_bunches_to_plot > 0:
        fig, axes = plt.subplots(
            num_bunches_to_plot, 1, 
            figsize=(15, 3 * num_bunches_to_plot), 
            squeeze=False
        )
        for i in range(num_bunches_to_plot):
            ax = axes[i, 0]
            bunch_start_s, bunch_end_s = bunch_edges_s[i]
            
            try:
                params = all_optimized_params[i]
                minor_period_s = params['period']
                minor_bunch_offset_s = params['phase']
            except IndexError:
                ax.text(0.5, 0.5, "Error: Missing params", ha='center', va='center')
                ax.set_title(f"Major Bunch {i+1} (Parameter Error)")
                continue
                
            plot_start_s = bunch_start_s
            plot_end_s = min(bunch_start_s + plot_window_s_minor, bunch_end_s)
            
            window_mask = (all_timestamps_s >= plot_start_s) & (all_timestamps_s < plot_end_s)
            timestamps_in_window = all_timestamps_s[window_mask]

            if len(timestamps_in_window) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"Major Bunch {i+1} (No Data)")
                continue

            ax.plot(timestamps_in_window, np.ones_like(timestamps_in_window), 
                    '|', markersize=10, label="Hits")
            
            t0_minor_s = bunch_start_s + minor_bunch_offset_s
            cycles_since_t0 = (plot_start_s - t0_minor_s) // minor_period_s
            first_cycle_start_s = t0_minor_s + (cycles_since_t0 * minor_period_s)
            
            current_gap_start_s = first_cycle_start_s + minor_bunch_duration_s
            while current_gap_start_s < plot_end_s:
                gap_end_s = current_gap_start_s + (minor_period_s - minor_bunch_duration_s)
                if current_gap_start_s < plot_end_s and gap_end_s > plot_start_s:
                    ax.axvspan(
                        max(current_gap_start_s, plot_start_s),
                        min(gap_end_s, plot_end_s),
                        color='r', alpha=0.3,
                        label='Cut (Outside)' if 'Cut (Outside)' not in ax.get_legend_handles_labels()[1] else ""
                    )
                current_gap_start_s += minor_period_s

            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_yticks([])
            title = (
                f"Minor Bunch Cut Verification (Major Bunch {i+1})\n"
                f"Optimized: Freq={1/minor_period_s:.3f} Hz, "
                f"Phase={minor_bunch_offset_s:.4f} s"
            )
            ax.set_title(title, fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
            ax.set_xlim(plot_start_s, plot_end_s)
            
        fig.suptitle("Minor Bunch Verification Plots", fontsize=16, y=1.02)
        plt.tight_layout()
        print("Displaying minor bunch plot...")
        plt.show()

    # --- Plot 5: Print Hit Frequencies ---
    freq_data = _print_hit_frequencies(
        bunch_edges_s,
        all_optimized_params,
        minor_bunch_duration_s,
        all_layers,
        all_inside_masks
    )

    # --- Plot 6: ToT vs. Phase ---
    plot_tot_vs_phase_heatmap_by_layer(
        all_phase_folded_times,
        all_tots,
        all_layers,
        base_period_s,
        minor_bunch_duration_s,         
        freq_data,                      
        normalize_y=normalize_tot_heatmaps, 
        cmap=heatmap_cmap                
    )

# --- Plot 7: Bunch Frequency Distribution ---
    _plot_bunch_frequency_distribution(
        all_bunch_freq_inside,
        all_bunch_layers_inside,
        all_bunch_hits_inside,
        all_bunch_freq_outside,
        all_bunch_layers_outside,
        all_bunch_hits_outside,
        dist_num_bins_in=dist_num_bins_in,
        dist_num_bins_out=dist_num_bins_out,
        log_y=dist_log_y,
        kde_bw_bins=kde_bw_bins  # <-- Pass through
    )
    
    # --- Plot 8: Bunch ToT Ratio Distribution ---
    _plot_bunch_tot_ratio_distribution(
        all_bunch_ratio_inside,
        all_bunch_layers_inside,
        all_bunch_hits_inside,
        all_bunch_ratio_outside,
        all_bunch_layers_outside,
        all_bunch_hits_outside,
        bunch_stat_tot_threshold,
        dist_num_bins_in=dist_num_bins_in,
        dist_num_bins_out=dist_num_bins_out,
        log_y=dist_log_y,
        kde_bw_bins=kde_bw_bins  # <-- Pass through
    )


def process_hits_with_threshold(
    dataset: Dataset,
    major_bunch_bin_s: float = 1.0,
    major_bunch_threshold_count: int = 4000,
    minor_bunch_freq_hz: float = 12.5,
    minor_bunch_duration_s: float = 0.07,
    minor_freq_scan_range_hz: float = 0.1,
    minor_freq_scan_step_hz: float = 0.001,
    minor_phase_scan_step_s: float = 0.001,
    minor_phase_scan_window_s: float = 0.01,
    trigger_ts_unit_seconds: float = 25e-9,
    normalize_tot_heatmaps: bool = False, 
    heatmap_cmap: str = 'plasma',
    bunch_stat_tot_threshold: float = 10.0,
    dist_log_y: bool = True,
    dist_num_bins_in: int = 100,
    dist_num_bins_out: int = 50,
    kde_bw_bins: float = 3.0
) -> Tuple[Dataset, Dataset]:
    """
    Sorts hits... (docstring as before) ...
    Args:
        ... (args as before) ...
        dist_log_y (bool): If True, sets the y-axis of distribution plots to log scale.
        dist_num_bins_in (int): The number of bins for the 'inside' distribution plots.
        dist_num_bins_out (int): The number of bins for the 'outside' distribution plots.
        kde_bw_bins (float): The KDE smoothing window size, expressed as a
                             number of histogram bins. Defaults to 3.
    """
    
    print("--- Starting Hit Processing Pipeline ---")
    
    # --- 1. Setup and Time Conversion ---
    try:
        all_timestamps_ts = dataset['TriggerTS']
        all_tots_global = dataset['ToT']
        all_layers_global = dataset['Layer']
    except KeyError as e:
        print(f"Error: Required key '{e}' not found in dataset.")
        return {}, {}

    all_timestamps_s = all_timestamps_ts * trigger_ts_unit_seconds
    
    # --- 2. Find Major Bunch Edges ---
    bunch_edges_s, hist_data = find_major_bunch_edges(
        all_timestamps_s,
        major_bunch_bin_s,
        major_bunch_threshold_count
    )
    base_period_s = 1.0 / minor_bunch_freq_hz
    
    if not bunch_edges_s:
        print("No major bunches were detected. Returning empty dicts.")
        _plot_diagnostic_charts(
            hist_data, major_bunch_threshold_count, bunch_edges_s,
            np.array([]), np.array([]), np.array([]), np.array([]),
            base_period_s, minor_bunch_duration_s,
            all_timestamps_s, [], 0, 0,
            normalize_tot_heatmaps=normalize_tot_heatmaps,
            heatmap_cmap=heatmap_cmap,
            bunch_stat_tot_threshold=bunch_stat_tot_threshold,
            all_bunch_freq_inside=np.array([]),
            all_bunch_freq_outside=np.array([]),
            all_bunch_layers_inside=np.array([]),
            all_bunch_layers_outside=np.array([]),
            all_bunch_ratio_inside=np.array([]),
            all_bunch_ratio_outside=np.array([]),
            all_bunch_hits_inside=np.array([]),
            all_bunch_hits_outside=np.array([]),
            dist_num_bins_in=dist_num_bins_in,
            dist_num_bins_out=dist_num_bins_out,
            dist_log_y=dist_log_y,
            kde_bw_bins=kde_bw_bins
        )
        return {}, {}
        
    # --- 3. Process Minor Bunches (Per-Bunch Loop) ---
    print("\n--- Processing Minor Bunches (Per-Bunch Optimization) ---")
    
    all_inside_indices: List[np.ndarray] = []
    all_outside_indices: List[np.ndarray] = []
    all_phase_folded_times: List[np.ndarray] = []
    all_phase_folded_tots: List[np.ndarray] = []
    all_phase_folded_layers: List[np.ndarray] = []
    all_phase_folded_inside_masks: List[np.ndarray] = []
    
    all_minor_bunch_freq_inside: List[np.ndarray] = []
    all_minor_bunch_freq_outside: List[np.ndarray] = []
    all_minor_bunch_ratio_inside: List[np.ndarray] = []
    all_minor_bunch_ratio_outside: List[np.ndarray] = []
    all_minor_bunch_layers_inside: List[np.ndarray] = []
    all_minor_bunch_layers_outside: List[np.ndarray] = []
    all_minor_bunch_hits_inside: List[np.ndarray] = []
    all_minor_bunch_hits_outside: List[np.ndarray] = []
    
    all_optimized_params: List[Dict] = []
    total_bunches = len(bunch_edges_s)
    
    for i, (start_s, end_s) in enumerate(bunch_edges_s):
        
        bunch_mask = (all_timestamps_s >= start_s) & (all_timestamps_s < end_s)
        (bunch_indices,) = np.where(bunch_mask)
        
        if len(bunch_indices) == 0:
            print(f"Skipping Bunch {i+1}/{total_bunches}: No hits found.")
            all_optimized_params.append({
                'freq': minor_bunch_freq_hz, 'phase': 0.0, 'period': 1.0/minor_bunch_freq_hz
            })
            continue
            
        bunch_timestamps_s = all_timestamps_s[bunch_indices]
        relative_timestamps_s = bunch_timestamps_s - start_s
        
        (opt_freq, 
         opt_phase, 
         opt_period, 
         score) = optimize_minor_bunch_parameters(
            relative_timestamps_s, minor_bunch_freq_hz, minor_freq_scan_range_hz,
            minor_freq_scan_step_hz, minor_phase_scan_step_s,
            minor_phase_scan_window_s, minor_bunch_duration_s,
            i, total_bunches
        )
        
        all_optimized_params.append({
            'freq': opt_freq, 'phase': opt_phase, 'period': opt_period
        })
        
        # --- REFACTORED: Per-Minor-Bunch Stat Calculation ---
        bunch_layers = all_layers_global[bunch_indices]
        bunch_tots = all_tots_global[bunch_indices]
        time_since_minor_t0_s = relative_timestamps_s - opt_phase
        time_in_minor_cycle_s = time_since_minor_t0_s % opt_period
        cycle_number = (time_since_minor_t0_s // opt_period).astype(int)
        mask_inside_this_bunch = (time_in_minor_cycle_s < minor_bunch_duration_s)
        
        duration_inside_s = minor_bunch_duration_s
        duration_outside_s = opt_period - minor_bunch_duration_s
        
        unique_cycles = np.unique(cycle_number)
        
        if duration_inside_s > 0 and duration_outside_s > 0:
            # Create a DataFrame for efficient grouping
            df = pd.DataFrame({
                'layer': bunch_layers,
                'cycle': cycle_number,
                'is_inside': mask_inside_this_bunch,
                'is_high_tot': bunch_tots > bunch_stat_tot_threshold
            })
            
            # Create a multi-index for all combinations
            # Replicates old logic: for *every* cycle, for *every* layer [1,2,3,4]
            all_combos_index = pd.MultiIndex.from_product(
                [[1, 2, 3, 4], unique_cycles],
                names=['layer', 'cycle']
            )

            # Group and count
            grouped = df.groupby(['layer', 'cycle', 'is_inside'])
            counts = grouped.size()
            high_tots = grouped['is_high_tot'].sum()

            # --- FIX: Unstack the 'is_inside' level ---
            counts_unstacked = counts.unstack(level='is_inside', fill_value=0)
            high_tots_unstacked = high_tots.unstack(level='is_inside', fill_value=0)

            # Reindex to add all missing (layer, cycle) combos
            counts_reindexed = counts_unstacked.reindex(all_combos_index, fill_value=0)
            high_tots_reindexed = high_tots_unstacked.reindex(all_combos_index, fill_value=0)

# --- Inside Stats ---
            stats_in = pd.DataFrame({
                'count': counts_reindexed.get(True, 0), 
                'high_tot': high_tots_reindexed.get(True, 0)
            }, index=counts_reindexed.index) 
            
            stats_in['low_tot'] = stats_in['count'] - stats_in['high_tot']
            stats_in['freq'] = stats_in['count'] / duration_inside_s
            
            # --- CORRECTED RATIO LOGIC (Inside) ---
            stats_in['ratio'] = 0.0 # <--- FIX: Initialize column
            stats_in.loc[stats_in['high_tot'] > 0, 'ratio'] = stats_in['low_tot'] / stats_in['high_tot']
            stats_in.loc[(stats_in['high_tot'] == 0) & (stats_in['low_tot'] > 0), 'ratio'] = np.inf
            # --- END CORRECTED RATIO LOGIC ---
            
            # --- Outside Stats ---
            stats_out = pd.DataFrame({
                'count': counts_reindexed.get(False, 0),
                'high_tot': high_tots_reindexed.get(False, 0)
            }, index=counts_reindexed.index)

            stats_out['low_tot'] = stats_out['count'] - stats_out['high_tot']
            stats_out['freq'] = stats_out['count'] / duration_outside_s

            # --- CORRECTED RATIO LOGIC (Outside) ---
            stats_out['ratio'] = 0.0 # <--- FIX: Initialize column
            stats_out.loc[stats_out['high_tot'] > 0, 'ratio'] = stats_out['low_tot'] / stats_out['high_tot']
            stats_out.loc[(stats_out['high_tot'] == 0) & (stats_out['low_tot'] > 0), 'ratio'] = np.inf
            # --- END CORRECTED RATIO LOGIC ---
            
            # --- Append to global lists ---
            all_minor_bunch_freq_inside.append(stats_in['freq'].values)
            all_minor_bunch_ratio_inside.append(stats_in['ratio'].values) # This line now works
            all_minor_bunch_layers_inside.append(stats_in.index.get_level_values('layer').values)
            all_minor_bunch_hits_inside.append(stats_in['count'].values)
            
            all_minor_bunch_freq_outside.append(stats_out['freq'].values)
            all_minor_bunch_ratio_outside.append(stats_out['ratio'].values)
            all_minor_bunch_layers_outside.append(stats_out.index.get_level_values('layer').values)
            all_minor_bunch_hits_outside.append(stats_out['count'].values)

        # (Store global indices - unchanged)
        all_inside_indices.append(bunch_indices[mask_inside_this_bunch])
        all_outside_indices.append(bunch_indices[~mask_inside_this_bunch])

        # (Store phase-folded data - unchanged)
        all_phase_folded_times.append(time_in_minor_cycle_s)
        all_phase_folded_tots.append(all_tots_global[bunch_indices])
        all_phase_folded_layers.append(all_layers_global[bunch_indices])
        all_phase_folded_inside_masks.append(mask_inside_this_bunch)


    print("\n--- Per-Bunch Optimization Complete ---")

    # --- 4. Aggregate Results and Generate Plots ---
    
    final_phase_folded_times = (np.concatenate(all_phase_folded_times) if all_phase_folded_times else np.array([]))
    final_phase_folded_tots = (np.concatenate(all_phase_folded_tots) if all_phase_folded_tots else np.array([]))
    final_phase_folded_layers = (np.concatenate(all_phase_folded_layers) if all_phase_folded_layers else np.array([]))
    final_phase_folded_inside_masks = (np.concatenate(all_phase_folded_inside_masks) if all_phase_folded_inside_masks else np.array([]))
    
    final_bunch_freq_inside = (np.concatenate(all_minor_bunch_freq_inside) if all_minor_bunch_freq_inside else np.array([]))
    final_bunch_freq_outside = (np.concatenate(all_minor_bunch_freq_outside) if all_minor_bunch_freq_outside else np.array([]))
    final_bunch_layers_inside = (np.concatenate(all_minor_bunch_layers_inside) if all_minor_bunch_layers_inside else np.array([]))
    final_bunch_layers_outside = (np.concatenate(all_minor_bunch_layers_outside) if all_minor_bunch_layers_outside else np.array([]))
    final_bunch_ratio_inside = (np.concatenate(all_minor_bunch_ratio_inside) if all_minor_bunch_ratio_inside else np.array([]))
    final_bunch_ratio_outside = (np.concatenate(all_minor_bunch_ratio_outside) if all_minor_bunch_ratio_outside else np.array([]))
    final_bunch_hits_inside = (np.concatenate(all_minor_bunch_hits_inside) if all_minor_bunch_hits_inside else np.array([]))
    final_bunch_hits_outside = (np.concatenate(all_minor_bunch_hits_outside) if all_minor_bunch_hits_outside else np.array([]))
    
    # Call diagnostic charts with updated args
    _plot_diagnostic_charts(
        hist_data,
        major_bunch_threshold_count,
        bunch_edges_s,
        final_phase_folded_times,     
        final_phase_folded_tots,      
        final_phase_folded_layers,    
        final_phase_folded_inside_masks, 
        base_period_s,
        minor_bunch_duration_s,
        all_timestamps_s,
        all_optimized_params,
        plot_window_s_major=bunch_edges_s[0][1] - bunch_edges_s[0][0] + 50.0,
        plot_window_s_minor=base_period_s * 10,
        normalize_tot_heatmaps=normalize_tot_heatmaps,
        heatmap_cmap=heatmap_cmap,
        bunch_stat_tot_threshold=bunch_stat_tot_threshold,
        all_bunch_freq_inside=final_bunch_freq_inside,
        all_bunch_freq_outside=final_bunch_freq_outside,
        all_bunch_layers_inside=final_bunch_layers_inside,
        all_bunch_layers_outside=final_bunch_layers_outside,
        all_bunch_ratio_inside=final_bunch_ratio_inside,
        all_bunch_ratio_outside=final_bunch_ratio_outside,
        all_bunch_hits_inside=final_bunch_hits_inside,
        all_bunch_hits_outside=final_bunch_hits_outside,
        dist_num_bins_in=dist_num_bins_in,
        dist_num_bins_out=dist_num_bins_out,
        dist_log_y=dist_log_y,
        kde_bw_bins=kde_bw_bins
    )

    # --- 5. Build the Output Dictionaries ---
    hits_inside, hits_outside = _build_output_dicts(
        dataset,
        all_inside_indices,
        all_outside_indices,
        all_timestamps_s
    )

    print("--- Hit Processing Pipeline Finished ---")
    
    return hits_inside, hits_outside

def _get_plot_data(data, timestamp_key, layer):
    """
    Helper function to safely extract x (hits_per_ts) and y (ToT)
    data for a specific layer.
    """
    try:
        layer_mask = (data['Layer'] == layer)
        ts_data = data[timestamp_key][layer_mask]
        y_data = data['ToT'][layer_mask]
        
        if len(ts_data) == 0 or len(y_data) == 0:
            return None, None
            
        unique_ts, inverse_indices, counts = np.unique(
            ts_data, 
            return_counts=True, 
            return_inverse=True
        )
        x_data = counts[inverse_indices]
        
        if x_data.size == 0 or y_data.size == 0:
            return None, None
            
        return x_data, y_data
        
    except KeyError as e:
        print(f"Error: Missing expected key {e} in data dictionary.")
        return None, None
    except Exception as e:
        print(f"An error occurred during data processing for layer {layer}: {e}")
        return None, None

def _normalize_hist(H, normalize_columns):
    """
    Helper function to normalize a 2D histogram.
    """
    if normalize_columns:
        column_sums = H.sum(axis=1)
        column_sums_safe = np.where(column_sums == 0, 1, column_sums)
        plot_data = H / column_sums_safe[:, np.newaxis]
    else:
        plot_data = H
    return plot_data


def _plot_joint_heatmap_panel(
    ax_hm,
    x_data: Optional[np.ndarray],
    y_data: Optional[np.ndarray],
    plot_data: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    norm,
    title: str,
    xlabel: str,
    tot_threshold: Optional[float],
    freq_text: str,
    cmap: str = 'plasma'
):
    """
    REFACTORED HELPER: Plots a single joint heatmap panel
    (heatmap + marginals) for compare_tot_vs_hits_plots_by_layer.
    """
    divider = make_axes_locatable(ax_hm)
    ax_histx = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_hm)
    ax_histy = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_hm)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.set_visible(False) # Hide by default, show for 'out' panel

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    has_data = (x_data is not None) and (y_data is not None)
    
    im = None
    if has_data:
        im = ax_hm.pcolormesh(
            xedges, yedges, plot_data.T,
            cmap=cmap, norm=norm, shading='auto'
        )
        ax_histx.hist(x_data, bins=xedges, histtype='step', color='black', linewidth=1.5)
        ax_histy.hist(y_data, bins=yedges, orientation='horizontal', histtype='step', color='black', linewidth=1.5)
        
        if tot_threshold is not None:
            ax_hm.axhline(y=tot_threshold, color='r', linestyle='--', linewidth=1.5, label=f'ToT Thresh ({tot_threshold})')
            ax_hm.legend(loc='upper right', fontsize='small')
            if freq_text:
                ax_hm.text(0.95, 0.05, freq_text, transform=ax_hm.transAxes,
                           color='white', ha='right', va='bottom', fontsize=12,
                           bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    else:
        ax_hm.text(0.5, 0.5, "No Data", transform=ax_hm.transAxes, ha='center', va='center')

    ax_hm.set_xlabel(xlabel)
    ax_histx.set_title(title)
    ax_histx.set_ylabel('Counts')
    ax_histy.set_xlabel('Counts')
    
    return im, cax, ax_histx, ax_histy


def compare_tot_vs_hits_plots_by_layer(
    data_in, 
    data_out, 
    timestamp_key='ext_TS', 
    max_hits_to_plot: Optional[int] = None, 
    normalize_columns: bool = True, 
    log_z_scale: bool = True,
    log_marginals: bool = False,
    tot_threshold: Optional[float] = None
):
    """
    REFACTORED: Compares ToT vs. Hits-per-Timestamp heatmaps... (as before)
    """
    
    print("--- Processing Data for Layered Comparison Plot ---")
    
    ts_to_seconds = 25e-9 
    
    # --- 1.A Calculate Total Duration (Global) ---
    total_duration_sec = 0.0
    global_min_ts = np.inf
    global_max_ts = -np.inf
    try:
        if data_in and timestamp_key in data_in and len(data_in[timestamp_key]) > 0:
            global_min_ts = min(global_min_ts, data_in[timestamp_key].min())
            global_max_ts = max(global_max_ts, data_in[timestamp_key].max())
        if data_out and timestamp_key in data_out and len(data_out[timestamp_key]) > 0:
            global_min_ts = min(global_min_ts, data_out[timestamp_key].min())
            global_max_ts = max(global_max_ts, data_out[timestamp_key].max())
        if np.isfinite(global_min_ts) and np.isfinite(global_max_ts):
            total_duration_ts = global_max_ts - global_min_ts
            if total_duration_ts > 0:
                total_duration_sec = total_duration_ts * ts_to_seconds
                print(f"Total experiment duration: {total_duration_sec:.2f} seconds")
        if total_duration_sec == 0:
            print("Warning: Could not determine valid time duration. Frequencies will be 0.")
    except Exception as e:
        print(f"Warning: Error calculating total duration: {e}. Frequencies will be 0.")

    # --- 1.B Find all layers ---
    layers_in = np.unique(data_in.get('Layer', []))
    layers_out = np.unique(data_out.get('Layer', []))
    all_layers = np.unique(np.concatenate((layers_in, layers_out))).astype(int)
    if len(all_layers) == 0:
        print("Error: No 'Layer' key found in either dataset.")
        return

    # --- 2. PRE-LOOP: Find global maxes and cache data ---
    print("Finding global axis limits...")
    global_x_max_count = 1
    global_y_max = 0
    layer_data_cache = {}
    for layer in all_layers:
        if layer == 0: continue
        x_in, y_in = _get_plot_data(data_in, timestamp_key, layer)
        x_out, y_out = _get_plot_data(data_out, timestamp_key, layer)
        has_in = x_in is not None
        has_out = x_out is not None
        layer_data_cache[layer] = {
            'in': (x_in, y_in), 'out': (x_out, y_out), 'has_data': has_in or has_out
        }
        if has_in:
            global_x_max_count = max(global_x_max_count, int(x_in.max()))
            global_y_max = max(global_y_max, int(y_in.max()))
        if has_out:
            global_x_max_count = max(global_x_max_count, int(x_out.max()))
            global_y_max = max(global_y_max, int(y_out.max()))

    # --- 3. Calculate Global Bins, Limits (ONCE) ---
    
    # Global Y-Bins (always linear)
    bins_y = np.arange(0, global_y_max + 2)
    yedges = bins_y
    
    # Global X-Bins (always linear)
    bins_x = np.arange(0.5, global_x_max_count + 1.5) 
    xedges = bins_x
    
    final_plot_limit_count = global_x_max_count
    if max_hits_to_plot:
        final_plot_limit_count = min(global_x_max_count, max_hits_to_plot)

    print(f"Global X-limit set to {final_plot_limit_count} hits. Global Y-limit set to {global_y_max} ToT.")

    # --- 4. MAIN PLOTTING LOOP ---
    for layer, data in layer_data_cache.items():
        
        print(f"\n--- Processing Layer {layer} ---")
        
        if not data['has_data']:
            print(f"Skipping Layer {layer}: No data found.")
            continue
            
        x_data_in, y_data_in = data['in']
        x_data_out, y_data_out = data['out']
        has_data_in = x_data_in is not None
        has_data_out = x_data_out is not None

        # --- 4.A Calculate Frequencies for text box ---
        freq_in = 0.0
        freq_out = 0.0
        freq_text_in = ""
        freq_text_out = ""
        if tot_threshold is not None and total_duration_sec > 0:
            if has_data_in:
                count_in = np.sum(y_data_in < tot_threshold)
                freq_in = count_in / total_duration_sec
                freq_text_in = f"Freq (ToT < {tot_threshold}): {freq_in:.1f} Hz"
            if has_data_out:
                count_out = np.sum(y_data_out < tot_threshold)
                freq_out = count_out / total_duration_sec
                freq_text_out = f"Freq (ToT < {tot_threshold}): {freq_out:.1f} Hz"
            print(f"  Layer {layer} Freq (ToT < {tot_threshold}): In={freq_in:.2f} Hz, Out={freq_out:.2f} Hz")

        # --- 4.B Create 2D Histograms (using global bins) ---
        if has_data_in:
            H_in, _, _ = np.histogram2d(x_data_in, y_data_in, bins=(bins_x, bins_y))
            plot_data_in = _normalize_hist(H_in, normalize_columns)
        else:
            plot_data_in = np.zeros((len(bins_x) - 1, len(bins_y) - 1))
        if has_data_out:
            H_out, _, _ = np.histogram2d(x_data_out, y_data_out, bins=(bins_x, bins_y))
            plot_data_out = _normalize_hist(H_out, normalize_columns)
        else:
            plot_data_out = np.zeros((len(bins_x) - 1, len(bins_y) - 1))

        # --- 4.C Determine Color Scale (per-layer) ---
        if normalize_columns:
            v_max = 1.0
            min_in = plot_data_in[plot_data_in > 0].min() if has_data_in and plot_data_in.sum() > 0 else 1.0
            min_out = plot_data_out[plot_data_out > 0].min() if has_data_out and plot_data_out.sum() > 0 else 1.0
            v_min = min(min_in, min_out)
        else:
            v_max = max(plot_data_in.max(), plot_data_out.max())
            if v_max == 0: v_max = 1.0
            min_in = plot_data_in[plot_data_in > 0].min() if has_data_in and plot_data_in.sum() > 0 else v_max
            min_out = plot_data_out[plot_data_out > 0].min() if has_data_out and plot_data_out.sum() > 0 else v_max
            v_min = min(min_in, min_out)
        if log_z_scale:
            if v_min <= 0 or v_min >= v_max: v_min = 1e-9
            norm = LogNorm(vmin=v_min, vmax=v_max)
        else:
            norm = Normalize(vmin=0, vmax=v_max)

        # --- 4.D Create Plots (REFACTORED) ---
        print(f"Plotting heatmap for Layer {layer}...")
        fig, (ax_hm_in, ax_hm_out) = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
        title_suffix = f"(Norm. Cols, Log Z)" if normalize_columns and log_z_scale else \
                       f"(Norm. Cols)" if normalize_columns else \
                       f"(Raw Counts, Log Z)" if log_z_scale else \
                       f"(Raw Counts)"
        
        xlabel = f'Hits per {timestamp_key}'

        # --- Panel 1: In-Bunch ---
        im_in, _, ax_histx_in, ax_histy_in = _plot_joint_heatmap_panel(
            ax_hm=ax_hm_in,
            x_data=x_data_in,
            y_data=y_data_in,
            plot_data=plot_data_in,
            xedges=xedges,
            yedges=yedges,
            norm=norm,
            title=f'In-Bunch Data {title_suffix}',
            xlabel=xlabel,
            tot_threshold=tot_threshold,
            freq_text=freq_text_in
        )
        ax_hm_in.set_ylabel('ToT')

        # --- Panel 2: Out-of-Bunch (with Colorbar) ---
        im_out, cax_out, ax_histx_out, ax_histy_out = _plot_joint_heatmap_panel(
            ax_hm=ax_hm_out,
            x_data=x_data_out,
            y_data=y_data_out,
            plot_data=plot_data_out,
            xedges=xedges,
            yedges=yedges,
            norm=norm,
            title=f'Out-of-Bunch Data {title_suffix}',
            xlabel=xlabel,
            tot_threshold=tot_threshold,
            freq_text=freq_text_out
        )
        
        if im_out:
            cax_out.set_visible(True)
            cbar_label = 'Norm. Prob.' if normalize_columns else 'Counts'
            fig.colorbar(im_out, cax=cax_out, label=cbar_label)

        # --- 4.E Set Ticks and Limits (using GLOBAL values) ---
        
        # Set Y-Limit (Global)
        ax_hm_in.set_ylim(yedges[0], yedges[-1]) 
        
        # Determine X-Limits
        plot_limit_x_min = xedges[0] # 0.5
        plot_limit_x_max = final_plot_limit_count + 0.5 
        
        ax_hm_in.set_xlim(plot_limit_x_min, plot_limit_x_max)
        ax_hm_out.set_xlim(plot_limit_x_min, plot_limit_x_max)

        if log_marginals:
            ax_histx_in.set_yscale('log')
            ax_histx_out.set_yscale('log')
            ax_histy_in.set_xscale('log')
            ax_histy_out.set_xscale('log')
            
            if has_data_in:
                ax_histx_in.set_ylim(bottom=0.5)
                ax_histy_in.set_xlim(left=0.5)
            if has_data_out:
                ax_histx_out.set_ylim(bottom=0.5)
                ax_histy_out.set_xlim(left=0.5)

        # --- Set linear ticks ---
        if final_plot_limit_count <= 20: step = 1
        elif final_plot_limit_count <= 50: step = 4
        else: step = 10
        
        tick_labels_count = np.arange(1, final_plot_limit_count + 1, step)
        tick_locations_count = tick_labels_count 
        count_labels = [f"{c:.0f}" for c in tick_labels_count]
        
        ax_hm_in.set_xticks(tick_locations_count)
        ax_hm_in.set_xticklabels(count_labels)
        ax_hm_out.set_xticks(tick_locations_count)
        ax_hm_out.set_xticklabels(count_labels)
        
        ax_hm_in.grid(False)
        ax_hm_out.grid(False)
        
        fig.suptitle(f'Layer {layer}: ToT vs. Hits per {timestamp_key} Comparison', fontsize=20, y=1.02)
        fig.tight_layout()
        plt.show()