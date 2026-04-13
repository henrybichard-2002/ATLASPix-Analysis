import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # Added for peak finding

def plot_hits_vs_time_by_layer(data, x_range=None, log_y=True,
                             timestamp_key='TriggerTS', layers_to_plot=[4, 3, 2, 1],
                             max_bins=1024, show_fft=True):
    """
    Plots the distribution of all hits vs. time, separated by layer.
    Optionally generates a second plot with a Fourier analysis of the hit rate.

    Binning for time plot: Set to one bin per unique timestamp, unless the number
    of unique timestamps exceeds `max_bins`, in which case `max_bins` are used.
    
    Binning for FFT plot: The number of bins is set to the number of
    unique time values in the range.
    
    FFT Plot Features: The 3 most prominent peaks are labeled with their frequency.

    Args:
        data (dict): The input data dictionary containing timestamps and layers.
        x_range (tuple, optional): (xmin, xmax) to set the plot's x-axis range.
                                   If None, the full data range is used.
        log_y (bool): Whether to set the y-axis to a logarithmic scale.
        timestamp_key (str): The key in `data` for the timestamp array.
        layers_to_plot (list): A list of layer numbers to plot.
        max_bins (int): The maximum number of bins to use for the time plot
                        if the number of unique timestamps is too large.
        show_fft (bool): If True, generates a second plot with the FFT power spectrum.
    """
    
    # Define standard colors
    layer_colors = {4: 'red', 3: 'orange', 2: 'green', 1: 'blue'}
    
    # --- 1. Data Extraction and Validation ---
    print(f"Plotting data from key: '{timestamp_key}'")
    try:
        Trig_ts_raw = data[timestamp_key]
        layer_all_hits = data['Layer']
    except KeyError as e:
        print(f"Error: Data dictionary missing required key: {e}")
        return
    
    if len(Trig_ts_raw) != len(layer_all_hits):
        print(f"Error: Timestamp array (len {len(Trig_ts_raw)}) and "
              f"Layer array (len {len(layer_all_hits)}) mismatch.")
        return
        
    if len(Trig_ts_raw) == 0:
        print("Error: No data to plot.")
        return

    # --- 2. Time Conversion ---
    s_per_ts = 25 * 1e-9  # 1 TS = 25 ns
    raw_ts_time_s = Trig_ts_raw * s_per_ts
    
    t_min_data = 0.0
    t_max_data = raw_ts_time_s[-1] if len(raw_ts_time_s) > 0 else 0.0
    if t_min_data == t_max_data and t_max_data == 0.0:
         t_max_data = raw_ts_time_s[0] if len(raw_ts_time_s) > 0 else 1.0 
    
    # --- 3. Setup Plot and Bins ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine plot range
    if x_range:
        plot_t_min, plot_t_max = x_range
    else:
        plot_t_min, plot_t_max = t_min_data, t_max_data
        
    if plot_t_min == plot_t_max:
        print("Warning: Data has zero time range. Adjusting range slightly to plot.")
        plot_t_max = plot_t_min + 1.0 # Add 1 second
        
    # Filter data to the selected time range *before* finding unique values
    range_mask = (raw_ts_time_s >= plot_t_min) & (raw_ts_time_s <= plot_t_max)
    times_in_range = raw_ts_time_s[range_mask]
    
    num_unique = 0
    num_bins = 0

    if times_in_range.size == 0:
        print("Warning: No data in the selected x_range. Plotting empty axes.")
        bins = np.linspace(plot_t_min, plot_t_max, 2)
        num_bins = 1
        unique_times = np.array([])
        num_unique = 0
    else:
        unique_times = np.unique(times_in_range)
        num_unique = len(unique_times)
        
        if num_unique == 1:
            time_val = unique_times[0]
            half_width = (s_per_ts / 2.0) if s_per_ts > 0 else 1e-9
            bins = np.array([time_val - half_width, time_val + half_width])
            num_bins = 1
            
        elif 1 < num_unique <= max_bins:
            num_bins = num_unique
            midpoints = (unique_times[:-1] + unique_times[1:]) / 2.0
            first_delta = (unique_times[1] - unique_times[0]) / 2.0
            last_delta = (unique_times[-1] - unique_times[-2]) / 2.0
            
            half_ts = (s_per_ts / 2.0) if s_per_ts > 0 else 1e-9
            if first_delta == 0: first_delta = half_ts
            if last_delta == 0: last_delta = half_ts
                
            first_edge = unique_times[0] - first_delta
            last_edge = unique_times[-1] + last_delta
            
            bins = np.concatenate(([first_edge], midpoints, [last_edge]))

        else: # num_unique > max_bins or num_unique == 0
            print(f"Info: {num_unique} unique times > max_bins ({max_bins}). Using {max_bins} linear bins.")
            num_bins = max_bins
            bins = np.linspace(plot_t_min, plot_t_max, num_bins + 1)
            
    # --- 4. Plotting ---
    
    ax.hist(raw_ts_time_s, bins=bins, histtype='step', 
            label='All Hits', color='black', linewidth=2)
    
    for layer in layers_to_plot:
        layer_mask = (layer_all_hits == layer)
        if np.any(layer_mask):
            ax.hist(raw_ts_time_s[layer_mask], bins=bins,
                    histtype='step', label=f'Layer {layer}',
                    color=layer_colors.get(layer, 'gray'), linewidth=1.5)

    # --- 5. Formatting ---
    ax.set_xlabel('Time (s) [Relative to first hit]')
    
    if num_bins == num_unique and num_unique > 1:
        ax.set_ylabel('Hits per Unique Time Bin')
    else:
        avg_bin_width_s = (plot_t_max - plot_t_min) / max(1, num_bins)
        ax.set_ylabel(f'Hits per Bin ({avg_bin_width_s:.2e} s)')
        
    ax.set_title('Hits vs. Time by Layer')
    
    if log_y:
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5)
        
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(plot_t_min, plot_t_max)
    fig.tight_layout()
    plt.show()

    # --- 6. Fourier Analysis Plot (Optional) ---
    if show_fft:
        print("Generating Fourier analysis plot...")
        
        # We must use a uniformly binned histogram for FFT.
        # N_fft is set to the number of unique time values in the range.
        
        if num_unique > 1:
            N_fft = num_unique
            print(f"Info: Using {N_fft} bins for Fourier analysis.")
            
            bin_width_s_fft = (plot_t_max - plot_t_min) / N_fft
            
            if bin_width_s_fft == 0:
                print("Error: Cannot perform FFT with zero time range.")
                return

            bins_fft = np.linspace(plot_t_min, plot_t_max, N_fft + 1)
            hist_fft, _ = np.histogram(times_in_range, bins=bins_fft)
            
            N = len(hist_fft)
            yf = np.fft.fft(hist_fft)[1:] # Remove DC component
            xf = np.fft.fftfreq(N, bin_width_s_fft)[1:] # Remove DC component
            
            power = np.abs(yf)**2
            
            positive_mask = xf > 0
            xf_pos = xf[positive_mask]
            power_pos = power[positive_mask]
            
            if xf_pos.size > 0:
                fig_fft, ax_fft = plt.subplots(figsize=(12, 6))
                
                ax_fft.plot(xf_pos, power_pos, color='blue', linewidth=1)
                
                # --- Peak Finding and Labeling ---
                try:
                    # Find all peaks with a prominence > 1/2 std dev of the power
                    min_prominence = np.std(power_pos) / 2.0
                    peaks, properties = find_peaks(power_pos, prominence=min_prominence)
                    
                    if len(peaks) > 0:
                        prominences = properties['prominences']
                        
                        # Get indices of the top 3 most prominent peaks
                        num_to_label = min(3, len(peaks))
                        top_indices_sorted = np.argsort(prominences)[-num_to_label:]
                        top_peak_indices = peaks[top_indices_sorted]
                        
                        top_freqs = xf_pos[top_peak_indices]
                        top_powers = power_pos[top_peak_indices]
                        
                        print(f"Top {num_to_label} peaks found:")
                        for i, (f, p) in enumerate(zip(top_freqs, top_powers)):
                            print(f"  {i+1}. Freq={f:.3f} Hz, Power={p:.2e}")
                            ax_fft.annotate(f'{f:.2f} Hz',
                                            xy=(f, p),
                                            xytext=(f +1, p * 1.5), # Position text above peak
                                            ha='right',
                                            arrowprops=dict(facecolor='black', shrink=0.05,
                                                            width=1, headwidth=4,
                                                            connectionstyle='arc3,rad=0.1'))
                    else:
                        print("No prominent peaks found.")
                        
                except Exception as e:
                    print(f"Error during peak finding: {e}")
                # --- End Peak Finding ---

                ax_fft.set_xlabel('Frequency (Hz)')
                ax_fft.set_ylabel('Power (a.u.)')
                ax_fft.set_title(f'Fourier Analysis of Hit Rate (Range: {plot_t_min:.2f}s to {plot_t_max:.2f}s)')
                
                ax_fft.set_yscale('log')
                ax_fft.set_xscale('log')
                
                ax_fft.grid(True, which='both', linestyle='--', alpha=0.6)
                fig_fft.tight_layout()
                plt.show()
            else:
                print("Warning: No positive frequencies to plot for FFT.")
        
        else:
            print("Warning: Not enough unique data points in range to perform FFT.")

def _bunch_contrast_cost_v2(params, *args):
    """
    Cost function (Version 2.1) for scipy.optimize.minimize.
    
    Now includes a bounds check for Nelder-Mead.
    
    Calculates the ratio of (hit rate density in gap) / (hit rate density in bunch).
    The goal is to MINIMIZE this ratio.
    """
    period, start, width = params
    relative_time_s, hit_counts, t_start_analysis, t_end_analysis, bounds_list = args

    # --- 1. Validate Parameters ---
    
    # Check if params are within the user-defined bounds
    for val, (b_min, b_max) in zip(params, bounds_list):
        if not (b_min <= val <= b_max):
            return np.inf # Penalty for being out of bounds
            
    gap_width = period - width
    
    # The gap MUST exist (width < period).
    if gap_width <= 1e-9:
        return np.inf

    # --- 2. Select Data in Time Window ---
    if t_end_analysis is None:
        t_end_analysis = relative_time_s[-1]
        
    analysis_mask = (relative_time_s >= t_start_analysis) & (relative_time_s <= t_end_analysis)
    
    if not np.any(analysis_mask):
        return np.inf # No data in this time range

    analysis_times = relative_time_s[analysis_mask]
    analysis_hits = hit_counts[analysis_mask]
    
    total_time_window = t_end_analysis - t_start_analysis
    if total_time_window <= 0:
        return np.inf

    # --- 3. Calculate Phase and Masks ---
    analysis_phase = (analysis_times - start) % period
    in_bunch_mask = (analysis_phase < width)
    
    # --- 4. Calculate Hit Rate Densities ---
    total_hits_in_bunch = np.sum(analysis_hits[in_bunch_mask])
    # Sum hits NOT in the bunch mask
    total_hits_in_gap = np.sum(analysis_hits[~in_bunch_mask])

    # Calculate the *total time duration* of each region
    total_time_in_bunch = total_time_window * (width / period)
    total_time_in_gap = total_time_window * (gap_width / period)

    if total_time_in_bunch <= 0 or total_time_in_gap <= 0:
        return np.inf
        
    rate_in_bunch = total_hits_in_bunch / total_time_in_bunch
    rate_in_gap = total_hits_in_gap / total_time_in_gap

    # Avoid division by zero if bunch rate is 0 (bad fit)
    if rate_in_bunch == 0:
        return np.inf 

    # --- 5. Return Cost ---
    cost = rate_in_gap / rate_in_bunch
    
    return cost


def optimize_bunch_parameters(data, timestamp_key, 
                              initial_params, 
                              search_deltas,
                              bunch_start_time=0.0, bunch_end_time=None):
    """
    Optimizes bunch parameters (period, start, width) by maximizing
    the contrast between in-bunch and out-of-bunch hit *rate densities*.
    
    Uses the 'Nelder-Mead' algorithm, which is robust for non-smooth
    cost functions.
    
    Args:
        data (dict): Dictionary containing the timestamp data.
        timestamp_key (str): The key for the timestamp array.
        
        initial_params (dict): A dict with initial guesses:
            {'period': 0.08, 'start': 73.0, 'width': 0.05}
            
        search_deltas (dict): A dict with the +/- search range (delta):
            {'period': 0.001, 'start': 0.01, 'width': 0.005}
            
        bunch_start_time (float): The time (in seconds) to START the
                                  analysis window for the cost function.
        bunch_end_time (float, optional): The time (in seconds) to END
                                          the analysis window.
                                          
    Returns:
        dict: A dictionary with the optimized parameters:
              {'period': ..., 'start': ..., 'width': ...}
              Returns None if optimization fails.
    """
    
    # --- 1. Data Preparation ---
    print(f"Preparing data from key: '{timestamp_key}'...")
    try:
        Trig_ts_raw = data[timestamp_key]
    except (KeyError, TypeError):
        print(f"Error: '{timestamp_key}' key not found or data is invalid.")
        return None

    if len(Trig_ts_raw) == 0:
        print(f"Error: '{timestamp_key}' array is empty.")
        return None
        
    s_per_ts = 25 * 1e-9  # 1 TS = 25 ns

    unique_ts_raw, hit_counts = np.unique(Trig_ts_raw, return_counts=True)
    
    if hit_counts.size == 0 or unique_ts_raw.size == 0:
        print("Error: No hits found after np.unique.")
        return None
        
    unique_ts_time_s = unique_ts_raw * s_per_ts
    relative_time_s = unique_ts_time_s - np.min(unique_ts_time_s)
    
    # --- 2. Setup Optimization ---
    
    x0 = [
        initial_params['period'], 
        initial_params['start'], 
        initial_params['width']
    ]
    
    # Define bounds as a list of tuples
    bounds_list = [
        (initial_params['period'] - search_deltas['period'], initial_params['period'] + search_deltas['period']),
        (initial_params['start'] - search_deltas['start'], initial_params['start'] + search_deltas['start']),
        (initial_params['width'] - search_deltas['width'], initial_params['width'] + search_deltas['width'])
    ]
    
    # Ensure bounds are valid (e.g., no negative time)
    bounds_list = [(max(1e-9, b[0]), b[1]) for b in bounds_list]
    
    # Set the analysis window for the cost function
    t_start_analysis = bunch_start_time
    t_end_analysis = bunch_end_time
    if t_end_analysis is None:
        t_end_analysis = relative_time_s[-1]
        
    # Pass the bounds_list *into* the cost function
    cost_args = (relative_time_s, hit_counts, t_start_analysis, t_end_analysis, bounds_list)
    
    print("Starting optimization ('Nelder-Mead')... (this may take a moment)")
    print(f"  Initial guess: Period={x0[0]:.4f}, Start={x0[1]:.4f}, Width={x0[2]:.4f}")
    print(f"  Analysis window: [{t_start_analysis:.2f}s, {t_end_analysis:.2f}s]")
    print(f"  Search bounds:")
    print(f"    Period: [{bounds_list[0][0]:.4f}, {bounds_list[0][1]:.4f}]")
    print(f"    Start:  [{bounds_list[1][0]:.4f}, {bounds_list[1][1]:.4f}]")
    print(f"    Width:  [{bounds_list[2][0]:.4f}, {bounds_list[2][1]:.4f}]")

    # --- 3. Run Optimization ---
    try:
        result = minimize(
            _bunch_contrast_cost_v2,  # Using the V2 cost function
            x0,
            args=cost_args,
            method='Nelder-Mead',  # *** CHANGED METHOD ***
            options={
                'xatol': 1e-7,  # Tolerance on parameter values
                'fatol': 1e-7,  # Tolerance on cost value
                'adaptive': True # Good for high-dimension/rugged problems
            }
        )
        
        if not result.success:
            print(f"\nOptimization FAILED: {result.message}")
            return None
            
        # --- 4. Process Results ---
        opt_period, opt_start, opt_width = result.x
        
        if opt_period <= opt_width:
             print(f"\nOptimization FAILED: Final width ({opt_width:.4f}) is >= period ({opt_period:.4f}).")
             return None
        
        print("\nOptimization SUCCEEDED.")
        print(f"  Final cost (rate_Gap / rate_Bunch): {result.fun:.6f}")
        print(f"  Optimized Period: {opt_period:.6f}")
        print(f"  Optimized Start:  {opt_start:.6f}")
        print(f"  Optimized Width:  {opt_width:.6f}")
        
        return {
            'period': opt_period,
            'start': opt_start,
            'width': opt_width
        }

    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        return None



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

# --- Top-Level Helpers (from original file) ---

def _gaussian(x, amplitude, mean, stddev):
    """Defines a Gaussian function for fitting. (Helper)"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def _plot_frequency_distribution_with_fit(ax, data, title, color,
                                          fit_range, num_bins,
                                          fit_label_prefix=""):
    """
    Helper function to plot a frequency distribution and its
    optional Gaussian fit.
    """
    ax.set_title(title)
    if data.size == 0:
        ax.text(0.5, 0.5, 'No data in range (> 0 hits)',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return

    # Get histogram data for plotting AND fitting
    # Note: num_bins can be an integer OR an array of bin edges
    counts, bin_edges = np.histogram(data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot the histogram
    ax.hist(data, bins=bin_edges, histtype='step',
            align='mid', linewidth=2, label=fit_label_prefix,
            color=color)
    
    # --- Gaussian Fit ---
    if fit_range is not None:
        try:
            fit_mask = (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1])
            x_fit = bin_centers[fit_mask]
            y_fit = counts[fit_mask]

            if len(x_fit) < 3:
                print(f"\n--- {fit_label_prefix} Fit Warning: Not enough data points in range to fit.")
            else:
                # Provide initial guesses
                amp_guess = np.max(y_fit)
                mean_guess = x_fit[np.argmax(y_fit)]
                std_guess = np.std(data)
                p0 = [amp_guess, mean_guess, std_guess]
                
                popt, _ = curve_fit(_gaussian, x_fit, y_fit, p0=p0, maxfev=5000)
                
                amp, mean, std = popt
                std = np.abs(std) # Std dev must be positive
                fwhm = 2 * std * np.sqrt(2 * np.log(2))
                
                print(f"\n--- {fit_label_prefix} Fit Results (Frequency) ---")
                print(f"  Fit Range: {fit_range[0]} to {fit_range[1]} Hz")
                print(f"  Mean (μ): {mean:.2f} Hz")
                print(f"  FWHM: {fwhm:.2f} Hz")
                print(f"  Std Dev (σ): {std:.2f} Hz")
                
                # Plot the fit
                x_plot = np.linspace(np.min(x_fit), np.max(x_fit), 200)
                y_plot = _gaussian(x_plot, amp, mean, std)
                ax.plot(x_plot, y_plot, 'r--', label=f'Fit: μ={mean:.1f}, FWHM={fwhm:.1f} Hz')
    
        except Exception as e:
            print(f"\n--- {fit_label_prefix} Fit Failed ---")
            print(f"  Error: {e}")

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

def _aggregate_raw_frequencies(raw_times_in_range, t_max_in_range, 
                             bunch_start_time, period, bunch_width, gap_width):
    """Inner helper to loop and aggregate frequencies from RAW hit times."""
    freqs_in_each_bunch = []
    freqs_in_each_gap = []
    t_current_bunch_start = bunch_start_time

    if raw_times_in_range.size == 0:
        return np.array([]), np.array([])
        
    while t_current_bunch_start < t_max_in_range:
        t_bunch_end = t_current_bunch_start + bunch_width
        t_gap_end = t_current_bunch_start + period
        
        bunch_mask = (raw_times_in_range >= t_current_bunch_start) & (raw_times_in_range < t_bunch_end)
        total_hits_in_this_bunch = np.sum(bunch_mask)
        
        if total_hits_in_this_bunch > 0:
            freqs_in_each_bunch.append(total_hits_in_this_bunch / bunch_width)
        
        gap_mask = (raw_times_in_range >= t_bunch_end) & (raw_times_in_range < t_gap_end)
        total_hits_in_this_gap = np.sum(gap_mask)
        
        if total_hits_in_this_gap > 0:
            freqs_in_each_gap.append(total_hits_in_this_gap / gap_width)
        
        t_current_bunch_start += period
            
    return np.array(freqs_in_each_bunch), np.array(freqs_in_each_gap)


# --- New Modular Sub-Functions ---

def _prepare_analysis_data(data, timestamp_key, period, bunch_width, 
                           bunch_start_time, bunch_end_time):
    """
    Validates and processes raw data into time arrays and masks.
    Corresponds to original Sections 1 & 2.
    """
    print(f"Processing hit data from key: '{timestamp_key}'")
    try:
        Trig_ts_raw = data[timestamp_key]
    except (KeyError, TypeError):
        print(f"Error: '{timestamp_key}' key not found or data is invalid.")
        return None

    if len(Trig_ts_raw) == 0:
        print(f"Error: '{timestamp_key}' array is empty.")
        return None
        
    s_per_ts = 25 * 1e-9  # 1 TS = 25 ns
    reference_len = len(Trig_ts_raw)
    
    res = {
        'Trig_ts_raw': Trig_ts_raw,
        'reference_len': reference_len,
        's_per_ts': s_per_ts,
        'layers_to_plot': [4, 3, 2, 1],
        'layer_colors': {4: 'red', 3: 'orange', 2: 'green', 1: 'blue'},
        'has_tot': False,
        'tot_all_hits': None,
        'has_layer': False,
        'layer_all_hits': None,
        'bunch_start_time': bunch_start_time # <-- FIX: Store this for later use
    }

    # Check for ToT data
    if 'ToT' not in data:
        print("Warning: 'ToT' key not found in data. Skipping Plot 7.")
    elif len(data['ToT']) != reference_len:
        print(f"Warning: 'ToT' array length ({len(data['ToT'])}) does not match "
              f"timestamp array length ({reference_len}). Skipping Plot 7.")
    else:
        res['has_tot'] = True
        res['tot_all_hits'] = data['ToT']
        print("Found 'ToT' data for Plot 7.")
        
    # Check for Layer data
    if 'Layer' not in data:
        print("Warning: 'Layer' key not found in data. Skipping layer-based analysis.")
    elif len(data['Layer']) != reference_len:
        print(f"Warning: 'Layer' array length ({len(data['Layer'])}) does not match "
              f"timestamp array length ({reference_len}). Skipping layer-based analysis.")
    else:
        res['has_layer'] = True
        res['layer_all_hits'] = data['Layer']
        print(f"Found 'Layer' data for layer-based analysis on Layers: {res['layers_to_plot']}")

    # Calculate Raw Relative Time
    raw_ts_time_s = Trig_ts_raw * s_per_ts
    if raw_ts_time_s.size == 0: 
        print("Error: No timestamps found after processing.")
        return None
    res['min_time_s'] = np.min(raw_ts_time_s) 
    res['raw_ts_time_s'] = raw_ts_time_s - res['min_time_s']
    
    print(f"Data spans {res['raw_ts_time_s'][-1]:.2f} seconds.")

    # Calculate Masks & Time Windows
    res['stats_range_str'] = f">= {bunch_start_time:.2f}s"
    raw_post_start_mask = (res['raw_ts_time_s'] >= bunch_start_time)

    if bunch_end_time is not None:
        res['stats_range_str'] = f"between {bunch_start_time:.2f}s and {bunch_end_time:.2f}s"
        raw_pre_end_mask = (res['raw_ts_time_s'] <= bunch_end_time)
        print(f"Analysis window set for {res['stats_range_str']}.")
    else:
        raw_pre_end_mask = np.ones_like(res['raw_ts_time_s'], dtype=bool)
        print(f"Analysis window set for {res['stats_range_str']}.")

    res['raw_stats_time_mask'] = raw_post_start_mask & raw_pre_end_mask
    
    # Phase Masks
    res['raw_time_from_start'] = (res['raw_ts_time_s'] - bunch_start_time) % period
    res['raw_in_bunch_mask_phase'] = (res['raw_time_from_start'] < bunch_width)
    
    return res

def _filter_data_for_return(data, prep_data, timestamp_key):
    """
    Filters all data columns based on time and phase masks.
    Corresponds to original Section 3.
    """
    final_raw_in_bunch_mask = prep_data['raw_in_bunch_mask_phase'] & prep_data['raw_stats_time_mask']
    final_raw_out_bunch_mask = (~prep_data['raw_in_bunch_mask_phase']) & prep_data['raw_stats_time_mask']

    data_in_bunches = {}
    data_out_of_bunches = {}
    reference_len = prep_data['reference_len']

    print("\nFiltering all columns for return dictionaries...")
    for key, value in data.items():
        try:
            value_array = np.asarray(value)
            if value_array.shape and value_array.shape[0] == reference_len:
                data_in_bunches[key] = value_array[final_raw_in_bunch_mask]
                data_out_of_bunches[key] = value_array[final_raw_out_bunch_mask]
                if key == timestamp_key: print(f"  > Filtered timestamp key '{key}'.")
                else: print(f"  > Filtered aligned data key '{key}'.")
            else:
                print(f"  > Copying metadata/unaligned key '{key}' (shape {value_array.shape})")
                data_in_bunches[key] = value
                data_out_of_bunches[key] = value
        except Exception as e:
            print(f"  > Copying metadata key '{key}' (Error: {e})")
            data_in_bunches[key] = value
            data_out_of_bunches[key] = value
            
    return data_in_bunches, data_out_of_bunches

def _calculate_window_stats(prep_data, period, bunch_width):
    """
    Calculates all statistics for the defined time window.
    Corresponds to original Sections 4 & 5.
    """
    stats_data = {
        'times_for_stats': prep_data['raw_ts_time_s'][prep_data['raw_stats_time_mask']],
        'ts_in_window_raw': prep_data['Trig_ts_raw'][prep_data['raw_stats_time_mask']]
    }

    if stats_data['times_for_stats'].size == 0:
        print("\n--- Statistics ---")
        print(f"Warning: No data found in the specified range ({prep_data['stats_range_str']}).")
        print("Statistics, Plot 2, and Frequency Plots will be empty.")
        
        # Set all data to empty arrays
        stats_data.update({
            'layers_for_stats': np.array([]),
            'tot_for_stats': np.array([]),
            'phase_for_stats_raw': np.array([]),
            'hit_counts_in_window': np.array([]),
            'in_bunch_hits_for_stats': np.array([]),
            'out_bunch_hits_for_stats': np.array([]),
            'layer_hit_counts_stats': {}
        })
        
    else:
        # Filter aligned data
        stats_data['phase_for_stats_raw'] = prep_data['raw_time_from_start'][prep_data['raw_stats_time_mask']]
        if prep_data['has_layer']:
            stats_data['layers_for_stats'] = prep_data['layer_all_hits'][prep_data['raw_stats_time_mask']]
        else:
            stats_data['layers_for_stats'] = np.array([])
            
        if prep_data['has_tot']:
            stats_data['tot_for_stats'] = prep_data['tot_all_hits'][prep_data['raw_stats_time_mask']]
        else:
            stats_data['tot_for_stats'] = np.array([])
            
        # --- Data for Plot 2 & Stats ---
        print("\nCalculating hits-per-timestamp for analysis window...")
        
        # 1. For "All Layers"
        unique_ts_raw_in_window, hit_counts_in_window = np.unique(stats_data['ts_in_window_raw'], return_counts=True)
        stats_data['hit_counts_in_window'] = hit_counts_in_window
        
        # Need phase for these unique timestamps
        unique_ts_time_s_in_window = unique_ts_raw_in_window * prep_data['s_per_ts']
        unique_ts_relative_in_window = unique_ts_time_s_in_window - prep_data['min_time_s']
        time_from_start_in_window = (unique_ts_relative_in_window - prep_data['bunch_start_time']) % period
        in_bunch_mask_in_window = (time_from_start_in_window < bunch_width)
        
        stats_data['in_bunch_hits_for_stats'] = hit_counts_in_window[in_bunch_mask_in_window]
        stats_data['out_bunch_hits_for_stats'] = hit_counts_in_window[~in_bunch_mask_in_window]
        
        # 2. For each layer
        layer_hit_counts_stats = {}
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                layer_mask_in_window = (stats_data['layers_for_stats'] == layer)
                ts_for_layer = stats_data['ts_in_window_raw'][layer_mask_in_window]
                if ts_for_layer.size > 0:
                    _ , hit_counts_layer = np.unique(ts_for_layer, return_counts=True)
                    layer_hit_counts_stats[layer] = hit_counts_layer
                else:
                    layer_hit_counts_stats[layer] = np.array([])
        stats_data['layer_hit_counts_stats'] = layer_hit_counts_stats
        
        # --- Print Statistics ---
        total_hits_for_stats = np.sum(hit_counts_in_window)
        total_ts_for_stats = len(hit_counts_in_window)
    
        print(f"\n--- Timestamp Statistics (for data {prep_data['stats_range_str']}) ---")
        print(f"Total Timestamps (in range): {total_ts_for_stats}")
        
        in_ts_count = len(stats_data['in_bunch_hits_for_stats'])
        in_ts_perc = (in_ts_count / total_ts_for_stats * 100) if total_ts_for_stats > 0 else 0
        print(f"  In-Bunch Timestamps: {in_ts_count} ({in_ts_perc:.1f}%)")
        
        out_ts_count = len(stats_data['out_bunch_hits_for_stats'])
        out_ts_perc = (out_ts_count / total_ts_for_stats * 100) if total_ts_for_stats > 0 else 0
        print(f"  Out-of-Bunch Timestamps: {out_ts_count} ({out_ts_perc:.1f}%)")

        print(f"Total Hits (in range): {total_hits_for_stats}")
        in_hits_sum = np.sum(stats_data['in_bunch_hits_for_stats'])
        in_hits_perc = (in_hits_sum / total_hits_for_stats * 100) if total_hits_for_stats > 0 else 0
        print(f"  In-Bunch Hits: {in_hits_sum} ({in_hits_perc:.1f}%)")
        
        out_hits_sum = np.sum(stats_data['out_bunch_hits_for_stats'])
        out_hits_perc = (out_hits_sum / total_hits_for_stats * 100) if total_hits_for_stats > 0 else 0
        print(f"  Out-of-Bunch Hits: {out_hits_sum} ({out_hits_perc:.1f}%)")

    return stats_data

def _plot_p1_hits_vs_time(prep_data, time_bins, period, bunch_width, bunch_start_time, time_range):
    """Generates Plot 1: Hits vs. Time (Binned, with Layer Overlays)"""
    print("\nGenerating Plot 1: Hits vs. Time (Binned, with Layer Overlays)")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    raw_ts_time_s = prep_data['raw_ts_time_s']
    t_min, t_max = raw_ts_time_s[0], raw_ts_time_s[-1]
    bins_plot1 = np.linspace(t_min, t_max, time_bins + 1)
    
    ax1.hist(raw_ts_time_s, bins=bins_plot1, histtype='step', 
             label='All Hits', color='black', linewidth=2)
    
    if prep_data['has_layer']:
        for layer in prep_data['layers_to_plot']:
            layer_mask = (prep_data['layer_all_hits'] == layer)
            if np.any(layer_mask):
                ax1.hist(raw_ts_time_s[layer_mask], bins=bins_plot1,
                         histtype='step', label=f'Layer {layer}',
                         color=prep_data['layer_colors'].get(layer), linewidth=1)

    # Plot bunch regions
    first_bunch_index = np.floor((t_min - bunch_start_time) / period)
    t_current = first_bunch_index * period + bunch_start_time
    span_label = 'Specified Bunch Region'
    while t_current < t_max:
        start = max(t_min, t_current)
        end = min(t_max, t_current + bunch_width)
        if end > start:
            ax1.axvspan(start, end, color='gray', alpha=0.3, label=span_label)
            span_label = '_nolegend_'
        t_current += period
        
    ax1.set_xlabel('Time (s) [Relative to first hit]')
    ax1.set_ylabel(f'Number of Hits (per {((t_max-t_min)/time_bins):.2e} s bin)')
    ax1.set_title('Hits per Time Bin (with Specified Bunch Regions)')
    ax1.legend(loc='upper right') 
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    if time_range is not None:
        try:
            xmin_s, xmax_s = time_range
            ax1.set_xlim(xmin_s, xmax_s)
            ax1.set_title(f'Hits per Time Bin (Zoomed: {xmin_s}-{xmax_s} s)')
        except Exception as e:
            print(f"Warning: Could not apply time range to Plot 1. Error: {e}")
    fig1.tight_layout()
    return fig1, ax1

def _plot_p2_hits_per_ts(stats_data, prep_data):
    """Generates Plot 2: Distribution of Hits per Timestamp (with Layer Overlays)"""
    print(f"\nGenerating Plot 2: Distribution of Hits (In vs. Out) [data {prep_data['stats_range_str']}]")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_title_2 = f'Distribution of Hits per Timestamp (Data {prep_data['stats_range_str']})'
    
    hit_counts_in_window = stats_data['hit_counts_in_window']
    if hit_counts_in_window.size == 0:
        ax2.text(0.5, 0.5, f'No data found {prep_data['stats_range_str']}', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    else:
        # --- FIXED BINNING LOGIC ---
        # Find global min/max across ALL datasets for this plot
        all_data_collections = [hit_counts_in_window]
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_l = stats_data['layer_hit_counts_stats'].get(layer)
                if data_l is not None and data_l.size > 0:
                    all_data_collections.append(data_l)
        
        combined_data = np.concatenate(all_data_collections)
        min_hits = np.min(combined_data)
        max_hits = np.max(combined_data)
        # Create bins centered on integers
        bins_plot2 = np.arange(min_hits, max_hits + 2) - 0.5 
        # --- END FIXED BINNING LOGIC ---
        
        # Plot "All Layers" In vs Out
        if len(stats_data['in_bunch_hits_for_stats']) > 0:
            ax2.hist(stats_data['in_bunch_hits_for_stats'], bins=bins_plot2, histtype='step', 
                     align='mid', linewidth=2, label='All Layers (In-Bunch)',
                     color='blue')
        if len(stats_data['out_bunch_hits_for_stats']) > 0:
            ax2.hist(stats_data['out_bunch_hits_for_stats'], bins=bins_plot2, histtype='step', 
                     align='mid', linewidth=2, linestyle='--', 
                     label='All Layers (Out-of-Bunch)', color='cyan')
                     
        # Plot Layer overlays (total hits)
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_l = stats_data['layer_hit_counts_stats'].get(layer)
                if data_l is not None and data_l.size > 0:
                    ax2.hist(data_l, bins=bins_plot2, histtype='step',
                             align='mid', linewidth=1.5,
                             label=f'Layer {layer} (Total)',
                             color=prep_data['layer_colors'].get(layer))

        if max_hits - min_hits < 20:
             ax2.set_xticks(np.arange(min_hits, max_hits + 1))
        ax2.set_xlabel('Number of Hits per Timestamp')
        ax2.set_ylabel('Frequency (Number of Timestamps)')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title(plot_title_2)
    fig2.tight_layout()
    return fig2, ax2

def _plot_p3_filtered_hits_vs_time(prep_data, time_bins, time_range):
    """Generates Plot 3: Filtered Hits vs. Time (Log Scale, Binned)"""
    print("\nGenerating Plot 3: Overlayed Filtered Hits (Log Scale, Binned)")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    raw_ts_time_s = prep_data['raw_ts_time_s']
    t_min, t_max = raw_ts_time_s[0], raw_ts_time_s[-1]
    bins_plot3 = np.linspace(t_min, t_max, time_bins + 1)
    
    if prep_data['has_layer']:
        for layer in prep_data['layers_to_plot']:
            layer_mask = (prep_data['layer_all_hits'] == layer)
            
            # In-Bunch hits for this layer
            in_bunch_mask = layer_mask & prep_data['raw_in_bunch_mask_phase']
            if np.any(in_bunch_mask):
                ax3.hist(raw_ts_time_s[in_bunch_mask], bins=bins_plot3,
                         histtype='step', label=f'Layer {layer} (In)',
                         color=prep_data['layer_colors'].get(layer), linewidth=2)
                         
            # Out-of-Bunch hits for this layer
            out_bunch_mask = layer_mask & ~prep_data['raw_in_bunch_mask_phase']
            if np.any(out_bunch_mask):
                ax3.hist(raw_ts_time_s[out_bunch_mask], bins=bins_plot3,
                         histtype='step', label=f'Layer {layer} (Out)',
                         color=prep_data['layer_colors'].get(layer), linewidth=1.5,
                         linestyle='--')
                         
    ax3.set_yscale('log')
    ax3.set_xlabel('Time (s) [Relative to first hit]')
    ax3.set_ylabel(f'Number of Hits (per {((t_max-t_min)/time_bins):.2e} s bin, Log Scale)')
    ax3.set_title('Filtered Hits vs. Time (Log Scale, Binned)')
    ax3.legend(loc='upper right') 
    ax3.grid(True, linestyle=':', alpha=0.5)
    if time_range is not None:
        try:
            xmin_s, xmax_s = time_range
            ax3.set_xlim(xmin_s, xmax_s)
            ax3.set_title(f'Filtered Hits vs. Time (Zoomed: {xmin_s}-{xmax_s} s, Log Scale)')
        except Exception as e:
            print(f"Warning: Could not apply time range to Plot 3. Error: {e}")
    fig3.tight_layout()
    return fig3, ax3

def _calculate_and_plot_p4_p5_freq_dist(stats_data, prep_data, period, bunch_width,
                                       bunch_start_time, num_bins,
                                       bunch_fit_range, gap_fit_range):
    """
    Aggregates frequencies and generates Plots 4 & 5.
    Corresponds to original Sections 9 & 10.
    """
    print(f"\n--- Frequency Aggregation (Ignoring 0-hit regions, data {prep_data['stats_range_str']}) ---")
    
    layer_freqs_bunch = {}
    layer_freqs_gap = {}
    times_for_stats = stats_data['times_for_stats']
    gap_width = period - bunch_width
    
    if times_for_stats.size == 0:
        freqs_in_each_bunch = np.array([])
        freqs_in_each_gap = np.array([])
        print("No data in range to aggregate.")
    else:
        t_max_in_range = times_for_stats[-1]
        
        # Aggregate ALL layers
        freqs_in_each_bunch, freqs_in_each_gap = _aggregate_raw_frequencies(
            times_for_stats, t_max_in_range, bunch_start_time, period, bunch_width, gap_width
        )
        
        if freqs_in_each_bunch.size > 0:
            print(f"Total bunches found (All Layers, > 0 hits): {len(freqs_in_each_bunch)}")
            print(f"  Avg Hit Freq/bunch (Hz): {np.mean(freqs_in_each_bunch):.2f}")
            print(f"  Std Hit Freq/bunch (Hz): {np.std(freqs_in_each_bunch):.2f}")
        else:
            print("Total bunches found (All Layers, > 0 hits): 0")

        if freqs_in_each_gap.size > 0:
            print(f"\nTotal gaps found (All Layers, > 0 hits): {len(freqs_in_each_gap)}")
            print(f"  Avg Hit Freq/gap (Hz): {np.mean(freqs_in_each_gap):.2f}")
            print(f"  Std Hit Freq/gap (Hz): {np.std(freqs_in_each_gap):.2f}")
        else:
            print("\nTotal gaps found (All Layers, > 0 hits): 0")

        # Aggregate for each layer
        if prep_data['has_layer'] and stats_data['layers_for_stats'].size > 0:
            print("\nAggregating by layer...")
            for layer in prep_data['layers_to_plot']:
                layer_mask = (stats_data['layers_for_stats'] == layer)
                times_for_this_layer = times_for_stats[layer_mask]
                
                if times_for_this_layer.size > 0:
                    b, g = _aggregate_raw_frequencies(
                        times_for_this_layer, t_max_in_range, bunch_start_time, period, bunch_width, gap_width
                    )
                    layer_freqs_bunch[layer] = b
                    layer_freqs_gap[layer] = g
                    print(f"  Layer {layer}: Found {len(b)} bunches, {len(g)} gaps (with > 0 hits)")
                else:
                    layer_freqs_bunch[layer] = np.array([])
                    layer_freqs_gap[layer] = np.array([])
                    print(f"  Layer {layer}: No hits found in time range.")
        elif prep_data['has_layer']:
            print("\nNo hits found in time range to aggregate by layer.")

    # --- Plot 4 & 5 ---
    print("\nGenerating Plots 4 & 5: Hit Frequency Distributions (with Layer Overlays)")
    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    title_suffix = f' (> 0 hits, {prep_data['stats_range_str']})'
    
    # Plot 4
    if freqs_in_each_bunch.size > 0:
        all_bunch_data = [freqs_in_each_bunch]
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                if layer_freqs_bunch.get(layer) is not None and layer_freqs_bunch[layer].size > 0:
                    all_bunch_data.append(layer_freqs_bunch[layer])
        
        combined_bunch_data = np.concatenate(all_bunch_data)
        min_b, max_b = np.min(combined_bunch_data), np.max(combined_bunch_data)
        bins_b = np.linspace(min_b, max_b, num_bins + 1) 

        _plot_frequency_distribution_with_fit(
            ax4, freqs_in_each_bunch, 
            'Distribution of Hit Frequency *per Bunch*' + title_suffix,
            'blue', bunch_fit_range, bins_b, "Bunch (All Layers)")
        
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_b = layer_freqs_bunch.get(layer)
                if data_b is not None and data_b.size > 0:
                    ax4.hist(data_b, bins=bins_b, histtype='step', 
                             label=f'Layer {layer}', color=prep_data['layer_colors'].get(layer),
                             linewidth=1.5)
            ax4.legend() 
    else:
        ax4.text(0.5, 0.5, 'No bunch data', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax4.set_title('Distribution of Hit Frequency *per Bunch*' + title_suffix)
        
    ax4.set_xlabel('Hit Frequency (Hz) per Bunch')
    ax4.set_ylabel('Frequency (Number of Bunches)')
    
    # Plot 5
    if freqs_in_each_gap.size > 0:
        all_gap_data = [freqs_in_each_gap]
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                if layer_freqs_gap.get(layer) is not None and layer_freqs_gap[layer].size > 0:
                    all_gap_data.append(layer_freqs_gap[layer])
        
        combined_gap_data = np.concatenate(all_gap_data)
        min_g, max_g = np.min(combined_gap_data), np.max(combined_gap_data)
        bins_g = np.linspace(min_g, max_g, num_bins + 1)
        
        _plot_frequency_distribution_with_fit(
            ax5, freqs_in_each_gap,
            'Distribution of Hit Frequency *per Gap*' + title_suffix,
            'orange', gap_fit_range, bins_g, "Gap (All Layers)")

        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_g = layer_freqs_gap.get(layer)
                if data_g is not None and data_g.size > 0:
                    ax5.hist(data_g, bins=bins_g, histtype='step', 
                             label=f'Layer {layer}', color=prep_data['layer_colors'].get(layer),
                             linewidth=1.5)
            ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No gap data', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
        ax5.set_title('Distribution of Hit Frequency *per Gap*' + title_suffix)

    ax5.set_xlabel('Hit Frequency (Hz) per Gap')
    ax5.set_ylabel('Frequency (Number of Gaps)')
    
    fig4.suptitle(f'Hit Frequency Aggregation Analysis (Period={period}s, Width={bunch_width}s)',
                  fontsize=14)
    plt.subplots_adjust(top=0.85)
    
    return fig4, (ax4, ax5)

def _plot_p6_phase_folded_hits(stats_data, prep_data, period, bunch_width, bunch_start_time):
    """Generates Plot 6: Phase-Folded Hit Distribution (with Layer Overlays)"""
    print(f"\nGenerating Plot 6: Phase-Folded Hit Distribution (with Layer Overlays) [data {prep_data['stats_range_str']}]")
    fig5, ax6 = plt.subplots(figsize=(10, 6))
    plot_title_6 = f'Phase-Folded Hit Distribution (Data {prep_data['stats_range_str']})'

    if stats_data['times_for_stats'].size == 0:
        ax6.text(0.5, 0.5, f'No data found {prep_data['stats_range_str']}',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax6.transAxes)
    else:
        phase_bins = np.linspace(0, period, 101) # 100 bins
        
        ax6.hist(stats_data['phase_for_stats_raw'], bins=phase_bins, histtype='step', 
                 label='All Layers', linewidth=2, color='black')
        
        if prep_data['has_layer'] and stats_data['layers_for_stats'].size > 0:
            for layer in prep_data['layers_to_plot']:
                layer_mask = (stats_data['layers_for_stats'] == layer)
                phase_for_this_layer = stats_data['phase_for_stats_raw'][layer_mask]
                
                if phase_for_this_layer.size > 0:
                    ax6.hist(phase_for_this_layer, bins=phase_bins,
                             histtype='step', label=f'Layer {layer}', 
                             color=prep_data['layer_colors'].get(layer), linewidth=1.5)
        
        ax6.axvspan(0, bunch_width, color='gray', alpha=0.3, label='Defined Bunch Region')
        ax6.set_xlabel(f'Time within Period (s) [relative to {bunch_start_time}s]')
        ax6.set_ylabel('Total Hits in Bin (Log Scale)')
        ax6.set_xlim(0, period)
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.6)

    ax6.set_title(plot_title_6)
    fig5.tight_layout()
    return fig5, ax6

def _plot_p7_phase_tot_heatmaps(stats_data, prep_data, period, bunch_width):
    """Generates Plot 7: Phase-Folded ToT Heatmaps (by Layer with Marginals)"""
    print(f"\nGenerating Plot 7: Phase-Folded ToT Heatmaps by Layer [data {prep_data['stats_range_str']}]")
    
    fig7 = plt.figure(figsize=(16, 14))
    plot_title_7 = f'Phase-Folded ToT Distribution by Layer (Data {prep_data['stats_range_str']})'
    fig7.suptitle(plot_title_7, fontsize=16, y=0.98)

    outer_gs = gridspec.GridSpec(2, 2, figure=fig7, hspace=0.4, wspace=0.4)
    
    if not prep_data['has_tot'] or not prep_data['has_layer'] or stats_data['times_for_stats'].size == 0:
        fig7.text(0.5, 0.5, f'No ToT, Layer, or Hit data found {prep_data['stats_range_str']}',
                  horizontalalignment='center', verticalalignment='center',
                  fontsize=14)
    else:
        phase_bins_7 = np.linspace(0, period, 101)
        tot_bins_7 = np.arange(0, 256) 
        
        all_hists = []
        data_to_plot = {}
        layers_to_plot = prep_data['layers_to_plot']

        for layer in layers_to_plot:
            layer_mask = (stats_data['layers_for_stats'] == layer)
            tot_layer = stats_data['tot_for_stats'][layer_mask]
            phase_layer = stats_data['phase_for_stats_raw'][layer_mask]
            data_to_plot[layer] = (phase_layer, tot_layer)
            
            if tot_layer.size > 0:
                h, _, _ = np.histogram2d(phase_layer, tot_layer, bins=(phase_bins_7, tot_bins_7))
                all_hists.append(h[h > 0])

        if not all_hists:
            ax = fig7.add_subplot(outer_gs[0])
            ax.text(0.5, 0.5, f'No hits found for Layers {layers_to_plot}',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes)
            ax.set_axis_off()
        else:
            h_flat = np.concatenate(all_hists)
            norm = LogNorm(vmin=np.min(h_flat), vmax=np.max(h_flat))
            
            images = [] 
            
            for i, layer in enumerate(layers_to_plot):
                inner_gs = gridspec.GridSpecFromSubplotSpec(
                    5, 5, subplot_spec=outer_gs[i], hspace=0, wspace=0
                )
                
                ax_main = fig7.add_subplot(inner_gs[1:, :-1])
                ax_marg_x = fig7.add_subplot(inner_gs[0, :-1], sharex=ax_main)
                ax_marg_y = fig7.add_subplot(inner_gs[1:, -1], sharey=ax_main)
                
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)

                phase_layer, tot_layer = data_to_plot[layer]
                
                if tot_layer.size == 0:
                    ax_main.text(0.5, 0.5, f'No hits for Layer {layer}',
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=ax_main.transAxes)
                else:
                    _, _, _, im = ax_main.hist2d(
                        phase_layer, tot_layer,
                        bins=(phase_bins_7, tot_bins_7),
                        cmap='plasma',
                        norm=norm
                    )
                    if not images: 
                        images.append(im)
                    
                    ax_marg_x.hist(phase_layer, bins=phase_bins_7, histtype='step', color='black')
                    ax_marg_y.hist(tot_layer, bins=tot_bins_7, histtype='step', 
                                   orientation='horizontal', color='black')

                ax_main.set_title(f'Layer {layer}')
                ax_main.axvspan(0, bunch_width, color='gray', alpha=0.3, label='Defined Bunch Region')
                ax_main.grid(True, linestyle='--', alpha=0.6)
                
                ax_marg_x.axvspan(0, bunch_width, color='gray', alpha=0.3)
                ax_marg_x.grid(True, linestyle='--', alpha=0.6)
                ax_marg_x.set_ylabel('Counts')
                
                ax_marg_y.grid(True, linestyle='--', alpha=0.6)
                ax_marg_y.set_xlabel('Counts')

                ax_main.set_xlabel('Time within Period (s)')
                ax_main.set_ylabel('ToT')

            if images:
                fig7.subplots_adjust(right=0.85, top=0.93)
                cbar_ax = fig7.add_axes([0.88, 0.15, 0.03, 0.7])
                cbar = fig7.colorbar(images[0], cax=cbar_ax)
                cbar.set_label('Hit Frequency (Log Scale)')
            else:
                 fig7.tight_layout(rect=[0, 0, 1, 0.95])
                 
    return fig7, None # Return fig and None for axes


# --- Main Wrapper Function ---

def analyze_bunch_data(data, period=0.08, bunch_width=0.05,
                       bunch_start_time=0.0, bunch_end_time=None,
                       time_range=None,
                       bunch_fit_range=None, gap_fit_range=None,
                       num_bins=128, timestamp_key='TriggerTS',
                       time_bins=1000):
    """
    Applies a periodic filter, generates 7 plot figures, and returns filtered data.
    
    This function is a wrapper that calls modular sub-functions to perform
    data preparation, statistical analysis, and plotting.
    
    Args:
        (Same as original function)
        
    Returns:
        (Same as original function)
    """
    
    # --- 0. Validate Inputs ---
    if period <= bunch_width:
        print(f"Error: Period ({period}s) must be greater than bunch_width ({bunch_width}s).")
        return None, None
    
    # --- 1. Data Preparation (Sections 1 & 2) ---
    prep_data = _prepare_analysis_data(
        data, timestamp_key, period, bunch_width, bunch_start_time, bunch_end_time
    )
    if prep_data is None:
        return None, None # Error handled inside function

    # --- 2. Filter Data for Return (Section 3) ---
    data_in_bunches, data_out_of_bunches = _filter_data_for_return(
        data, prep_data, timestamp_key
    )

    # --- 3. Calculate Window Statistics (Sections 4 & 5) ---
    stats_data = _calculate_window_stats(
        prep_data, period, bunch_width
    )

    # --- 4. Generate All Plots (Sections 6-12) ---
    _plot_p1_hits_vs_time(
        prep_data, time_bins, period, bunch_width, bunch_start_time, time_range
    )
    
    _plot_p2_hits_per_ts(stats_data, prep_data)
    
    _plot_p3_filtered_hits_vs_time(prep_data, time_bins, time_range)
    
    _calculate_and_plot_p4_p5_freq_dist(
        stats_data, prep_data, period, bunch_width, bunch_start_time, 
        num_bins, bunch_fit_range, gap_fit_range
    )
    
    _plot_p6_phase_folded_hits(
        stats_data, prep_data, period, bunch_width, bunch_start_time
    )
    
    _plot_p7_phase_tot_heatmaps(
        stats_data, prep_data, period, bunch_width
    )

    # --- 13. Finalize ---
    plt.show() 
    
    print(f"\nReturning {len(data_in_bunches[timestamp_key])} in-bunch hits and {len(data_out_of_bunches[timestamp_key])} out-of-bunch hits (from {prep_data['stats_range_str']}).")
    return data_in_bunches, data_out_of_bunches

def _calculate_window_stats(prep_data, period, bunch_width):
    """
    Calculates all statistics for the defined time window.
    Corresponds to original Sections 4 & 5.
    """
    stats_data = {
        'times_for_stats': prep_data['raw_ts_time_s'][prep_data['raw_stats_time_mask']],
        'ts_in_window_raw': prep_data['Trig_ts_raw'][prep_data['raw_stats_time_mask']]
    }

    if stats_data['times_for_stats'].size == 0:
        print("\n--- Statistics ---")
        print(f"Warning: No data found in the specified range ({prep_data['stats_range_str']}).")
        print("Statistics, Plot 2, and Frequency Plots will be empty.")
        
        # Set all data to empty arrays
        stats_data.update({
            'layers_for_stats': np.array([]),
            'tot_for_stats': np.array([]),
            'phase_for_stats_raw': np.array([]),
            'hit_counts_in_window': np.array([]),
            'in_bunch_hits_for_stats': np.array([]),
            'out_bunch_hits_for_stats': np.array([]),
            'layer_hit_counts_stats': {}
        })
        
    else:
        # Filter aligned data
        stats_data['phase_for_stats_raw'] = prep_data['raw_time_from_start'][prep_data['raw_stats_time_mask']]
        if prep_data['has_layer']:
            stats_data['layers_for_stats'] = prep_data['layer_all_hits'][prep_data['raw_stats_time_mask']]
        else:
            stats_data['layers_for_stats'] = np.array([])
            
        if prep_data['has_tot']:
            stats_data['tot_for_stats'] = prep_data['tot_all_hits'][prep_data['raw_stats_time_mask']]
        else:
            stats_data['tot_for_stats'] = np.array([])
            
        # --- Data for Plot 2 & Stats ---
        print("\nCalculating hits-per-timestamp for analysis window...")
        
        # 1. For "All Layers"
        unique_ts_raw_in_window, hit_counts_in_window = np.unique(stats_data['ts_in_window_raw'], return_counts=True)
        stats_data['hit_counts_in_window'] = hit_counts_in_window
        
        # Need phase for these unique timestamps
        unique_ts_time_s_in_window = unique_ts_raw_in_window * prep_data['s_per_ts']
        unique_ts_relative_in_window = unique_ts_time_s_in_window - prep_data['min_time_s']
        time_from_start_in_window = (unique_ts_relative_in_window - prep_data['bunch_start_time']) % period
        in_bunch_mask_in_window = (time_from_start_in_window < bunch_width)
        
        stats_data['in_bunch_hits_for_stats'] = hit_counts_in_window[in_bunch_mask_in_window]
        stats_data['out_bunch_hits_for_stats'] = hit_counts_in_window[~in_bunch_mask_in_window]
        
        # 2. For each layer
        layer_hit_counts_stats = {}
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                layer_mask_in_window = (stats_data['layers_for_stats'] == layer)
                ts_for_layer = stats_data['ts_in_window_raw'][layer_mask_in_window]
                if ts_for_layer.size > 0:
                    _ , hit_counts_layer = np.unique(ts_for_layer, return_counts=True)
                    layer_hit_counts_stats[layer] = hit_counts_layer
                else:
                    layer_hit_counts_stats[layer] = np.array([])
        stats_data['layer_hit_counts_stats'] = layer_hit_counts_stats
        
        # --- Print Statistics ---
        total_hits_for_stats = np.sum(hit_counts_in_window)
        total_ts_for_stats = len(hit_counts_in_window)
    
        print(f"\n--- Timestamp Statistics (for data {prep_data['stats_range_str']}) ---")
        print(f"Total Timestamps (in range): {total_ts_for_stats}")
        
        in_ts_count = len(stats_data['in_bunch_hits_for_stats'])
        in_ts_perc = (in_ts_count / total_ts_for_stats * 100) if total_ts_for_stats > 0 else 0
        print(f"  In-Bunch Timestamps: {in_ts_count} ({in_ts_perc:.1f}%)")
        
        out_ts_count = len(stats_data['out_bunch_hits_for_stats'])
        out_ts_perc = (out_ts_count / total_ts_for_stats * 100) if total_ts_for_stats > 0 else 0
        print(f"  Out-of-Bunch Timestamps: {out_ts_count} ({out_ts_perc:.1f}%)")

        print(f"Total Hits (in range): {total_hits_for_stats}")
        in_hits_sum = np.sum(stats_data['in_bunch_hits_for_stats'])
        in_hits_perc = (in_hits_sum / total_hits_for_stats * 100) if total_hits_for_stats > 0 else 0
        print(f"  In-Bunch Hits: {in_hits_sum} ({in_hits_perc:.1f}%)")
        
        out_hits_sum = np.sum(stats_data['out_bunch_hits_for_stats'])
        out_hits_perc = (out_hits_sum / total_hits_for_stats * 100) if total_hits_for_stats > 0 else 0
        print(f"  Out-of-Bunch Hits: {out_hits_sum} ({out_hits_perc:.1f}%)")

    return stats_data

def _plot_p1_hits_vs_time(prep_data, time_bins, period, bunch_width, bunch_start_time, time_range):
    """Generates Plot 1: Hits vs. Time (Binned, with Layer Overlays)"""
    print("\nGenerating Plot 1: Hits vs. Time (Binned, with Layer Overlays)")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    raw_ts_time_s = prep_data['raw_ts_time_s']
    t_min, t_max = raw_ts_time_s[0], raw_ts_time_s[-1]
    bins_plot1 = np.linspace(t_min, t_max, time_bins + 1)
    
    ax1.hist(raw_ts_time_s, bins=bins_plot1, histtype='step', 
             label='All Hits', color='black', linewidth=2)
    
    if prep_data['has_layer']:
        for layer in prep_data['layers_to_plot']:
            layer_mask = (prep_data['layer_all_hits'] == layer)
            if np.any(layer_mask):
                ax1.hist(raw_ts_time_s[layer_mask], bins=bins_plot1,
                         histtype='step', label=f'Layer {layer}',
                         color=prep_data['layer_colors'].get(layer), linewidth=1)

    # Plot bunch regions
    first_bunch_index = np.floor((t_min - bunch_start_time) / period)
    t_current = first_bunch_index * period + bunch_start_time
    span_label = 'Specified Bunch Region'
    while t_current < t_max:
        start = max(t_min, t_current)
        end = min(t_max, t_current + bunch_width)
        if end > start:
            ax1.axvspan(start, end, color='gray', alpha=0.3, label=span_label)
            span_label = '_nolegend_'
        t_current += period
        
    ax1.set_xlabel('Time (s) [Relative to first hit]')
    ax1.set_ylabel(f'Number of Hits (per {((t_max-t_min)/time_bins):.2e} s bin)')
    ax1.set_title('Hits per Time Bin (with Specified Bunch Regions)')
    ax1.legend(loc='upper right') 
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    if time_range is not None:
        try:
            xmin_s, xmax_s = time_range
            ax1.set_xlim(xmin_s, xmax_s)
            ax1.set_title(f'Hits per Time Bin (Zoomed: {xmin_s}-{xmax_s} s)')
        except Exception as e:
            print(f"Warning: Could not apply time range to Plot 1. Error: {e}")
    fig1.tight_layout()
    return fig1, ax1

def _plot_p2_hits_per_ts(stats_data, prep_data):
    """Generates Plot 2: Distribution of Hits per Timestamp (with Layer Overlays)"""
    print(f"\nGenerating Plot 2: Distribution of Hits (In vs. Out) [data {prep_data['stats_range_str']}]")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_title_2 = f'Distribution of Hits per Timestamp (Data {prep_data['stats_range_str']})'
    
    hit_counts_in_window = stats_data['hit_counts_in_window']
    if hit_counts_in_window.size == 0:
        ax2.text(0.5, 0.5, f'No data found {prep_data['stats_range_str']}', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    else:
        # --- FIXED BINNING LOGIC ---
        # Find global min/max across ALL datasets for this plot
        all_data_collections = [hit_counts_in_window]
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_l = stats_data['layer_hit_counts_stats'].get(layer)
                if data_l is not None and data_l.size > 0:
                    all_data_collections.append(data_l)
        
        combined_data = np.concatenate(all_data_collections)
        min_hits = np.min(combined_data)
        max_hits = np.max(combined_data)
        # Create bins centered on integers
        bins_plot2 = np.arange(min_hits, max_hits + 2) - 0.5 
        # --- END FIXED BINNING LOGIC ---
        
        # Plot "All Layers" In vs Out
        if len(stats_data['in_bunch_hits_for_stats']) > 0:
            ax2.hist(stats_data['in_bunch_hits_for_stats'], bins=bins_plot2, histtype='step', 
                     align='mid', linewidth=2, label='All Layers (In-Bunch)',
                     color='blue')
        if len(stats_data['out_bunch_hits_for_stats']) > 0:
            ax2.hist(stats_data['out_bunch_hits_for_stats'], bins=bins_plot2, histtype='step', 
                     align='mid', linewidth=2, linestyle='--', 
                     label='All Layers (Out-of-Bunch)', color='cyan')
                     
        # Plot Layer overlays (total hits)
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_l = stats_data['layer_hit_counts_stats'].get(layer)
                if data_l is not None and data_l.size > 0:
                    ax2.hist(data_l, bins=bins_plot2, histtype='step',
                             align='mid', linewidth=1.5,
                             label=f'Layer {layer} (Total)',
                             color=prep_data['layer_colors'].get(layer))

        if max_hits - min_hits < 20:
             ax2.set_xticks(np.arange(min_hits, max_hits + 1))
        ax2.set_xlabel('Number of Hits per Timestamp')
        ax2.set_ylabel('Frequency (Number of Timestamps)')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title(plot_title_2)
    fig2.tight_layout()
    return fig2, ax2

def _plot_p3_filtered_hits_vs_time(prep_data, time_bins, time_range):
    """Generates Plot 3: Filtered Hits vs. Time (Log Scale, Binned)"""
    print("\nGenerating Plot 3: Overlayed Filtered Hits (Log Scale, Binned)")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    raw_ts_time_s = prep_data['raw_ts_time_s']
    t_min, t_max = raw_ts_time_s[0], raw_ts_time_s[-1]
    bins_plot3 = np.linspace(t_min, t_max, time_bins + 1)
    
    if prep_data['has_layer']:
        for layer in prep_data['layers_to_plot']:
            layer_mask = (prep_data['layer_all_hits'] == layer)
            
            # In-Bunch hits for this layer
            in_bunch_mask = layer_mask & prep_data['raw_in_bunch_mask_phase']
            if np.any(in_bunch_mask):
                ax3.hist(raw_ts_time_s[in_bunch_mask], bins=bins_plot3,
                         histtype='step', label=f'Layer {layer} (In)',
                         color=prep_data['layer_colors'].get(layer), linewidth=2)
                         
            # Out-of-Bunch hits for this layer
            out_bunch_mask = layer_mask & ~prep_data['raw_in_bunch_mask_phase']
            if np.any(out_bunch_mask):
                ax3.hist(raw_ts_time_s[out_bunch_mask], bins=bins_plot3,
                         histtype='step', label=f'Layer {layer} (Out)',
                         color=prep_data['layer_colors'].get(layer), linewidth=1.5,
                         linestyle='--')
                         
    ax3.set_yscale('log')
    ax3.set_xlabel('Time (s) [Relative to first hit]')
    ax3.set_ylabel(f'Number of Hits (per {((t_max-t_min)/time_bins):.2e} s bin, Log Scale)')
    ax3.set_title('Filtered Hits vs. Time (Log Scale, Binned)')
    ax3.legend(loc='upper right') 
    ax3.grid(True, linestyle=':', alpha=0.5)
    if time_range is not None:
        try:
            xmin_s, xmax_s = time_range
            ax3.set_xlim(xmin_s, xmax_s)
            ax3.set_title(f'Filtered Hits vs. Time (Zoomed: {xmin_s}-{xmax_s} s, Log Scale)')
        except Exception as e:
            print(f"Warning: Could not apply time range to Plot 3. Error: {e}")
    fig3.tight_layout()
    return fig3, ax3

def _calculate_and_plot_p4_p5_freq_dist(stats_data, prep_data, period, bunch_width,
                                       bunch_start_time, num_bins,
                                       bunch_fit_range, gap_fit_range):
    """
    Aggregates frequencies and generates Plots 4 & 5.
    Corresponds to original Sections 9 & 10.
    """
    print(f"\n--- Frequency Aggregation (Ignoring 0-hit regions, data {prep_data['stats_range_str']}) ---")
    
    layer_freqs_bunch = {}
    layer_freqs_gap = {}
    times_for_stats = stats_data['times_for_stats']
    gap_width = period - bunch_width
    
    if times_for_stats.size == 0:
        freqs_in_each_bunch = np.array([])
        freqs_in_each_gap = np.array([])
        print("No data in range to aggregate.")
    else:
        t_max_in_range = times_for_stats[-1]
        
        # Aggregate ALL layers
        freqs_in_each_bunch, freqs_in_each_gap = _aggregate_raw_frequencies(
            times_for_stats, t_max_in_range, bunch_start_time, period, bunch_width, gap_width
        )
        
        if freqs_in_each_bunch.size > 0:
            print(f"Total bunches found (All Layers, > 0 hits): {len(freqs_in_each_bunch)}")
            print(f"  Avg Hit Freq/bunch (Hz): {np.mean(freqs_in_each_bunch):.2f}")
            print(f"  Std Hit Freq/bunch (Hz): {np.std(freqs_in_each_bunch):.2f}")
        else:
            print("Total bunches found (All Layers, > 0 hits): 0")

        if freqs_in_each_gap.size > 0:
            print(f"\nTotal gaps found (All Layers, > 0 hits): {len(freqs_in_each_gap)}")
            print(f"  Avg Hit Freq/gap (Hz): {np.mean(freqs_in_each_gap):.2f}")
            print(f"  Std Hit Freq/gap (Hz): {np.std(freqs_in_each_gap):.2f}")
        else:
            print("\nTotal gaps found (All Layers, > 0 hits): 0")

        # Aggregate for each layer
        if prep_data['has_layer'] and stats_data['layers_for_stats'].size > 0:
            print("\nAggregating by layer...")
            for layer in prep_data['layers_to_plot']:
                layer_mask = (stats_data['layers_for_stats'] == layer)
                times_for_this_layer = times_for_stats[layer_mask]
                
                if times_for_this_layer.size > 0:
                    b, g = _aggregate_raw_frequencies(
                        times_for_this_layer, t_max_in_range, bunch_start_time, period, bunch_width, gap_width
                    )
                    layer_freqs_bunch[layer] = b
                    layer_freqs_gap[layer] = g
                    print(f"  Layer {layer}: Found {len(b)} bunches, {len(g)} gaps (with > 0 hits)")
                else:
                    layer_freqs_bunch[layer] = np.array([])
                    layer_freqs_gap[layer] = np.array([])
                    print(f"  Layer {layer}: No hits found in time range.")
        elif prep_data['has_layer']:
            print("\nNo hits found in time range to aggregate by layer.")

    # --- Plot 4 & 5 ---
    print("\nGenerating Plots 4 & 5: Hit Frequency Distributions (with Layer Overlays)")
    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    title_suffix = f' (> 0 hits, {prep_data['stats_range_str']})'
    
    # Plot 4
    if freqs_in_each_bunch.size > 0:
        all_bunch_data = [freqs_in_each_bunch]
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                if layer_freqs_bunch.get(layer) is not None and layer_freqs_bunch[layer].size > 0:
                    all_bunch_data.append(layer_freqs_bunch[layer])
        
        combined_bunch_data = np.concatenate(all_bunch_data)
        min_b, max_b = np.min(combined_bunch_data), np.max(combined_bunch_data)
        bins_b = np.linspace(min_b, max_b, num_bins + 1) 

        _plot_frequency_distribution_with_fit(
            ax4, freqs_in_each_bunch, 
            'Distribution of Hit Frequency *per Bunch*' + title_suffix,
            'blue', bunch_fit_range, bins_b, "Bunch (All Layers)")
        
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_b = layer_freqs_bunch.get(layer)
                if data_b is not None and data_b.size > 0:
                    ax4.hist(data_b, bins=bins_b, histtype='step', 
                             label=f'Layer {layer}', color=prep_data['layer_colors'].get(layer),
                             linewidth=1.5)
            ax4.legend() 
    else:
        ax4.text(0.5, 0.5, 'No bunch data', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax4.set_title('Distribution of Hit Frequency *per Bunch*' + title_suffix)
        
    ax4.set_xlabel('Hit Frequency (Hz) per Bunch')
    ax4.set_ylabel('Frequency (Number of Bunches)')
    
    # Plot 5
    if freqs_in_each_gap.size > 0:
        all_gap_data = [freqs_in_each_gap]
        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                if layer_freqs_gap.get(layer) is not None and layer_freqs_gap[layer].size > 0:
                    all_gap_data.append(layer_freqs_gap[layer])
        
        combined_gap_data = np.concatenate(all_gap_data)
        min_g, max_g = np.min(combined_gap_data), np.max(combined_gap_data)
        bins_g = np.linspace(min_g, max_g, num_bins + 1)
        
        _plot_frequency_distribution_with_fit(
            ax5, freqs_in_each_gap,
            'Distribution of Hit Frequency *per Gap*' + title_suffix,
            'orange', gap_fit_range, bins_g, "Gap (All Layers)")

        if prep_data['has_layer']:
            for layer in prep_data['layers_to_plot']:
                data_g = layer_freqs_gap.get(layer)
                if data_g is not None and data_g.size > 0:
                    ax5.hist(data_g, bins=bins_g, histtype='step', 
                             label=f'Layer {layer}', color=prep_data['layer_colors'].get(layer),
                             linewidth=1.5)
            ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No gap data', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
        ax5.set_title('Distribution of Hit Frequency *per Gap*' + title_suffix)

    ax5.set_xlabel('Hit Frequency (Hz) per Gap')
    ax5.set_ylabel('Frequency (Number of Gaps)')
    
    fig4.suptitle(f'Hit Frequency Aggregation Analysis (Period={period}s, Width={bunch_width}s)',
                  fontsize=14)
    plt.subplots_adjust(top=0.85)
    
    return fig4, (ax4, ax5)

def _plot_p6_phase_folded_hits(stats_data, prep_data, period, bunch_width, bunch_start_time):
    """Generates Plot 6: Phase-Folded Hit Distribution (with Layer Overlays)"""
    print(f"\nGenerating Plot 6: Phase-Folded Hit Distribution (with Layer Overlays) [data {prep_data['stats_range_str']}]")
    fig5, ax6 = plt.subplots(figsize=(10, 6))
    plot_title_6 = f'Phase-Folded Hit Distribution (Data {prep_data['stats_range_str']})'

    if stats_data['times_for_stats'].size == 0:
        ax6.text(0.5, 0.5, f'No data found {prep_data['stats_range_str']}',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax6.transAxes)
    else:
        phase_bins = np.linspace(0, period, 101) # 100 bins
        
        ax6.hist(stats_data['phase_for_stats_raw'], bins=phase_bins, histtype='step', 
                 label='All Layers', linewidth=2, color='black')
        
        if prep_data['has_layer'] and stats_data['layers_for_stats'].size > 0:
            for layer in prep_data['layers_to_plot']:
                layer_mask = (stats_data['layers_for_stats'] == layer)
                phase_for_this_layer = stats_data['phase_for_stats_raw'][layer_mask]
                
                if phase_for_this_layer.size > 0:
                    ax6.hist(phase_for_this_layer, bins=phase_bins,
                             histtype='step', label=f'Layer {layer}', 
                             color=prep_data['layer_colors'].get(layer), linewidth=1.5)
        
        ax6.axvspan(0, bunch_width, color='gray', alpha=0.3, label='Defined Bunch Region')
        ax6.set_xlabel(f'Time within Period (s) [relative to {bunch_start_time}s]')
        ax6.set_ylabel('Total Hits in Bin (Log Scale)')
        ax6.set_xlim(0, period)
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.6)

    ax6.set_title(plot_title_6)
    fig5.tight_layout()
    return fig5, ax6

def _plot_p7_phase_tot_heatmaps(stats_data, prep_data, period, bunch_width):
    """Generates Plot 7: Phase-Folded ToT Heatmaps (by Layer with Marginals)"""
    print(f"\nGenerating Plot 7: Phase-Folded ToT Heatmaps by Layer [data {prep_data['stats_range_str']}]")
    
    fig7 = plt.figure(figsize=(16, 14))
    plot_title_7 = f'Phase-Folded ToT Distribution by Layer (Data {prep_data['stats_range_str']})'
    fig7.suptitle(plot_title_7, fontsize=16, y=0.98)

    outer_gs = gridspec.GridSpec(2, 2, figure=fig7, hspace=0.4, wspace=0.4)
    
    if not prep_data['has_tot'] or not prep_data['has_layer'] or stats_data['times_for_stats'].size == 0:
        fig7.text(0.5, 0.5, f'No ToT, Layer, or Hit data found {prep_data['stats_range_str']}',
                  horizontalalignment='center', verticalalignment='center',
                  fontsize=14)
    else:
        phase_bins_7 = np.linspace(0, period, 101)
        tot_bins_7 = np.arange(0, 256) 
        
        all_hists = []
        data_to_plot = {}
        layers_to_plot = prep_data['layers_to_plot']

        for layer in layers_to_plot:
            layer_mask = (stats_data['layers_for_stats'] == layer)
            tot_layer = stats_data['tot_for_stats'][layer_mask]
            phase_layer = stats_data['phase_for_stats_raw'][layer_mask]
            data_to_plot[layer] = (phase_layer, tot_layer)
            
            if tot_layer.size > 0:
                h, _, _ = np.histogram2d(phase_layer, tot_layer, bins=(phase_bins_7, tot_bins_7))
                all_hists.append(h[h > 0])

        if not all_hists:
            ax = fig7.add_subplot(outer_gs[0])
            ax.text(0.5, 0.5, f'No hits found for Layers {layers_to_plot}',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax.transAxes)
            ax.set_axis_off()
        else:
            h_flat = np.concatenate(all_hists)
            norm = LogNorm(vmin=np.min(h_flat), vmax=np.max(h_flat))
            
            images = [] 
            
            for i, layer in enumerate(layers_to_plot):
                inner_gs = gridspec.GridSpecFromSubplotSpec(
                    5, 5, subplot_spec=outer_gs[i], hspace=0, wspace=0
                )
                
                ax_main = fig7.add_subplot(inner_gs[1:, :-1])
                ax_marg_x = fig7.add_subplot(inner_gs[0, :-1], sharex=ax_main)
                ax_marg_y = fig7.add_subplot(inner_gs[1:, -1], sharey=ax_main)
                
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)

                phase_layer, tot_layer = data_to_plot[layer]
                
                if tot_layer.size == 0:
                    ax_main.text(0.5, 0.5, f'No hits for Layer {layer}',
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=ax_main.transAxes)
                else:
                    _, _, _, im = ax_main.hist2d(
                        phase_layer, tot_layer,
                        bins=(phase_bins_7, tot_bins_7),
                        cmap='plasma',
                        norm=norm
                    )
                    if not images: 
                        images.append(im)
                    
                    ax_marg_x.hist(phase_layer, bins=phase_bins_7, histtype='step', color='black')
                    ax_marg_y.hist(tot_layer, bins=tot_bins_7, histtype='step', 
                                   orientation='horizontal', color='black')

                ax_main.set_title(f'Layer {layer}')
                ax_main.axvspan(0, bunch_width, color='gray', alpha=0.3, label='Defined Bunch Region')
                ax_main.grid(True, linestyle='--', alpha=0.6)
                
                ax_marg_x.axvspan(0, bunch_width, color='gray', alpha=0.3)
                ax_marg_x.grid(True, linestyle='--', alpha=0.6)
                ax_marg_x.set_ylabel('Counts')
                
                ax_marg_y.grid(True, linestyle='--', alpha=0.6)
                ax_marg_y.set_xlabel('Counts')

                ax_main.set_xlabel('Time within Period (s)')
                ax_main.set_ylabel('ToT')

            if images:
                fig7.subplots_adjust(right=0.85, top=0.93)
                cbar_ax = fig7.add_axes([0.88, 0.15, 0.03, 0.7])
                cbar = fig7.colorbar(images[0], cax=cbar_ax)
                cbar.set_label('Hit Frequency (Log Scale)')
            else:
                 fig7.tight_layout(rect=[0, 0, 1, 0.95])
                 
    return fig7, None # Return fig and None for axes


# --- Main Wrapper Function ---

def analyze_bunch_data(data, period=0.08, bunch_width=0.05,
                       bunch_start_time=0.0, bunch_end_time=None,
                       time_range=None,
                       bunch_fit_range=None, gap_fit_range=None,
                       num_bins=128, timestamp_key='TriggerTS',
                       time_bins=1000):
    """
    Applies a periodic filter, generates 7 plot figures, and returns filtered data.
    
    This function is a wrapper that calls modular sub-functions to perform
    data preparation, statistical analysis, and plotting.
    
    Args:
        (Same as original function)
        
    Returns:
        (Same as original function)
    """
    
    # --- 0. Validate Inputs ---
    if period <= bunch_width:
        print(f"Error: Period ({period}s) must be greater than bunch_width ({bunch_width}s).")
        return None, None
    
    # --- 1. Data Preparation (Sections 1 & 2) ---
    prep_data = _prepare_analysis_data(
        data, timestamp_key, period, bunch_width, bunch_start_time, bunch_end_time
    )
    if prep_data is None:
        return None, None # Error handled inside function

    # --- 2. Filter Data for Return (Section 3) ---
    data_in_bunches, data_out_of_bunches = _filter_data_for_return(
        data, prep_data, timestamp_key
    )

    # --- 3. Calculate Window Statistics (Sections 4 & 5) ---
    stats_data = _calculate_window_stats(
        prep_data, period, bunch_width
    )

    # --- 4. Generate All Plots (Sections 6-12) ---
    _plot_p1_hits_vs_time(
        prep_data, time_bins, period, bunch_width, bunch_start_time, time_range
    )
    
    _plot_p2_hits_per_ts(stats_data, prep_data)
    
    _plot_p3_filtered_hits_vs_time(prep_data, time_bins, time_range)
    
    _calculate_and_plot_p4_p5_freq_dist(
        stats_data, prep_data, period, bunch_width, bunch_start_time, 
        num_bins, bunch_fit_range, gap_fit_range
    )
    
    _plot_p6_phase_folded_hits(
        stats_data, prep_data, period, bunch_width, bunch_start_time
    )
    
    _plot_p7_phase_tot_heatmaps(
        stats_data, prep_data, period, bunch_width
    )

    # --- 13. Finalize ---
    plt.show() 
    
    print(f"\nReturning {len(data_in_bunches[timestamp_key])} in-bunch hits and {len(data_out_of_bunches[timestamp_key])} out-of-bunch hits (from {prep_data['stats_range_str']}).")
    return data_in_bunches, data_out_of_bunches
from matplotlib.colors import LogNorm, Normalize # Import LogNorm and Normalize


from mpl_toolkits.axes_grid1 import make_axes_locatable # Added this import

def plot_tot_vs_hits_per_timestamp(data_raw, timestamp_key='TriggerTS', 
                                   max_hits_to_plot=None, 
                                   normalize_columns=True, 
                                   log_z_scale=True):
    """
    Plots a 2D heatmap of ToT (y-axis) vs. the number of hits per 
    specified timestamp (x-axis), with marginal 1D histograms.
    
    Each vertical column (corresponding to a specific number of hits) is normalized
    to 1, showing the probability distribution of ToT for that hit count.

    Parameters:
    ----------
    data_raw : dict
        A dictionary containing numpy arrays, as described in the data format.
        Must contain at least the specified `timestamp_key` and 'ToT' keys.
        
    timestamp_key : str, optional
        The key in data_raw to use for the x-axis, e.g., 'ext_TS' or 'TriggerTS'.
        Defaults to 'ext_TS'.
        
    max_hits_to_plot : int, optional
        If provided, the x-axis of the heatmap will be limited to this value.
        This is useful for focusing on regions with lower hit multiplicity.
        If None (default), the x-axis will show all hit multiplicities.
        
    normalize_columns : bool, optional
        If True (default), each vertical column is normalized to sum to 1.
        If False, raw counts are plotted.
        
    log_z_scale : bool, optional
        If True (default), the z-axis (color scale) is logarithmic.
        If False, the z-axis is linear.
    """
    
    print(f"Calculating number of hits per {timestamp_key}...")
    try:
        unique_ts, inverse_indices, counts = np.unique(
            data_raw[timestamp_key], 
            return_counts=True, 
            return_inverse=True
        )
        y_data = data_raw['ToT']
    except KeyError as e:
        print(f"Error: Missing expected key {e} in data_raw dictionary.")
        print(f"  (Expected keys: '{timestamp_key}' and 'ToT')")
        return
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return

    # x_data is the count of hits for each *original* hit's timestamp
    x_data = counts[inverse_indices]
    
    if x_data.size == 0 or y_data.size == 0:
        print(f"Error: '{timestamp_key}' or 'ToT' data is empty after processing.")
        return
    
    print("Creating 2D histogram...")
    # --- Define bins ---
    x_max = int(x_data.max())
    bins_x = np.arange(1, x_max + 2) # Edges: [1, 2, ..., x_max + 1]
    
    y_max = int(y_data.max())
    bins_y = np.arange(0, y_max + 2) # Edges: [0, 1, ..., y_max + 1]

    # --- Create 2D histogram ---
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=(bins_x, bins_y))
    
    # --- Handle normalization ---
    if normalize_columns:
        print("Normalizing histogram columns...")
        column_sums = H.sum(axis=1)
        column_sums_safe = np.where(column_sums == 0, 1, column_sums)
        plot_data = H / column_sums_safe[:, np.newaxis]
        cbar_label = 'Normalized Probability Density'
        v_max = 1.0
    else:
        print("Using raw counts for histogram.")
        plot_data = H
        cbar_label = 'Counts'
        v_max = plot_data.max()

    print("Plotting heatmap with marginals...")

    # --- Set up color normalization (log or linear) ---
    if log_z_scale:
        non_zero_vals = plot_data[plot_data > 0]
        v_min = non_zero_vals.min() if non_zero_vals.size > 0 else 1e-6 
        
        if v_min >= v_max:
             print(f"Warning: v_min ({v_min}) >= v_max ({v_max}) in LogNorm. Resetting.")
             if non_zero_vals.size > 0:
                 v_max = non_zero_vals.max() 
             else:
                 v_max = 1.0 
             if v_min >= v_max: 
                 print("Warning: Cannot use LogNorm, switching to linear scale.")
                 norm = Normalize(vmin=0, vmax=v_max)
                 log_z_scale = False 
             else:
                 norm = LogNorm(vmin=v_min, vmax=v_max)
        else:
             norm = LogNorm(vmin=v_min, vmax=v_max)
    else:
        norm = Normalize(vmin=0, vmax=v_max) 

    # --- *** MODIFIED PLOTTING SECTION *** ---
    
    # Create the figure and the main heatmap axis
    fig, ax_heatmap = plt.subplots(figsize=(12, 8))
    
    # Use make_axes_locatable to create the divider for new axes
    divider = make_axes_locatable(ax_heatmap)
    
    # Create axis for x-marginal (top)
    ax_histx = divider.append_axes("top", size="20%", pad=0.1, sharex=ax_heatmap)
    
    # Create axis for y-marginal (right)
    ax_histy = divider.append_axes("right", size="20%", pad=0.1, sharey=ax_heatmap)
    
    # Create axis for colorbar (right of y-marginal)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Hide labels on marginal plots that are shared
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Plot the main heatmap
    im = ax_heatmap.imshow(
        plot_data.T, 
        origin='lower', 
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='plasma', 
        interpolation='none', 
        norm=norm 
    )
    
    # Plot the marginal histograms
    # Use the original 1D data and the same bins
    ax_histx.hist(x_data, bins=xedges, histtype='step', color='black', linewidth=1.5)
    ax_histy.hist(y_data, bins=yedges, orientation='horizontal', histtype='step', color='black', linewidth=1.5)
    
    # Set marginal plot labels
    ax_histx.set_ylabel('Counts')
    ax_histy.set_xlabel('Counts')

    # Add the colorbar to its dedicated axis
    fig.colorbar(im, cax=cax, label=cbar_label)
    
    # --- Main Heatmap Formatting ---
    
    ax_heatmap.set_xlabel(f'Number of Hits per {timestamp_key}')
    ax_heatmap.set_ylabel('ToT')
    
    if normalize_columns:
        title = f'Normalized ToT vs. Hits per {timestamp_key}'
    else:
        title = f'ToT vs. Hits per {timestamp_key} (Raw Counts)'
    if log_z_scale:
        title += ' [Log Scale]'
    
    # Set title on the top marginal plot to make space
    ax_histx.set_title(title)
    
    # Adjust x-axis limit if requested
    if max_hits_to_plot:
        ax_heatmap.set_xlim(xedges[0], max_hits_to_plot + 1)
        plot_limit_x = max_hits_to_plot
    else:
        ax_heatmap.set_xlim(xedges[0], xedges[-1])
        plot_limit_x = x_max

    ax_heatmap.set_ylim(yedges[0], yedges[-1])
    
    # --- Set x-ticks (centered) ---
    if plot_limit_x <= 20:
        step = 1
    elif plot_limit_x <= 50:
        step = 4
    else:
        step = 10
        
    tick_labels = np.arange(1, plot_limit_x + 1, step)
    tick_locations = tick_labels + 0.5
    
    ax_heatmap.set_xticks(tick_locations)
    ax_heatmap.set_xticklabels(tick_labels)
    # --- End of x-tick section ---

    ax_heatmap.grid(False) 
    plt.show()
    
    # Return the main figure and an object containing all axes
    all_axes = {'heatmap': ax_heatmap, 'x_marginal': ax_histx, 'y_marginal': ax_histy, 'colorbar': cax}
    return fig, all_axes


def _get_plot_data(data, timestamp_key):
    """
    Helper function to safely extract x (hits_per_ts) and y (ToT)
    data from a raw data dictionary.
    """
    try:
        ts_data = data[timestamp_key]
        y_data = data['ToT']
        
        if len(ts_data) == 0 or len(y_data) == 0:
            print(f"Warning: Empty '{timestamp_key}' or 'ToT' array found.")
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
        print(f"An error occurred during data processing: {e}")
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

from typing import Dict, List, Optional

def compare_tot_vs_hits_plots(data_in, data_out, timestamp_key='ext_TS', 
                              max_hits_to_plot=None, 
                              normalize_columns=True, 
                              log_z_scale=True,
                              tot_threshold: Optional[float] = None):
    """
    Compares the ToT vs. Hits-per-Timestamp heatmap for two datasets
    (e.g., in-bunch vs. out-of-bunch) side-by-side on a single figure.
    
    The plots share a common y-axis (ToT) and a common color scale (z-axis)
    for direct comparison.

    Parameters:
    ----------
    data_in : dict
        The first dataset (e.g., in-bunch hits).
    data_out : dict
        The second dataset (e.g., out-of-bunch hits).
    timestamp_key : str, optional
        Key for the timestamp data (e.g., 'ext_TS' or 'TriggerTS').
    max_hits_to_plot : int, optional
        Limits the x-axis for both plots.
    normalize_columns : bool, optional
        If True, normalizes heatmap columns for both plots.
    log_z_scale : bool, optional
        Uses a logarithmic color scale for both plots.
    tot_threshold : float, optional
        If set, calculates and displays the frequency (Hz) of hits
        with ToT < this value and draws a horizontal line.
    """
    
    print("--- Processing Data for Comparison Plot ---")
    
    # --- 1.A Calculate Total Duration ---
    ts_to_seconds = 25e-9  # 25 ns
    total_duration_sec = 0.0
    global_min_ts = np.inf
    global_max_ts = -np.inf

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
        else:
            print("Warning: Could not determine valid time duration. Frequencies will be 0.")
    else:
        print("Warning: No valid timestamp data found. Frequencies will be 0.")

    # --- 1.B Get Data for both plots ---
    x_data_in, y_data_in = _get_plot_data(data_in, timestamp_key)
    x_data_out, y_data_out = _get_plot_data(data_out, timestamp_key)

    has_data_in = x_data_in is not None
    has_data_out = x_data_out is not None

    if not has_data_in and not has_data_out:
        print("Error: Both input datasets are empty. Cannot plot.")
        return None, None

    # --- 1.C Calculate Frequencies ---
    freq_in = 0.0
    freq_out = 0.0
    if tot_threshold is not None and total_duration_sec > 0:
        if has_data_in:
            count_in = np.sum(y_data_in < tot_threshold)
            freq_in = count_in / total_duration_sec
            print(f"  In-Bunch: {count_in} hits < {tot_threshold} ToT -> {freq_in:.2f} Hz")
            
        if has_data_out:
            count_out = np.sum(y_data_out < tot_threshold)
            freq_out = count_out / total_duration_sec
            print(f"  Out-Bunch: {count_out} hits < {tot_threshold} ToT -> {freq_out:.2f} Hz")

    # --- 2. Determine Global Bins ---
    x_max = 1
    y_max = 0

    if has_data_in:
        x_max = max(x_max, int(x_data_in.max()))
        y_max = max(y_max, int(y_data_in.max()))
    if has_data_out:
        x_max = max(x_max, int(x_data_out.max()))
        y_max = max(y_max, int(y_data_out.max()))
        
    bins_x = np.arange(1, x_max + 2)
    bins_y = np.arange(0, y_max + 2)
    xedges = bins_x
    yedges = bins_y

    # --- 3. Create 2D Histograms ---
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

    # --- 4. Determine Global Color Scale ---
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
        if v_min >= v_max:
            print("Warning: Cannot use LogNorm (vmin >= vmax), switching to linear scale.")
            norm = Normalize(vmin=0, vmax=v_max)
            log_z_scale = False
        else:
            norm = LogNorm(vmin=v_min, vmax=v_max)
    else:
        norm = Normalize(vmin=0, vmax=v_max)

    # --- 5. Create Plots ---
    print("Plotting comparison heatmap...")
    fig, (ax_hm_in, ax_hm_out) = plt.subplots(1, 2, figsize=(22, 10), sharey=True)
    
    title_suffix = f"(Norm. Cols, Log Z)" if normalize_columns and log_z_scale else \
                   f"(Norm. Cols)" if normalize_columns else \
                   f"(Raw Counts, Log Z)" if log_z_scale else \
                   f"(Raw Counts)"
    
    # --- Panel 1: In-Bunch ---
    divider_in = make_axes_locatable(ax_hm_in)
    ax_histx_in = divider_in.append_axes("top", size="20%", pad=0.1, sharex=ax_hm_in)
    ax_histy_in = divider_in.append_axes("right", size="20%", pad=0.1, sharey=ax_hm_in)
    ax_histx_in.tick_params(axis="x", labelbottom=False)
    ax_histy_in.tick_params(axis="y", labelleft=False)

    if has_data_in:
        im_in = ax_hm_in.imshow(
            plot_data_in.T, origin='lower', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='plasma', interpolation='none', norm=norm
        )
        ax_histx_in.hist(x_data_in, bins=xedges, histtype='step', color='black', linewidth=1.5)
        ax_histy_in.hist(y_data_in, bins=yedges, orientation='horizontal', histtype='step', color='black', linewidth=1.5)
        
        if tot_threshold is not None:
            # --- NEW: Add horizontal line ---
            ax_hm_in.axhline(y=tot_threshold, color='r', linestyle='--', linewidth=1.5, label=f'ToT Thresh ({tot_threshold})')
            ax_hm_in.legend(loc='upper right', fontsize='small')
            
            # Add Frequency Text
            if total_duration_sec > 0:
                text_str = f"Freq (ToT < {tot_threshold}): {freq_in:.1f} Hz"
                ax_hm_in.text(0.95, 0.05, text_str, transform=ax_hm_in.transAxes,
                              color='white', ha='right', va='bottom', fontsize=12,
                              bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    else:
        ax_hm_in.text(0.5, 0.5, "No In-Bunch Data", transform=ax_hm_in.transAxes, ha='center', va='center')

    ax_hm_in.set_xlabel(f'Hits per {timestamp_key}')
    ax_hm_in.set_ylabel('ToT')
    ax_histx_in.set_title(f'In-Bunch Data {title_suffix}')
    ax_histx_in.set_ylabel('Counts')
    ax_histy_in.set_xlabel('Counts')

    # --- Panel 2: Out-of-Bunch (with Colorbar) ---
    divider_out = make_axes_locatable(ax_hm_out)
    ax_histx_out = divider_out.append_axes("top", size="20%", pad=0.1, sharex=ax_hm_out)
    ax_histy_out = divider_out.append_axes("right", size="20%", pad=0.1, sharey=ax_hm_out)
    cax = divider_out.append_axes("right", size="5%", pad=0.1)
    ax_histx_out.tick_params(axis="x", labelbottom=False)
    ax_histy_out.tick_params(axis="y", labelleft=False)

    if has_data_out:
        im_out = ax_hm_out.imshow(
            plot_data_out.T, origin='lower', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='plasma', interpolation='none', norm=norm
        )
        ax_histx_out.hist(x_data_out, bins=xedges, histtype='step', color='black', linewidth=1.5)
        
        # --- FIX: Plot to ax_histy_out, not ax_histy_in ---
        ax_histy_out.hist(y_data_out, bins=yedges, orientation='horizontal', histtype='step', color='black', linewidth=1.5)
        
        cbar_label = 'Norm. Prob.' if normalize_columns else 'Counts'
        fig.colorbar(im_out, cax=cax, label=cbar_label)
        
        if tot_threshold is not None:
            # --- NEW: Add horizontal line ---
            ax_hm_out.axhline(y=tot_threshold, color='r', linestyle='--', linewidth=1.5, label=f'ToT Thresh ({tot_threshold})')
            ax_hm_out.legend(loc='upper right', fontsize='small')
            
            # Add Frequency Text
            if total_duration_sec > 0:
                text_str = f"Freq (ToT < {tot_threshold}): {freq_out:.1f} Hz"
                ax_hm_out.text(0.95, 0.05, text_str, transform=ax_hm_out.transAxes,
                               color='white', ha='right', va='bottom', fontsize=12,
                               bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    else:
        ax_hm_out.text(0.5, 0.5, "No Out-of-Bunch Data", transform=ax_hm_out.transAxes, ha='center', va='center')

    ax_hm_out.set_xlabel(f'Hits per {timestamp_key}')
    ax_histx_out.set_title(f'Out-of-Bunch Data {title_suffix}')
    ax_histx_out.set_ylabel('Counts')
    ax_histy_out.set_xlabel('Counts')

    # --- 6. Set Ticks and Limits for both plots ---
    if max_hits_to_plot:
        plot_limit_x = max_hits_to_plot
        ax_hm_in.set_xlim(xedges[0], max_hits_to_plot + 1)
        ax_hm_out.set_xlim(xedges[0], max_hits_to_plot + 1)
    else:
        plot_limit_x = x_max
        ax_hm_in.set_xlim(xedges[0], xedges[-1])
        ax_hm_out.set_xlim(xedges[0], xedges[-1])

    ax_hm_in.set_ylim(yedges[0], yedges[-1])

    if plot_limit_x <= 20: step = 1
    elif plot_limit_x <= 50: step = 4
    else: step = 10
        
    tick_labels = np.arange(1, plot_limit_x + 1, step)
    tick_locations = tick_labels + 0.5
    
    ax_hm_in.set_xticks(tick_locations)
    ax_hm_in.set_xticklabels(tick_labels)
    ax_hm_out.set_xticks(tick_locations)
    ax_hm_out.set_xticklabels(tick_labels)
    
    ax_hm_in.grid(False)
    ax_hm_out.grid(False)
    
    fig.suptitle(f'ToT vs. Hits per {timestamp_key} Comparison', fontsize=20, y=1.02)
    fig.tight_layout()
    plt.show()
    
    return fig