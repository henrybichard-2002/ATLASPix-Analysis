# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 16:16:45 2026

@author: henry
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, Tuple, List, Optional


from data_loading_optimized import load_data_numpy
from utils import numpy_to_dataframe, layer_split


Dataset = Dict[str, np.ndarray]

class BunchClassifierPipeline:
    def __init__(
        self,
        trigger_ts_unit_s: float = 25e-9,
        major_bunch_bin_s: float = 1.0,
        major_bunch_thresholds: List[int] = [2000, 2000, 2000, 2000], 
        minor_freq_hz: float = 12.5,
        minor_duration_s: float = 0.07,
        scan_range_hz: float = 0.1,
        scan_step_hz: float = 0.001,
        phase_scan_window_s: float = 0.1,
        phase_scan_step_s: float = 0.0001,
        hint_freq_range_hz: float = 0.02,
        hint_phase_range_s: float = 0.02 
    ):
        self.ts_unit = trigger_ts_unit_s
        self.major_bin = major_bunch_bin_s
        self.major_thresh_list = major_bunch_thresholds 
        self.base_freq = minor_freq_hz
        self.duration = minor_duration_s
        self.scan_range = scan_range_hz
        self.scan_step = scan_step_hz
        self.phase_window = phase_scan_window_s
        self.phase_step = phase_scan_step_s
        
        self.hint_freq_range = hint_freq_range_hz
        self.hint_phase_range = hint_phase_range_s
        
        self.layer_colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red'}

    def process(self, data: Dataset, filename_label: str, plot_result: bool = True, plot_diagnostics: bool = True) -> Dataset:
        """
        Main execution method. 
        """
        print(f"--- Starting Pipeline for: {filename_label} ---")
        start_time_global = time.time()
        
        if 'TriggerTS' not in data or 'Layer' not in data:
            raise KeyError("Dataset missing 'TriggerTS' or 'Layer'")
        
        out_data = data.copy()
        total_len = len(data['TriggerTS'])
        
        # Output Arrays
        status_array = np.full(total_len, -1, dtype=np.int8)
        ptof_array = np.full(total_len, -1, dtype=np.int16)

        # Global Data Views
        global_ts_s = data['TriggerTS'] * self.ts_unit
        global_layers = data['Layer']
        global_tots = data.get('ToT', np.zeros(total_len))

        if plot_diagnostics:
            self._plot_all_layers_major_diagnostic(global_ts_s, global_layers, filename_label)

        process_order = [4, 3, 2, 1]
        opt_hint = None 

        for layer_id in process_order:
            print(f"\n=== PROCESSING LAYER {layer_id} ===")
            
            layer_mask = (global_layers == layer_id)
            if np.sum(layer_mask) == 0:
                print(f"  No hits found.")
                continue
                
            layer_ts_s = global_ts_s[layer_mask]
            
            try:
                current_thresh = self.major_thresh_list[layer_id - 1]
            except IndexError:
                current_thresh = 4000

            major_edges = self._find_major_bunch_edges(layer_ts_s, current_thresh)
            
            if not major_edges:
                print(f"  Warning: No major bunches detected.")
                continue
            

            print(f"  Optimizing {len(major_edges)} Major Bunches...")
            
            layer_heat_bins = []
            layer_heat_tots = []
            layer_folded_distributions = [] 
            
            current_layer_freqs = []
            current_layer_phases = []

            for i, (start_s, end_s) in enumerate(major_edges):
                
                bunch_mask_local = (layer_ts_s >= start_s) & (layer_ts_s < end_s)
                bunch_times_s = layer_ts_s[bunch_mask_local]
                
                if bunch_times_s.size == 0: continue

                rel_times = bunch_times_s - start_s
                

                if opt_hint:
                    best_freq, best_phase, best_period = self._optimize_bunch_params(
                        rel_times, center_freq=opt_hint[0], center_phase=opt_hint[1], narrow_scan=True
                    )
                else:
                    best_freq, best_phase, best_period = self._optimize_bunch_params(rel_times, narrow_scan=False)

                current_layer_freqs.append(best_freq)
                current_layer_phases.append(best_phase)

    
                time_in_cycle = (rel_times - best_phase) % best_period
                
                if plot_diagnostics:
                    layer_folded_distributions.append({
                        'data': time_in_cycle,
                        'freq': best_freq,
                        'bunch_idx': i + 1
                    })


                normalized_phase = time_in_cycle / best_period
                scaled_phase = normalized_phase * 512
                bunch_ptof_bins = np.clip(np.floor(scaled_phase), 0, 511).astype(np.int16)
                
                # --- UPDATE ARRAYS ---
                layer_indices_global = np.where(layer_mask)[0]
                bunch_indices_global = layer_indices_global[bunch_mask_local]
                
                status_array[bunch_indices_global] = 0 # Major only
                ptof_array[bunch_indices_global] = bunch_ptof_bins
                
                inside_minor_mask = time_in_cycle < self.duration
                hits_inside_minor = bunch_indices_global[inside_minor_mask]
                status_array[hits_inside_minor] = 1 # Minor inside

                if plot_result:
                    layer_heat_bins.append(bunch_ptof_bins)
                    layer_heat_tots.append(global_tots[bunch_indices_global])


            if plot_diagnostics and layer_folded_distributions:
                self._plot_layer_overlays(layer_folded_distributions, layer_id, filename_label)

            if current_layer_freqs:
                avg_freq = np.mean(current_layer_freqs)
                avg_phase = np.mean(current_layer_phases)
                opt_hint = (avg_freq, avg_phase)
                print(f"  -> Next Layer Hint: F={avg_freq:.4f}Hz, P={avg_phase:.4f}s")

            if plot_result and layer_heat_bins:
                self._plot_verification_bins(layer_heat_bins, layer_heat_tots, layer_id, filename_label)

        out_data['BunchStatus'] = status_array
        out_data['pToF'] = ptof_array
        
        print(f"\n--- Pipeline Complete ({time.time() - start_time_global:.2f}s) ---")
        return out_data

    # --- CORE ALGORITHMS (Unchanged) ---
    def _find_major_bunch_edges(self, times_s: np.ndarray, threshold: int) -> List[Tuple[float, float]]:
        if times_s.size == 0: return []
        t_min, t_max = times_s[0], times_s[-1]
        bins = int((t_max - t_min) / self.major_bin)
        if bins <= 0: return []
        counts, bin_edges = np.histogram(times_s, bins=bins, range=(t_min, t_max))
        active_bins = counts >= threshold
        padded = np.concatenate(([False], active_bins, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        edges = []
        for s, e in zip(starts, ends):
            edges.append((bin_edges[s], bin_edges[e]))
        return edges

    def _optimize_bunch_params(self, rel_times: np.ndarray, center_freq=None, center_phase=None, narrow_scan=False):
        if narrow_scan and center_freq is not None:
            freq_scan_range = self.hint_freq_range
            phase_scan_range = self.hint_phase_range 
            freqs = np.arange(center_freq - freq_scan_range, center_freq + freq_scan_range, self.scan_step)
        else:
            freqs = np.arange(self.base_freq - self.scan_range, self.base_freq + self.scan_range + self.scan_step, self.scan_step)

        best_score = -len(rel_times)
        best_freq = self.base_freq if center_freq is None else center_freq
        best_phase = 0.0
        best_period = 1.0 / best_freq
        total_hits = len(rel_times)
        
        for freq in freqs:
            if freq <= 0: continue
            period = 1.0 / freq
            folded = rel_times % period
            
            if narrow_scan and center_phase is not None:
                p_start = center_phase - phase_scan_range
                p_end = center_phase + phase_scan_range
                phases = np.arange(p_start, p_end, self.phase_step)
                folded_sorted = np.sort(folded)
            else:
                nbins = int(period / 0.01)
                if nbins < 1: nbins = 1
                counts, edges = np.histogram(folded, bins=nbins, range=(0, period))
                if len(counts) == 0: continue
                est_phase = (edges[np.argmax(counts)] + edges[np.argmax(counts)+1]) / 2.0
                p_start = est_phase - self.phase_window
                p_end = est_phase + self.phase_window
                phases = np.arange(p_start, p_end, self.phase_step)
                folded_sorted = np.sort(folded)

            for phase in phases:
                ph = phase % period
                w_start = ph
                w_end = (ph + self.duration) % period
                
                if w_start < w_end:
                    i_start = np.searchsorted(folded_sorted, w_start, 'left')
                    i_end = np.searchsorted(folded_sorted, w_end, 'right')
                    hits_in = i_end - i_start
                else:
                    i_start_1 = np.searchsorted(folded_sorted, w_start, 'left')
                    i_end_1 = total_hits
                    i_start_2 = 0
                    i_end_2 = np.searchsorted(folded_sorted, w_end, 'right')
                    hits_in = (i_end_1 - i_start_1) + (i_end_2 - i_start_2)
                
                score = 2 * hits_in - total_hits
                if score > best_score:
                    best_score = score
                    best_freq = freq
                    best_phase = ph
                    best_period = period

        return best_freq, best_phase, best_period


    def _plot_all_layers_major_diagnostic(self, global_ts, global_layers, filename):
        print("Generating Global Major Bunch Diagnostic...")
        plt.figure(figsize=(15, 6))

        t_min, t_max = global_ts[0], global_ts[-1]
        bins = int((t_max - t_min) / self.major_bin)
        if bins <= 0: bins = 1

        for layer_id in [1, 2, 3, 4]:
            mask = (global_layers == layer_id)
            if np.sum(mask) == 0: continue

            ts_layer = global_ts[mask]
            counts, bin_edges = np.histogram(ts_layer, bins=bins, range=(t_min, t_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            color = self.layer_colors.get(layer_id, 'black')
            thresh = self.major_thresh_list[layer_id - 1]

            plt.plot(bin_centers, counts, drawstyle='steps-mid', 
                     label=f'Layer {layer_id}', color=color, alpha=0.8)
            plt.axhline(thresh, color=color, linestyle='--', alpha=0.5, linewidth=1.5)

        plt.xlabel("Time (s)")
        plt.ylabel(f"Counts per {self.major_bin}s bin")
        plt.title(f"{filename}: Major Bunch Detection (All Layers)")
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def _plot_layer_overlays(self, folded_data_list, layer_id, filename):
        print(f"Generating Overlay Diagnostic for Layer {layer_id}...")
        plt.figure(figsize=(14, 7))
        
        max_freq = max(d['freq'] for d in folded_data_list)
        bins = np.linspace(0, 1.0/max_freq, 128) 
        colors = plt.cm.jet(np.linspace(0, 1, len(folded_data_list)))

        for i, item in enumerate(folded_data_list):
            data = item['data']
            plt.hist(
                data, bins=bins, histtype='step', 
                color=colors[i], alpha=0.6, linewidth=1.5,
                label=f'Bunch {item["bunch_idx"]}'
            )

        plt.axvline(self.duration, color='black', linestyle='--', linewidth=2, label='Minor Cut')
        plt.xlabel("Time in Minor Cycle (s)")
        plt.ylabel("Counts per Bin")
        plt.title(f"{filename} | Layer {layer_id}: Phase Folded Distributions")
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, fontsize='small')
        plt.tight_layout()
        plt.show()

    def _plot_verification_bins(self, bin_list, tot_list, layer_id, filename):
        flat_bins = np.concatenate(bin_list)
        flat_tots = np.concatenate(tot_list)
        fig, ax = plt.subplots(figsize=(10, 5))
        h = ax.hist2d(flat_bins, flat_tots, bins=[np.arange(0,513), np.arange(0,256)], norm=LogNorm(), cmap='jet')
        approx_cut_bin = (self.duration * self.base_freq) * 512
        ax.axvline(approx_cut_bin, color='red', linestyle='--')
        plt.colorbar(h[3], ax=ax, label='Hits')
        ax.set_title(f"{filename} | Layer {layer_id}: Final Heatmap (0-511 Bins)")
        plt.show()

# =============================================================================
# 2. ToT RATIO ANALYSIS FUNCTION
# =============================================================================

def plot_tot_ratio_vs_ptof(data: dict, filename: str, tot_threshold: int = 50):
    """
    Calculates Ratio = (Hits <= Threshold) / (Hits > Threshold) per pToF.
    Plots with statistical error bars.
    """
    print(f"--- Analyzing ToT Ratio (Low/High) vs pToF (Threshold={tot_threshold}) ---")
    
    df = pd.DataFrame({
        'Layer': data['Layer'],
        'ToT': data['ToT'],
        'pToF': data['pToF']
    })
    
    df = df[(df['pToF'] >= 0) & (df['pToF'] <= 511)]
    if df.empty:
        print("No valid pToF data found for ToT Ratio Analysis.")
        return

    df['is_high_tot'] = df['ToT'] > tot_threshold
    counts = df.groupby(['Layer', 'pToF'])['is_high_tot'].value_counts().unstack(fill_value=0)
    
    if True not in counts.columns: counts[True] = 0   
    if False not in counts.columns: counts[False] = 0 
    
    N_high = counts[True]
    N_low = counts[False]
    
    denom_safe = N_high.replace(0, np.nan)
    counts['ratio'] = N_low / denom_safe
    
    term_low = np.divide(1.0, N_low, out=np.zeros_like(N_low, dtype=float), where=N_low!=0)
    term_high = np.divide(1.0, N_high, out=np.zeros_like(N_high, dtype=float), where=N_high!=0)
    counts['error'] = counts['ratio'] * np.sqrt(term_low + term_high)
    
    unique_layers = sorted(counts.index.get_level_values('Layer').unique())
    colors = ['royalblue', 'crimson', 'forestgreen', 'darkorange']
    
    plt.figure(figsize=(14, 7))
    for i, layer in enumerate(unique_layers):
        layer_data = counts.loc[layer]
        full_range = np.arange(0, 512)
        layer_data = layer_data.reindex(full_range)
        
        plt.errorbar(
            layer_data.index, 
            layer_data['ratio'], 
            yerr=layer_data['error'],
            fmt='o', markersize=3, elinewidth=1, capsize=2, alpha=0.7,
            label=f'Layer {layer}', color=colors[i % len(colors)]
        )

    plt.title(f"{filename}: Ratio (Low ToT / High ToT) vs pToF [Threshold={tot_threshold}]")
    plt.xlabel("pToF [156.25us]")
    plt.ylabel("Ratio ($N_{low} / N_{high}$)")
    plt.xlim(0, 511)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Path to the specific file
    target_file = r"C:\Users\henry\ATLASpix-analysis\data\202204061308_udp_beamonall_6Gev_kit_0_decode.dat" ###perp
    #target_file = r"C:\Users\henry\ATLASpix-analysis\data\202204071531_udp_beamonall_angle6_6Gev_kit_4_decode.dat"
    
    n_lines_to_load = 50000000
    
    # Extract filename for titles and saving
    file_label = os.path.splitext(os.path.basename(target_file))[0]
    output_txt_path = os.path.join(os.path.dirname(target_file), f"{file_label}_PROCESSED.txt")

    print(f"Target File: {target_file}")
    print(f"Output will be saved to: {output_txt_path}")

    # --- 1. LOAD DATA ---
    data_raw = load_data_numpy(target_file, n_lines=n_lines_to_load)
    if data_raw is None:
        print("Error: File not found or empty.")
        exit()

    # --- 2. RUN PIPELINE ---
    thresholds = [500, 500, 800, 800] 

    pipeline = BunchClassifierPipeline(
        trigger_ts_unit_s=25e-9,
        major_bunch_bin_s = 0.5,
        major_bunch_thresholds=thresholds,
        minor_freq_hz=12.5,
        minor_duration_s=0.07,
        phase_scan_window_s=0.1,
        phase_scan_step_s=0.0001,
        hint_freq_range_hz=0.02, 
        hint_phase_range_s=0.1 
    )

    processed_data = pipeline.process(
        data_raw, 
        filename_label=file_label, 
        plot_result=True, 
        plot_diagnostics=True
    )

    plot_tot_ratio_vs_ptof(processed_data, filename=file_label, tot_threshold=15)

    print("\nSaving processed data to text file...")
    
    df_export = numpy_to_dataframe(processed_data)

    df_export.to_csv(output_txt_path, sep='\t', index=False)
    
    print(f"Successfully saved {len(df_export)} rows to:")
    print(output_txt_path)