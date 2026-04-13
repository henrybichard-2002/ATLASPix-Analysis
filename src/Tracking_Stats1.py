import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_track_composition_summary(tracks_dict: dict):
    """
    Plots:
    1. Pie charts of track composition (Clean vs Mixed vs Xtalk) for each Track Length.
    2. A STACKED bar chart showing missing clusters per layer, broken down by track type.
    """
    if not tracks_dict:
        print("No data provided.")
        return

    # --- 1. Data Parsing & Classification ---
    n_tracks = len(tracks_dict['L4_ID'])
    
    # Storage
    track_lengths = np.zeros(n_tracks, dtype=int)
    track_types = np.full(n_tracks, 'Unknown', dtype=object)
    
    # Store indices of missing hits per layer for easy counting later
    missing_indices = {1: [], 2: [], 3: [], 4: []}
    
    # Helper for robust value extraction
    def get_val(key, i):
        if key not in tracks_dict: return None
        return tracks_dict[key][i]

    # Helper for Xtalk check
    def is_xtalk_value(val):
        if isinstance(val, (int, float, np.number)): return val != 0
        if isinstance(val, str): return val != '0'
        if isinstance(val, (list, np.ndarray)): 
            return any(v != 0 for v in val)
        return False

    # Iterate tracks
    for i in range(n_tracks):
        valid_hits = 0
        clean_hits = 0
        xtalk_hits = 0
        
        for L in [4, 3, 2, 1]:
            lid = get_val(f'L{L}_ID', i)
            
            # Check Missing
            if lid == -1:
                missing_indices[L].append(i)
                continue
            
            valid_hits += 1
            
            # Check Type
            xt = get_val(f'xtalk{L}', i)
            if is_xtalk_value(xt):
                xtalk_hits += 1
            else:
                clean_hits += 1
            
        track_lengths[i] = valid_hits
        
        # Classification
        if valid_hits == 0: track_types[i] = 'Empty'
        elif clean_hits == valid_hits: track_types[i] = 'Pure Clean'
        elif xtalk_hits == valid_hits: track_types[i] = 'Pure Xtalk'
        else: track_types[i] = 'Contaminated (Mixed)'

    # --- 2. Plotting Setup ---
    unique_lengths = sorted(np.unique(track_lengths[track_lengths > 0]), reverse=True)
    n_plots = len(unique_lengths) + 1 
    
    fig = plt.figure(figsize=(6 * n_plots, 7))
    
    # Consistent Colors
    colors = {
        'Pure Clean': '#1f77b4',       # Blue
        'Contaminated (Mixed)': '#ff7f0e', # Orange
        'Pure Xtalk': '#d62728'        # Red
    }
    
    # --- A. PIE CHARTS (Track Types by Length) ---
    for idx, length in enumerate(unique_lengths):
        ax = fig.add_subplot(1, n_plots, idx + 1)
        
        mask = (track_lengths == length)
        types = track_types[mask]
        counts = pd.Series(types).value_counts()
        
        labels = [l for l in ['Pure Clean', 'Contaminated (Mixed)', 'Pure Xtalk'] if l in counts.index]
        sizes = [counts[l] for l in labels]
        pie_colors = [colors[l] for l in labels]
        
        total = sum(sizes)
        if total > 0:
            wedges, texts, autotexts = ax.pie(
                sizes, autopct='%1.1f%%', startangle=140, 
                colors=pie_colors, pctdistance=0.85, textprops={'fontsize': 11}
            )
            ax.legend(wedges, labels, title="Track Type", loc="lower center", bbox_to_anchor=(0.5, -0.2))
            ax.set_title(f"Track Length {length}\n(N={total})", fontsize=14, fontweight='bold')
            ax.add_artist(plt.Circle((0,0),0.70,fc='white'))
        else:
            ax.text(0.5, 0.5, "No Tracks", ha='center')

    # --- B. STACKED BAR CHART (Missing Cluster Composition) ---
    ax_bar = fig.add_subplot(1, n_plots, n_plots)
    
    layers = [1, 2, 3, 4]
    categories = ['Pure Clean', 'Contaminated (Mixed)', 'Pure Xtalk']
    
    # Build data for stacking
    # We want % of TOTAL tracks
    bar_data = {cat: [] for cat in categories}
    
    for L in layers:
        idxs = missing_indices[L]
        if not idxs:
            for cat in categories: bar_data[cat].append(0)
            continue
            
        # Get types of tracks that are missing this layer
        types_missing_L = track_types[idxs]
        counts = pd.Series(types_missing_L).value_counts()
        
        for cat in categories:
            count = counts.get(cat, 0)
            percent = (count / n_tracks) * 100
            bar_data[cat].append(percent)

    # Plot Bars
    bottoms = np.zeros(len(layers))
    
    for cat in categories:
        values = bar_data[cat]
        ax_bar.bar(layers, values, bottom=bottoms, label=cat, color=colors[cat], 
                   alpha=0.8, edgecolor='white', width=0.6)
        bottoms += np.array(values)

    ax_bar.set_xticks(layers)
    ax_bar.set_xlabel("Layer ID", fontsize=12)
    ax_bar.set_ylabel("% of Total Tracks Missing this Layer", fontsize=12)
    ax_bar.set_title("Missing Cluster Composition", fontsize=14, fontweight='bold')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add Total % Labels on top
    for i, total in enumerate(bottoms):
        if total > 0:
            ax_bar.text(layers[i], total + 0.5, f'{total:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')

    # Specific Legend for Bar Chart (if needed, or rely on Pie Legends)
    # ax_bar.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_cluster_size_better(tracks_dict: dict, cluster_data: dict):
    """
    Plots Cluster Size distributions using Boxen Plots for clearer comparison 
    between track types (Clean, Contaminated, Xtalk).
    """
    if not tracks_dict or not cluster_data:
        print("No data provided.")
        return

    print("--- Processing Cluster Size Data ---")

    # --- 1. Data Parsing & Classification ---
    n_tracks = len(tracks_dict['L4_ID'])
    c_ids = cluster_data['clusterID']
    sort_idx = np.argsort(c_ids)
    sorted_c_ids = c_ids[sort_idx]
    
    def is_cluster_xtalk(val):
        if isinstance(val, (int, float, np.number)): return val != 0
        if isinstance(val, str): return val != '0'
        if isinstance(val, (list, np.ndarray)): return any(v != 0 for v in val)
        return False

    is_pure_clean = np.ones(n_tracks, dtype=bool)
    is_pure_xtalk = np.ones(n_tracks, dtype=bool)
    
    for L in [4, 3, 2, 1]:
        xt_col = tracks_dict[f'xtalk{L}']
        id_col = tracks_dict[f'L{L}_ID']
        
        # Determine status
        current_is_xtalk = [(i != -1 and is_cluster_xtalk(x)) for i, x in zip(id_col, xt_col)]
        current_is_clean = [(i != -1 and not is_cluster_xtalk(x)) for i, x in zip(id_col, xt_col)]
        
        is_pure_clean[np.array(current_is_xtalk)] = False
        is_pure_xtalk[np.array(current_is_clean)] = False

    is_mixed = (~is_pure_clean) & (~is_pure_xtalk)
    
    # --- FIX: Initialize with object to avoid string truncation ---
    track_types = np.full(n_tracks, 'Unknown', dtype=object)
    track_types[is_pure_clean] = 'Pure Clean'
    track_types[is_pure_xtalk] = 'Pure Xtalk'
    track_types[is_mixed] = 'Contaminated'
    
    # --- 2. Extract Data ---
    data_records = []
    
    for L in [4, 3, 2, 1]:
        ids = tracks_dict[f'L{L}_ID']
        valid_mask = (ids != -1)
        
        valid_ids = ids[valid_mask]
        valid_types = track_types[valid_mask]
        
        # Lookup widths
        search_pos = np.searchsorted(sorted_c_ids, valid_ids)
        search_pos = np.clip(search_pos, 0, len(sorted_c_ids)-1)
        
        found_ids = sorted_c_ids[search_pos]
        match_mask = (found_ids == valid_ids)
        
        if np.sum(match_mask) == 0: continue
            
        final_orig_idx = sort_idx[search_pos[match_mask]]
        final_types = valid_types[match_mask]
        
        w_cols = cluster_data['width_col'][final_orig_idx]
        w_rows = cluster_data['width_row'][final_orig_idx]
        
        df_layer = pd.DataFrame({
            'Layer': f'Layer {L}',
            'Type': final_types,
            'Col Width': w_cols,
            'Row Width': w_rows
        })
        data_records.append(df_layer)

    if not data_records: 
        print("No valid cluster data found.")
        return

    df_all = pd.concat(data_records)
    
    # Melt
    df_melt = df_all.melt(id_vars=['Layer', 'Type'], 
                          value_vars=['Col Width', 'Row Width'], 
                          var_name='Dimension', value_name='Size')

    # --- 3. Plotting ---
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Column Widths
    ax1 = plt.subplot(2, 1, 1)
    sns.boxenplot(
        data=df_melt[df_melt['Dimension']=='Col Width'], 
        x='Layer', y='Size', hue='Type',
        palette={'Pure Clean': '#1f77b4', 'Contaminated': '#ff7f0e', 'Pure Xtalk': '#d62728'},
        ax=ax1
    )
    ax1.set_title("Column Cluster Size Distribution", fontsize=14, fontweight='bold')
    ax1.set_xlabel("")
    #ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

    # Plot 2: Row Widths
    ax2 = plt.subplot(2, 1, 2)
    sns.boxenplot(
        data=df_melt[df_melt['Dimension']=='Row Width'], 
        x='Layer', y='Size', hue='Type',
        palette={'Pure Clean': '#1f77b4', 'Contaminated': '#ff7f0e', 'Pure Xtalk': '#d62728'},
        ax=ax2
    )
    ax2.set_title("Row Cluster Size Distribution", fontsize=14, fontweight='bold')
    #ax2.set_yscale('log')
    ax2.legend().remove()
    ax2.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)

    plt.suptitle("Cluster Size Comparison (Log Scale)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- Summary Stats ---
    print("\n--- Mean Cluster Sizes ---")
    summary = df_melt.groupby(['Layer', 'Dimension', 'Type'])['Size'].mean().unstack()
    print(summary.round(2))
    
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_energy_vs_size_box(tracks_dict: dict):
    """
    Fast visualization of Energy vs Cluster Size using Box Plots.
    Separated by Layer (Rows) and colored by Track Type.
    """
    if not tracks_dict:
        print("No data provided.")
        return

    print("--- Analyzing Energy vs Cluster Size (Box Plot) ---")

    # --- 1. Data Parsing & Classification ---
    n_tracks = len(tracks_dict['L4_ID'])
    
    # Classify Tracks
    def is_xtalk_val(x):
        if isinstance(x, (int, float, np.number)): return x != 0
        if isinstance(x, str): return x != '0'
        if isinstance(x, (list, np.ndarray)): return any(v != 0 for v in x)
        return False

    is_pure_clean = np.ones(n_tracks, dtype=bool)
    is_pure_xtalk = np.ones(n_tracks, dtype=bool)

    for L in [4, 3, 2, 1]:
        xt_col = tracks_dict[f'xtalk{L}']
        id_col = tracks_dict[f'L{L}_ID']
        
        # Vectorized check
        curr_xtalk = [(i != -1 and is_xtalk_val(x)) for i, x in zip(id_col, xt_col)]
        curr_clean = [(i != -1 and not is_xtalk_val(x)) for i, x in zip(id_col, xt_col)]
        
        is_pure_clean[np.array(curr_xtalk)] = False
        is_pure_xtalk[np.array(curr_clean)] = False

    track_types = np.full(n_tracks, 'Contaminated', dtype=object)
    track_types[is_pure_clean] = 'Clean'
    track_types[is_pure_xtalk] = 'Xtalk'

    # --- 2. Build DataFrame ---
    records = []
    
    for L in [4, 3, 2, 1]:
        tot_col = np.array(tracks_dict[f'tot{L}'])
        
        # Get hits (handle potential missing columns safely)
        if f'nhits{L}' in tracks_dict:
             nhits_col = np.array(tracks_dict[f'nhits{L}'])
        else:
             ptof = tracks_dict[f'pToF{L}']
             nhits_col = np.array([len(x) if isinstance(x, list) else 1 for x in ptof])
        
        ids = np.array(tracks_dict[f'L{L}_ID'])
        valid = (ids != -1)
        
        df_L = pd.DataFrame({
            'Layer': f'Layer {L}',
            'Type': track_types[valid],
            'ToT': tot_col[valid],
            'Hits': nhits_col[valid]
        })
        records.append(df_L)
        
    if not records: return
    df = pd.concat(records)
    
    # Order types for consistent coloring
    df['Type'] = pd.Categorical(df['Type'], categories=['Clean', 'Contaminated', 'Xtalk'], ordered=True)
    
    g = sns.FacetGrid(df, row='Layer', height=3, aspect=2.5, sharex=True, sharey=True)
    
    g.map_dataframe(sns.boxplot, x='Hits', y='ToT', hue='Type', 
                    palette={'Clean': '#1f77b4', 'Contaminated': '#ff7f0e', 'Xtalk': '#d62728'},
                    fliersize=2, linewidth=1) # fliersize=2 makes outliers small dots
    
    g.add_legend(title="Track Type")
    g.set_axis_labels("Cluster Size (Hits)", "Energy (ToT)")
    g.set(ylim=(0, 600), xlim=(-0.5, 7.5)) # Limit hits to meaningful range
    
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    g.fig.suptitle("Energy vs Cluster Size Profile", y=1.02, fontsize=16)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_energy_density_mixed_style(tracks_dict: dict, x_limit: float = 200.0, bw_adjust: float = 1.0):
    """
    Plots energy density distributions with mixed styles:
    - Pure Clean / Contaminated: Smoothed KDE (Filled)
    - Pure Xtalk: Step Histogram (Dashed Line, No Fill) - NO KDE
    """
    if not tracks_dict:
        print("No data provided.")
        return

    print(f"--- Analyzing Energy Density (Mixed Styles, bw={bw_adjust}) ---")

    # --- 1. Classify Tracks ---
    n_tracks = len(tracks_dict['L4_ID'])

    def is_xtalk_val(x):
        if isinstance(x, (int, float, np.number)): return x != 0
        if isinstance(x, str): return x != '0'
        if isinstance(x, (list, np.ndarray)): return any(v != 0 for v in x)
        return False

    is_pure_clean = np.ones(n_tracks, dtype=bool)
    is_pure_xtalk = np.ones(n_tracks, dtype=bool)

    for L in [4, 3, 2, 1]:
        xt_col = tracks_dict[f'xtalk{L}']
        id_col = tracks_dict[f'L{L}_ID']

        curr_xtalk = [(i != -1 and is_xtalk_val(x)) for i, x in zip(id_col, xt_col)]
        curr_clean = [(i != -1 and not is_xtalk_val(x)) for i, x in zip(id_col, xt_col)]

        is_pure_clean[np.array(curr_xtalk)] = False
        is_pure_xtalk[np.array(curr_clean)] = False

    track_types = np.full(n_tracks, 'Contaminated', dtype=object)
    track_types[is_pure_clean] = 'Pure Clean'
    track_types[is_pure_xtalk] = 'Pure Xtalk'

    # --- 2. Build Dataframe ---
    plot_data = []

    for L in [4, 3, 2, 1]:
        tot_col = np.array(tracks_dict[f'tot{L}'])
        id_col = np.array(tracks_dict[f'L{L}_ID'])

        if f'nhits{L}' in tracks_dict:
             nhits_col = np.array(tracks_dict[f'nhits{L}'])
        else:
             ptof = tracks_dict[f'pToF{L}']
             nhits_col = np.array([len(x) if isinstance(x, list) else 1 for x in ptof])

        valid_mask = (id_col != -1) & (nhits_col > 0)
        avg_energy = tot_col[valid_mask] / nhits_col[valid_mask]

        df_L = pd.DataFrame({
            'Layer': f'Layer {L}',
            'Type': track_types[valid_mask],
            'Avg Energy/Pixel': avg_energy
        })
        plot_data.append(df_L)

    if not plot_data: return
    df_all = pd.concat(plot_data)

    # Ordering
    df_all['Type'] = pd.Categorical(df_all['Type'], 
                                    categories=['Pure Clean', 'Contaminated', 'Pure Xtalk'], 
                                    ordered=True)

    # --- 3. Plotting ---
    colors = {'Pure Clean': '#1f77b4', 'Contaminated': '#ff7f0e', 'Pure Xtalk': '#d62728'}

    # Initialize Grid
    g = sns.FacetGrid(df_all, col='Layer', col_wrap=2, 
                      sharex=True, sharey=False, height=4, aspect=1.5)

    # A. Plot KDEs ONLY for Clean and Contaminated
    # We filter using hue_order so Xtalk is ignored by this call
    g.map_dataframe(sns.kdeplot, x='Avg Energy/Pixel', hue='Type',
                    hue_order=['Pure Clean', 'Contaminated'],
                    palette=colors,
                    fill=True, alpha=0.1, linewidth=2,
                    common_norm=False, 
                    bw_adjust=bw_adjust, 
                    clip=(0, x_limit))

    # B. Plot Step Histogram ONLY for Pure Xtalk
    # We filter using hue_order so only Xtalk is processed here
    g.map_dataframe(sns.histplot, x='Avg Energy/Pixel', hue='Type',
                    hue_order=['Pure Xtalk'],
                    palette=colors,
                    element="step", fill=False,
                    stat="density", common_norm=False,
                    linewidth=2.5, linestyle='--',
                    bins=30, binrange=(0, x_limit))

    # --- 4. Custom Legend ---
    # Since we mixed plot types, we build the legend manually to be clear
    legend_elements = [
        Patch(facecolor=colors['Pure Clean'], edgecolor=colors['Pure Clean'], alpha=0.3, label='Pure Clean (Smooth)'),
        Patch(facecolor=colors['Contaminated'], edgecolor=colors['Contaminated'], alpha=0.3, label='Contaminated (Smooth)'),
        Line2D([0], [0], color=colors['Pure Xtalk'], lw=2.5, linestyle='--', label='Pure Xtalk (Raw Steps)')
    ]

    # Add legend to the figure (top right usually works well)
    g.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    # --- Formatting ---
    g.set_titles("{col_name}")
    g.set_axis_labels("Avg Energy per Pixel (ToT / Hits)", "Density")

    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, x_limit)

    plt.subplots_adjust(top=0.88, right=0.85) # Make room for legend
    g.fig.suptitle(f"Energy Density Profile (Mixed Styles)", fontsize=16, x=0.45)
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ptof_by_type(tracks_dict: dict):
    """
    Plots pToF distributions (0-511, with 512 as overflow/-1)
    separated by Track Type (Clean, Contaminated, Xtalk).
    """
    if not tracks_dict:
        print("No data provided.")
        return

    print("--- Analyzing pToF Distribution by Type ---")

    n_tracks = len(tracks_dict['L4_ID'])

    # --- 1. Classify Tracks ---
    def is_xtalk_val(x):
        if isinstance(x, (int, float, np.number)): return x != 0
        if isinstance(x, str): return x != '0'
        if isinstance(x, (list, np.ndarray)): return any(v != 0 for v in x)
        return False

    is_pure_clean = np.ones(n_tracks, dtype=bool)
    is_pure_xtalk = np.ones(n_tracks, dtype=bool)

    for L in [4, 3, 2, 1]:
        xt_col = tracks_dict[f'xtalk{L}']
        id_col = tracks_dict[f'L{L}_ID']

        curr_xtalk = [(i != -1 and is_xtalk_val(x)) for i, x in zip(id_col, xt_col)]
        curr_clean = [(i != -1 and not is_xtalk_val(x)) for i, x in zip(id_col, xt_col)]

        is_pure_clean[np.array(curr_xtalk)] = False
        is_pure_xtalk[np.array(curr_clean)] = False

    track_types = np.full(n_tracks, 'Contaminated', dtype=object)
    track_types[is_pure_clean] = 'Pure Clean'
    track_types[is_pure_xtalk] = 'Pure Xtalk'

    # --- 2. Extract pToF ---
    ptof_values = []

    for i in range(n_tracks):
        track_ptof = -1
        # Priority L4 -> L1
        for L in [4, 3, 2, 1]:
            val = tracks_dict[f'pToF{L}'][i]
            if isinstance(val, (list, np.ndarray)):
                if len(val) > 0: val = val[0]
                else: continue

            if val is not None and val != -1 and val != '-1':
                try:
                    track_ptof = int(val)
                    break 
                except:
                    continue

        # Map -1 or out-of-range to 512
        if track_ptof < 0 or track_ptof > 511:
            ptof_values.append(512)
        else:
            ptof_values.append(track_ptof)

    # --- 3. Build DataFrame ---
    df = pd.DataFrame({
        'pToF': ptof_values,
        'Type': track_types
    })

    # Ordering
    df['Type'] = pd.Categorical(df['Type'], categories=['Pure Clean', 'Contaminated', 'Pure Xtalk'], ordered=True)

    # --- 4. Plotting ---
    # We use a FacetGrid (Rows) to see each distribution clearly without overlap
    # We use "layer" type plots or histograms

    colors = {'Pure Clean': '#1f77b4', 'Contaminated': '#ff7f0e', 'Pure Xtalk': '#d62728'}

    g = sns.FacetGrid(df, row='Type', hue='Type', palette=colors, 
                      height=3, aspect=3, sharex=True, sharey=False)

    # Map Histogram
    # binwidth=4 gives roughly 128 bins for the 0-512 range
    g.map(sns.histplot, 'pToF', bins=np.arange(0, 514, 4), element="step", fill=True, alpha=0.4)

    # Add vertical line for the overflow bin
    def plot_boundaries(**kwargs):
        plt.axvline(511.5, color='black', linestyle='--', alpha=0.3)
        plt.text(515, plt.gca().get_ylim()[1]*0.8, "Invalid (-1)", fontsize=9, color='red')

    g.map(plot_boundaries)

    # --- Formatting ---
    g.set_titles("{row_name} Tracks")
    g.set_axis_labels("pToF Value (0-511)", "Count")
    g.set(xlim=(0, 530))

    # Add Grid
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.3)

        # Log scale option?
        # ax.set_yscale('log') # Uncomment if Xtalk counts are invisible compared to Clean

    plt.subplots_adjust(top=0.92)
    g.fig.suptitle("pToF Distribution by Track Type", fontsize=16)
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_and_analyze_ghosts(tracks: dict, time_window: float = 100.0):
    """
    Finds and plots pairs where:
    1. Reference: A Complete (4-hit) Pure Clean Track.
    2. Candidate: A Pure Xtalk Track with 2 or 3 hits.
    3. Condition: Occur within time_window, but share NO clusters (disjoint).
    """
    print(f"--- Searching for Ghost Tracks (Window={time_window}) ---")
    
    n_tracks = len(tracks['L4_ID'])
    
    # --- 1. Robust Classification ---
    def is_xtalk_val(x):
        if isinstance(x, (int, float, np.number)): return x != 0
        if isinstance(x, str): return x != '0'
        if isinstance(x, (list, np.ndarray)): return any(v != 0 for v in x)
        return False

    # Store properties for fast filtering
    is_pure_clean = np.ones(n_tracks, dtype=bool)
    is_pure_xtalk = np.ones(n_tracks, dtype=bool)
    n_hits = np.zeros(n_tracks, dtype=int)
    
    # Pre-calculate time/pos centers
    t_sum = np.zeros(n_tracks); x_sum = np.zeros(n_tracks); y_sum = np.zeros(n_tracks)
    
    # Map for ID checking later
    ids_map = {L: tracks[f'L{L}_ID'] for L in [4,3,2,1]}

    for L in [4, 3, 2, 1]:
        ids = tracks[f'L{L}_ID']
        xt = tracks[f'xtalk{L}']
        
        # Identify valid hits
        valid = (ids != -1)
        n_hits += valid.astype(int)
        
        # Update Clean/Xtalk Status
        curr_xtalk = [(i != -1 and is_xtalk_val(x)) for i, x in zip(ids, xt)]
        curr_clean = [(i != -1 and not is_xtalk_val(x)) for i, x in zip(ids, xt)]
        
        is_pure_clean[np.array(curr_xtalk)] = False
        is_pure_xtalk[np.array(curr_clean)] = False
        
        # Accumulate sums for mean calc
        # (Use 0 for invalid to avoid NaNs, we divide by count later)
        ts = np.array(tracks[f't{L}']); xs = np.array(tracks[f'x{L}']); ys = np.array(tracks[f'y{L}'])
        
        t_sum += np.where(valid, ts, 0)
        x_sum += np.where(valid, xs, 0)
        y_sum += np.where(valid, ys, 0)

    # Calculate Means
    track_t = np.divide(t_sum, n_hits, out=np.zeros_like(t_sum), where=n_hits!=0)
    track_x = np.divide(x_sum, n_hits, out=np.zeros_like(x_sum), where=n_hits!=0)
    track_y = np.divide(y_sum, n_hits, out=np.zeros_like(y_sum), where=n_hits!=0)

    # --- 2. Define Groups ---
    # Group A: Complete (4-hit) Pure Clean
    idx_clean = np.where(is_pure_clean & (n_hits == 4))[0]
    
    # Group B: Partial (2 or 3-hit) Pure Xtalk
    idx_ghost = np.where(is_pure_xtalk & ((n_hits == 2) | (n_hits == 3)))[0]
    
    print(f"  Ref Tracks (Clean 4-hit): {len(idx_clean)}")
    print(f"  Ghost Candidates (Xtalk 2/3-hit): {len(idx_ghost)}")
    
    if len(idx_clean) == 0 or len(idx_ghost) == 0:
        print("Not enough tracks to correlate.")
        return

    # --- 3. Fast Temporal Matching ---
    # Sort ghosts by time
    ghost_order = np.argsort(track_t[idx_ghost])
    idx_ghost_sorted = idx_ghost[ghost_order]
    times_ghost_sorted = track_t[idx_ghost_sorted]
    
    pairs = []
    
    for c_idx in idx_clean:
        t_c = track_t[c_idx]
        
        # Binary Search window
        start = np.searchsorted(times_ghost_sorted, t_c - time_window, side='left')
        end = np.searchsorted(times_ghost_sorted, t_c + time_window, side='right')
        
        candidates = idx_ghost_sorted[start:end]
        
        if len(candidates) == 0: continue
            
        # Get Clean ID Set
        c_set = {ids_map[L][c_idx] for L in [4,3,2,1] if ids_map[L][c_idx] != -1}
        
        for g_idx in candidates:
            # Disjoint Check
            g_set = {ids_map[L][g_idx] for L in [4,3,2,1] if ids_map[L][g_idx] != -1}
            
            if c_set.isdisjoint(g_set):
                # We found a ghost!
                pairs.append({
                    'Clean_Idx': c_idx,
                    'Ghost_Idx': g_idx,
                    'Ghost_Hits': n_hits[g_idx],
                    'dt': track_t[g_idx] - t_c,
                    'dx': track_x[g_idx] - track_x[c_idx],
                    'dy': track_y[g_idx] - track_y[c_idx]
                })

    print(f"  Found {len(pairs)} correlated ghost pairs.")
    if len(pairs) == 0: return

    df_pairs = pd.DataFrame(pairs)

    # --- 4. Visualization ---
    fig = plt.figure(figsize=(14, 6))
    
    # Plot A: Spatial Offset Heatmap
    # If ghosts are real crosstalk, they often appear at fixed offsets (e.g. adjacent columns)
    ax1 = plt.subplot(1, 2, 1)
    
    # Filter very far outliers for cleaner plot
    df_plot = df_pairs[(df_pairs['dx'].abs() < 50) & (df_pairs['dy'].abs() < 50)]
    
    h = ax1.hist2d(df_plot['dx'], df_plot['dy'], bins=60, 
                   range=[[-30, 30], [-30, 30]], cmap='inferno', cmin=1)
    fig.colorbar(h[3], ax=ax1, label='Count')
    
    ax1.set_title("Spatial Location of Ghost Track\n(Relative to True Track)")
    ax1.set_xlabel(r"$\Delta$ Column (pixels)")
    ax1.set_ylabel(r"$\Delta$ Row (pixels)")
    ax1.axvline(0, color='white', linestyle='--', alpha=0.3)
    ax1.axhline(0, color='white', linestyle='--', alpha=0.3)
    
    # Plot B: Breakdown by Ghost Length
    ax2 = plt.subplot(1, 2, 2)
    sns.histplot(data=df_pairs, x='dt', hue='Ghost_Hits', palette='viridis', 
                 element='step', bins=40, ax=ax2)
    
    ax2.set_title("Time Difference Distribution")
    ax2.set_xlabel("Time Diff (Ghost - True)")
    
    plt.tight_layout()
    plt.show()
    
    return df_pairs

# Example Usage
# ghost_df = find_and_analyze_ghosts(tracks, time_window=100)