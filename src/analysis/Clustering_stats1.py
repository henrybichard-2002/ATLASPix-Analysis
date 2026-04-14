import numpy as np
import matplotlib.pyplot as plt

def plot_xtalk_composition(data_dict, mode='clusters', title=None):
    """
    Plots crosstalk composition with a custom defining title.
    
    Parameters:
    -----------
    data_dict : dict
        Either 'final_clusters' (for mode='clusters') or 'best_tracks' (for mode='tracks').
    mode : str
        'clusters' -> Stacked bar chart of HIT composition per layer.
        'tracks'   -> Pie chart of TRACK purity classification.
    title : str, optional
        A defining string to label the graph (e.g., "Raw Data", "Best Unique Tracks").
    """
    
    # ==========================================
    # MODE 1: CLUSTER HIT COMPOSITION (Stacked Bar)
    # ==========================================
    if mode == 'clusters':
        layers = [4, 3, 2, 1]
        
        # Stats structure: {Layer: {0: count, 1: count, 2: count}}
        stats = {L: {0: 0, 1: 0, 2: 0} for L in layers}
        
        print(f"--- Analyzing Individual Hits ({title if title else 'Clusters'}) ---")
        
        for L in layers:
            # Filter for this layer
            if 'Layer' not in data_dict:
                print("Error: Input dictionary missing 'Layer' key.")
                return
            
            mask = (data_dict['Layer'] == L)
            if np.sum(mask) == 0: continue
            
            # Get raw xtalk data
            raw_data = data_dict['xtalk_type'][mask]
            
            # Iterate through every cluster
            for entry in raw_data:
                # Normalize entry to a list
                if isinstance(entry, (list, np.ndarray)):
                    items = entry
                else:
                    items = [entry]
                
                # Count items
                for item in items:
                    try:
                        val = int(item)
                        if val in [0, 1, 2]:
                            stats[L][val] += 1
                        else:
                            stats[L][0] += 1 # Noise -> Clean
                    except (ValueError, TypeError):
                        stats[L][0] += 1 # Nulls -> Clean

        # Prepare Plotting Data
        clean_pct, type1_pct, type2_pct = [], [], []
        labels = []
        
        print(f"{'Layer':<6} | {'Clean':<8} | {'Type 1':<8} | {'Type 2':<8}")
        print("-" * 40)

        for L in layers:
            total = sum(stats[L].values())
            if total > 0:
                c = stats[L][0] / total * 100
                t1 = stats[L][1] / total * 100
                t2 = stats[L][2] / total * 100
            else:
                c, t1, t2 = 0, 0, 0
            
            clean_pct.append(c)
            type1_pct.append(t1)
            type2_pct.append(t2)
            labels.append(f"Layer {L}\n(N={total})")
            
            print(f"L{L:<5} | {stats[L][0]:<8} | {stats[L][1]:<8} | {stats[L][2]:<8}")

        # Plot Stacked Bar
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(layers))
        width = 0.6
        
        p1 = ax.bar(x, clean_pct, width, label='Clean (Type 0)', color='lightgrey', edgecolor='black')
        p2 = ax.bar(x, type1_pct, width, bottom=clean_pct, label='Type 1', color='#1f77b4', edgecolor='black', hatch='//')
        
        bot_t2 = [c + t for c, t in zip(clean_pct, type1_pct)]
        p3 = ax.bar(x, type2_pct, width, bottom=bot_t2, label='Type 2', color='#d62728', edgecolor='black', hatch='..')
        
        # Add labels
        for rects in [p1, p2, p3]:
            for rect in rects:
                h = rect.get_height()
                if h > 2.0:
                    ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + h/2.,
                            f"{h:.1f}%", ha='center', va='center', fontsize=9, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
        
        # Set Title
        main_title = title if title else "Composition of Individual Hits (Flattened Clusters)"
        ax.set_title(main_title, fontsize=14)
        
        ax.set_ylabel("Percentage of Hits (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ==========================================
    # MODE 2: TRACK PURITY (Pie Chart)
    # ==========================================
    elif mode == 'tracks':
        print(f"--- Analyzing Track Purity ({title if title else 'Unknown Data'}) ---")
        
        # We need to check xtalk4, xtalk3, xtalk2, xtalk1 for each track
        n_tracks = len(data_dict['L4_ID'])
        
        # Counters
        n_pure_clean = 0       # All clusters are pure clean (0 or [0,0])
        n_pure_xtalk = 0       # ALL clusters have at least some xtalk
        n_contaminated = 0     # Mixed: some clean clusters, some xtalk clusters
        
        # Helper to check if a single cluster entry has ANY xtalk
        def has_xtalk(entry):
            if isinstance(entry, (list, np.ndarray)):
                # If list [0, 1], sum is > 0. If [0,0], sum is 0.
                # Check if ANY element is > 0
                for x in entry:
                    try: 
                        if int(x) > 0: return True
                    except: pass
                return False
            else:
                try: return int(entry) > 0
                except: return False

        # Iterate tracks
        for i in range(n_tracks):
            # Check the 4 layers for this track
            # We only check layers that actually have hits (ID != -1)
            cluster_statuses = [] # True if xtalk, False if clean
            
            for L in [4, 3, 2, 1]:
                if data_dict[f'L{L}_ID'][i] != -1:
                    val = data_dict[f'xtalk{L}'][i]
                    cluster_statuses.append(has_xtalk(val))
            
            if not cluster_statuses: continue # Empty track?
            
            if all(cluster_statuses):
                n_pure_xtalk += 1
            elif not any(cluster_statuses): # All False
                n_pure_clean += 1
            else:
                n_contaminated += 1
                
        # Pie Chart Data
        counts = [n_pure_clean, n_contaminated, n_pure_xtalk]
        labels = ['Pure Clean Track\n(All clusters clean)', 
                  'Contaminated Track\n(Mixed Clean/Xtalk)', 
                  'Pure Xtalk Track\n(All clusters have xtalk)']
        colors = ['lightgrey', 'orange', '#d62728']
        explode = (0.05, 0, 0) # Pop out the clean slice
        
        total = sum(counts)
        if total == 0:
            print("No tracks found.")
            return

        print(f"Total Tracks: {total}")
        print(f"Pure Clean:   {n_pure_clean} ({n_pure_clean/total:.1%})")
        print(f"Contaminated: {n_contaminated} ({n_contaminated/total:.1%})")
        print(f"Pure Xtalk:   {n_pure_xtalk} ({n_pure_xtalk/total:.1%})")

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(counts, explode=explode, labels=labels, autopct='%1.1f%%',
                                          shadow=True, startangle=140, colors=colors)
        
        # Formatting text
        plt.setp(autotexts, size=10, weight="bold", color="black")
        
        # Set Title
        main_title = title if title else "Track Purity Classification"
        ax.set_title(f"{main_title}\n(N={total})", fontsize=14)
        plt.show()

# ==========================================
# USAGE
# ==========================================
plot_xtalk_composition(final_clusters, mode='clusters', title="Raw Cluster Composition")
plot_xtalk_composition(all_clsts, mode='clusters', title="All Track-Associated Clusters")
plot_xtalk_composition(all_trks, mode='tracks', title="All Tracks (Redundant included)")
plot_xtalk_composition(best_clsts, mode='clusters', title="Best Unique Track Clusters")
plot_xtalk_composition(best_trks, mode='tracks', title="Best Unique Tracks")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_xtalk_scatter_all_layers(cluster_dict, x_key, y_key, num_hits=None, xtalk_types=None, alpha=0.6):
    """
    Plots a single Scatter Plot with Marginal Histograms for ALL layers combined.
    
    - Scatter: Balanced sampling (visual density is normalized by the smallest group).
    - Marginals: TRUE SUM distribution (shows actual counts of the full dataset).
    
    Parameters:
    -----------
    cluster_dict : dict
        The dictionary of cluster data arrays.
    x_key, y_key : str
        Keys for axes data (e.g., 'avg_ToT', 'pToF').
    num_hits : int, optional
        Filter by exact hit count.
    xtalk_types : list, optional
        Filter by crosstalk types [0, 1, 2, 3].
    alpha : float
        Transparency.
    """
    
    # 1. Configuration
    if xtalk_types is None: 
        xtalk_types = [0, 1, 2, 3]
        
    all_groups = [
        {'label': 'Clean (0)',      'val': 0, 'color': 'grey',      'marker': 'o'},
        {'label': 'Capacitive (1)', 'val': 1, 'color': '#1f77b4',   'marker': '^'}, # Blue
        {'label': 'Ambiguous (2)',  'val': 2, 'color': '#d62728',   'marker': 's'}, # Red
        {'label': 'Mixed (3)',      'val': 3, 'color': '#2ca02c',   'marker': 'D'}  # Green
    ]
    
    groups_def = [g for g in all_groups if g['val'] in xtalk_types]
    
    # Setup Figure Layout (Single large plot with margins)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 4, wspace=0.05, hspace=0.05,
                           width_ratios=[0.2, 1, 1, 1], 
                           height_ratios=[1, 1, 1, 0.2])
    
    # Axes placement:
    # Scatter: Bottom-Left 3x3 (Rows 1-3, Cols 0-2)
    # Top Hist: Row 0, Cols 0-2
    # Right Hist: Rows 1-3, Col 3
    
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_xhist = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    
    # Hide inner ticks
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)
    
    hit_str = f"Size: {num_hits} hits" if num_hits else "Size: All"
    print(f"--- Plotting {y_key} vs {x_key} (All Layers, {hit_str}) ---")

    # --- Helper: Robust Data Extractor (MIN/Earliest for lists) ---
    def get_clean_array(key, mask):
        raw = cluster_dict[key][mask]
        clean = np.empty(len(raw), dtype=float)
        for i, val in enumerate(raw):
            try:
                if isinstance(val, (list, np.ndarray, tuple)):
                    # Take MIN (earliest/smallest)
                    clean[i] = np.min(val) if len(val) > 0 else np.nan
                else:
                    clean[i] = float(val)
            except:
                clean[i] = np.nan
        return clean

    # --- Helper: Integer Bins ---
    def get_int_bins(data):
        valid = data[~np.isnan(data)]
        if len(valid) == 0: return np.linspace(0, 1, 10)
        mn, mx = np.floor(np.min(valid)), np.ceil(np.max(valid))
        if mx - mn > 200: return np.linspace(mn, mx, 60)
        return np.arange(mn - 0.5, mx + 1.5, 1)

    # 1. Filter Data (Global Mask)
    # Start with all True
    mask = np.ones(len(cluster_dict['Layer']), dtype=bool)
    
    if num_hits is not None:
        mask &= (cluster_dict['n_hits'] == num_hits)
        
    # Check if we have data
    if np.sum(mask) == 0:
        ax_main.text(0.5, 0.5, "No Data", ha='center', transform=ax_main.transAxes)
        plt.show()
        return

    # 2. Extract Data
    l_x = get_clean_array(x_key, mask)
    l_y = get_clean_array(y_key, mask)
    raw_xtalk = cluster_dict['xtalk_type'][mask]
    
    # 3. Process Xtalk Types
    l_xtalk = np.full(len(raw_xtalk), -1, dtype=int)
    for i, val in enumerate(raw_xtalk):
        try:
            if isinstance(val, (list, np.ndarray, tuple)):
                l_xtalk[i] = 3
            else:
                v = int(val)
                l_xtalk[i] = v if v in [0,1,2] else -1
        except: pass

    # 4. Calculate Bins
    bins_x = get_int_bins(l_x)
    bins_y = get_int_bins(l_y)

    # 5. Identify Indices & Balance
    indices = {}
    counts = []
    for g in groups_def:
        idxs = np.where(l_xtalk == g['val'])[0]
        indices[g['label']] = idxs
        if len(idxs) > 0: counts.append(len(idxs))
    
    # Scatter plot subsampling limit (smallest group)
    limit = min(counts) if counts else 0
    
    # 6. Plotting
    plotted = False
    for g in groups_def:
        idxs = indices[g['label']]
        if len(idxs) == 0: continue
        
        # --- A) Scatter (Balanced Subsampling) ---
        if len(idxs) > limit:
            scatter_idxs = np.random.choice(idxs, limit, replace=False)
        else:
            scatter_idxs = idxs
            
        ax_main.scatter(l_x[scatter_idxs], l_y[scatter_idxs], 
                        label=f"{g['label']}", c=g['color'], marker=g['marker'],
                        alpha=alpha, s=25, edgecolors='none')
        
        # --- B) Marginal Hists (True Sum / Full Count) ---
        # Note: density=False ensures we show the real magnitude of this group
        ax_xhist.hist(l_x[idxs], bins=bins_x, histtype='step', 
                      color=g['color'], linewidth=2, density=False)
        
        ax_yhist.hist(l_y[idxs], bins=bins_y, histtype='step', orientation='horizontal',
                      color=g['color'], linewidth=2, density=False)
        
        plotted = True

    # Formatting
    ax_main.set_xlabel(x_key, fontsize=12)
    ax_main.set_ylabel(y_key, fontsize=12)
    ax_main.grid(True, linestyle=':', alpha=0.5)
    
    # Hide Hist frames for cleaner look
    ax_xhist.axis('off')
    ax_yhist.axis('off')
    
    if plotted:
        # Legend shows the subsampled count (n=) to remind user it's balanced
        handles, labels = ax_main.get_legend_handles_labels()
        # Append info about subsampling to legend title or first item? 
        # Actually standard legend is fine, but maybe clarify in title
        ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9, title=f"Scatter n={limit}")
    else:
        ax_main.text(0.5, 0.5, "No Matching Data", ha='center', transform=ax_main.transAxes)

    plt.suptitle(f"Cluster Properties: {y_key} vs {x_key}\n(All Layers Combined) [{hit_str}]", fontsize=16)
    plt.show()

# ==========================================
# EXAMPLE USAGE
# ==========================================
plot_xtalk_scatter_all_layers(all_clsts, 'avg_ToT', 'pToF', num_hits=None, xtalk_types=[0, 1, 2, 3])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import math
import textwrap

def run_advanced_cluster_analysis(cluster_dict, features=None, target_types=None):
    """
    Runs a complete diagnostic pipeline for cluster separation.
    
    UPDATES:
    - Lists used features on the plot.
    - Uses constrained_layout for better formatting.
    """

    print(f"{'='*80}")
    print(f"{'ADVANCED CLUSTER SEPARATION PIPELINE':^80}")
    print(f"{'='*80}")

    # --- 0. TARGET SELECTION ---
    valid_map = {
        1: 'Capacitive (Type 1)',
        2: 'Ambiguous (Type 2)',
        3: 'Mixed (Type 3)'
    }

    if target_types is None:
        active_targets = [1, 2, 3]
    else:
        if isinstance(target_types, (int, float)):
            target_types = [int(target_types)]
        active_targets = [t for t in target_types if t in valid_map]

    if not active_targets:
        print("Error: No valid target types selected.")
        return

    # --- 1. DATA PREP ---
    if features is None:
        features = ['avg_ToT', 'sum_ToT', 'n_hits', 'width_col', 'width_row', 'pToF']
        if 'duration' in cluster_dict: features.append('duration')
        if 'track_chi2' in cluster_dict: features.append('track_chi2')

    # Format feature string for display
    feature_text = "Features Used:\n" + "\n".join([f"- {f}" for f in features])

    # Data Extraction
    data_map = {}
    def get_clean_col(key):
        if key not in cluster_dict: return None
        raw = cluster_dict[key]
        clean = np.empty(len(raw), dtype=float)
        for i, val in enumerate(raw):
            try:
                if isinstance(val, (list, np.ndarray, tuple)):
                    v = np.min(val) if len(val) > 0 else np.nan
                else:
                    v = float(val)
                if v == -1 or np.isnan(v): v = np.nan
                clean[i] = v
            except:
                clean[i] = np.nan
        return clean

    for f in features:
        col = get_clean_col(f)
        if col is not None: data_map[f] = col

    raw_xtalk = cluster_dict['xtalk_type']
    labels = np.full(len(raw_xtalk), -1, dtype=int)
    for i, val in enumerate(raw_xtalk):
        try:
            if isinstance(val, (list, np.ndarray)): labels[i] = 3
            else:
                v = int(val)
                labels[i] = v if v in [0,1,2] else -1
        except: pass

    df = pd.DataFrame(data_map)
    df['label'] = labels
    df = df[df['label'] != -1].dropna()

    if len(df) == 0: return

    # --- 2. CORRELATION ANALYSIS ---
    types_to_plot = [(0, 'Clean (Type 0)')]
    for t in active_targets:
        types_to_plot.append((t, valid_map[t]))

    num_plots = len(types_to_plot)
    cols = 2 if num_plots > 1 else 1
    rows = math.ceil(num_plots / cols)

    # Use constrained_layout for automatic spacing fixes
    fig_corr = plt.figure(figsize=(6 * cols, 5 * rows), layout="constrained")
    
    # Add Feature List to Figure
    fig_corr.text(0.01, 0.99, feature_text, fontsize=9, va='top', ha='left', 
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    for i, (lbl, name) in enumerate(types_to_plot):
        ax = fig_corr.add_subplot(rows, cols, i + 1)
        subset = df[df['label'] == lbl].drop(columns=['label'])

        if len(subset) < 5:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center')
            ax.set_title(f"{name}")
            continue

        corr = subset.corr(method='pearson')
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                    fmt=".2f", ax=ax, cbar=True, annot_kws={"size": 8})
        ax.set_title(f"{name} (N={len(subset)})")

    fig_corr.suptitle("Feature Correlation Matrices", fontsize=16, y=1.02)
    plt.show()

    # --- 3. PCA & QDA ANALYSIS ---
    # Dynamic Height: 5 inches per row
    fig_qda = plt.figure(figsize=(18, 5 * len(active_targets)), layout="constrained")
    
    # Add Feature List to Figure
    fig_qda.text(0.01, 0.99, feature_text, fontsize=9, va='top', ha='left', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # GridSpec with explicit spacing
    outer_grid = gridspec.GridSpec(len(active_targets), 1, figure=fig_qda, hspace=0.3)

    for row_idx, xt_type in enumerate(active_targets):
        type_name = valid_map[xt_type]
        subset = df[df['label'].isin([0, xt_type])]

        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[row_idx], wspace=0.2)

        if len(subset) < 50 or subset['label'].nunique() < 2:
            ax = fig_qda.add_subplot(inner_grid[1])
            ax.text(0.5, 0.5, f"Insufficient Data for {type_name}", ha='center')
            continue

        X = subset[features].values
        y = subset['label'].values 
        y_binary = np.where(y == 0, 0, 1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2, whiten=True)
        X_pca = pca.fit_transform(X_scaled)
        var_expl = pca.explained_variance_ratio_

        qda = QuadraticDiscriminantAnalysis(reg_param=0.05)
        qda.fit(X_pca, y_binary)
        y_prob = qda.predict_proba(X_pca)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_binary, y_prob)
        roc_auc = auc(fpr, tpr)
        J_scores = tpr - fpr
        best_J_idx = np.argmax(J_scores)
        best_thresh = thresholds[best_J_idx]

        # --- PLOT 1: PCA ---
        ax_pca = fig_qda.add_subplot(inner_grid[0])
        idx_0 = np.where(y_binary == 0)[0]
        idx_1 = np.where(y_binary == 1)[0]
        limit = min(len(idx_0), len(idx_1), 2000)
        
        samp_0 = np.random.choice(idx_0, limit, replace=False)
        samp_1 = np.random.choice(idx_1, limit, replace=False)
        jitter_0 = np.random.normal(0, 0.05, (limit, 2))
        jitter_1 = np.random.normal(0, 0.05, (limit, 2))

        ax_pca.scatter(X_pca[samp_0, 0] + jitter_0[:,0], X_pca[samp_0, 1] + jitter_0[:,1], 
                       c='#3b4cc0', alpha=0.4, s=10, label='Clean', edgecolor='none')
        ax_pca.scatter(X_pca[samp_1, 0] + jitter_1[:,0], X_pca[samp_1, 1] + jitter_1[:,1], 
                       c='#b40426', alpha=0.4, s=10, label=type_name, edgecolor='none')

        # Decision Boundary
        x_min, x_max = ax_pca.get_xlim(); y_min, y_max = ax_pca.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_min-0.5, x_max+0.5, 100), np.linspace(y_min-0.5, y_max+0.5, 100))
        Z = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax_pca.contour(xx, yy, Z, levels=[best_thresh], colors='black', linestyles='--', linewidths=2)

        ax_pca.set_title(f"{type_name}\nPCA Space")
        ax_pca.set_xlabel(f"PC1 ({var_expl[0]:.1%})")
        ax_pca.set_ylabel(f"PC2 ({var_expl[1]:.1%})")
        ax_pca.legend(loc='upper right', fontsize=8)

        # --- PLOT 2: Prob Dist ---
        ax_dist = fig_qda.add_subplot(inner_grid[1])
        ax_dist.hist(y_prob[y_binary==0], bins=40, color='#3b4cc0', alpha=0.5, label='Clean', density=True)
        ax_dist.hist(y_prob[y_binary==1], bins=40, color='#b40426', alpha=0.5, label=type_name, density=True)
        ax_dist.axvline(best_thresh, color='k', linestyle='--', label='Youden Cut')
        ax_dist.set_title("QDA Probability")
        ax_dist.set_xlabel("P(Xtalk)")
        ax_dist.legend(fontsize=8)

        # --- PLOT 3: ROC ---
        ax_roc = fig_qda.add_subplot(inner_grid[2])
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.3f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax_roc.plot(fpr[best_J_idx], tpr[best_J_idx], 'go')
        ax_roc.text(fpr[best_J_idx]+0.05, tpr[best_J_idx]-0.15, f"TPR={tpr[best_J_idx]:.2f}\nFPR={fpr[best_J_idx]:.2f}", fontsize=8, color='green')
        ax_roc.set_title(f"ROC Curve")
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)

    fig_qda.suptitle("Advanced Separation (PCA + QDA) by Type", fontsize=16)
    plt.show()


def perform_bdt_analysis(cluster_dict, features=None, target_types=None):
    """
    Performs BDT classification with improved formatting and feature listing.
    """

    print(f"{'='*80}")
    print(f"{'GRADIENT BOOSTED DECISION TREE (BDT) ANALYSIS':^80}")
    print(f"{'='*80}")

    valid_map = {1: 'Capacitive (Type 1)', 2: 'Ambiguous (Type 2)', 3: 'Mixed (Type 3)'}

    if target_types is None:
        active_targets = [1, 2, 3]
    else:
        if isinstance(target_types, (int, float)): target_types = [int(target_types)]
        active_targets = [t for t in target_types if t in valid_map]

    if not active_targets: return

    if features is None:
        features = ['avg_ToT', 'sum_ToT', 'n_hits', 'width_col', 'width_row', 'pToF']
        if 'duration' in cluster_dict: features.append('duration')
        if 'track_chi2' in cluster_dict: features.append('track_chi2')

    feature_text = "Features Used:\n" + "\n".join([f"- {f}" for f in features])

    # Data Extraction (Same as above)
    data_map = {}
    def get_clean_col(key):
        if key not in cluster_dict: return None
        raw = cluster_dict[key]
        clean = np.empty(len(raw), dtype=float)
        for i, val in enumerate(raw):
            try:
                if isinstance(val, (list, np.ndarray, tuple)): v = np.min(val) if len(val) > 0 else np.nan
                else: v = float(val)
                if v == -1 or np.isnan(v): v = np.nan
                clean[i] = v
            except: clean[i] = np.nan
        return clean

    for f in features:
        col = get_clean_col(f)
        if col is not None: data_map[f] = col

    raw_xtalk = cluster_dict['xtalk_type']
    labels = np.full(len(raw_xtalk), -1, dtype=int)
    for i, val in enumerate(raw_xtalk):
        try:
            if isinstance(val, (list, np.ndarray)): labels[i] = 3
            else:
                v = int(val)
                labels[i] = v if v in [0,1,2] else -1
        except: pass

    df = pd.DataFrame(data_map)
    df['label'] = labels
    df = df[df['label'] != -1].dropna()
    if len(df) == 0: return

    # --- PLOTTING ---
    fig = plt.figure(figsize=(18, 5 * len(active_targets)), layout="constrained")
    
    # Add Feature List Text Box
    fig.text(0.01, 0.99, feature_text, fontsize=9, va='top', ha='left', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    outer_grid = gridspec.GridSpec(len(active_targets), 1, figure=fig, hspace=0.3)

    for row_idx, xt_type in enumerate(active_targets):
        type_name = valid_map[xt_type]
        subset = df[df['label'].isin([0, xt_type])]

        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[row_idx], wspace=0.2)

        if len(subset) < 50 or subset['label'].nunique() < 2:
            ax = fig.add_subplot(inner_grid[1])
            ax.text(0.5, 0.5, f"Insufficient Data for {type_name}", ha='center')
            continue

        X = subset[features].values
        y = subset['label'].values 
        y_binary = np.where(y == 0, 0, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.7, random_state=42)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        J_scores = tpr - fpr
        best_J_idx = np.argmax(J_scores)
        best_thresh = thresholds[best_J_idx]

        # --- PLOT 1: Importance ---
        ax_feat = fig.add_subplot(inner_grid[0])
        importances = clf.feature_importances_
        indices = np.argsort(importances)
        ax_feat.barh(range(len(indices)), importances[indices], color='#1f77b4', align='center')
        ax_feat.set_yticks(range(len(indices)))
        ax_feat.set_yticklabels([features[i] for i in indices])
        ax_feat.set_xlabel('Relative Importance')
        ax_feat.set_title(f'{type_name}\nFeature Importance')
        ax_feat.grid(axis='x', alpha=0.3)

        # --- PLOT 2: Dist ---
        ax_dist = fig.add_subplot(inner_grid[1])
        ax_dist.hist(y_score[y_binary==0], bins=50, color='blue', alpha=0.5, label='Clean', density=True)
        ax_dist.hist(y_score[y_binary==1], bins=50, color='red', alpha=0.5, label=type_name, density=True)
        ax_dist.axvline(best_thresh, color='k', linestyle='--', label=f'Cut ({best_thresh:.2f})')
        ax_dist.set_title(f"BDT Score Separation")
        ax_dist.set_xlabel(f"P({type_name})")
        ax_dist.legend(loc='upper center', fontsize=9)
        ax_dist.grid(True, alpha=0.3)

        # --- PLOT 3: ROC ---
        ax_roc = fig.add_subplot(inner_grid[2])
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax_roc.plot(fpr[best_J_idx], tpr[best_J_idx], 'go')
        ax_roc.text(fpr[best_J_idx]+0.05, tpr[best_J_idx]-0.15, f"TPR={tpr[best_J_idx]:.2f}\nFPR={fpr[best_J_idx]:.2f}", fontsize=8, color='green')
        ax_roc.set_title(f"ROC Curve")
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)

    fig.suptitle("Gradient Boosted Decision Tree (BDT) Analysis by Type", fontsize=16)
    plt.show()

tables = run_advanced_cluster_analysis(all_clsts, target_types=1, features = ['avg_ToT', 'sum_ToT', 'cog_row'])
perform_bdt_analysis(all_clsts, target_types=[1,2, 3])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import textwrap

def run_manifold_learning_analysis(cluster_dict, features=None, target_types=None, max_samples=2000):
    """
    Performs t-SNE Manifold Learning to visualize non-linear cluster separation.
    
    Args:
        cluster_dict: Dictionary containing cluster data.
        features: List of features to use.
        target_types: Int or List of Ints [1, 2, 3].
        max_samples: Maximum number of points per class to plot (t-SNE is slow).
    """

    print(f"{'='*80}")
    print(f"{'MANIFOLD LEARNING (t-SNE) ANALYSIS':^80}")
    print(f"{'='*80}")

    # --- 0. TARGET SELECTION ---
    valid_map = {
        1: 'Capacitive (Type 1)',
        2: 'Ambiguous (Type 2)',
        3: 'Mixed (Type 3)'
    }

    if target_types is None:
        active_targets = [1, 2, 3]
    else:
        if isinstance(target_types, (int, float)):
            target_types = [int(target_types)]
        active_targets = [t for t in target_types if t in valid_map]

    if not active_targets:
        print("Error: No valid target types selected.")
        return

    # --- 1. DATA PREP ---
    if features is None:
        features = ['avg_ToT', 'sum_ToT', 'n_hits', 'width_col', 'width_row', 'pToF']
        if 'duration' in cluster_dict: features.append('duration')
        if 'track_chi2' in cluster_dict: features.append('track_chi2')

    print(f"Features: {features}")
    
    # Format feature string for display
    feature_text = "Features Used:\n" + "\n".join([f"- {f}" for f in features])

    # Robust Data Extraction
    data_map = {}
    def get_clean_col(key):
        if key not in cluster_dict: return None
        raw = cluster_dict[key]
        clean = np.empty(len(raw), dtype=float)
        for i, val in enumerate(raw):
            try:
                if isinstance(val, (list, np.ndarray, tuple)):
                    v = np.min(val) if len(val) > 0 else np.nan
                else:
                    v = float(val)
                if v == -1 or np.isnan(v): v = np.nan
                clean[i] = v
            except:
                clean[i] = np.nan
        return clean

    for f in features:
        col = get_clean_col(f)
        if col is not None: data_map[f] = col

    # Extract Labels
    raw_xtalk = cluster_dict['xtalk_type']
    labels = np.full(len(raw_xtalk), -1, dtype=int)
    for i, val in enumerate(raw_xtalk):
        try:
            if isinstance(val, (list, np.ndarray)): labels[i] = 3
            else:
                v = int(val)
                labels[i] = v if v in [0,1,2] else -1
        except: pass

    df = pd.DataFrame(data_map)
    df['label'] = labels
    df = df[df['label'] != -1].dropna()

    if len(df) == 0:
        print("Error: No valid data found.")
        return

    # --- 2. PLOTTING SETUP ---
    # We will plot 2 columns per target (Perplexity 30 and Perplexity 50)
    fig = plt.figure(figsize=(14, 6 * len(active_targets)), layout="constrained")
    
    # Add Feature List Text Box
    fig.text(0.01, 0.99, feature_text, fontsize=9, va='top', ha='left', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    outer_grid = gridspec.GridSpec(len(active_targets), 1, figure=fig, hspace=0.3)

    # --- 3. ANALYSIS LOOP ---
    for row_idx, xt_type in enumerate(active_targets):
        type_name = valid_map[xt_type]
        
        # Grid for this row (2 columns)
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[row_idx], wspace=0.1)

        # Filter Data: Clean (0) vs Current Xtalk
        subset = df[df['label'].isin([0, xt_type])]
        
        # Check counts
        idx_0 = subset[subset['label'] == 0].index
        idx_xt = subset[subset['label'] == xt_type].index

        if len(idx_xt) < 10:
            ax = fig.add_subplot(inner_grid[0])
            ax.text(0.5, 0.5, f"Insufficient Data for {type_name}", ha='center')
            continue

        # --- DOWNSAMPLING (CRITICAL FOR t-SNE SPEED) ---
        # We take a balanced sample up to max_samples
        n_samples = min(len(idx_0), len(idx_xt), max_samples)
        
        samp_0 = np.random.choice(idx_0, n_samples, replace=False)
        samp_xt = np.random.choice(idx_xt, n_samples, replace=False)
        
        # Combine and shuffle
        indices = np.concatenate([samp_0, samp_xt])
        np.random.shuffle(indices)
        
        X_sub = df.loc[indices, features].values
        y_sub = df.loc[indices, 'label'].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)

        # --- RUN t-SNE (Two Perplexities) ---
        perplexities = [30, 50]
        
        for col_idx, perp in enumerate(perplexities):
            ax = fig.add_subplot(inner_grid[col_idx])
            
            print(f"Running t-SNE (Type {xt_type}, Perp={perp}, N={len(X_sub)})...")
            
            tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
            X_embedded = tsne.fit_transform(X_scaled)
            
            # Plot Clean
            mask_0 = (y_sub == 0)
            ax.scatter(X_embedded[mask_0, 0], X_embedded[mask_0, 1], 
                       c='#3b4cc0', label='Clean', alpha=0.5, s=15, edgecolor='none')
            
            # Plot Xtalk
            mask_xt = (y_sub == xt_type)
            ax.scatter(X_embedded[mask_xt, 0], X_embedded[mask_xt, 1], 
                       c='#b40426', label=type_name, alpha=0.5, s=15, edgecolor='none')

            ax.set_title(f"{type_name}\nt-SNE (Perplexity={perp})")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            
            # Remove ticks for cleaner "manifold" look
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col_idx == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.suptitle("Manifold Learning: Feature Space Topology", fontsize=16)
    plt.show()

run_manifold_learning_analysis(all_clsts, target_types=3, features = ['avg_ToT', 'sum_ToT', 'cog_row', 'track_chi2'])