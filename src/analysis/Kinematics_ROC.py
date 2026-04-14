# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:16:17 2026

@author: henry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon
from scipy.optimize import curve_fit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_master_kinematics_and_roc(all_trks, final_clusters, z_positions=[3, 2, 1, 0]):
    print("--- 1. Processing Tracks and Residuals ---")
    df_trks = pd.DataFrame(all_trks)
    df_clsts = pd.DataFrame(final_clusters)

    shared_cols = ['L1_ID', 'L2_ID', 'L3_ID']
    df_trks['group_size'] = df_trks.groupby(shared_cols)['chi2'].transform('count')
    df_trks['chi2_rank'] = df_trks.groupby(shared_cols)['chi2'].rank(method='first')

    # Extract Baseline Clean Tracks
    df_clean_base = df_trks[df_trks['group_size'] == 1].copy()

    # ENFORCE STRICT UNIQUENESS: Sort by best fit, then drop any shared clusters across all 4 layers
    df_clean = df_clean_base.sort_values('chi2')
    for col in ['L1_ID', 'L2_ID', 'L3_ID', 'L4_ID']:
        df_clean = df_clean.drop_duplicates(subset=[col], keep='first')

    print(f"Clean Tracks (Base):   {len(df_clean_base):,}")
    print(f"Clean Tracks (Unique): {len(df_clean):,} (Platinum Sample)")

    # Extract Combinatorial Splits
    df_split = df_trks[df_trks['group_size'] > 1].copy()
    df_2a = df_split[df_split['chi2_rank'] == 1].copy()
    df_2b = df_split[df_split['chi2_rank'] > 1].copy()

    # Calculate Residuals (Vectorized)
    Z = np.array(z_positions)
    fit_z, target_z = Z[0:3], Z[3]

    def process_df(df):
        X_fit, Y_fit = df[['x1', 'x2', 'x3']].values, df[['y1', 'y2', 'y3']].values
        z_mean = np.mean(fit_z)
        denom = np.sum((fit_z - z_mean)**2)

        mx = np.sum((fit_z - z_mean) * (X_fit - np.mean(X_fit, axis=1)[:, None]), axis=1) / denom
        x_exp = mx * target_z + (np.mean(X_fit, axis=1) - mx * z_mean)

        my = np.sum((fit_z - z_mean) * (Y_fit - np.mean(Y_fit, axis=1)[:, None]), axis=1) / denom
        y_exp = my * target_z + (np.mean(Y_fit, axis=1) - my * z_mean)

        df = df.copy() 
        df['dX'] = df['x4'].values - x_exp
        df['dY'] = df['y4'].values - y_exp
        df['dR'] = np.sqrt(df['dX']**2 + df['dY']**2)

        return df.merge(df_clsts[['clusterID', 'sum_ToT', 'n_hits']], 
                        left_on='L4_ID', right_on='clusterID', how='left')

    df_clean = process_df(df_clean)
    df_2a = process_df(df_2a)
    df_2b = process_df(df_2b)

    # --- 2. TRAIN MACHINE LEARNING (LDA) ---
    print("--- 2. Training LDA Models ---")
    features = ['dX', 'dY', 'dR', 'sum_ToT', 'n_hits']

    def get_roc(df_pos, df_neg):
        p_filt, n_filt = df_pos[df_pos['chi2'] < 100], df_neg[df_neg['chi2'] < 100]
        df_p, df_n = p_filt[features].dropna(), n_filt[features].dropna()

        X = np.vstack([df_p.values, df_n.values])
        y = np.concatenate([np.ones(len(df_p)), np.zeros(len(df_n))])

        y_prob = LinearDiscriminantAnalysis().fit(X, y).predict_proba(X)[:, 1]
        return roc_curve(y, y_prob)

    def get_optimal_youden(fpr, tpr):
        """Calculates Youden's J statistic and returns the optimal point."""
        J = tpr - fpr
        idx = np.argmax(J)
        return fpr[idx], tpr[idx], J[idx]

    # Calculate ROC and Youden's J for both models
    fpr_ab, tpr_ab, _ = get_roc(df_2a, df_2b)
    auc_ab = auc(fpr_ab, tpr_ab)
    opt_fpr_ab, opt_tpr_ab, max_j_ab = get_optimal_youden(fpr_ab, tpr_ab)

    fpr_cb, tpr_cb, _ = get_roc(df_clean, df_2b)
    auc_cb = auc(fpr_cb, tpr_cb)
    opt_fpr_cb, opt_tpr_cb, max_j_cb = get_optimal_youden(fpr_cb, tpr_cb)

    # --- 3. RENDER MASTER GRAPHIC ---
    print("--- 3. Rendering Publication Graphic ---")
    plt.rcParams.update({'font.family': 'serif', 'font.size': 16})
    fig, ax = plt.subplots(figsize=(14, 8))

    x_eval = np.linspace(0.1, 30, 500)
    colors = {'Clean': '#1f77b4', '2a': '#2ca02c', '2b': '#d62728'}

    def plot_gamma_mixture(data, label, color):
        chi2_raw = data['chi2'].dropna().values
        chi2_vis = chi2_raw[chi2_raw <= 40]
        if len(chi2_vis) < 2: return

        # 1. Create a histogram to fit against
        counts, bin_edges = np.histogram(chi2_vis, bins=60, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 2. Define Mixture PDF: c * Gamma + (1-c) * Exponential
        def mixture_model(x, c, shape, scale, decay):
            return c * gamma.pdf(x, a=shape, loc=0, scale=scale) + \
                   (1 - c) * expon.pdf(x, loc=0, scale=decay)

        # 3. Fit the model using robust boundaries
        p0 = [0.8, 2.0, 1.5, 10.0]
        bounds = ([0.0, 1.01, 0.1, 1.0], [1.0, 15.0, 10.0, 100.0])

        try:
            popt, _ = curve_fit(mixture_model, bin_centers, counts, p0=p0, bounds=bounds)
            pdf = mixture_model(x_eval, *popt)
            mpv = x_eval[np.argmax(pdf)]
            fit_label = f"{label} (N={len(chi2_raw):,}) | Core MPV = {mpv:.1f}"
        except:
            pdf = None
            fit_label = f"{label} (Fit Failed)"

        if pdf is not None:
            ax.plot(x_eval, pdf, color=color, lw=3, label=fit_label)

        # Plot normalized raw histogram 
        ax.hist(chi2_vis, bins=30, density=True, color=color, alpha=0.15, histtype='stepfilled')
        ax.hist(chi2_vis, bins=30, density=True, color=color, alpha=0.8, histtype='step', lw=1.5)

    plot_gamma_mixture(df_clean, "Clean Tracks (Platinum)", colors['Clean'])
    plot_gamma_mixture(df_2a, "Cluster 2a (Winner)", colors['2a'])
    plot_gamma_mixture(df_2b, "Cluster 2b (Loser)", colors['2b'])

    ax.set_title(r"Combinatorial $\chi^2$ (Gamma + Exp. Tail Fit) & LDA Performance", fontweight='bold', pad=15)
    ax.set_xlabel(r"Combinatorial Track $\chi^2$", fontweight='bold', labelpad=10)
    ax.set_ylabel("Probability Density", fontweight='bold', labelpad=10)
    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=0)
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', edgecolor='black', framealpha=1.0)

    plt.tight_layout()

    # --- 4. ROC INSET WITH YOUDEN'S J ---
    axins = inset_axes(ax, width="35%", height="45%", loc='center right', borderpad=4)

    # Plot lines
    axins.plot(fpr_ab, tpr_ab, color=colors['2a'], lw=2.5, label=f'2a vs 2b (AUC = {auc_ab:.3f})')
    axins.plot(fpr_cb, tpr_cb, color=colors['Clean'], lw=2.5, label=f'Clean vs 2b (AUC = {auc_cb:.3f})')
    axins.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')

    # Plot optimal points
    axins.scatter([opt_fpr_ab], [opt_tpr_ab], color=colors['2a'], s=60, edgecolor='black', zorder=5)
    axins.scatter([opt_fpr_cb], [opt_tpr_cb], color=colors['Clean'], s=60, edgecolor='black', zorder=5)

    # Annotate optimal points
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.95)

    axins.annotate(f"Max J: {max_j_ab:.2f}\nTPR: {opt_tpr_ab:.2f} | FPR: {opt_fpr_ab:.2f}",
                   xy=(opt_fpr_ab, opt_tpr_ab), xytext=(opt_fpr_ab + 0.05, opt_tpr_ab - 0.25),
                   arrowprops=dict(arrowstyle="->", color="black", connectionstyle="arc3,rad=-0.2"),
                   fontsize=9, bbox=bbox_props)

    axins.annotate(f"Max J: {max_j_cb:.2f}\nTPR: {opt_tpr_cb:.2f} | FPR: {opt_fpr_cb:.2f}",
                   xy=(opt_fpr_cb, opt_tpr_cb), xytext=(opt_fpr_cb + 0.15, opt_tpr_cb - 0.10),
                   arrowprops=dict(arrowstyle="->", color="black", connectionstyle="arc3,rad=-0.2"),
                   fontsize=9, bbox=bbox_props)

    axins.set_title(r"LDA ROC (No $\chi^2$ feature)", fontsize=13, fontweight='bold')
    axins.set_xlabel("False Pos. Rate", fontsize=11, fontweight='bold')
    axins.set_ylabel("True Pos. Rate", fontsize=11, fontweight='bold')
    axins.tick_params(labelsize=10)
    axins.grid(linestyle=':', alpha=0.5)
    axins.legend(loc="lower right", edgecolor='black', fontsize=10)

    plt.show()


plot_master_kinematics_and_roc(all_trks, final_clusters, z_positions=[3, 2, 1, 0])
