import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import maxwell

def plot_tot_analysis(
    df: pd.DataFrame,
    n_to_plot: list[int] = None,
    summary_plot_title: str = 'Fitted Mean ToT vs. Hit Order in Cluster',
    boxplot_title: str = 'ToT Distribution by Hit Order',
    save_plots: bool = False,
    filename_prefix: str = 'tot_analysis'
):

    required_cols = ['ClusterID', 'TS', 'ToT', 'Layer']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain: {required_cols}")

    df_processed = df.copy()
    noise_indices = df_processed[df_processed['ClusterID'] == -1].index
    if not noise_indices.empty:
        new_ids = range(-1, -1 - len(noise_indices), -1)
        df_processed.loc[noise_indices, 'ClusterID'] = new_ids
    df_processed = df_processed.sort_values(['ClusterID', 'TS'])
    df_processed['n'] = df_processed.groupby('ClusterID').cumcount() + 1

    summary_results = []
    for (layer, n_val), group_df in df_processed.groupby(['Layer', 'n']):
        tot_data = group_df['ToT']
        if len(tot_data) < 2: continue
        loc, scale = maxwell.fit(tot_data, floc=0)
        fitted_mean = scale * 2 * np.sqrt(2 / np.pi)
        sem = tot_data.std() / np.sqrt(len(tot_data)) if len(tot_data) > 1 else 0
        summary_results.append({'Layer': layer, 'n': n_val, 'fitted_mean': fitted_mean, 'sem': sem})
    summary_df = pd.DataFrame(summary_results)
    
    palette = sns.color_palette("viridis", n_colors=df_processed['Layer'].nunique())

    if n_to_plot:
        if not isinstance(n_to_plot, list):
            raise TypeError("`n_to_plot` must be a list of integers.")

        for n_val in n_to_plot:
            fig_n, ax_n = plt.subplots(figsize=(10, 7))
            df_n_specific = df_processed[df_processed['n'] == n_val]
            if df_n_specific.empty:
                print(f"Warning: No data found for n = {n_val}. Skipping plot.")
                continue

            bins = np.linspace(df_n_specific['ToT'].min(), df_n_specific['ToT'].max(), 40)
            bin_width = bins[1] - bins[0]

            for i, layer in enumerate(sorted(df_n_specific['Layer'].unique())):
                tot_data = df_n_specific[df_n_specific['Layer'] == layer]['ToT']
                if tot_data.empty or len(tot_data) < 2: continue
                ax_n.hist(tot_data, bins=bins, histtype='step', linewidth=2,
                          density=False, color=palette[i], label='_nolegend_')
                loc, scale = maxwell.fit(tot_data, floc=0)
                fitted_mean = scale * 2 * np.sqrt(2 / np.pi)
                x_fit = np.linspace(bins[0], bins[-1], 200)
                y_pdf = maxwell.pdf(x_fit, loc=loc, scale=scale)
                y_fit = y_pdf * (len(tot_data) * bin_width)
                label = f'Layer {layer} (μ={fitted_mean:.2f}, a={scale:.2f})'
                ax_n.plot(x_fit, y_fit, '--', color=palette[i], label=label)

            ax_n.set_title(f'ToT Distribution for n = {n_val}', fontsize=16)
            ax_n.set_xlabel('ToT', fontsize=12)
            ax_n.set_ylabel('Frequency (Counts)', fontsize=12)
            ax_n.set_xlim(0, max(tot_data))
            ax_n.legend(title='Layer & Fit Parameters')
            ax_n.grid(True, linestyle='--', alpha=0.6)
            fig_n.tight_layout()
            if save_plots:
                plt.savefig(f"{filename_prefix}_distribution_n{n_val}.png")

    # --- Plot 2: Fitted Mean ToT vs. Hit Order ---
    fig_summary, ax_summary = plt.subplots(figsize=(12, 7))
    for i, layer in enumerate(sorted(summary_df['Layer'].unique())):
        layer_data = summary_df[summary_df['Layer'] == layer]
        ax_summary.errorbar(x=layer_data['n'], y=layer_data['fitted_mean'], yerr=layer_data['sem'],
                            fmt='-o', capsize=5, label=f'Layer {layer}', color=palette[i])
    ax_summary.set_title(summary_plot_title, fontsize=16)
    ax_summary.set_xlabel('n (Order of Hit in Cluster)', fontsize=12)
    ax_summary.set_ylabel('Fitted Mean ToT (with SEM)', fontsize=12)
    ax_summary.legend(title='Layer')
    ax_summary.grid(True, which='both', linestyle='--', linewidth=0.5)
    if not summary_df.empty:
        ax_summary.set_xticks(np.arange(1, summary_df['n'].max() + 1))
    fig_summary.tight_layout()
    if save_plots:
        plt.savefig(f"{filename_prefix}_average_fitted.png")

    # --- Plot 3: Box and Whisker Plot ---
    fig_box, ax_box = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='n', y='ToT', hue='Layer', data=df_processed, ax=ax_box,
                palette=palette, hue_order=sorted(df_processed['Layer'].unique()))
    ax_box.set_title(boxplot_title, fontsize=16)
    ax_box.set_xlabel('n (Order of Hit in Cluster)', fontsize=12)
    ax_box.set_ylabel('ToT Distribution', fontsize=12)
    ax_box.legend(title='Layer', loc='upper right')
    ax_box.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig_box.tight_layout()
    if save_plots:
        plt.savefig(f"{filename_prefix}_boxplot.png")

    if not save_plots:
        plt.show()