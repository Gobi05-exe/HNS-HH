"""
Generate a clean, professional, publication-ready Parameter Sensitivity & Interaction Heatmap
Fixes Reviewer Comment 4: "Fig. 12 parameter value index some blocks in the figure is missing"
Removes NaN-padded missing blocks and replaces ambiguous 'Parameter Value Index' with explicit,
annotated parameter values across all hyperparameter configurations.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'experiments', 'results', 'param_sensitivity.csv')
PAPER_IMG_PATH = os.path.join(BASE_DIR, 'paper', 'MTRPP_Paper', 'images', 'fig14_param_heatmap.png')
ANALYSIS_IMG_PATH = os.path.join(BASE_DIR, 'experiments', 'results', 'analysis_plots', 'sec6_param_heatmap.png')

def load_data():
    with open(CSV_PATH, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    
    params = {}
    for r in rows:
        p = r['Parameter'].strip()
        v = r['Value'].strip()
        b_sol = float(r['B_sol'])
        av_sol = float(r['AV_sol'])
        t_avg = float(r['T_avg(s)'])
        if p not in params:
            params[p] = []
        params[p].append({'val': v, 'B_sol': b_sol, 'AV_sol': av_sol, 'T_avg': t_avg})
    return params

def plot_heatmap():
    params = load_data()
    
    # We will build a multi-row structured layout where each parameter row has its own dedicated
    # cells spanning exactly its tested values, completely eliminating any missing blocks!
    param_keys = [
        ('T_itr (max_iterations)', 'Search Iterations ($T_{\\text{itr}}$)'),
        ('temperature', 'Exploration Temp ($\\tau$)'),
        ('dc_threshold', 'D&C Threshold ($N_{\\text{D\\&C}}$)'),
        ('dc_target_cluster_size', 'Cluster Size ($S_{\\text{cluster}}$)'),
        ('use_gnn_guidance', 'GNN Guidance Policy'),
        ('intensive_polish', 'Intensive ALNS Polish')
    ]
    
    fig = plt.figure(figsize=(11, 7.5), dpi=300)
    gs = gridspec.GridSpec(len(param_keys), 1, height_ratios=[1]*len(param_keys), hspace=0.6)
    
    cmap = plt.cm.RdYlGn
    
    for idx, (p_key, display_name) in enumerate(param_keys):
        entries = params[p_key]
        n_vals = len(entries)
        avs = np.array([e['AV_sol'] for e in entries])
        
        # Min-max normalization per parameter row
        if avs.max() > avs.min():
            norm_vals = (avs - avs.min()) / (avs.max() - avs.min())
        else:
            norm_vals = np.ones_like(avs)
            
        ax = fig.add_subplot(gs[idx])
        
        # Display 1 x n_vals matrix for this parameter
        matrix = norm_vals.reshape(1, n_vals)
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and explicit value labels
        ax.set_yticks([0])
        ax.set_yticklabels([display_name], fontsize=10, fontweight='bold')
        ax.set_xticks(range(n_vals))
        ax.set_xticklabels([f"val = {e['val']}" for e in entries], fontsize=9)
        
        # Annotate each cell with both the exact AV_sol and normalized score
        for c in range(n_vals):
            score = norm_vals[c]
            raw_val = entries[c]['AV_sol']
            text_color = 'white' if (score < 0.25 or score > 0.85) else 'black'
            ax.text(c, 0, f"{raw_val:,.0f}\n({score:.2f})", ha='center', va='center',
                    color=text_color, fontsize=8.5, fontweight='bold')
            
        # Clean up spines
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1.0)
            
    fig.suptitle('Normalized Hyperparameter Performance Map (All Valid Configurations Evaluated)',
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Add a horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Normalized Objective Performance (0.0 = Parameter Worst, 1.0 = Parameter Best)', fontsize=9.5)
    
    # Ensure dirs exist
    os.makedirs(os.path.dirname(PAPER_IMG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ANALYSIS_IMG_PATH), exist_ok=True)
    
    fig.savefig(PAPER_IMG_PATH, bbox_inches='tight', dpi=300)
    fig.savefig(ANALYSIS_IMG_PATH, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Successfully generated clean heatmap without missing blocks:")
    print(f"  -> {PAPER_IMG_PATH}")
    print(f"  -> {ANALYSIS_IMG_PATH}")

if __name__ == '__main__':
    plot_heatmap()
