"""MTRPP Comprehensive Analysis - Sections 6-9: Param Sensitivity, Robustness, Efficiency, Summary"""
import json, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results')
PLOTS = os.path.join(RESULTS, 'analysis_plots')
PAPER = os.path.join(os.path.dirname(__file__), '..', 'paper', 'MTRPP_Paper', 'images')
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                      'axes.grid': True, 'grid.alpha': 0.3, 'axes.titlesize': 12, 'axes.labelsize': 11,
                      'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9})

import re as _re
def safe_float(v):
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    m = _re.match(r'tensor\(([\d.eE+-]+)\)', s)
    if m: return float(m.group(1))
    return float(s)

def safe_int(v):
    return int(safe_float(v))


def load_csv(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p): return []
    with open(p, newline='') as f: return list(csv.DictReader(f))

def load_eval(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def size_group(n):
    if n <= 20: return 'Small (n<=20)'
    if n <= 100: return 'Medium (n<=100)'
    return 'Large (n>100)'

# ============================================================
# SECTION 6: PARAMETER SENSITIVITY
# ============================================================
def section6_param_sensitivity():
    print("\n" + "="*70)
    print("SECTION 6: PARAMETER SENSITIVITY")
    print("="*70)
    rows = load_csv('param_sensitivity.csv')
    if not rows:
        print("No param_sensitivity.csv found."); return

    header = f"{'Parameter':<25} {'Value':<10} {'B_sol':>12} {'AV_sol':>12} {'T_avg(s)':>10}"
    print(header); print("-"*len(header))

    params = defaultdict(list)
    for r in rows:
        p = r['Parameter']; v = r['Value']
        b = safe_float(r['B_sol']); av = safe_float(r['AV_sol']); t = safe_float(r['T_avg(s)'])
        print(f"{p:<25} {v:<10} {b:>12.0f} {av:>12.1f} {t:>10.2f}")
        params[p].append({'val': v, 'B_sol': b, 'AV_sol': av, 'T_avg': t})

    # Identify best settings and stable ranges
    print("\n--- Best Settings & Stability ---")
    for p, entries in params.items():
        avs = [e['AV_sol'] for e in entries]
        best_idx = np.argmin(avs)  # Lower AV_sol = lower total latency = better
        # Actually for MTRPP profit maximization, higher is better
        best_idx = np.argmax(avs)
        spread = (max(avs) - min(avs)) / np.mean(avs) * 100
        print(f"  {p}: Best={entries[best_idx]['val']} (AV_sol={entries[best_idx]['AV_sol']:.1f}), "
              f"Spread={spread:.3f}%")

    # Plot: Parameter sensitivity subplots
    param_names = list(params.keys())
    n_params = len(param_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, p in enumerate(param_names):
        if idx >= 6: break
        ax = axes[idx]
        entries = params[p]
        vals = [str(e['val']) for e in entries]
        avs = [e['AV_sol'] for e in entries]
        times = [e['T_avg'] for e in entries]

        color1 = 'steelblue'; color2 = 'coral'
        ax.bar(range(len(vals)), avs, color=color1, alpha=0.7, label='AV_sol')
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals, fontsize=8)
        ax.set_xlabel(p.split('(')[0].strip(), fontsize=9)
        ax.set_ylabel('AV_sol', color=color1, fontsize=9)

        ax2 = ax.twinx()
        ax2.plot(range(len(vals)), times, 'o-', color=color2, lw=2, ms=6, label='T_avg')
        ax2.set_ylabel('T_avg (s)', color=color2, fontsize=9)

    # Hide unused subplots
    for idx in range(n_params, 6):
        axes[idx].set_visible(False)

    plt.suptitle('Parameter Sensitivity Analysis', fontweight='bold', fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec6_param_sensitivity.png')); plt.savefig(os.path.join(PAPER, 'fig13_param_sensitivity_analysis.png')); plt.close()

    # Plot: Heatmap of normalized AV_sol across parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    matrix = []
    ylabels = []
    for p in param_names:
        entries = params[p]
        avs = np.array([e['AV_sol'] for e in entries])
        norm = (avs - avs.min()) / (avs.max() - avs.min() + 1e-10)
        matrix.append(norm)
        ylabels.append(p.split('(')[0].strip())

    max_cols = max(len(row) for row in matrix)
    padded = np.full((len(matrix), max_cols), np.nan)
    for i, row in enumerate(matrix):
        padded[i, :len(row)] = row

    im = ax.imshow(padded, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_xlabel('Parameter Value Index'); ax.set_title('Normalized AV_sol Heatmap (green=best)')
    plt.colorbar(im, ax=ax, label='Normalized AV_sol')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec6_param_heatmap.png')); plt.close()

# ============================================================
# SECTION 7: ROBUSTNESS
# ============================================================
def section7_robustness():
    print("\n" + "="*70)
    print("SECTION 7: ROBUSTNESS ANALYSIS")
    print("="*70)
    rows = load_csv('robustness.csv')
    if not rows:
        print("No robustness.csv found."); return

    header = f"{'Instance':<22} {'N':>4} {'B_sol':>10} {'AV_sol':>10} {'StdDev':>8} {'B-AV Gap%':>9} {'CoeffVar%':>9}"
    print(header); print("-"*len(header))

    gaps, cvs, stds = [], [], []
    names_for_plot, gaps_for_plot, cvs_for_plot = [], [], []

    for r in rows:
        n = safe_int(r['N']); b = safe_float(r['B_sol']); av = safe_float(r['AV_sol'])
        sd = safe_float(r['StdDev']); gap = safe_float(r['B_AV_Gap(%)']); cv = safe_float(r['CoeffVar(%)'])
        print(f"{r['Instance']:<22} {n:>4} {b:>10.0f} {av:>10.1f} {sd:>8.2f} {gap:>9.4f} {cv:>9.4f}")
        gaps.append(gap); cvs.append(cv); stds.append(sd)
        names_for_plot.append(r['Instance'])
        gaps_for_plot.append(gap); cvs_for_plot.append(cv)

    zero_std = sum(1 for s in stds if s == 0)
    print(f"\n--- Robustness Summary ---")
    print(f"Total instances: {len(rows)}")
    print(f"Perfectly stable (StdDev=0): {zero_std}/{len(rows)} ({100*zero_std/len(rows):.1f}%)")
    print(f"Avg B-AV Gap: {np.mean(gaps):.4f}%")
    print(f"Max B-AV Gap: {np.max(gaps):.4f}%")
    print(f"Avg CoeffVar: {np.mean(cvs):.4f}%")
    print(f"Max CoeffVar: {np.max(cvs):.4f}%")
    print(f"Avg StdDev: {np.mean(stds):.2f}")

    # Plot: Robustness - B-AV gap and CoeffVar
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by gap for cleaner visualization
    sorted_idx = np.argsort(gaps_for_plot)[::-1]
    top_n = min(20, len(sorted_idx))  # Show top 20

    ax = axes[0]
    top_names = [names_for_plot[i] for i in sorted_idx[:top_n]]
    top_gaps = [gaps_for_plot[i] for i in sorted_idx[:top_n]]
    ax.barh(range(top_n), top_gaps, color='steelblue', alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_names, fontsize=7)
    ax.set_xlabel('B_sol vs AV_sol Gap (%)'); ax.set_title('Top 20: B-AV Gap')
    ax.invert_yaxis()

    ax = axes[1]
    sorted_cv = np.argsort(cvs_for_plot)[::-1]
    top_cv_names = [names_for_plot[i] for i in sorted_cv[:top_n]]
    top_cvs = [cvs_for_plot[i] for i in sorted_cv[:top_n]]
    ax.barh(range(top_n), top_cvs, color='coral', alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_cv_names, fontsize=7)
    ax.set_xlabel('Coefficient of Variation (%)'); ax.set_title('Top 20: CoeffVar')
    ax.invert_yaxis()

    plt.suptitle('Robustness Analysis', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec7_robustness.png')); plt.savefig(os.path.join(PAPER, 'fig15_robustness_top20.png')); plt.close()

    # Plot: Stability pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    stable = sum(1 for g in gaps if g == 0)
    near_stable = sum(1 for g in gaps if 0 < g <= 0.1)
    moderate = sum(1 for g in gaps if 0.1 < g <= 0.5)
    variable = sum(1 for g in gaps if g > 0.5)
    sizes = [stable, near_stable, moderate, variable]
    labels_pie = [f'Perfectly Stable\n({stable})', f'Near-Stable\n({near_stable})',
                  f'Moderate\n({moderate})', f'Variable\n({variable})']
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c']
    # Remove zero slices
    nonzero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors) if s > 0]
    if nonzero:
        ax.pie([x[0] for x in nonzero], labels=[x[1] for x in nonzero],
               colors=[x[2] for x in nonzero], autopct='%1.1f%%', startangle=90)
    ax.set_title('Solution Stability Distribution')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec7_stability_pie.png')); plt.savefig(os.path.join(PAPER, 'fig16_solution_stability.png')); plt.close()

# ============================================================
# SECTION 8: EFFICIENCY
# ============================================================
def section8_efficiency():
    print("\n" + "="*70)
    print("SECTION 8: EFFICIENCY ANALYSIS")
    print("="*70)
    rows = load_csv('efficiency.csv')
    if not rows:
        print("No efficiency.csv found."); return

    header = f"{'Group':<12} {'Instances':>10} {'Avg Runtime(s)':>15} {'Total Runtime(s)':>16}"
    print(header); print("-"*len(header))

    groups, avg_rt, total_rt = [], [], []
    for r in rows:
        avg_str = r.get('Avg_Runtime(s)', '')
        total_str = r.get('Total_Runtime(s)', '')
        print(f"{r['Group']:<12} {r['Instances']:>10} {avg_str:>15} {total_str:>16}")
        if r['Group'] != 'TOTAL' and avg_str:
            groups.append(r['Group'])
            avg_rt.append(float(avg_str))
            total_rt.append(float(total_str))

    if groups:
        print(f"\nTotal pipeline runtime: {sum(total_rt):.1f}s ({sum(total_rt)/3600:.2f} hours)")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(groups)]

        axes[0].bar(groups, avg_rt, color=colors, alpha=0.85, edgecolor='white')
        axes[0].set_ylabel('Avg Runtime (s)'); axes[0].set_title('Average Runtime per Instance')
        for i, v in enumerate(avg_rt):
            axes[0].text(i, v + 1, f'{v:.1f}s', ha='center', fontweight='bold')

        axes[1].bar(groups, total_rt, color=colors, alpha=0.85, edgecolor='white')
        axes[1].set_ylabel('Total Runtime (s)'); axes[1].set_title('Total Runtime per Group')
        for i, v in enumerate(total_rt):
            axes[1].text(i, v + 50, f'{v:.0f}s', ha='center', fontweight='bold')

        plt.suptitle('Efficiency Analysis', fontweight='bold')
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec8_efficiency.png')); plt.savefig(os.path.join(PAPER, 'fig08_efficiency_runtime.png')); plt.close()

# ============================================================
# SECTION 9: SUMMARY RESULTS
# ============================================================
def section9_summary():
    print("\n" + "="*70)
    print("SECTION 9: COMPREHENSIVE SUMMARY")
    print("="*70)

    data = load_eval('evaluation_raw.json')
    if not data:
        print("No evaluation_raw.json found."); return

    instances = data['instances']
    n = len(instances)
    matched = sum(1 for i in instances if safe_float(i['gap_B_sol']) == 0)
    near = sum(1 for i in instances if 0 < safe_float(i['gap_B_sol']) <= 1)
    above1 = sum(1 for i in instances if safe_float(i['gap_B_sol']) > 1)
    avg_gap_b = np.mean([safe_float(i['gap_B_sol']) for i in instances])
    avg_gap_av = np.mean([safe_float(i['gap_AV_sol']) for i in instances])
    avg_t = np.mean([safe_float(i['T_avg']) for i in instances])
    zero_std = sum(1 for i in instances if safe_float(i['std_dev']) == 0)

    # Multi-iteration comparison
    best_per_instance = {}
    for fname, label in [('evaluation_100.json','100'), ('evaluation_500.json','500'),
                          ('evaluation_raw.json','1000'), ('evaluation_2000.json','2000')]:
        d = load_eval(fname)
        if d:
            for inst in d['instances']:
                name = inst['name']
                if name not in best_per_instance or safe_float(inst['B_sol']) > best_per_instance[name]['B_sol']:
                    best_per_instance[name] = {'B_sol': safe_float(inst['B_sol']), 'config': label}

    best_config_counts = defaultdict(int)
    for v in best_per_instance.values():
        best_config_counts[v['config']] += 1

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    MTRPP EXPERIMENTAL SUMMARY                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Total test instances:              {n:>4}                          ║
║  Matched UB exactly (gap=0%):       {matched:>4} ({100*matched/n:>5.1f}%)               ║
║  Near-optimal (gap≤1%):             {near:>4} ({100*near/n:>5.1f}%)               ║
║  Gap > 1%:                          {above1:>4} ({100*above1/n:>5.1f}%)               ║
║                                                                  ║
║  Avg Gap to UB (B_sol):            {avg_gap_b:>6.4f}%                      ║
║  Avg Gap to UB (AV_sol):           {avg_gap_av:>6.4f}%                      ║
║  Avg Runtime:                      {avg_t:>7.2f}s                       ║
║  Perfectly stable (std=0):          {zero_std:>4}/{n:<4} ({100*zero_std/n:>5.1f}%)            ║
╠══════════════════════════════════════════════════════════════════╣
║  BEST CONFIG WINS (across 100/500/1000/2000 iterations):         ║""")
    for config, count in sorted(best_config_counts.items(), key=lambda x: -x[1]):
        print(f"║    Itr={config:<6}: best on {count:>3}/{len(best_per_instance):<3} instances                        ║")
    print(f"""╚══════════════════════════════════════════════════════════════════╝""")

    # Final summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Gap category pie
    ax = axes[0, 0]
    cats = [matched, near, above1]
    cat_labels = [f'Matched UB\n({matched})', f'Gap ≤1%\n({near})', f'Gap >1%\n({above1})']
    cat_colors = ['#2ecc71', '#3498db', '#e74c3c']
    nonzero = [(s, l, c) for s, l, c in zip(cats, cat_labels, cat_colors) if s > 0]
    ax.pie([x[0] for x in nonzero], labels=[x[1] for x in nonzero],
           colors=[x[2] for x in nonzero], autopct='%1.1f%%', startangle=90)
    ax.set_title('Gap to UB Categories')

    # 2) Gap by size group
    ax = axes[0, 1]
    groups = defaultdict(list)
    for inst in instances:
        groups[size_group(safe_int(inst['n_customers']))].append(safe_float(inst['gap_B_sol']))
    grp_names = ['Small (n<=20)', 'Medium (n<=100)', 'Large (n>100)']
    grp_data = [groups.get(g, []) for g in grp_names]
    grp_data_clean = [d for d in grp_data if d]
    grp_labels_clean = [g for g, d in zip(grp_names, grp_data) if d]
    bp = ax.boxplot(grp_data_clean, labels=grp_labels_clean, patch_artist=True)
    colors_box = ['#2ecc71', '#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel('Gap to UB (%)'); ax.set_title('Gap Distribution by Size')

    # 3) Iteration convergence
    ax = axes[1, 0]
    itr_labels = ['100', '500', '1000', '2000']
    itr_avg_gaps = []
    for fname in ['evaluation_100.json', 'evaluation_500.json', 'evaluation_raw.json', 'evaluation_2000.json']:
        d = load_eval(fname)
        if d:
            itr_avg_gaps.append(np.mean([safe_float(i['gap_B_sol']) for i in d['instances']]))
        else:
            itr_avg_gaps.append(None)
    valid = [(l, g) for l, g in zip(itr_labels, itr_avg_gaps) if g is not None]
    if valid:
        ax.plot([v[0] for v in valid], [v[1] for v in valid], 'o-', color='steelblue', lw=2, ms=8)
        ax.set_xlabel('Max Iterations'); ax.set_ylabel('Avg Gap to UB (%)')
        ax.set_title('Convergence: Gap vs Iterations')

    # 4) Runtime by size
    ax = axes[1, 1]
    rt_groups = defaultdict(list)
    for inst in instances:
        rt_groups[size_group(safe_int(inst['n_customers']))].append(safe_float(inst['T_avg']))
    grp_rt_data = [rt_groups.get(g, []) for g in grp_names]
    grp_rt_clean = [d for d in grp_rt_data if d]
    grp_rt_labels = [g for g, d in zip(grp_names, grp_rt_data) if d]
    bp2 = ax.boxplot(grp_rt_clean, labels=grp_rt_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel('Runtime (s)'); ax.set_title('Runtime Distribution by Size')

    plt.suptitle('Comprehensive Summary', fontweight='bold', fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec9_summary.png')); plt.savefig(os.path.join(PAPER, 'fig11_results_summary.png')); plt.close()

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("MTRPP COMPREHENSIVE ANALYSIS - PART 2 (Sections 6-9)")
    print("="*70)

    section6_param_sensitivity()
    section7_robustness()
    section8_efficiency()
    section9_summary()

    print(f"\n\nAll plots saved to: {PLOTS}")
    print("Analysis complete!")
