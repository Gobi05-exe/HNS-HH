"""MTRPP Comprehensive Analysis - Sections 1-5: Metrics, Benchmark, Scalability, Stats, Ablation"""
import json, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results')
PLOTS = os.path.join(RESULTS, 'analysis_plots')
os.makedirs(PLOTS, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight',
                      'axes.grid': True, 'grid.alpha': 0.3})

import re
def safe_float(v):
    """Convert value to float, handling 'tensor(...)' strings from PyTorch."""
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    m = re.match(r'tensor\(([\d.eE+-]+)\)', s)
    if m: return float(m.group(1))
    return float(s)

def safe_int(v):
    return int(safe_float(v))

def load_eval(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def load_csv(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p): return []
    with open(p, newline='') as f: return list(csv.DictReader(f))

def size_group(n):
    if n <= 20: return 'Small (n≤20)'
    if n <= 100: return 'Medium (n≤100)'
    return 'Large (n>100)'

# ============================================================
# SECTION 1: INSTANCE-LEVEL METRICS
# ============================================================
def section1_metrics(data):
    print("\n" + "="*70)
    print("SECTION 1: INSTANCE-LEVEL METRICS (max_iterations=1000)")
    print("="*70)
    instances = data['instances']
    header = f"{'Instance':<22} {'N':>4} {'R':>2} {'Reference':>10} {'B_sol':>10} {'AV_sol':>10} {'T_avg':>7} {'Gap_B%':>7} {'Gap_AV%':>7} {'StdDev':>8}"
    print(header)
    print("-"*len(header))
    
    total_b, total_av, total_gap_b, total_gap_av, total_t, total_std = [], [], [], [], [], []
    matched_ub = 0
    
    for inst in sorted(instances, key=lambda x: (x['n_customers'], x['name'])):
        ref=safe_float(inst['reference']); bs=safe_float(inst['B_sol']); av=safe_float(inst['AV_sol'])
        ta=safe_float(inst['T_avg']); gb=safe_float(inst['gap_B_sol']); ga=safe_float(inst['gap_AV_sol'])
        sd=safe_float(inst['std_dev']); nc=safe_int(inst['n_customers']); nr=safe_int(inst['n_repairmen'])
        print(f"{inst['name']:<22} {nc:>4} {nr:>2} "
              f"{ref:>10.0f} {bs:>10.0f} {av:>10.1f} "
              f"{ta:>7.2f} {gb:>7.4f} {ga:>7.4f} "
              f"{sd:>8.2f}")
        total_b.append(bs); total_av.append(av)
        total_gap_b.append(gb); total_gap_av.append(ga)
        total_t.append(ta); total_std.append(sd)
        if gb == 0: matched_ub += 1
    
    n = len(instances)
    print("-"*len(header))
    print(f"{'AVERAGE':<22} {'':>4} {'':>2} {'':>10} {'':>10} {'':>10} "
          f"{np.mean(total_t):>7.2f} {np.mean(total_gap_b):>7.4f} {np.mean(total_gap_av):>7.4f} "
          f"{np.mean(total_std):>8.2f}")
    print(f"\nTotal instances: {n}")
    print(f"Matched UB (gap=0%): {matched_ub}/{n} ({100*matched_ub/n:.1f}%)")
    print(f"Avg Gap to UB (B_sol): {np.mean(total_gap_b):.4f}%")
    print(f"Avg Gap to UB (AV_sol): {np.mean(total_gap_av):.4f}%")
    print(f"Avg Runtime: {np.mean(total_t):.2f}s")
    
    # Plot: Gap distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(total_gap_b, bins=15, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('Gap to UB (%)'); axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of B_sol Gaps')
    axes[0].axvline(np.mean(total_gap_b), color='red', ls='--', label=f'Mean={np.mean(total_gap_b):.3f}%')
    axes[0].legend()
    
    axes[1].hist(total_gap_av, bins=15, color='coral', edgecolor='white', alpha=0.85)
    axes[1].set_xlabel('Gap to UB (%)'); axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of AV_sol Gaps')
    axes[1].axvline(np.mean(total_gap_av), color='red', ls='--', label=f'Mean={np.mean(total_gap_av):.3f}%')
    axes[1].legend()
    plt.suptitle('Section 1: Gap Distribution', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec1_gap_distribution.png')); plt.close()
    return instances

# ============================================================
# SECTION 2: BENCHMARK COMPARISON
# ============================================================
def section2_benchmark():
    print("\n" + "="*70)
    print("SECTION 2: BENCHMARK COMPARISON")
    print("="*70)
    rows = load_csv('benchmark_comparison.csv')
    if not rows:
        print("No benchmark_comparison.csv found."); return
    
    improved, matched, worse = 0, 0, 0
    gaps_b, gaps_av = [], []
    by_size = defaultdict(lambda: {'gaps_b': [], 'gaps_av': [], 'times': [], 'count': 0})
    
    header = f"{'Instance':<22} {'N':>4} {'R':>2} {'Reference':>10} {'B_sol':>10} {'AV_sol':>10} {'T_avg':>7} {'Gap_B%':>7} {'Gap_AV%':>7}"
    print(header)
    print("-"*len(header))
    
    for r in rows:
        n = safe_int(r['N']); gap_b = safe_float(r['Gap_B(%)']); gap_av = safe_float(r['Gap_AV(%)'])
        t = safe_float(r['T_avg']); b = safe_float(r['B_sol']); ref = safe_float(r['Reference'])
        av = safe_float(r['AV_sol'])
        
        print(f"{r['Instance']:<22} {n:>4} {r['R']:>2} {ref:>10.0f} {b:>10.0f} {av:>10.1f} "
              f"{t:>7.2f} {gap_b:>7.4f} {gap_av:>7.4f}")
        
        gaps_b.append(gap_b); gaps_av.append(gap_av)
        grp = size_group(n)
        by_size[grp]['gaps_b'].append(gap_b)
        by_size[grp]['gaps_av'].append(gap_av)
        by_size[grp]['times'].append(t)
        by_size[grp]['count'] += 1
        
        if gap_b == 0: matched += 1
        elif gap_b < 0: improved += 1
        else: worse += 1
    
    print("-"*len(header))
    print(f"\nOverall Avg Gap_B: {np.mean(gaps_b):.4f}%, Avg Gap_AV: {np.mean(gaps_av):.4f}%")
    print(f"Matched UB: {matched}, Worse than UB: {worse}, Improved over UB: {improved}")
    
    print(f"\n{'Group':<18} {'Count':>6} {'Avg Gap_B%':>10} {'Avg Gap_AV%':>11} {'Avg T(s)':>8}")
    for grp in ['Small (n≤20)', 'Medium (n≤100)', 'Large (n>100)']:
        if grp in by_size:
            d = by_size[grp]
            print(f"{grp:<18} {d['count']:>6} {np.mean(d['gaps_b']):>10.4f} "
                  f"{np.mean(d['gaps_av']):>11.4f} {np.mean(d['times']):>8.2f}")
    
    # Plot: Per-instance gap bar chart
    names = [r['Instance'] for r in rows]
    gb = [safe_float(r['Gap_B(%)']) for r in rows]
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ['green' if g == 0 else 'steelblue' if g < 1 else 'coral' for g in gb]
    ax.bar(range(len(names)), gb, color=colors, edgecolor='white', alpha=0.85)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('Gap to UB (%)'); ax.set_title('Section 2: Per-Instance Gap to Upper Bound (B_sol)')
    ax.axhline(np.mean(gb), color='red', ls='--', lw=1.5, label=f'Mean={np.mean(gb):.3f}%')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, 'sec2_benchmark_gaps.png')); plt.close()

# ============================================================
# SECTION 3: SCALABILITY ANALYSIS
# ============================================================
def section3_scalability(data):
    print("\n" + "="*70)
    print("SECTION 3: SCALABILITY ANALYSIS")
    print("="*70)
    
    groups = defaultdict(lambda: {'b': [], 'av': [], 't': [], 'gap_b': [], 'gap_av': [], 'n': []})
    for inst in data['instances']:
        grp = size_group(safe_int(inst['n_customers']))
        groups[grp]['b'].append(safe_float(inst['B_sol'])); groups[grp]['av'].append(safe_float(inst['AV_sol']))
        groups[grp]['t'].append(safe_float(inst['T_avg'])); groups[grp]['gap_b'].append(safe_float(inst['gap_B_sol']))
        groups[grp]['gap_av'].append(safe_float(inst['gap_AV_sol'])); groups[grp]['n'].append(safe_int(inst['n_customers']))
    
    header = f"{'Group':<18} {'Count':>6} {'Avg B_sol':>12} {'Avg AV_sol':>12} {'Avg T(s)':>8} {'Avg Gap_B%':>10} {'Avg Gap_AV%':>11}"
    print(header); print("-"*len(header))
    
    for grp in ['Small (n≤20)', 'Medium (n≤100)', 'Large (n>100)']:
        if grp in groups:
            d = groups[grp]
            print(f"{grp:<18} {len(d['b']):>6} {np.mean(d['b']):>12.1f} {np.mean(d['av']):>12.1f} "
                  f"{np.mean(d['t']):>8.2f} {np.mean(d['gap_b']):>10.4f} {np.mean(d['gap_av']):>11.4f}")
    
    # Plot: Scalability - gap and runtime vs problem size
    all_n = [safe_int(inst['n_customers']) for inst in data['instances']]
    all_gap = [safe_float(inst['gap_B_sol']) for inst in data['instances']]
    all_t = [safe_float(inst['T_avg']) for inst in data['instances']]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(all_n, all_gap, c='steelblue', alpha=0.7, edgecolors='white', s=50)
    axes[0].set_xlabel('Problem Size (n)'); axes[0].set_ylabel('Gap to UB (%)')
    axes[0].set_title('Gap vs Problem Size')
    
    axes[1].scatter(all_n, all_t, c='coral', alpha=0.7, edgecolors='white', s=50)
    axes[1].set_xlabel('Problem Size (n)'); axes[1].set_ylabel('Avg Runtime (s)')
    axes[1].set_title('Runtime vs Problem Size')
    plt.suptitle('Section 3: Scalability Analysis', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec3_scalability.png')); plt.close()
    
    # Plot: Grouped bar chart
    grp_names = [g for g in ['Small (n≤20)', 'Medium (n≤100)', 'Large (n>100)'] if g in groups]
    avg_gaps = [np.mean(groups[g]['gap_b']) for g in grp_names]
    avg_times = [np.mean(groups[g]['t']) for g in grp_names]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(grp_names)); w = 0.35
    bars1 = ax1.bar(x - w/2, avg_gaps, w, label='Avg Gap_B (%)', color='steelblue', alpha=0.85)
    ax1.set_ylabel('Gap to UB (%)', color='steelblue')
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, avg_times, w, label='Avg Runtime (s)', color='coral', alpha=0.85)
    ax2.set_ylabel('Runtime (s)', color='coral')
    ax1.set_xticks(x); ax1.set_xticklabels(grp_names)
    ax1.set_title('Section 3: Performance by Size Group')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec3_grouped_bars.png')); plt.close()

# ============================================================
# SECTION 4: STATISTICAL TESTS (Friedman + Nemenyi)
# ============================================================
def section4_statistical(eval_files):
    print("\n" + "="*70)
    print("SECTION 4: STATISTICAL TESTS")
    print("="*70)
    
    configs = {}
    for fname, label in eval_files:
        d = load_eval(fname)
        if d: configs[label] = d
    
    if len(configs) < 2:
        print("Need at least 2 iteration configs for statistical tests."); return
    
    # Get common instances
    all_names = None
    for label, d in configs.items():
        names = {inst['name'] for inst in d['instances']}
        all_names = names if all_names is None else all_names & names
    all_names = sorted(all_names)
    
    # Build performance matrix: rows=instances, cols=configs
    labels = list(configs.keys())
    perf = np.zeros((len(all_names), len(labels)))
    for j, label in enumerate(labels):
        lookup = {inst['name']: inst for inst in configs[label]['instances']}
        for i, name in enumerate(all_names):
            perf[i, j] = safe_float(lookup[name]['AV_sol'])
    
    # Compute ranks (higher AV_sol = better for profit, so rank 1 = highest)
    ranks = np.zeros_like(perf)
    for i in range(perf.shape[0]):
        order = np.argsort(-perf[i])  # descending
        for rank_val, idx in enumerate(order):
            ranks[i, idx] = rank_val + 1
    
    avg_ranks = ranks.mean(axis=0)
    print(f"\n{'Config':<20} {'Avg Rank':>10} {'Avg AV_sol':>12}")
    for j, label in enumerate(labels):
        col_avg = perf[:, j].mean()
        print(f"{label:<20} {avg_ranks[j]:>10.3f} {col_avg:>12.1f}")
    
    # Friedman test (chi-squared approximation)
    n_inst, k = perf.shape
    if k >= 2:
        chi2 = 12 * n_inst / (k * (k + 1)) * (np.sum(avg_ranks**2) - k * (k + 1)**2 / 4)
        # degrees of freedom = k-1
        # Critical values: chi2(3, 0.05) = 7.815
        dof = k - 1
        print(f"\nFriedman test statistic: χ² = {chi2:.4f} (df={dof})")
        if chi2 > 7.815 and dof == 3:
            print("Result: SIGNIFICANT at α=0.05 (χ²_crit = 7.815)")
        elif chi2 > 6.251 and dof == 3:
            print("Result: SIGNIFICANT at α=0.10 but not at α=0.05")
        else:
            print("Result: No significant difference detected (α=0.05)")
        
        # Nemenyi CD
        q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}
        if k in q_alpha:
            cd = q_alpha[k] * np.sqrt(k * (k + 1) / (6 * n_inst))
            print(f"Nemenyi Critical Difference (CD): {cd:.4f}")
            print("\nPairwise rank differences:")
            for i in range(k):
                for j in range(i+1, k):
                    diff = abs(avg_ranks[i] - avg_ranks[j])
                    sig = "SIG" if diff > cd else "n.s."
                    print(f"  {labels[i]} vs {labels[j]}: |{avg_ranks[i]:.3f} - {avg_ranks[j]:.3f}| = {diff:.3f} [{sig}]")
    
    # Plot: Average ranks
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, avg_ranks, color=colors, edgecolor='white', alpha=0.85)
    ax.set_ylabel('Average Rank (lower=better)')
    ax.set_title('Section 4: Average Ranks Across Iteration Configs')
    for bar, val in zip(bars, avg_ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec4_avg_ranks.png')); plt.close()

# ============================================================
# SECTION 5: ABLATION STUDY
# ============================================================
def section5_ablation():
    print("\n" + "="*70)
    print("SECTION 5: ABLATION STUDY")
    print("="*70)
    rows = load_csv('ablation.csv')
    if not rows:
        print("No ablation.csv found."); return
    
    header = f"{'Rank':>4} {'Component':<28} {'Baseline AV_sol':>14} {'Without AV_sol':>14} {'Drop(%)':>8}"
    print(header); print("-"*len(header))
    
    for r in rows:
        print(f"{r['Rank']:>4} {r['Component']:<28} {safe_float(r['Baseline_AV_sol']):>14.1f} "
              f"{safe_float(r['AV_sol_Without']):>14.1f} {safe_float(r['Drop(%)']):>8.4f}")
    
    active = [r for r in rows if safe_float(r['Drop(%)']) > 0]
    inactive = [r for r in rows if safe_float(r['Drop(%)']) == 0]
    print(f"\nActive components (drop>0): {len(active)}")
    print(f"Inactive components (drop=0): {len(inactive)}")
    if active:
        print(f"Most critical: {active[0]['Component']} (drop={active[0]['Drop(%)']}%)")
    
    # Plot: Ablation bar chart
    names = [r['Component'] for r in rows]
    drops = [safe_float(r['Drop(%)']) for r in rows]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c' if d > 0.01 else '#f39c12' if d > 0 else '#95a5a6' for d in drops]
    bars = ax.barh(range(len(names)), drops, color=colors, edgecolor='white', alpha=0.85)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel('Performance Drop (%)')
    ax.set_title('Section 5: Ablation Study - Component Importance')
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS, 'sec5_ablation.png')); plt.close()

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("MTRPP COMPREHENSIVE ANALYSIS - PART 1 (Sections 1-5)")
    print("="*70)
    
    data = load_eval('evaluation_raw.json')
    if not data:
        print("ERROR: evaluation_raw.json not found!"); sys.exit(1)
    
    section1_metrics(data)
    section2_benchmark()
    section3_scalability(data)
    section4_statistical([
        ('evaluation_100.json', 'Itr=100'),
        ('evaluation_500.json', 'Itr=500'),
        ('evaluation_raw.json', 'Itr=1000'),
        ('evaluation_2000.json', 'Itr=2000'),
    ])
    section5_ablation()
    
    print(f"\n\nPlots saved to: {PLOTS}")
    print("Run comprehensive_analysis_part2.py for sections 6-9.")
