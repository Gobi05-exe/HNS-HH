"""
Statistical Tests: Friedman test and Nemenyi post-hoc test.
Compares GNN solver across multiple runs against the reference solution.
"""
import json
import argparse
import os
import numpy as np
from scipy import stats


def nemenyi_cd(k, n, alpha=0.05):
    """Compute Nemenyi critical difference.
    k = number of methods, n = number of instances.
    Uses q_alpha values for common k values.
    """
    # q_alpha table for alpha=0.05 (from Demsar 2006)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k, 2.728)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def main():
    parser = argparse.ArgumentParser(description="Statistical Tests")
    parser.add_argument("--input", type=str, default="experiments/results/evaluation_raw.json")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    instances = data['instances']
    num_runs = data['config']['num_runs']

    report_path = os.path.join(args.output_dir, "statistical_tests.txt")
    with open(report_path, 'w') as f:
        f.write("STATISTICAL TESTS REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Build rank matrix: each row = instance, each column = method
        # Methods: Run1, Run2, ..., RunN, Reference
        n_instances = len(instances)
        n_methods = num_runs + 1  # runs + reference

        if n_instances < 3:
            f.write("Not enough instances for Friedman test (need >= 3).\n")
            print(f"Report saved to {report_path}")
            return

        rank_matrix = np.zeros((n_instances, n_methods))

        for i, inst in enumerate(instances):
            objectives = list(inst.get('all_objectives', []))
            ref = inst.get('reference', 0)

            # Pad if fewer runs
            while len(objectives) < num_runs:
                objectives.append(0)

            all_vals = objectives[:num_runs] + [ref if ref else 0]
            # Rank: higher objective = better = lower rank
            ranks = stats.rankdata([-v for v in all_vals])
            rank_matrix[i, :] = ranks

        # Average ranks
        avg_ranks = np.mean(rank_matrix, axis=0)

        f.write("Average Ranks (lower is better):\n")
        for r in range(num_runs):
            f.write(f"  Run {r+1}: {avg_ranks[r]:.3f}\n")
        f.write(f"  Reference: {avg_ranks[-1]:.3f}\n\n")

        # Friedman test
        columns = [rank_matrix[:, j] for j in range(n_methods)]
        try:
            stat_val, p_val = stats.friedmanchisquare(*columns)
            f.write(f"Friedman Test:\n")
            f.write(f"  Chi-squared: {stat_val:.4f}\n")
            f.write(f"  p-value: {p_val:.6f}\n")
            f.write(f"  Significant (p<0.05): {'Yes' if p_val < 0.05 else 'No'}\n\n")

            # Nemenyi post-hoc
            cd = nemenyi_cd(n_methods, n_instances)
            f.write(f"Nemenyi Post-Hoc Test:\n")
            f.write(f"  Critical Difference (CD): {cd:.4f}\n\n")

            f.write("  Pairwise Comparisons (|rank_diff| > CD means significant):\n")
            method_names = [f"Run{r+1}" for r in range(num_runs)] + ["Reference"]
            for a in range(n_methods):
                for b in range(a + 1, n_methods):
                    diff = abs(avg_ranks[a] - avg_ranks[b])
                    sig = "SIGNIFICANT" if diff > cd else "not significant"
                    f.write(f"    {method_names[a]} vs {method_names[b]}: "
                            f"|{avg_ranks[a]:.3f} - {avg_ranks[b]:.3f}| = {diff:.3f} ({sig})\n")

        except Exception as e:
            f.write(f"Friedman test failed: {e}\n")

    print(f"Statistical tests saved to {report_path}")


if __name__ == "__main__":
    main()
