"""
Benchmark Comparison: GNN solver vs Avci/Lu reference solutions.
Separates results by dataset origin (Avci vs Lu).
"""
import json
import csv
import argparse
import os
import numpy as np


def classify_dataset(name):
    """Classify instance as Avci or Lu based on naming convention."""
    # Avci instances: typically <=100 nodes, names like "10.1.txt_2", "20.5.txt_3"
    # Lu instances: typically >=200 nodes, names like "200.1.txt_2", "500.1.txt_3"
    parts = name.split('.')
    try:
        n = int(parts[0])
        return "Lu" if n >= 200 else "Avci"
    except (ValueError, IndexError):
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="Benchmark Comparison")
    parser.add_argument("--input", type=str, default="experiments/results/evaluation_raw.json")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    instances = data['instances']

    # Classify
    for inst in instances:
        inst['dataset'] = classify_dataset(inst['name'])

    # Write per-instance CSV
    csv_path = os.path.join(args.output_dir, "benchmark_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Instance", "N", "R", "Reference",
                         "B_sol", "AV_sol", "T_avg", "Gap_B(%)", "Gap_AV(%)", "StdDev"])
        for inst in sorted(instances, key=lambda x: (x['dataset'], x['n_customers'], x['name'])):
            writer.writerow([
                inst['dataset'],
                inst['name'],
                inst['n_customers'],
                inst['n_repairmen'],
                inst.get('reference', ''),
                f"{inst['B_sol']:.1f}",
                f"{inst['AV_sol']:.1f}",
                f"{inst['T_avg']:.2f}",
                f"{inst['gap_B_sol']:.4f}" if inst.get('gap_B_sol') is not None else '',
                f"{inst['gap_AV_sol']:.4f}" if inst.get('gap_AV_sol') is not None else '',
                f"{inst['std_dev']:.2f}",
            ])

    # Summary report
    report_path = os.path.join(args.output_dir, "benchmark_comparison.txt")
    with open(report_path, 'w') as f:
        f.write("MTRPP BENCHMARK COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        for ds_name in ["Avci", "Lu", "All"]:
            if ds_name == "All":
                subset = instances
            else:
                subset = [i for i in instances if i['dataset'] == ds_name]

            if not subset:
                continue

            gaps_b = [i['gap_B_sol'] for i in subset if i.get('gap_B_sol') is not None]
            gaps_av = [i['gap_AV_sol'] for i in subset if i.get('gap_AV_sol') is not None]
            times = [i['T_avg'] for i in subset]

            improved = sum(1 for g in gaps_b if g < 0)
            matched = sum(1 for g in gaps_b if abs(g) < 0.01)
            worse = sum(1 for g in gaps_b if g > 0.01)

            f.write(f"--- {ds_name} Dataset ---\n")
            f.write(f"  Instances:           {len(subset)}\n")
            f.write(f"  Avg Gap B_sol:       {np.mean(gaps_b):.4f}%\n" if gaps_b else "")
            f.write(f"  Avg Gap AV_sol:      {np.mean(gaps_av):.4f}%\n" if gaps_av else "")
            f.write(f"  Avg Runtime:         {np.mean(times):.2f}s\n" if times else "")
            f.write(f"  Improved vs ref:     {improved}\n")
            f.write(f"  Matched ref:         {matched}\n")
            f.write(f"  Worse than ref:      {worse}\n")
            f.write("\n")

    print(f"Benchmark comparison saved to {csv_path}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
