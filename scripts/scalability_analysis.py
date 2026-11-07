"""
Scalability Analysis: Group by Small/Medium/Large and report performance.
"""
import json
import csv
import argparse
import os
import numpy as np


def size_group(n):
    if n <= 20:
        return "Small"
    elif n <= 100:
        return "Medium"
    else:
        return "Large"


def main():
    parser = argparse.ArgumentParser(description="Scalability Analysis")
    parser.add_argument("--input", type=str, default="experiments/results/evaluation_raw.json")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    groups = {"Small": [], "Medium": [], "Large": []}
    for inst in data['instances']:
        g = size_group(inst['n_customers'])
        groups[g].append(inst)

    csv_path = os.path.join(args.output_dir, "scalability.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "Instances", "Size_Range", "Avg_B_sol", "Avg_AV_sol",
                         "Avg_T_avg", "Avg_Gap_B(%)", "Avg_Gap_AV(%)"])

        for gname in ["Small", "Medium", "Large"]:
            subset = groups[gname]
            if not subset:
                writer.writerow([gname, 0, "", "", "", "", "", ""])
                continue

            sizes = [i['n_customers'] for i in subset]
            b_sols = [i['B_sol'] for i in subset]
            av_sols = [i['AV_sol'] for i in subset]
            times = [i['T_avg'] for i in subset]
            gaps_b = [i['gap_B_sol'] for i in subset if i.get('gap_B_sol') is not None]
            gaps_av = [i['gap_AV_sol'] for i in subset if i.get('gap_AV_sol') is not None]

            writer.writerow([
                gname,
                len(subset),
                f"{min(sizes)}-{max(sizes)}",
                f"{np.mean(b_sols):.1f}",
                f"{np.mean(av_sols):.1f}",
                f"{np.mean(times):.2f}",
                f"{np.mean(gaps_b):.4f}" if gaps_b else "",
                f"{np.mean(gaps_av):.4f}" if gaps_av else "",
            ])

    print(f"Scalability analysis saved to {csv_path}")


if __name__ == "__main__":
    main()
