"""
Robustness and Efficiency Analysis.
"""
import json, csv, argparse, os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/results/evaluation_raw.json")
    parser.add_argument("--output_dir", default="experiments/results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)
    instances = data['instances']

    # Robustness
    rob_path = os.path.join(args.output_dir, "robustness.csv")
    with open(rob_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Instance", "N", "B_sol", "AV_sol", "StdDev",
                     "B_AV_Gap(%)", "CoeffVar(%)"])
        for inst in instances:
            b = inst['B_sol']
            av = inst['AV_sol']
            sd = inst['std_dev']
            gap_bav = ((b - av) / b * 100) if b > 0 else 0
            cv = (sd / av * 100) if av > 0 else 0
            w.writerow([inst['name'], inst['n_customers'],
                        f"{b:.1f}", f"{av:.1f}", f"{sd:.2f}",
                        f"{gap_bav:.4f}", f"{cv:.4f}"])

    # Efficiency
    def size_group(n):
        if n <= 20: return "Small"
        elif n <= 100: return "Medium"
        return "Large"

    groups = {}
    total_time = 0
    for inst in instances:
        g = size_group(inst['n_customers'])
        if g not in groups:
            groups[g] = []
        groups[g].append(inst['T_avg'])
        total_time += inst['T_avg'] * data['config']['num_runs']

    eff_path = os.path.join(args.output_dir, "efficiency.csv")
    with open(eff_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Group", "Instances", "Avg_Runtime(s)", "Total_Runtime(s)"])
        for g in ["Small", "Medium", "Large"]:
            times = groups.get(g, [])
            if times:
                w.writerow([g, len(times), f"{np.mean(times):.2f}",
                            f"{sum(times)*data['config']['num_runs']:.1f}"])
        w.writerow(["TOTAL", len(instances), "", f"{total_time:.1f}"])

    print(f"Robustness: {rob_path}")
    print(f"Efficiency: {eff_path}")

if __name__ == "__main__":
    main()
