"""
Generate Summary Report.
Reads all result files and produces a consolidated report.
"""
import os, sys, csv, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/results")
    args = parser.parse_args()

    report_path = os.path.join(args.results_dir, "SUMMARY_REPORT.txt")
    with open(report_path, 'w') as out:
        out.write("MTRPP COMPREHENSIVE EXPERIMENTAL SUMMARY\n\n")

        # 1. Benchmark Comparison Summary
        bench_txt = os.path.join(args.results_dir, "benchmark_comparison.txt")
        out.write("1. BENCHMARK COMPARISON\n")
        if os.path.exists(bench_txt):
            with open(bench_txt, 'r') as f:
                content = f.read()
                # Extract "All Dataset" block
                if "--- All Dataset ---" in content:
                    block = content.split("--- All Dataset ---")[1]
                    out.write(block.strip() + "\n\n")
                else:
                    out.write("Summary not found in benchmark_comparison.txt\n\n")
        else:
            out.write("benchmark_comparison.txt not found.\n\n")

        # 2. Scalability Analysis
        scalability_csv = os.path.join(args.results_dir, "scalability.csv")
        out.write("2. SCALABILITY ANALYSIS (Averages)\n")
        if os.path.exists(scalability_csv):
            with open(scalability_csv, 'r') as f:
                reader = csv.DictReader(f)
                out.write(f"{'Group':<10} | {'Instances':<10} | {'Avg B_sol':<10} | {'Avg Gap(%)':<10} | {'Avg Time(s)':<10}\n")
                for row in reader:
                    out.write(f"{row['Group']:<10} | {row['Instances']:<10} | {row.get('Avg_B_sol', ''):<10} | {row.get('Avg_Gap_B(%)', ''):<10} | {row.get('Avg_T_avg', ''):<10}\n")
            out.write("\n")
        else:
            out.write("scalability.csv not found.\n\n")

        # 3. Ablation Study
        ablation_csv = os.path.join(args.results_dir, "ablation.csv")
        out.write("3. ABLATION STUDY (Top 5 Components)\n")
        if os.path.exists(ablation_csv):
            with open(ablation_csv, 'r') as f:
                reader = csv.DictReader(f)
                out.write(f"{'Rank':<5} | {'Component':<25} | {'Drop (%)':<10}\n")
                count = 0
                for row in reader:
                    if count >= 5: break
                    out.write(f"{row['Rank']:<5} | {row['Component']:<25} | {row['Drop(%)']:<10}\n")
                    count += 1
            out.write("\n")
        else:
            out.write("ablation.csv not found.\n\n")
            
        # 4. Parameter Sensitivity
        param_csv = os.path.join(args.results_dir, "param_sensitivity.csv")
        out.write("4. PARAMETER SENSITIVITY\n")
        if os.path.exists(param_csv):
            with open(param_csv, 'r') as f:
                reader = csv.DictReader(f)
                out.write(f"{'Parameter':<20} | {'Value':<10} | {'Avg AV_sol':<15}\n")
                for row in reader:
                    out.write(f"{row['Parameter']:<20} | {row['Value']:<10} | {row['AV_sol']:<15}\n")
            out.write("\n")
        else:
            out.write("param_sensitivity.csv not found.\n\n")

        # 5. Statistical Tests
        stat_txt = os.path.join(args.results_dir, "statistical_tests.txt")
        out.write("5. STATISTICAL TESTS\n")
        if os.path.exists(stat_txt):
            with open(stat_txt, 'r') as f:
                lines = f.readlines()
                # Find Friedman test block
                for i, line in enumerate(lines):
                    if "Friedman Test:" in line:
                        for j in range(4):
                            if i + j < len(lines): out.write(lines[i+j])
                        break
            out.write("\n")
        else:
            out.write("statistical_tests.txt not found.\n\n")
            
        out.write("============================================================\n")
        out.write("END OF REPORT\n")
        out.write("============================================================\n")

    print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main()
