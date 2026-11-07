"""
Parameter Sensitivity: Vary key parameters and measure impact.
Reflects metrics requested from 'Solving Large Instances of MTRPP' (Nrst, Titr, etc.)
and approach-specific parameters (GNN guidance, DC thresholds, Slack thresholds).
"""
import os
import sys
import json
import csv
import time
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mtrpp_local import (
    create_large_model, load_instances, evaluate_on_instances, set_seed, CONFIG,
    ControllerConfig
)

def save_full_json(results, config_dict, out_path):
    output_data = {
        "config": config_dict,
        "gnn_guided": {
            "summary": {
                "avg_objective": results.get("avg_objective"),
                "median_objective": results.get("median_objective"),
                "std_objective": results.get("std_objective"),
                "avg_time": results.get("avg_time"),
                "avg_gap": results.get("avg_gap")
            },
            "instances": results.get("results", [])
        }
    }
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)


def run_config(instances, model, max_iters, temperature, param_name, param_val, output_dir, intensive=False, gnn_guidance=True):
    """Run evaluation with specific config. Returns avg objective, max objective and avg time."""
    t0 = time.time()
    
    # Temporarily patch EnhancedConfig if we want to disable GNN
    from mtrpp_local import EnhancedConfig
    orig_gnn = EnhancedConfig.use_gnn_guidance
    EnhancedConfig.use_gnn_guidance = gnn_guidance
    
    results = evaluate_on_instances(
        instances, model,
        batch_size=16, max_iterations=max_iters,
        temperature=temperature, intensive=intensive
    )
    
    
    # Restore
    EnhancedConfig.use_gnn_guidance = orig_gnn
    
    elapsed = time.time() - t0
    
    # Save detailed json
    cfg = {"max_iterations": max_iters, "temperature": temperature, "param_name": param_name, "param_val": param_val}
    out_path = os.path.join(output_dir, f"param_{param_name}_{param_val}.json")
    save_full_json(results, cfg, out_path)
    
    instance_results = results.get('results', [])
    objs = [r['objective'] for r in instance_results]
    return float(np.mean(objs)) if objs else 0.0, float(np.max(objs)) if objs else 0.0, elapsed

def main():
    parser = argparse.ArgumentParser(description="Parameter Sensitivity")
    parser.add_argument("--dataset", type=str, default="dataset/test_split_FINAL.pt")
    parser.add_argument("--checkpoint", type=str, default="experiments/models/best_model.pt")
    parser.add_argument("--max_instances", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    instances = load_instances([args.dataset], max_instances=args.max_instances)
    # We want a mix of medium/large instances to effectively test Divide & Conquer
    test_subset = [i for i in instances if i.n_customers >= 50][:10]
    if not test_subset:
        test_subset = instances[:10]
    print(f"Sensitivity set: {len(test_subset)} instances")

    # Load model
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = create_large_model(
        hidden_dim=cp.get('hidden_dim', 128),
        num_layers=cp.get('num_layers', 6),
        heads=cp.get('heads', 8)
    )
    sd = cp['model_state_dict']
    clean_sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)

    results = []

    def record(param, val, b_sol, av_sol, t_total):
        results.append({
            'parameter': param, 'value': str(val),
            'B_sol': b_sol, 'AV_sol': av_sol, 'T_avg': t_total / max(1, len(test_subset))
        })
        print(f"    {param}={val}: B_sol={b_sol:.1f}, AV_sol={av_sol:.1f}, T_avg={t_total/len(test_subset):.2f}s")

    print("\n--- 1. Varying max_iterations (T_itr) ---")
    # Base configuration: 0.05 temp, DC threshold 150, intensive=False
    CONFIG['dc_threshold'] = 150
    CONFIG['dc_target_cluster_size'] = 50
    for iters in [50, 100, 200, 500, 1000]:
        set_seed(42)
        av, b, t = run_config(test_subset, model, iters, 0.05, 'T_itr', iters, args.output_dir)
        record('T_itr (max_iterations)', iters, b, av, t)

    print("\n--- 2. Varying Temperature (Exploration parameter) ---")
    for temp in [0.01, 0.05, 0.1, 0.5, 1.0]:
        set_seed(42)
        av, b, t = run_config(test_subset, model, 200, temp, 'temperature', temp, args.output_dir)
        record('temperature', temp, b, av, t)
        
    print("\n--- 3. Varying Divide & Conquer Threshold ---")
    # Affects when the problem splits (specific to our large-scale approach)
    orig_dc = CONFIG.get('dc_threshold', 150)
    for dc_thresh in [50, 100, 150, 200, 500]:
        CONFIG['dc_threshold'] = dc_thresh
        set_seed(42)
        av, b, t = run_config(test_subset, model, 200, 0.05, 'dc_threshold', dc_thresh, args.output_dir)
        record('dc_threshold', dc_thresh, b, av, t)
    CONFIG['dc_threshold'] = orig_dc
    
    print("\n--- 4. Varying Divide & Conquer Cluster Size ---")
    orig_cluster = CONFIG.get('dc_target_cluster_size', 50)
    for cluster_sz in [20, 30, 50, 80]:
        CONFIG['dc_target_cluster_size'] = cluster_sz
        set_seed(42)
        av, b, t = run_config(test_subset, model, 200, 0.05, 'dc_target_cluster_size', cluster_sz, args.output_dir)
        record('dc_target_cluster_size', cluster_sz, b, av, t)
    CONFIG['dc_target_cluster_size'] = orig_cluster
    
    print("\n--- 5. Varying GNN Guidance (Ablating NN from HH) ---")
    for gnn_on in [True, False]:
        set_seed(42)
        av, b, t = run_config(test_subset, model, 200, 0.05, 'use_gnn_guidance', gnn_on, args.output_dir, gnn_guidance=gnn_on)
        record('use_gnn_guidance', gnn_on, b, av, t)

    print("\n--- 6. Varying Intensive Polish (ALNS Refinement) ---")
    for intensive in [False, True]:
        set_seed(42)
        av, b, t = run_config(test_subset, model, 200, 0.05, 'intensive_polish', intensive, args.output_dir, intensive=intensive)
        record('intensive_polish', intensive, b, av, t)

    # Save
    csv_path = os.path.join(args.output_dir, "param_sensitivity.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "B_sol", "AV_sol", "T_avg(s)"])
        for r in results:
            writer.writerow([r['parameter'], r['value'],
                             f"{r['B_sol']:.1f}", f"{r['AV_sol']:.1f}", f"{r['T_avg']:.2f}"])

    print(f"\nParameter sensitivity saved to {csv_path}")


if __name__ == "__main__":
    main()
