"""
Ablation Study: Remove one heuristic component at a time and measure impact.
"""
import os
import sys
import json
import csv
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mtrpp_local import (
    create_large_model, load_instances, evaluate_on_instances, set_seed, CONFIG,
    LH1_RemcrRoute, LH2_RempRoute, LH3_RemlRoute, LH4_SwapRoute,
    LH5_TwoOptRoute, LH6_RemlIroute, LH7_RemcrIroute, LH8_SwapIroute,
    LH9_RemprIroute, LH10_RempaddIroute, LH11_Remove, LH12_Add,
    LH13_DestroyRepair, LH14_SlackTriage, EnhancedHyperHeuristic
)

HEURISTIC_CLASSES = {
    'LH1_RemcrRoute': LH1_RemcrRoute,
    'LH2_RempRoute': LH2_RempRoute,
    'LH3_RemlRoute': LH3_RemlRoute,
    'LH4_SwapRoute': LH4_SwapRoute,
    'LH5_TwoOptRoute': LH5_TwoOptRoute,
    'LH6_RemlIroute': LH6_RemlIroute,
    'LH7_RemcrIroute': LH7_RemcrIroute,
    'LH8_SwapIroute': LH8_SwapIroute,
    'LH9_RemprIroute': LH9_RemprIroute,
    'LH10_RempaddIroute': LH10_RempaddIroute,
    'LH11_Remove': LH11_Remove,
    'LH12_Add': LH12_Add,
    'LH13_DestroyRepair': LH13_DestroyRepair,
    'LH14_SlackTriage': LH14_SlackTriage,
}

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


def run_ablation(instances, model, exclude_class, name, output_dir, max_iters=200):
    """Run evaluation with one heuristic class excluded via monkey-patching."""
    original_init = EnhancedHyperHeuristic.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Filter out the excluded heuristic
        self.heuristics = [h for h in self.heuristics
                           if not isinstance(h, exclude_class)]

    EnhancedHyperHeuristic.__init__ = patched_init
    try:
        results = evaluate_on_instances(
            instances, model,
            batch_size=16, max_iterations=max_iters,
            temperature=0.05, intensive=False
        )
        instance_results = results.get('results', [])
        objs = [r['objective'] for r in instance_results]
        avg_obj = float(np.mean(objs)) if objs else 0.0
        
        # Save detailed json
        cfg = {"max_iterations": max_iters, "temperature": 0.05, "ablated": name}
        out_path = os.path.join(output_dir, f"ablation_{name}.json")
        save_full_json(results, cfg, out_path)
    finally:
        EnhancedHyperHeuristic.__init__ = original_init

    return avg_obj


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--dataset", type=str, default="dataset/test_split_FINAL.pt")
    parser.add_argument("--checkpoint", type=str, default="experiments/models/best_model.pt")
    parser.add_argument("--max_instances", type=int, default=15)
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    CONFIG['dc_threshold'] = 150

    # Load medium-sized instances for ablation
    instances = load_instances([args.dataset], max_instances=args.max_instances)
    instances = [i for i in instances if 20 <= i.n_customers <= 100]
    if not instances:
        instances = load_instances([args.dataset], max_instances=args.max_instances)
    print(f"Ablation set: {len(instances)} instances")

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

    # Baseline: full heuristic set
    set_seed(42)
    print("\nRunning baseline (all heuristics)...")
    baseline_obj = run_ablation(instances, model, type(None), "Baseline", args.output_dir, args.max_iterations)
    print(f"  Baseline AV_sol: {baseline_obj:.1f}")

    # Ablate each component
    results = []
    for name, cls in HEURISTIC_CLASSES.items():
        set_seed(42)
        print(f"\nAblating {name}...")
        ablated_obj = run_ablation(instances, model, cls, name, args.output_dir, args.max_iterations)
        drop = ((baseline_obj - ablated_obj) / baseline_obj * 100.0) if baseline_obj > 0 else 0.0
        results.append({
            'component': name,
            'AV_sol_without': ablated_obj,
            'drop_pct': drop,
        })
        print(f"  AV_sol without: {ablated_obj:.1f} (drop: {drop:.2f}%)")

    # Rank by importance (larger drop = more important)
    results.sort(key=lambda x: x['drop_pct'], reverse=True)
    for i, r in enumerate(results):
        r['rank'] = i + 1

    # Save
    csv_path = os.path.join(args.output_dir, "ablation.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Component", "Baseline_AV_sol", "AV_sol_Without", "Drop(%)"])
        for r in results:
            writer.writerow([
                r['rank'], r['component'],
                f"{baseline_obj:.1f}", f"{r['AV_sol_without']:.1f}", f"{r['drop_pct']:.4f}"
            ])

    print(f"\nAblation results saved to {csv_path}")


if __name__ == "__main__":
    main()
