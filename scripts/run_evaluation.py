"""
MTRPP Multi-Run Evaluation
Runs the GNN solver N times per instance, collects B_sol, AV_sol, T_avg, std_dev.
"""
import os
import sys
import time
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mtrpp_local import (
    LargeCustomerRankingGNN, create_large_model,
    load_instances, evaluate_on_instances, set_seed, CONFIG
)


def main():
    parser = argparse.ArgumentParser(description="Multi-run MTRPP Evaluation")
    parser.add_argument("--dataset", type=str, default="dataset/test_split_FINAL.pt")
    parser.add_argument("--checkpoint", type=str, default="experiments/models/best_model.pt")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--intensive", action="store_true", default=True)
    parser.add_argument("--output", type=str, default="experiments/results/evaluation_raw.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    CONFIG['dc_threshold'] = 150
    CONFIG['dc_target_cluster_size'] = 50
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("MTRPP MULTI-RUN EVALUATION")
    print(f"Device: {device}")
    print(f"Runs per instance: {args.num_runs}")

    # Load dataset
    instances = load_instances([args.dataset], max_instances=args.max_instances)
    print(f"Test instances: {len(instances)}")

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
    print("Model loaded successfully")

    # Collect per-instance, per-run results
    all_results = {}  # name -> list of {objective, runtime, gap}
    for inst in instances:
        all_results[inst.name] = {
            'n_customers': inst.n_customers,
            'n_repairmen': inst.n_repairmen,
            'reference': getattr(inst, 'final_profit', None),
            'runs': []
        }

    for run_id in range(args.num_runs):
        set_seed(args.seed + run_id * 1000)
        print(f"\n--- Run {run_id + 1}/{args.num_runs} ---")

        t0 = time.time()
        results = evaluate_on_instances(
            instances, model,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations,
            temperature=args.temperature,
            intensive=args.intensive
        )
        total_time = time.time() - t0

        instance_results = results.get('results', [])
        print(f"  Evaluated {len(instance_results)} instances in {total_time:.1f}s")

        for res in instance_results:
            name = res.get('name')
            if name in all_results:
                ref = all_results[name]['reference']
                obj = res.get('objective', 0)
                gap = ((ref - obj) / ref * 100.0) if ref and ref > 0 else None
                all_results[name]['runs'].append({
                    'objective': obj,
                    'runtime': res.get('time', total_time / len(instance_results)),
                    'gap_pct': gap,
                    'visited': res.get('visited', None),
                    'routes': res.get('routes', []),
                })

    # Aggregate
    summary = []
    for name, data in all_results.items():
        runs = data['runs']
        if not runs:
            continue
        objs = [r['objective'] for r in runs]
        times = [r['runtime'] for r in runs]
        gaps = [r['gap_pct'] for r in runs if r['gap_pct'] is not None]

        entry = {
            'name': name,
            'n_customers': data['n_customers'],
            'n_repairmen': data['n_repairmen'],
            'reference': data['reference'],
            'B_sol': max(objs),
            'AV_sol': float(np.mean(objs)),
            'T_avg': float(np.mean(times)),
            'std_dev': float(np.std(objs)),
            'gap_B_sol': ((data['reference'] - max(objs)) / data['reference'] * 100.0)
                         if data['reference'] and data['reference'] > 0 else None,
            'gap_AV_sol': float(np.mean(gaps)) if gaps else None,
            'routes': runs[objs.index(max(objs))]['routes'] if objs else [],
            'visited': runs[objs.index(max(objs))]['visited'] if objs else [],
            'all_objectives': objs,
            'all_runtimes': times,
            'all_gaps': gaps,
        }
        summary.append(entry)

    output_data = {
        'config': {
            'num_runs': args.num_runs,
            'max_iterations': args.max_iterations,
            'temperature': args.temperature,
            'intensive': args.intensive,
            'checkpoint': args.checkpoint,
        },
        'instances': summary,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    # Print summary
    all_gaps = [s['gap_B_sol'] for s in summary if s['gap_B_sol'] is not None]
    improved = sum(1 for g in all_gaps if g < 0)
    matched = sum(1 for g in all_gaps if g is not None and abs(g) < 0.01)
    worse = sum(1 for g in all_gaps if g is not None and g > 0.01)

    print("\nEVALUATION SUMMARY")
    print(f"  Instances: {len(instances)}")
    print(f"  Avg gap (B_sol): {np.mean(all_gaps):.4f}%" if all_gaps else "  No gaps computed")
    print(f"  Improved vs ref: {improved}")
    print(f"  Matched ref:     {matched}")
    print(f"  Worse than ref:  {worse}")
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
