"""
MTRPP Route Visualizer (Adapted for run_evaluation.py JSON format)
==================================================================
Reads evaluation JSON results and the test dataset (.pt) to produce
per-instance route plots. Each route is drawn in a distinct color,
with the depot highlighted and edges connecting visited nodes in order.

Usage:
    python scripts/visualize_routes_adapted.py \
        --eval_json experiments/results/evaluation_raw.json \
        --dataset   dataset/test_split_FINAL.pt \
        --output    experiments/route_plots/
"""

import argparse
import json
import os
import sys
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Colour palette (up to 12 routes) ────────────────────────────────
ROUTE_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#6A0572", "#1B998B", "#FF6B6B", "#4ECDC4",
    "#C77DFF", "#7209B7",
]


def parse_num(v):
    """Parse a number from various formats including 'tensor(...)' strings."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    m = re.search(r'[-+]?[0-9]*\.?[0-9]+', s)
    return float(m.group()) if m else None


def load_coordinates(dataset_path: str) -> dict:
    """
    Load the test split and build a lightweight mapping:
        instance_name -> coords_np_array [N+1, 2]
    Uses a pickle cache to avoid reloading the heavy .pt file.
    """
    cache_path = dataset_path + ".coord_cache.pkl"

    if os.path.exists(cache_path):
        print(f"  Using cached coordinates from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  Loading dataset from {dataset_path} (first time, may take a minute)...")
    import torch
    graphs = torch.load(dataset_path, map_location="cpu", weights_only=False)

    coord_map = {}
    for g in graphs:
        name = g.instance_name
        x = g.x.numpy()
        coord_map[name] = x[:, :2].copy()  # just x,y — discard profit & edges
    del graphs  # free memory

    # Cache for fast future loads
    with open(cache_path, "wb") as f:
        pickle.dump(coord_map, f)
    print(f"  Cached {len(coord_map)} coordinate sets to {cache_path}")

    return coord_map


def load_eval_results(json_path: str) -> list:
    """Load evaluation JSON and return the list of instance results."""
    with open(json_path, "r") as f:
        data = json.load(f)
    # Support both formats: run_evaluation.py uses 'instances',
    # older format uses 'gnn_guided.instances'
    if "instances" in data:
        return data["instances"]
    elif "gnn_guided" in data:
        return data["gnn_guided"]["instances"]
    else:
        raise ValueError("Unknown JSON format: no 'instances' or 'gnn_guided' key found")


def plot_instance(instance: dict, coords: np.ndarray, output_dir: str):
    """
    Plot a single instance's routes on a 2-D plane.
    """
    name = instance["name"]
    routes = instance["routes"]
    n_customers = int(parse_num(instance["n_customers"]))
    
    # Handle field name differences between JSON formats
    gap = parse_num(instance.get("gap", instance.get("gap_B_sol")))
    obj = parse_num(instance.get("objective", instance.get("B_sol")))
    opt = parse_num(instance.get("optimal", instance.get("reference")))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    # ── Collect visited nodes ────────────────────────────────────────
    visited_set = set()
    for route in routes:
        visited_set.update(route)

    # ── Plot unvisited nodes ─────────────────────────────────────────
    for i in range(1, len(coords)):
        if i not in visited_set:
            ax.scatter(coords[i, 0], coords[i, 1], c="#3D4450",
                       s=12, zorder=1, alpha=0.4)

    # ── Plot each route ──────────────────────────────────────────────
    legend_handles = []
    for r_idx, route in enumerate(routes):
        color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]

        # Path: depot -> route nodes (do NOT return to depot as per MTRPP literature)
        full_path = [0] + route
        path_coords = coords[full_path]

        # Draw edges
        ax.plot(path_coords[:, 0], path_coords[:, 1],
                color=color, linewidth=1.2, alpha=0.75, zorder=2)

        # Draw visited nodes
        route_coords = coords[route]
        ax.scatter(route_coords[:, 0], route_coords[:, 1],
                   c=color, s=25, zorder=3, edgecolors="white",
                   linewidths=0.3, alpha=0.9)

        patch = mpatches.Patch(color=color,
                               label=f"Route {r_idx + 1} ({len(route)} nodes)")
        legend_handles.append(patch)

    # ── Plot depot ───────────────────────────────────────────────────
    ax.scatter(coords[0, 0], coords[0, 1], c="#FFD700", s=200,
               marker="*", zorder=5, edgecolors="white", linewidths=0.8,
               label="Depot")

    # ── Annotations ──────────────────────────────────────────────────
    gap_str = f"{gap:+.2f}%" if gap is not None else "N/A"
    opt_str = f"{opt:,.0f}" if opt is not None else "N/A"
    obj_str = f"{obj:,.0f}" if obj is not None else "N/A"
    
    title_text = (f"{name}  |  N={n_customers}  |  "
                  f"Profit={obj_str}  |  Reference={opt_str}  |  "
                  f"Gap={gap_str}")
    ax.set_title(title_text, fontsize=11, color="white", fontweight="bold",
                 pad=12)
    ax.tick_params(colors="#8B949E", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363D")

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              facecolor="#21262D", edgecolor="#30363D", labelcolor="white",
              framealpha=0.9)

    ax.set_xlabel("X", color="#8B949E", fontsize=9)
    ax.set_ylabel("Y", color="#8B949E", fontsize=9)

    # ── Save ─────────────────────────────────────────────────────────
    safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = os.path.join(output_dir, f"{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MTRPP routes for all test instances.")
    parser.add_argument("--eval_json", type=str, required=True,
                        help="Path to evaluation JSON")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to test_split_FINAL.pt")
    parser.add_argument("--output", type=str, default="experiments/route_plots",
                        help="Output directory for PNG plots")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    coord_map = load_coordinates(args.dataset)
    instances = load_eval_results(args.eval_json)

    # Match and plot
    matched, skipped = 0, 0
    for inst in instances:
        json_name = inst["name"]

        # Try exact match, then base-name match
        if json_name in coord_map:
            coords = coord_map[json_name]
        else:
            base = json_name.rsplit("_", 1)[0] if "_" in json_name else json_name
            if base in coord_map:
                coords = coord_map[base]
            else:
                print(f"  ⚠ No coordinates for '{json_name}', skipping.")
                skipped += 1
                continue

        out_path = plot_instance(inst, coords, args.output)
        matched += 1
        print(f"  ✅ [{matched:2d}/{len(instances)}] {json_name} -> {os.path.basename(out_path)}")

    print(f"\n{'='*60}")
    print(f"Done! {matched} plots saved to {args.output}/")
    if skipped:
        print(f"  ⚠ {skipped} instances skipped (no coordinate data).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
