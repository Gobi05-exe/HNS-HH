# %% [code] {"jupyter":{"outputs_hidden":false}}
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Future imports must be first
from __future__ import annotations



"""
MTRPP RL Training - Self-Contained Jupyter-Style Script

This script replicates the exact behavior of run_rl.sh and contains all necessary
modules inlined. It can be run as a regular Python script or opened in Jupyter
using the # %% cell markers.

Usage:
    python mtrpp_rl_notebook.py
    
Or open in Jupyter/VS Code for cell-by-cell execution.

Author: Generated from MTRPP_Project codebase
"""

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # MTRPP Reinforcement Learning Training
# 
# This notebook trains a Graph Neural Network (GNN) using advantage actor-critic
# to solve the Multi-Trip Repairman Problem with Profits (MTRPP).
# #
# The GNN learns to predict which customers are most promising to include in
# heuristic operations, improving search efficiency.

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# CONFIGURATION - Modify these values as needed
# ==============================================================================

CONFIG = {
    # Dataset paths (relative to script location)  
    "dataset_paths": [
        "dataset/avci_graphs_with_results.pt",
        "dataset/lu_graphs_with_results.pt",
    ],
    
    # Data split (train/test) - 90/10 for more training data
    "train_split": 0.9,  # 90% train, 10% test
    "max_instances": None,  # Use all instances
    
    # Training parameters (Optimized for 8-10h run: 1 epoch, full episodes)
    "epochs": 1,                    # Single dedicated epoch
    "episodes_per_instance": 1,     # Single pass (to allow full solution within time)
    "max_steps_per_episode": 300,   # High limit (finish solutions fully, don't truncate)
    
    # Model architecture (Optimized for 4GB GPU Stability)
    "hidden_dim": 128,
    "num_layers": 6,                # Target for 48h surge
    "num_heads": 8,                 # Target for 48h surge
    
    # RL hyperparameters
    "learning_rate": 5e-4,          # Aggressive LR to maximize single update value
    "gamma": 0.99,
    "entropy_coef": 0.02,           # Slightly higher entropy for more exploration early on
    "temperature": 1.0,
    "temperature_decay": 0.999,      # Decay per epoch: 1.0 -> 0.80 -> 0.64 -> 0.51 -> 0.41 -> 0.33
    "min_temperature": 0.1,         # Very greedy in final epoch
    # k is adaptive: max(3, min(n_customers // 10, 50)) — computed at runtime
    
    # Divide & Conquer
    "dc_threshold": 150,          # Minimum n_customers to activate D&C
    "dc_target_cluster_size": 50, # Target sub-problem size
    "dc_boundary_repair_iters": 80, # Increased for deep breakthrough optimization
    "dc_profit_aware": True,      # Use profit-weighted clustering
    
    # Reward shaping
    "reward_clip": 10.0,            # Clip normalized reward to [-10, 10]
    
    # Checkpointing
    "checkpoint_every": 50,         # Save checkpoint every N instances
    
    # Evaluation parameters
    "eval_temperature": 0.2,       # Near-greedy evaluation for clearest signal
    "eval_iterations": 25,          # Increased from 20 for better search budget
    "eval_restarts": 1,             # Set to >1 for best-of-N restarts (trades time for quality)
    
    # Output
    "output_dir": "checkpoints_rl",
    "seed": 42,
}

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# IMPORTS
# ==============================================================================

import os
# Set single-threaded mode for libraries to avoid oversubscription in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Reduction in VRAM fragmentation to fix OOM on 4GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import json
import contextlib
import random
import time
import pickle
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Set, List, Tuple, Dict, Any, Union, TYPE_CHECKING
from copy import deepcopy
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

try:
    from torch_geometric.nn import TransformerConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import softmax as pyg_softmax
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # Set multiprocessing start method for stability on Linux (avoid fork+cuda deadlock)
    try:
        if os.name != 'nt':
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. Install with:")
    print("  pip install torch-geometric")
    Data = None
    Batch = None

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. D&C decomposition unavailable.")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# UTILITIES
# ==============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif torch is not None and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist() if obj.dim() > 0 else obj.item()
        return json.JSONEncoder.default(self, obj)


# ==============================================================================
# CORE MODULE: Instance
# ==============================================================================

class Instance:
    """
    MTRPP Problem Instance
    
    Represents a Multi-Trip Repairman Problem with Profits instance containing:
    - Customer locations (coordinates)
    - Profit deadlines (time-dependent profit decay)
    - Travel times between locations
    - Number of repairmen (vehicles)
    """
    
    def __init__(
        self,
        n_customers: int,
        n_repairmen: int,
        coords: np.ndarray,
        profits: np.ndarray,
        travel_time: np.ndarray,
        instance_name: str = "unnamed"
    ):
        """
        Args:
            n_customers: Number of customers (excluding depot)
            n_repairmen: Number of repairmen/vehicles
            coords: Coordinates [n_customers+1, 2] (index 0 is depot)
            profits: Profit deadlines [n_customers+1] (profit[0] is depot, usually 0)
            travel_time: Travel time matrix [n+1, n+1]
            instance_name: Name identifier for the instance
        """
        self.n_customers = n_customers
        self.n_repairmen = n_repairmen
        self.coords = np.array(coords, dtype=np.float64)
        self.profits = np.array(profits, dtype=np.float64)
        self.travel_time = np.array(travel_time, dtype=np.float64)
        self.name = instance_name
        
        assert self.coords.shape[0] == n_customers + 1
        assert self.profits.shape[0] == n_customers + 1
        assert self.travel_time.shape == (n_customers + 1, n_customers + 1)
        
        # Precompute static graph data
        self.precompute_graph_data()
        
    def precompute_graph_data(self):
        """Precompute static graph structures for efficient GNN data construction."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            self.static_edge_index = None
            self.static_edge_attr = None
            self.static_node_features = None
            return

        n = self.n_customers + 1
        
        # --- 1. Agnostic Node Features ---
        # Instead of raw X,Y, we use Polar coordinates + Local Density
        # to ensure the model is Translation and Rotation Agnostic.
        
        # 1.1 Radial Distance (Dist to Depot)
        depot_coords = self.coords[0]
        distances_to_depot = np.linalg.norm(self.coords - depot_coords, axis=1)
        radial_norm = distances_to_depot / (np.max(distances_to_depot) + 1e-6)
        
        # 1.2 Angular Position (Theta)
        relative_coords = self.coords - depot_coords
        angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
        angle_norm = (angles + np.pi) / (2 * np.pi) # Map to [0, 1]
        
        # 1.3 Local Density (Avg distance to 5-NN)
        # 1.4 Cluster Value (Sum of profits of 5-NN)
        n_neighbors = min(5, self.n_customers)
        density = np.zeros(n)
        cluster_val = np.zeros(n)
        
        for i in range(n):
            dists = self.travel_time[i]
            nearest_indices = np.argsort(dists)[1:n_neighbors+1]
            density[i] = np.mean(dists[nearest_indices])
            cluster_val[i] = np.sum(self.profits[nearest_indices])
            
        density_norm = density / (np.max(density) + 1e-6)
        cluster_norm = cluster_val / (np.max(cluster_val) + 1e-6)
        profit_norm = self.profits / (np.max(self.profits) + 1e-6)
        
        # Static Node Features [Radial, Angle, Density, ClusterVal, Profit]
        self.static_node_features = np.column_stack([
            radial_norm,
            angle_norm,
            density_norm,
            cluster_norm,
            profit_norm
        ]).astype(np.float32)
        
        # --- 2. Static Edge Index and Attributes ---
        edge_index = []
        edge_attr_static = [] # [travel_norm]
        travel_max = np.max(self.travel_time) + 1e-6

        # KNN Sparsification for memory safety (K=25)
        # This reduces edge memory by ~4-6x while keeping all relevant local moves
        K = min(25, n - 1)
        
        for i in range(n):
            # Get distances from i to all other nodes
            dists = self.travel_time[i, :]
            # Get indices of K nearest nodes (excluding i)
            nearest = np.argsort(dists)
            count = 0
            # Always keep depot connections if i is not depot
            if i != 0:
                edge_index.append([i, 0])
                edge_index.append([0, i])
                edge_attr_static.append(self.travel_time[i, 0] / travel_max)
                edge_attr_static.append(self.travel_time[0, i] / travel_max)
            
            for j in nearest:
                if i == j or j == 0: continue # Already handled or self
                edge_index.append([i, j])
                edge_attr_static.append(self.travel_time[i, j] / travel_max)
                count += 1
                if count >= K: break
        
        # Enrichment: Add profit-ratio edges regardless of KNN budget
        # This allows the GNN to see high-value "long jumps"
        for i in range(n):
            profit_travel_ratios = []
            for j in range(1, n):
                if j != i:
                    ratio = self.profits[j] / max(self.travel_time[i, j], 1e-6)
                    profit_travel_ratios.append((j, ratio))
            
            # Add top 3 by profit/travel ratio
            profit_travel_ratios.sort(key=lambda x: -x[1])
            for j, _ in profit_travel_ratios[:3]:
                # We'll handle uniqueness later in the unique_edges dict
                edge_index.append([i, j])
                edge_attr_static.append(self.travel_time[i, j] / travel_max)

        # Remove duplicates (depot might have been added twice)
        unique_edges = {}
        for (u, v), attr in zip(edge_index, edge_attr_static):
            if (u, v) not in unique_edges:
                unique_edges[(u, v)] = attr
        
        final_index = []
        final_attr = []
        for (u, v), attr in unique_edges.items():
            final_index.append([u, v])
            final_attr.append(attr)
            
        self.static_edge_index = torch.tensor(final_index, dtype=torch.long).t().contiguous()
        self.static_edge_attr_base = torch.tensor(final_attr, dtype=torch.float32).unsqueeze(1) # [E, 1]
    
    def get_travel_time(self, from_node: int, to_node: int) -> float:
        return self.travel_time[from_node, to_node]
    
    def get_profit_deadline(self, customer: int) -> float:
        return self.profits[customer]
    
    def compute_profit(self, customer: int, arrival_time: float) -> float:
        """Compute profit for arriving at customer at given time."""
        deadline = self.profits[customer]
        if arrival_time >= deadline:
            return 0.0
        return deadline - arrival_time
    
    @classmethod
    def from_pyg_data(cls, data: Data, instance_name: str = "unnamed") -> "Instance":
        """Create Instance from PyTorch Geometric Data object."""
        x = data.x.numpy() if hasattr(data.x, 'numpy') else np.array(data.x)
        
        n_nodes = x.shape[0]
        n_customers = n_nodes - 1
        
        # Extract coordinates (features 0, 1)
        coords = x[:, :2].copy()
        
        # Extract profits (feature 2) - need to denormalize if normalized
        profits_raw = x[:, 2].copy()
        
        # Handle normalization
        profit_max = np.max(profits_raw) if np.max(profits_raw) > 2 else 1.0
        if profit_max <= 1.0:
            # Profits appear normalized - check for metadata
            if hasattr(data, 'profit_max'):
                profit_max = float(data.profit_max)
            else:
                profit_max = 100.0  # Default
        
        profits = profits_raw * profit_max if np.max(profits_raw) <= 1.0 else profits_raw
        
        # Handle coordinates normalization
        coord_max = np.max(np.abs(coords)) if np.max(np.abs(coords)) > 2 else 1.0
        if coord_max <= 1.0:
            if hasattr(data, 'coord_max'):
                coord_max = float(data.coord_max)
            else:
                coord_max = 100.0
        
        coords = coords * coord_max if np.max(np.abs(coords)) <= 1.0 else coords
        
        # Build travel time matrix (Euclidean distances)
        travel_time = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist = np.round(np.sqrt((coords[i, 0] - coords[j, 0])**2 + 
                                           (coords[i, 1] - coords[j, 1])**2))
                    travel_time[i, j] = dist
        
        # Number of repairmen — try stored attribute, then parse from filename
        n_repairmen = int(getattr(data, 'n_repairmen', -1))
        if n_repairmen < 0:
            # Parse from filename: "10.1.txt_2" -> R=2
            if instance_name.endswith(('.txt_2', '.txt_3', '.txt_4')):
                n_repairmen = int(instance_name.rsplit('_', 1)[-1])
            elif hasattr(data, 'routes') and isinstance(data.routes, list) and len(data.routes) > 0:
                # Infer from number of routes in solution
                n_repairmen = len(data.routes)
            else:
                n_repairmen = 3  # true last resort
        
        return cls(
            n_customers=n_customers,
            n_repairmen=n_repairmen,
            coords=coords,
            profits=profits,
            travel_time=travel_time,
            instance_name=instance_name
        )
    
    def __repr__(self) -> str:
        return f"Instance(name={self.name}, customers={self.n_customers}, repairmen={self.n_repairmen})"

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# CORE MODULE: Solution and Route
# ==============================================================================

class Route:
    """
    Represents a single repairman's route.
    
    A route is a sequence of customer visits, implicitly starting and ending at depot (0).
    """
    
    def __init__(self, instance: Instance, repairman_id: int = 0):
        self.instance = instance
        self.repairman_id = repairman_id
        self.customers: List[int] = []
        
        # Cache for objective computation
        self._cached_objective: Optional[float] = None
        self._cached_arrival_times: Optional[List[float]] = None
    
    def _invalidate_cache(self) -> None:
        self._cached_objective = None
        self._cached_arrival_times = None
    
    def insert(self, position: int, customer: int) -> None:
        """Insert customer at position."""
        self.customers.insert(position, customer)
        self._invalidate_cache()
    
    def remove_at(self, position: int) -> int:
        """Remove and return customer at position."""
        customer = self.customers.pop(position)
        self._invalidate_cache()
        return customer
    
    def swap(self, i: int, j: int) -> None:
        """Swap customers at positions i and j."""
        self.customers[i], self.customers[j] = self.customers[j], self.customers[i]
        self._invalidate_cache()
    
    def reverse_segment(self, i: int, j: int) -> None:
        """Reverse the segment from i to j (inclusive)."""
        self.customers[i:j+1] = self.customers[i:j+1][::-1]
        self._invalidate_cache()
    
    def compute_arrival_times(self) -> List[float]:
        """Compute arrival time at each customer."""
        if self._cached_arrival_times is not None:
            return self._cached_arrival_times
        
        if not self.customers:
            self._cached_arrival_times = []
            return []
        
        arrival_times = []
        current_time = 0.0
        prev = 0  # Start at depot
        
        for customer in self.customers:
            current_time += self.instance.get_travel_time(prev, customer)
            arrival_times.append(current_time)
            prev = customer
        
        self._cached_arrival_times = arrival_times
        return arrival_times
    
    def compute_travel_time(self) -> float:
        """Compute total travel time for this route."""
        if not self.customers:
            return 0.0
        
        arrival_times = self.compute_arrival_times()
        if not arrival_times:
            return 0.0
            
        # Arrival time at last customer is the total travel time from depot
        return arrival_times[-1]
    
    def compute_objective(self) -> float:
        """Compute total profit collected on this route."""
        if self._cached_objective is not None:
            return self._cached_objective
        
        if not self.customers:
            self._cached_objective = 0.0
            return 0.0
        
        arrival_times = self.compute_arrival_times()
        total_profit = 0.0
        
        for customer, arr_time in zip(self.customers, arrival_times):
            profit = self.instance.compute_profit(customer, arr_time)
            total_profit += profit
        
        self._cached_objective = total_profit
        return total_profit

    def prune_late_nodes(self) -> bool:
        """
        Remove any nodes that are visited after their profit deadline.
        Returns True if any nodes were removed.
        """
        if not self.customers:
            return False
            
        initial_count = len(self.customers)
        new_customers = []
        prev = 0
        t = 0.0
        
        for cust in self.customers:
            t += float(self.instance.get_travel_time(prev, cust))
            deadline = float(self.instance.profits[cust])
            
            if t < (deadline + 1e-3):
                new_customers.append(cust)
                prev = cust
            else:
                # Node is late, skip it. t remains at prev node.
                pass
        
        if len(new_customers) != initial_count:
            self.customers = new_customers
            self._invalidate_cache()
            return True
        return False
    
    def evaluate_removal(self, position: int) -> float:
        """Compute objective change if we remove customer at position."""
        # Optimization: Simulate removal without mutation
        # This keeps the current State cached.
        
        current_obj = self.compute_objective()
        
        simul_profit = 0.0
        current_time = 0.0
        prev = 0
        
        for k in range(len(self.customers)):
            if k == position:
                continue
            
            cust = self.customers[k]
            current_time += self.instance.get_travel_time(prev, cust)
            simul_profit += self.instance.compute_profit(cust, current_time)
            prev = cust
            
        return simul_profit - current_obj

    def evaluate_insertion(self, customer: int, position: int) -> float:
        """Compute objective change if we insert customer at position."""
        current_obj = self.compute_objective()
        
        simul_profit = 0.0
        current_time = 0.0
        prev = 0
        
        # Iterate through simulated sequence
        # Sequence: customers[:position] + [customer] + customers[position:]
        
        # 1. Before position
        for k in range(position):
            cust = self.customers[k]
            current_time += self.instance.get_travel_time(prev, cust)
            simul_profit += self.instance.compute_profit(cust, current_time)
            prev = cust
            
        # 2. The inserted customer
        current_time += self.instance.get_travel_time(prev, customer)
        simul_profit += self.instance.compute_profit(customer, current_time)
        prev = customer
        
        # 3. After position
        for k in range(position, len(self.customers)):
            cust = self.customers[k]
            current_time += self.instance.get_travel_time(prev, cust)
            simul_profit += self.instance.compute_profit(cust, current_time)
            prev = cust
            
        return simul_profit - current_obj

    def evaluate_swap(self, i: int, j: int) -> float:
        """Compute objective change if we swap customers at i and j."""
        current_obj = self.compute_objective()
        
        simul_profit = 0.0
        current_time = 0.0
        prev = 0
        
        for k in range(len(self.customers)):
            cust = self.customers[k]
            if k == i:
                cust = self.customers[j]
            elif k == j:
                cust = self.customers[i]
            
            current_time += self.instance.get_travel_time(prev, cust)
            simul_profit += self.instance.compute_profit(cust, current_time)
            prev = cust
            
        return simul_profit - current_obj

    def evaluate_reverse_segment(self, i: int, j: int) -> float:
        """Compute objective change if we reverse segment [i, j]."""
        current_obj = self.compute_objective()
        
        simul_profit = 0.0
        current_time = 0.0
        prev = 0
        
        # 0 to i-1: normal
        for k in range(i):
            cust = self.customers[k]
            current_time += self.instance.get_travel_time(prev, cust)
            simul_profit += self.instance.compute_profit(cust, current_time)
            prev = cust
        
        # i to j: reversed
        # effectively accessing indices j, j-1, ..., i
        for k in range(j, i - 1, -1):
            cust = self.customers[k]
            current_time += self.instance.get_travel_time(prev, cust)
            simul_profit += self.instance.compute_profit(cust, current_time)
            prev = cust
            
        # j+1 to end: normal
        for k in range(j + 1, len(self.customers)):
            cust = self.customers[k]
            current_time += self.instance.get_travel_time(prev, cust)
            simul_profit += self.instance.compute_profit(cust, current_time)
            prev = cust
            
        return simul_profit - current_obj
    
    def find_best_insertion(self, customer: int) -> Tuple[int, float]:
        """Find best position to insert customer. Returns (position, delta)."""
        best_pos = 0
        best_delta = float('-inf')
        
        old_obj = self.compute_objective()
        
        for pos in range(len(self.customers) + 1):
            # Optimized: Use evaluate_insertion instead of insert/compute/remove
            delta = self.evaluate_insertion(customer, pos)
            
            if delta > best_delta:
                best_delta = delta
                best_pos = pos
        
        return best_pos, best_delta
    
    def copy(self) -> "Route":
        new_route = Route(self.instance, self.repairman_id)
        new_route.customers = self.customers.copy()
        return new_route
    
    def __len__(self) -> int:
        return len(self.customers)
    
    def __getitem__(self, index: int) -> int:
        return self.customers[index]
    
    def __iter__(self):
        return iter(self.customers)
    
    def __repr__(self) -> str:
        return f"Route({self.repairman_id}): {self.customers}"


class Solution:
    """
    Complete solution to an MTRPP instance.
    
    Contains R routes (one per repairman) and tracks visited customers.
    """
    
    def __init__(self, instance: Instance, n_routes: Union[int, List[Route], List[List[int]]] = None):
        self.instance = instance
        
        if n_routes is None:
            self.n_routes = instance.n_repairmen
            self.routes = [Route(instance, i) for i in range(self.n_routes)]
        elif isinstance(n_routes, int):
            self.n_routes = n_routes
            self.routes = [Route(instance, i) for i in range(self.n_routes)]
        elif isinstance(n_routes, list):
            self.n_routes = len(n_routes)
            # Check if list of Route objects or list of customer lists
            if len(n_routes) > 0 and isinstance(n_routes[0], Route):
                self.routes = [r.copy() for r in n_routes] # Key: copy to avoid shared ref issues
            else:
                # Handle both List[List[int]] and List[Dict] (our internal output format)
                self.routes = []
                for i, r_data in enumerate(n_routes):
                    new_route = Route(instance, i)
                    if isinstance(r_data, dict) and 'customers' in r_data:
                        new_route.customers = list(r_data['customers'])
                    else:
                        # Assume list of customers
                        new_route.customers = list(r_data)
                    new_route._invalidate_cache()
                    self.routes.append(new_route)
        else:
            raise TypeError(f"Invalid type for n_routes: {type(n_routes)}")
        
        # Cache
        self._visited_set: Optional[Set[int]] = None
    
    def _invalidate_visited_cache(self) -> None:
        self._visited_set = None
    
    def compute_objective(self) -> float:
        """Total profit across all routes."""
        return sum(route.compute_objective() for route in self.routes)
    
    def get_visited_set(self) -> Set[int]:
        """Get set of all visited customers."""
        if self._visited_set is not None:
            return self._visited_set
        
        visited = set()
        for route in self.routes:
            visited.update(route.customers)
        
        self._visited_set = visited
        return visited
    
    def get_unvisited_set(self) -> Set[int]:
        """Get set of unvisited customers."""
        all_customers = set(range(1, self.instance.n_customers + 1))
        return all_customers - self.get_visited_set()
    
    def is_customer_visited(self, customer: int) -> bool:
        return customer in self.get_visited_set()
    
    def find_customer_route(self, customer: int) -> Optional[int]:
        """Find which route contains a customer."""
        for i, route in enumerate(self.routes):
            if customer in route.customers:
                return i
        return None
    
    def get_customer_position(self, customer: int) -> Optional[Tuple[int, int]]:
        """Get (route_idx, position) of customer."""
        for i, route in enumerate(self.routes):
            for j, c in enumerate(route.customers):
                if c == customer:
                    return (i, j)
        return None
    
    def get_visited_count(self) -> int:
        return len(self.get_visited_set())
    
    def get_customer_map(self) -> Dict[int, Tuple[int, int]]:
        """
        Build and return a map of customer -> (route_idx, position).
        Computed on-demand, O(N).
        """
        mapping = {}
        for r_idx, route in enumerate(self.routes):
            for pos, c in enumerate(route.customers):
                mapping[c] = (r_idx, pos)
        return mapping
    
    def find_best_insertion_global(self, customer: int) -> Tuple[int, int, float]:
        """Find best insertion across all routes. Returns (route_idx, position, delta)."""
        best_route = 0
        best_pos = 0
        best_delta = float('-inf')
        
        for route_idx, route in enumerate(self.routes):
            pos, delta = route.find_best_insertion(customer)
            if delta > best_delta:
                best_delta = delta
                best_route = route_idx
                best_pos = pos
        
        return best_route, best_pos, best_delta
    
    def compute_arrival_times_dict(self) -> Dict[int, float]:
        """Return dict mapping customer -> arrival time."""
        result = {}
        for route in self.routes:
            arrivals = route.compute_arrival_times()
            for customer, arr_time in zip(route.customers, arrivals):
                result[customer] = arr_time
        return result
    
    def is_feasible(self) -> bool:
        """Check if solution is feasible (no duplicates)."""
        seen = set()
        for route in self.routes:
            for c in route.customers:
                if c in seen:
                    return False
                seen.add(c)
        return True
    
    def assert_feasible(self) -> None:
        """Assert feasibility."""
        assert self.is_feasible(), "Solution is infeasible"
    
    def copy(self) -> "Solution":
        new_solution = Solution(self.instance, self.n_routes)
        new_solution.routes = [route.copy() for route in self.routes]
        return new_solution
    
    def validate_ordering(self) -> bool:
        """
        Strict check: simulate arrival times along all routes and ensure:
          - No duplicate visits across routes
          - For each visited customer i: arrival_time < profit_deadline p_i
        Returns True if ordering is valid.
        """
        inst = self.instance
        visited = set()
        for route in self.routes:
            prev = 0  # depot
            t = 0.0
            for cust in route.customers:
                if cust == 0:
                    continue
                if cust in visited:
                    return False
                if prev < 0 or cust < 0:
                    return False
                
                t += float(inst.travel_time[prev, cust])
                # MTRPP physics: if arrival >= deadline -> unreachable/invalid profit
                # Using 1e-3 epsilon for float stability
                if t >= (float(inst.profits[cust]) + 1e-3):
                    return False
                
                visited.add(cust)
                prev = cust
        return True

    def get_solution_routes(self) -> List[Dict]:
        """Return solution routes in a structured format for output."""
        routes_info = []
        for i, route in enumerate(self.routes):
            routes_info.append({
                'repairman_id': i,
                'customers': list(route.customers),
                'objective': route.compute_objective(),
                'travel_time': route.compute_arrival_times()[-1] if route.customers else 0.0,
                'arrival_times': route.compute_arrival_times(),
                'n_customers': len(route.customers),
            })
        return routes_info
    
    def __repr__(self) -> str:
        obj = self.compute_objective()
        visited = self.get_visited_count()
        return f"Solution(obj={obj:.2f}, visited={visited}/{self.instance.n_customers})"

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# HEURISTICS MODULE: Base Classes
# ==============================================================================

class Heuristic(ABC):
    """Abstract base class for low-level heuristics."""
    
    name: str = "BaseHeuristic"
    
    @abstractmethod
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        """Apply the heuristic to the solution."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class IntraRouteHeuristic(Heuristic):
    """Heuristic that operates within a single route."""
    pass


class InterRouteHeuristic(Heuristic):
    """Heuristic that operates across multiple routes."""
    pass


class GreedyHeuristic(Heuristic):
    """Greedy improvement heuristic."""
    pass

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# HEURISTICS MODULE: Greedy Heuristics (LH11, LH12) + Constructive
# ==============================================================================

class LH11_Remove(GreedyHeuristic):
    """LH11 — Remove the single node whose removal maximizes improvement."""
    
    name = "LH11_Remove"
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        current_obj = new_solution.compute_objective()
        
        best_improvement = 0.0
        best_removal = None
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            
            for pos in range(len(route)):
                customer = route[pos]
                
                if candidates is not None and customer not in candidates:
                    continue
                
                delta = route.evaluate_removal(pos)
                
                if delta > best_improvement:
                    best_improvement = delta
                    best_removal = (route_idx, pos, customer)
        
        if best_removal is not None:
            route_idx, pos, customer = best_removal
            new_solution.routes[route_idx].remove_at(pos)
            return new_solution
        
        return None


class LH12_Add(GreedyHeuristic):
    """LH12 — Insert the single unvisited node that maximizes improvement."""
    
    name = "LH12_Add"
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        
        unvisited = new_solution.get_unvisited_set()
        
        if candidates is not None:
            unvisited = unvisited & candidates
        
        if not unvisited:
            return None
        
        best_improvement = 0.0
        best_insertion = None
        
        for customer in unvisited:
            route_idx, pos, delta = new_solution.find_best_insertion_global(customer)
            
            if delta > best_improvement:
                best_improvement = delta
                best_insertion = (route_idx, pos, customer)
        
        if best_insertion is not None:
            route_idx, pos, customer = best_insertion
            new_solution.routes[route_idx].insert(pos, customer)
            return new_solution
        
        return None


class LH13_DestroyRepair(InterRouteHeuristic):
    """LH13 — ALNS-style Destroy & Repair. Rips out a large segment and optimally reinserts."""
    
    name = "LH13_DestroyRepair"
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        
        # Pick a random route with at least 4 nodes
        valid_routes = [i for i, r in enumerate(new_solution.routes) if len(r) >= 4]
        if not valid_routes:
            return None
            
        route_idx = random.choice(valid_routes)
        route = new_solution.routes[route_idx]
        
        # Destroy 15% to 30% of the route (contiguous chunk)
        destroy_pct = random.uniform(0.15, 0.30)
        destroy_size = max(2, int(len(route) * destroy_pct))
        
        start_idx = random.randint(0, len(route) - destroy_size)
        
        removed = []
        for _ in range(destroy_size):
            removed.append(route.remove_at(start_idx))
            
        # Repair: Optimal greedy reinsertion across ALL routes
        random.shuffle(removed)
        for customer in removed:
            best_idx, best_pos, _ = new_solution.find_best_insertion_global(customer)
            if best_idx is not None:
                new_solution.routes[best_idx].insert(best_pos, customer)
                
        return new_solution


class LH14_SlackTriage(IntraRouteHeuristic):
    """LH14 — Look-Ahead Slack Triage. 
    
    Removes 'barely-alive' nodes whose marginal profit is low AND whose removal
    would advance subsequent nodes enough to unlock substantially higher profit
    for them. Targets the MTRPP cascade effect.
    """
    name = "LH14_SlackTriage"

    def __init__(self, slack_threshold_pct: float = 0.05, lookahead: int = 3):
        self.slack_threshold_pct = slack_threshold_pct
        self.lookahead = lookahead

    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        # Fast pre-check: are there any barely-alive candidate nodes at all?
        found_any = False
        for route in solution.routes:
            arrivals = route.compute_arrival_times()
            for pos, c in enumerate(route.customers):
                if candidates is not None and c not in candidates:
                    continue
                deadline = solution.instance.profits[c]
                if solution.instance.compute_profit(c, arrivals[pos]) < self.slack_threshold_pct * deadline:
                    found_any = True
                    break
            if found_any: break
        
        if not found_any:
            return None

        new_solution = solution.copy()
        modified = False

        for route_idx, route in enumerate(new_solution.routes):
            if len(route) < self.lookahead + 1:
                continue

            arrivals = route.compute_arrival_times()
            best_remove_pos = None
            best_cascade_gain = 0.0

            for pos in range(len(route) - self.lookahead):
                c = route[pos]
                if candidates is not None and c not in candidates:
                    continue

                deadline = route.instance.profits[c]
                realised = route.instance.compute_profit(c, arrivals[pos])

                # Only consider 'barely-alive' nodes
                if realised > self.slack_threshold_pct * deadline:
                    continue

                # Simulate removal: how much time do downstream nodes gain?
                prev_c = route[pos - 1] if pos > 0 else 0
                next_c = route[pos + 1]
                
                time_saved = (arrivals[pos] - (arrivals[pos - 1] if pos > 0 else 0.0)) + \
                             route.instance.get_travel_time(prev_c, next_c) - \
                             route.instance.get_travel_time(c, next_c)

                # Measure cascade profit gain for next `lookahead` nodes
                cascade_gain = -realised  # we lose this node's profit
                for k in range(1, self.lookahead + 1):
                    if pos + k >= len(route):
                        break
                    cn = route[pos + k]
                    old_arr = arrivals[pos + k]
                    new_arr = max(0.0, old_arr - time_saved)
                    cascade_gain += (
                        route.instance.compute_profit(cn, new_arr) -
                        route.instance.compute_profit(cn, old_arr)
                    )

                if cascade_gain > best_cascade_gain:
                    best_cascade_gain = cascade_gain
                    best_remove_pos = pos

            if best_remove_pos is not None:
                route.remove_at(best_remove_pos)
                modified = True

        return new_solution if modified else None


class ConstructiveHeuristic:
    """Build initial solution using greedy insertion."""
    
    def __init__(self, instance: Instance):
        self.instance = instance
    
    def construct(self, seed: Optional[int] = None) -> Solution:
        """Build initial solution using deterministic greedy insertion."""
        if seed is not None:
            random.seed(seed)
        solution = Solution(instance=self.instance)
        
        priorities = []
        for c in range(1, self.instance.n_customers + 1):
            dist_from_depot = self.instance.get_travel_time(0, c)
            profit_deadline = self.instance.get_profit_deadline(c)
            
            if dist_from_depot > 0:
                potential_profit = max(profit_deadline - dist_from_depot, 0)
                priority = potential_profit / dist_from_depot if potential_profit > 0 else 0
            else:
                priority = profit_deadline
            priorities.append((c, priority))
        
        priorities.sort(key=lambda x: x[1], reverse=True)
        unvisited = [c for c, p in priorities if p > 0]
        
        for customer in unvisited:
            best_route = None
            best_pos = None
            max_improvement = -float('inf')
            
            for r_idx in range(solution.n_routes):
                route = solution.routes[r_idx]
                pos, delta = route.find_best_insertion(customer)
                
                if pos is not None and delta > max_improvement:
                    max_improvement = delta
                    best_route = r_idx
                    best_pos = pos
            
            if best_route is not None:
                solution.routes[best_route].insert(best_pos, customer)
                
        return solution

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# HEURISTICS MODULE: Intra-Route Heuristics (LH1-LH5)
# ==============================================================================

class LH1_RemcrRoute(IntraRouteHeuristic):
    """LH1 — Remove crn consecutive nodes, shuffle, reinsert in same route."""
    
    name = "LH1_RemcrRoute"
    
    def __init__(self, crn: int = 3):
        self.crn = crn
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        modified = False
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            
            if len(route) < 2:
                continue
            
            actual_crn = min(self.crn, len(route))
            max_start = len(route) - actual_crn
            if max_start < 0:
                continue
            
            start_pos = random.randint(0, max_start)
            
            removed = []
            for _ in range(actual_crn):
                if start_pos < len(route):
                    customer = route.remove_at(start_pos)
                    if candidates is None or customer in candidates:
                        removed.append(customer)
                    else:
                        route.insert(start_pos, customer)
            
            if not removed:
                continue
            
            modified = True
            random.shuffle(removed)
            
            for customer in removed:
                best_pos, _ = route.find_best_insertion(customer)
                route.insert(best_pos, customer)
        
        return new_solution if modified else None


class LH2_RempRoute(IntraRouteHeuristic):
    """LH2 — Remove nodes by probability, shuffle, reinsert in same route."""
    
    name = "LH2_RempRoute"
    
    def __init__(
        self, 
        max_prob: float = 0.5, 
        min_prob: float = 0.1,
        max_iter_for_decay: int = 500
    ):
        self.max_prob = max_prob
        self.min_prob = min_prob
        self.max_iter_for_decay = max_iter_for_decay
    
    def _compute_probability(self, iteration: int) -> float:
        if iteration >= self.max_iter_for_decay:
            return self.min_prob
        slope = (self.max_prob - self.min_prob) / self.max_iter_for_decay
        return slope * (self.max_iter_for_decay - iteration) + self.min_prob
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        prrd = self._compute_probability(iteration)
        modified = False
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            if len(route) == 0:
                continue
            
            removed = []
            positions_to_remove = []
            
            for pos in range(len(route)):
                customer = route[pos]
                if candidates is not None and customer not in candidates:
                    continue
                if random.random() < prrd:
                    positions_to_remove.append(pos)
            
            for pos in reversed(positions_to_remove):
                customer = route.remove_at(pos)
                removed.append(customer)
                modified = True
            
            if not removed:
                continue
            
            random.shuffle(removed)
            
            for customer in removed:
                best_pos, _ = route.find_best_insertion(customer)
                route.insert(best_pos, customer)
        
        return new_solution if modified else None


class LH3_RemlRoute(IntraRouteHeuristic):
    """LH3 — Remove last rl nodes from route, shuffle, reinsert in same route."""
    
    name = "LH3_RemlRoute"
    
    def __init__(self, rl: int = 2):
        self.rl = rl
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        modified = False
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            if len(route) == 0:
                continue
            
            actual_rl = min(self.rl, len(route))
            removed = []
            
            for _ in range(actual_rl):
                if len(route) > 0:
                    customer = route.remove_at(len(route) - 1)
                    if candidates is None or customer in candidates:
                        removed.append(customer)
                    else:
                        route.insert(len(route), customer)
            
            if not removed:
                continue
            
            modified = True
            random.shuffle(removed)
            
            for customer in removed:
                best_pos, _ = route.find_best_insertion(customer)
                route.insert(best_pos, customer)
        
        return new_solution if modified else None


class LH4_SwapRoute(IntraRouteHeuristic):
    """LH4 — Swap two nodes within the same route."""
    
    name = "LH4_SwapRoute"
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        
        best_improvement = 0.0
        best_swap = None
        
        # Optimized candidate iteration
        if candidates is not None:
            customer_map = new_solution.get_customer_map()
            candidate_list = [c for c in candidates if c in customer_map]
            
            # Iterate each candidate
            for c1 in candidate_list:
                r1_idx, p1 = customer_map[c1]
                route = new_solution.routes[r1_idx]
                
                if len(route) < 2:
                    continue
                    
                current_obj = route.compute_objective()
                
                # Swap with every other node in SAME route
                for p2 in range(len(route)):
                    if p1 == p2:
                        continue
                    
                    # Avoid double checking pairs (only check if p1 < p2 OR if route[p2] is not in candidates)
                    # Simpler: just check all, since swap(i,j) == swap(j,i). 
                    # But if both are candidates, we check twice. Minor overhead.
                    # Strict ordering: if route[p2] in candidates, only process if p1 < p2
                    c2 = route[p2]
                    if c2 in candidates and p2 < p1:
                        continue
                        
                    # Optimized: Use evaluate_swap without mutation
                    improvement = route.evaluate_swap(p1, p2)
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (r1_idx, p1, p2)
                    
        else:
            # Baseline (no candidates) - naive O(N^2) per route
            for route_idx in range(new_solution.n_routes):
                route = new_solution.routes[route_idx]
                if len(route) < 2:
                    continue
                current_obj = route.compute_objective()
                for i in range(len(route)):
                    for j in range(i + 1, len(route)):
                        # Optimized: Use evaluate_swap without mutation
                        improvement = route.evaluate_swap(i, j)
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (route_idx, i, j)
        
        if best_swap is not None:
            route_idx, i, j = best_swap
            new_solution.routes[route_idx].swap(i, j)
            return new_solution
        
        return None


class LH5_TwoOptRoute(IntraRouteHeuristic):
    """LH5 — 2-Opt move: reverse a segment within a route."""
    
    name = "LH5_TwoOptRoute"
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        
        best_improvement = 0.0
        best_move = None
        
        # Optimized candidate iteration
        if candidates is not None:
            customer_map = new_solution.get_customer_map()
            candidate_list = [c for c in candidates if c in customer_map]
            
            for c in candidate_list:
                r_idx, pos = customer_map[c]
                route = new_solution.routes[r_idx]
                
                if len(route) < 3:
                    continue
                    
                current_obj = route.compute_objective()
                
                # Check 2-opt segments starting or ending at 'pos'
                # 2-opt involves (i, j). We can treat 'pos' as 'i' or 'j'.
                
                # Case 1: pos is start of segment (i)
                i = pos
                for j in range(i + 2, len(route)):
                    # Strict strategy: valid if I OR J is in candidates.
                    # Since we iterate all candidates, checking (pos=i, j=any) covers it.
                    
                    # Optimized: Use evaluate_reverse_segment without mutation
                    improvement = route.evaluate_reverse_segment(i, j)
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (r_idx, i, j)
                
                # Case 2: pos is end of segment (j)
                j = pos
                for i in range(j - 2, -1, -1):
                    if route[i] in candidates:
                        continue
                        
                    # Optimized: Use evaluate_reverse_segment without mutation
                    improvement = route.evaluate_reverse_segment(i, j)
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (r_idx, i, j)
                    
        else:
            # Baseline (no candidates)
            for route_idx in range(new_solution.n_routes):
                route = new_solution.routes[route_idx]
                if len(route) < 3:
                    continue
                current_obj = route.compute_objective()
                for i in range(len(route)):
                    for j in range(i + 2, len(route)):
                        # Optimized: Use evaluate_reverse_segment without mutation
                        improvement = route.evaluate_reverse_segment(i, j)
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = (route_idx, i, j)
        
        if best_move is not None:
            route_idx, i, j = best_move
            new_solution.routes[route_idx].reverse_segment(i, j)
            return new_solution
        
        return None

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# HEURISTICS MODULE: Inter-Route Heuristics (LH6-LH8)
# ==============================================================================

class LH6_RemlIroute(InterRouteHeuristic):
    """LH6 — Remove last rl nodes from each route, reinsert globally."""
    
    name = "LH6_RemlIroute"
    
    def __init__(self, rl: int = 2):
        self.rl = rl
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        
        removed = []
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            
            actual_rl = min(self.rl, len(route))
            
            for _ in range(actual_rl):
                if len(route) > 0:
                    customer = route.remove_at(len(route) - 1)
                    if candidates is None or customer in candidates:
                        removed.append(customer)
                    else:
                        route.insert(len(route), customer)
        
        if not removed:
            return None
        
        random.shuffle(removed)
        
        for customer in removed:
            best_route, best_pos, _ = new_solution.find_best_insertion_global(customer)
            new_solution.routes[best_route].insert(best_pos, customer)
        
        return new_solution


class LH8_SwapIroute(InterRouteHeuristic):
    """LH8 — Swap nodes between different routes."""
    
    name = "LH8_SwapIroute"
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        if solution.n_routes < 2:
            return None
        
        new_solution = solution.copy()
        # Precompute initial objectives of all routes to avoid recomputing solution-level obj
        route_objs = [r.compute_objective() for r in new_solution.routes]
        
        best_improvement = 0.0
        best_swap = None
        
        # Optimized candidate iteration
        if candidates is not None:
            customer_map = new_solution.get_customer_map()
            candidate_list = [c for c in candidates if c in customer_map]
            
            for c1 in candidate_list:
                r1_idx, p1 = customer_map[c1]
                route1 = new_solution.routes[r1_idx]
                obj1_old = route_objs[r1_idx]
                
                # Check against ALL other customers in OTHER routes
                for r2_idx in range(new_solution.n_routes):
                    if r1_idx == r2_idx:
                        continue
                        
                    route2 = new_solution.routes[r2_idx]
                    obj2_old = route_objs[r2_idx]
                    
                    if len(route2) == 0:
                        continue
                        
                    for p2 in range(len(route2)):
                        c2 = route2[p2]
                        
                        # Tie breaking: if c2 is also candidate, allow only one order?
                        # Since r1 != r2, (r1, p1) vs (r2, p2) is distinct from (r2, p2) vs (r1, p1).
                        # But swap is symmetric.
                        # Enforce r1 < r2 if both in candidates? No, complicated.
                        # Simple check: if c2 in candidates, only process if r1_idx < r2_idx
                        if c2 in candidates and r1_idx > r2_idx:
                            continue
                            
                        # Apply swap
                        route1.customers[p1] = c2
                        route2.customers[p2] = c1
                        route1._invalidate_cache()
                        route2._invalidate_cache()
                        
                        # Improved delta evaluation (O(L) instead of O(N))
                        new_obj1 = route1.compute_objective()
                        new_obj2 = route2.compute_objective()
                        
                        delta = (new_obj1 + new_obj2) - (obj1_old + obj2_old)
                        
                        if delta > best_improvement:
                            best_improvement = delta
                            best_swap = (r1_idx, p1, r2_idx, p2)
                            
                        # Revert
                        route1.customers[p1] = c1
                        route2.customers[p2] = c2
                        route1._invalidate_cache()
                        route2._invalidate_cache()
                        
        else:
            # Baseline O(N^2)
            for r1 in range(new_solution.n_routes):
                route1 = new_solution.routes[r1]
                if len(route1) == 0: continue
                for r2 in range(r1 + 1, new_solution.n_routes):
                    route2 = new_solution.routes[r2]
                    if len(route2) == 0: continue
                    
                    obj1_old = list(route1.customers) # dummy
                    obj1_val = route1.compute_objective()
                    obj2_val = route2.compute_objective()
                    
                    for p1 in range(len(route1)):
                        for p2 in range(len(route2)):
                            c1 = route1[p1]
                            c2 = route2[p2]
                            
                            route1.customers[p1] = c2
                            route2.customers[p2] = c1
                            route1._invalidate_cache()
                            route2._invalidate_cache()
                            
                            delta = (route1.compute_objective() + route2.compute_objective()) - (obj1_val + obj2_val)
                            
                            if delta > best_improvement:
                                best_improvement = delta
                                best_swap = (r1, p1, r2, p2)
                            
                            route1.customers[p1] = c1
                            route2.customers[p2] = c2
                            route1._invalidate_cache()
                            route2._invalidate_cache()
                            
        if best_swap is not None:
            r1, p1, r2, p2 = best_swap
            c1 = new_solution.routes[r1].customers[p1]
            c2 = new_solution.routes[r2].customers[p2]
            new_solution.routes[r1].customers[p1] = c2
            new_solution.routes[r2].customers[p2] = c1
            new_solution.routes[r1]._invalidate_cache()
            new_solution.routes[r2]._invalidate_cache()
            return new_solution
        
        return None


class LH7_RemcrIroute(InterRouteHeuristic):
    """LH7 — Remove crn consecutive nodes, reinsert globally at best positions."""
    
    name = "LH7_RemcrIroute"
    
    def __init__(self, crn: int = 3):
        self.crn = crn
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        removed = []
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            if len(route) < 1:
                continue
            
            actual_crn = min(self.crn, len(route))
            max_start = len(route) - actual_crn
            if max_start < 0:
                continue
            
            start_pos = random.randint(0, max_start)
            
            for _ in range(actual_crn):
                if start_pos < len(route):
                    customer = route.remove_at(start_pos)
                    if candidates is None or customer in candidates:
                        removed.append(customer)
                    else:
                        route.insert(start_pos, customer)
        
        if not removed:
            return None
        
        random.shuffle(removed)
        
        for customer in removed:
            best_route, best_pos, _ = new_solution.find_best_insertion_global(customer)
            new_solution.routes[best_route].insert(best_pos, customer)
        
        return new_solution


class LH9_RemprIroute(InterRouteHeuristic):
    """LH9 — Remove by probability, reinsert globally at best positions."""
    
    name = "LH9_RemprIroute"
    
    def __init__(
        self, 
        max_prob: float = 0.5, 
        min_prob: float = 0.1,
        max_iter_for_decay: int = 500
    ):
        self.max_prob = max_prob
        self.min_prob = min_prob
        self.max_iter_for_decay = max_iter_for_decay
    
    def _compute_probability(self, iteration: int) -> float:
        if iteration >= self.max_iter_for_decay:
            return self.min_prob
        slope = (self.max_prob - self.min_prob) / self.max_iter_for_decay
        return slope * (self.max_iter_for_decay - iteration) + self.min_prob
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        prrd = self._compute_probability(iteration)
        removed = []
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            positions_to_remove = []
            
            for pos in range(len(route)):
                customer = route[pos]
                if candidates is not None and customer not in candidates:
                    continue
                if random.random() < prrd:
                    positions_to_remove.append(pos)
            
            for pos in reversed(positions_to_remove):
                customer = route.remove_at(pos)
                removed.append(customer)
        
        if not removed:
            return None
        
        random.shuffle(removed)
        
        for customer in removed:
            best_route, best_pos, _ = new_solution.find_best_insertion_global(customer)
            new_solution.routes[best_route].insert(best_pos, customer)
        
        return new_solution


class LH10_RempaddIroute(InterRouteHeuristic):
    """LH10 — Remove by probability + add unvisited, reinsert only if improves."""
    
    name = "LH10_RempaddIroute"
    
    def __init__(
        self, 
        max_prob: float = 0.5, 
        min_prob: float = 0.1,
        max_iter_for_decay: int = 500
    ):
        self.max_prob = max_prob
        self.min_prob = min_prob
        self.max_iter_for_decay = max_iter_for_decay
    
    def _compute_probability(self, iteration: int) -> float:
        if iteration >= self.max_iter_for_decay:
            return self.min_prob
        slope = (self.max_prob - self.min_prob) / self.max_iter_for_decay
        return slope * (self.max_iter_for_decay - iteration) + self.min_prob
    
    def apply(
        self, 
        solution: Solution, 
        candidates: Optional[Set[int]] = None,
        iteration: int = 0,
        max_iterations: int = 1000,
        **kwargs
    ) -> Optional[Solution]:
        new_solution = solution.copy()
        prrd = self._compute_probability(iteration)
        removed = []
        
        for route_idx in range(new_solution.n_routes):
            route = new_solution.routes[route_idx]
            positions_to_remove = []
            
            for pos in range(len(route)):
                customer = route[pos]
                if candidates is not None and customer not in candidates:
                    continue
                if random.random() < prrd:
                    positions_to_remove.append(pos)
            
            for pos in reversed(positions_to_remove):
                customer = route.remove_at(pos)
                removed.append(customer)
        
        # Add unvisited nodes to pool
        unvisited = new_solution.get_unvisited_set()
        if candidates is not None:
            unvisited = unvisited & candidates
        
        pool = list(set(removed) | unvisited)
        if not pool:
            return None
        
        random.shuffle(pool)
        modified = len(removed) > 0
        
        for customer in pool:
            if new_solution.is_customer_visited(customer):
                continue
            
            best_route, best_pos, delta = new_solution.find_best_insertion_global(customer)
            if delta > 0:  # Only insert if improves
                new_solution.routes[best_route].insert(best_pos, customer)
                modified = True
        
        return new_solution if modified else None

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# GRAPH BUILDER MODULE
# ==============================================================================

def build_solution_graph(solution: Solution) -> Data:
    """Build a PyTorch Geometric Data object using precomputed static data."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric required for graph building")
    
    instance = solution.instance
    n = instance.n_customers + 1
    
    # Use precomputed static features if available
    if not hasattr(instance, 'static_edge_index') or instance.static_edge_index is None:
        instance.precompute_graph_data()
        
    # Dynamic (8): visited, route_id, slack, position, nearest-unvisited distance,
    # neighbourhood density, time pressure (earliest arrival / deadline),
    # repairman-tip proximity. Deadline urgency is used only in D&C clustering.
    dynamic_features = np.zeros((n, 8), dtype=np.float32)
    
    visited_set = solution.get_visited_set()
    customer_map = solution.get_customer_map()
    arrival_times = {}
    route_ends = []
    
    for r_idx, route in enumerate(solution.routes):
        times = route.compute_arrival_times()
        if route.customers:
            route_ends.append(route.customers[-1])
            for c, t in zip(route.customers, times):
                arrival_times[c] = t
        else:
            route_ends.append(0) # depot is the tip
            
    profit_max = np.max(instance.profits) + 1e-6
    max_routes = max(solution.n_routes, 1)
    max_dist = np.max(instance.travel_time) + 1e-6

    # Defaults for all nodes
    dynamic_features[1:, 1] = -1.0 # Unknown RouteID
    dynamic_features[1:, 3] = -1.0 # Unknown Position
    
    # Fill Visited/Route/Slack/Position
    for c in range(n):
        if c in visited_set or c == 0:
            dynamic_features[c, 0] = 1.0 # Visited
            if c != 0:
                r_idx, pos = customer_map[c]
                dynamic_features[c, 1] = r_idx / max_routes
                dynamic_features[c, 3] = pos / max(len(solution.routes[r_idx]), 1)
                slack = max(0.0, instance.profits[c] - arrival_times.get(c, 0.0))
                dynamic_features[c, 2] = slack / profit_max
            else:
                dynamic_features[c, 2] = 1.0 # depot slack
        else:
            # Unvisited Slack based on best possible arrival
            min_arr = min(t + instance.get_travel_time(last, c) for last, t in 
                         [(re, arrival_times.get(re, 0.0)) for re in route_ends])
            dynamic_features[c, 2] = max(0.0, instance.profits[c] - min_arr) / profit_max
            dynamic_features[c, 6] = min(min_arr / (instance.profits[c] + 1e-6), 1.0)

    # Spatial context: Nearest Unvisited and Density
    unvisited = [c for c in range(1, n) if c not in visited_set]
    if unvisited:
        dists_to_unvisited = instance.travel_time[:, unvisited]
        dynamic_features[:, 4] = np.min(dists_to_unvisited, axis=1) / max_dist
        dynamic_features[:, 5] = np.sum(dists_to_unvisited < max_dist*0.2, axis=1) / max(len(unvisited), 1)

    for c in range(1, n):
        d_to_rep = min(instance.get_travel_time(re, c) for re in route_ends)
        dynamic_features[c, 7] = d_to_rep / max_dist

    # --- 2. Edge Features ---
    sources = instance.static_edge_index[0]
    targets = instance.static_edge_index[1]
    
    current_arrivals = instance.travel_time[0, :].copy() 
    for c, t in arrival_times.items():
        current_arrivals[c] = t
        
    src_arrivals = torch.from_numpy(current_arrivals[sources.numpy()]).float()
    travel_time_vals = torch.from_numpy(instance.travel_time[sources.numpy(), targets.numpy()]).float()
    potential_arrivals = src_arrivals + travel_time_vals
    target_deadlines = torch.from_numpy(instance.profits[targets.numpy()]).float()
    
    # MTRPP Feasibility: No return to depot check needed
    feasible_mask = (potential_arrivals < target_deadlines).float().unsqueeze(1)
    edge_attr = torch.cat([instance.static_edge_attr_base, feasible_mask], dim=1)
    
    # Static (5): radial, angle, density, cluster value, profit
    # Dynamic (8): visited, route_id, slack, position, unvisited dist,
    # neighbourhood density, time pressure, repairman proximity
    x = torch.zeros((n, 13), dtype=torch.float32)
    x[:, :5] = torch.from_numpy(instance.static_node_features)
    x[:, 5:13] = torch.from_numpy(dynamic_features)
    
    # --- Final SOTA Checks ---
    # Ensure depot is fully marked
    x[0, 5] = 1.0 # Visited
    x[0, 6] = 0.0 # RouteID 0
    x[0, 8] = 0.0 # Position 0
    
    # Node Feasibility Mask
    node_feasible = torch.ones(n, dtype=torch.float32)
    node_feasible[0] = 0.0
    for cust_idx in range(1, n):
        if instance.profits[cust_idx] <= 0:
            node_feasible[cust_idx] = 0.0
            continue
        can_reach = any(t + instance.travel_time[re, cust_idx] < instance.profits[cust_idx] 
                       for re in route_ends for last, t in [(re, arrival_times.get(re, 0.0))])
        if not can_reach:
            node_feasible[cust_idx] = 0.0

    data = Data(x=x, edge_index=instance.static_edge_index, edge_attr=edge_attr)
    data.node_feasible = node_feasible
    data.n_customers = instance.n_customers
    data.current_objective = solution.compute_objective()
    data.visited_count = solution.get_visited_count()
    
    return data


def get_node_feature_dim() -> int:
    return 13


def get_edge_feature_dim() -> int:
    return 2

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# GNN MODEL MODULE
# ==============================================================================

class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, c_g: torch.Tensor) -> torch.Tensor:
        return self.net(c_g).squeeze(-1)


class LargeCustomerRankingGNN(nn.Module):
    """
    Larger GNN model for RL-based training.
    Optimized for 4GB GPU with reduced dimensions.
    """
    
    def __init__(
        self,
        node_input_dim: int = 13,
        edge_input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 2,
        dropout: float = 0.1,
        use_global_context: bool = True
    ):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_global_context = use_global_context
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.mp_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.mp_layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=hidden_dim, # Matches edge_encoder output
                    concat=True
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        readout_in = hidden_dim * 2 if use_global_context else hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden_dim), # *2 to accommodate global pool
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.value_head = ValueHead(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: Data, return_logits: bool = False, return_value: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = data.x
        # Backward compatibility: slice features if model expects fewer than provided
        if x.size(-1) > self.node_input_dim:
            x = x[:, :self.node_input_dim]
            
        edge_index = data.edge_index
        
        h = self.node_encoder(x)
        e = self.edge_encoder(data.edge_attr) if hasattr(data, 'edge_attr') else None
        
        from torch.utils.checkpoint import checkpoint
        
        for mp_layer, layer_norm in zip(self.mp_layers, self.layer_norms):
            if self.training:
                # Use checkpointing to save memory
                h_new = checkpoint(mp_layer, h, edge_index, e, use_reentrant=False)
            else:
                h_new = mp_layer(h, edge_index, edge_attr=e)
            h = layer_norm(h + self.dropout(h_new))
            
        # Global context vector
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        c_g = global_mean_pool(h, batch)
        
        # Value prediction (for Critic)
        value = self.value_head(c_g)
        
        # Policy prediction (Actor)
        if self.use_global_context:
            c_g_expanded = c_g[batch]
            h_concat = torch.cat([h, c_g_expanded], dim=-1)
        else:
            h_concat = h
            
        scores = self.readout(h_concat).squeeze(-1)
        
        output = scores if return_logits else torch.sigmoid(scores)
        if return_value:
            return output, value
        return output
    
    def get_action_probs(self, data: Data, temperature: float = 1.0) -> torch.Tensor:
        logits = self(data, return_logits=True)
        
        # Apply hard feasibility mask
        if hasattr(data, 'node_feasible'):
            infeasible_mask = (data.node_feasible < 0.5)
            # Handle possible batching of the mask
            if hasattr(data, 'batch') and data.batch is not None:
                # node_feasible is [BatchTotalNodes]
                logits[infeasible_mask] = float('-inf')
            else:
                logits[infeasible_mask] = float('-inf')
        
        # Handle batching
        if hasattr(data, 'batch') and data.batch is not None:
             # Mask depots (first node of each graph)
             if hasattr(data, 'ptr'):
                 logits[data.ptr[:-1]] = float('-inf')
             
             return pyg_softmax(logits / temperature, data.batch)
        else:
            logits[0] = float('-inf')  # Mask depot
            return F.softmax(logits / temperature, dim=0)
    
    def sample_actions(
        self,
        data: Data,
        k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.get_action_probs(data, temperature)
        
        # Handle batching
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.num_graphs
            actions_list = []
            log_probs_list = []
            
            ptr = data.ptr
            for i in range(batch_size):
                start, end = ptr[i], ptr[i+1]
                p = probs[start:end]
                
                n_cust = p.size(0) - 1
                k_i = min(k, n_cust)
                
                if k_i <= 0:
                     actions_list.append(torch.tensor([], device=p.device, dtype=torch.long))
                     log_probs_list.append(torch.tensor([], device=p.device))
                     continue

                gumbel = -torch.log(-torch.log(torch.rand_like(p) + 1e-10) + 1e-10)
                perturbed = torch.log(p + 1e-10) + gumbel
                perturbed[0] = float('-inf') 
                
                _, acts = torch.topk(perturbed, k_i)
                lps = torch.log(p[acts] + 1e-10)
                
                actions_list.append(acts)
                log_probs_list.append(lps)
                
            return actions_list, log_probs_list
            
        else:
            n_customers = probs.size(0) - 1
            k = min(k, n_customers)
            
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
            perturbed = torch.log(probs + 1e-10) + gumbel_noise
            perturbed[0] = float('-inf')
            
            _, actions = torch.topk(perturbed, k)
            log_probs = torch.log(probs[actions] + 1e-10)
            
            return actions, log_probs


def create_large_model(
    node_dim: int = None,
    edge_dim: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 4,           # Reduced for stability
    heads: int = 4,                # Reduced for stability
    use_global_context: bool = True,
    **kwargs
) -> LargeCustomerRankingGNN:
    """Factory function to create a large GNN model for RL training."""
    if node_dim is None:
        node_dim = get_node_feature_dim()
    return LargeCustomerRankingGNN(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
        use_global_context=use_global_context,
        **kwargs
    )

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# RL TRAINING MODULE
# ==============================================================================

@dataclass
class RLConfig:
    """Configuration for RL training."""
    
    max_steps_per_episode: int = 25  # Harmonized with golden reference
    k_actions: int = 5
    batch_size: int = 4  # Reduced batch size for local buffer
    
    learning_rate: float = 1e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01   # Base entropy booster
    entropy_coef_max: float = 0.05   # Ceiling for adaptive boost
    entropy_floor: float = 0.3      # Target minimum entropy (ratio of max)
    baseline_ema: float = 0.08
    
    episodes_per_instance: int = 1
    warmup_episodes: int = 50
    temperature: float = 2.0
    sampling_temperature: float = 2.0
    temperature_decay: float = 0.999
    min_temperature: float = 0.5
    
    reward_clip: float = 10.0          # Clip normalized reward to [-clip, clip]
    coverage_bonus: float = 2.0         # Bonus per node visited
    distance_penalty_weight: float = 0.1 # Penalty for inefficient jumps
    checkpoint_every: int = 50
    grad_accum_steps: int = 1
         # Checkpoint every N instances
    
    max_grad_norm: float = 1.0
    log_interval: int = 10


# ==============================================================================
# RL TRAINING UTILITIES
# ==============================================================================

def compute_td_lambda_returns(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute TD(lambda) returns for Actor-Critic training.
    Interpolates between 1-step TD and Monte Carlo returns to balance bias/variance.
    """
    T = len(rewards)
    returns = [0.0] * T
    gae = 0.0  # Generalised Advantage Estimate
    
    # values is expected to have T + 1 elements [V_0, ..., V_T]
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        returns[t] = gae + values[t]
    
    return returns


class RLTrainer:
    """Advantage actor-critic trainer for the Large GNN model."""
    
    def __init__(
        self,
        model: nn.Module,
        config: RLConfig = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config or RLConfig()
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Cosine annealing LR scheduler (None until train_on_instances sets it)
        self.scheduler = None
        
        self.heuristics = [
            # Intra-route heuristics (LH1-LH5)
            LH1_RemcrRoute(crn=3),
            LH2_RempRoute(),
            LH3_RemlRoute(rl=2),
            LH4_SwapRoute(),
            LH5_TwoOptRoute(),
            # Inter-route heuristics (LH6-LH10)
            LH6_RemlIroute(rl=2),
            LH7_RemcrIroute(crn=2),
            LH8_SwapIroute(),
            LH9_RemprIroute(),
            LH10_RempaddIroute(),
            # Greedy heuristics (LH11-LH12)
            LH11_Remove(),
            LH12_Add(),
            LH13_DestroyRepair()
        ]
        
        # Per-instance-size baselines (key: size_bin -> float)
        self.baselines = {}  # binned by n_customers // 50 * 50
        self.episode_rewards = []
        self.episode_lengths = []
        self.temperature = self.config.temperature
        
        # Mixed-precision (FP16) support for VRAM conservation
        self.use_amp = (device == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp) if self.use_amp else None
        
        self.episodes_run = 0
    
    def _get_size_bin(self, instance: Instance) -> int:
        """Get instance size bin for per-size baseline."""
        return (instance.n_customers // 50) * 50
    
    def _get_baseline(self, size_bin: int) -> float:
        """Get baseline for a given size bin."""
        return self.baselines.get(size_bin, 0.0)
    
    def _update_baseline(self, size_bin: int, reward: float):
        """Update baseline for a given size bin with EMA."""
        alpha = self.config.baseline_ema
        old = self.baselines.get(size_bin, reward)
        self.baselines[size_bin] = (1 - alpha) * old + alpha * reward
    
    def run_episode(
        self,
        instance: Instance,
        initial_solution: Optional[Solution] = None
    ) -> Tuple[float, List[float], float]:
        """Run one episode of search with A2C (Advantage Actor-Critic) and TD(lambda)."""
        if initial_solution is None:
            constructor = ConstructiveHeuristic(instance)
            solution = constructor.construct()
        else:
            solution = initial_solution.copy()
        
        episode_log_probs = []
        episode_values = []
        episode_rewards = []
        episode_entropies = []
        
        clip_val = self.config.reward_clip
        total_profit = max(sum(instance.profits[1:]), 1.0)
        paper_best = getattr(instance, 'final_profit', None)
        max_dist = np.max(instance.travel_time) + 1e-6
        
        for step in range(self.config.max_steps_per_episode):
            old_obj = solution.compute_objective()
            
            graph = build_solution_graph(solution).to(self.device)
            
            # Policy & Value forward pass
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # get_action_probs calls forward(return_logits=True)
                # But we need both logits and value. Let's use forward directly.
                logits, value = self.model(graph, return_logits=True, return_value=True)
            
            # Policy distribution (mask depot)
            logits[0] = float('-inf')
            probs = F.softmax(logits / self.temperature, dim=0)
            
            # Sample actions (K is dynamic based on entropy)
            n_customers = instance.n_customers
            probs_cust = probs[1:]
            entropy_k = -(probs_cust * torch.log(probs_cust + 1e-10)).sum().item()
            max_entropy = math.log(max(n_customers, 2))
            norm_entropy = entropy_k / max(max_entropy, 1e-6)
            k_min = max(3, n_customers // 20)
            k_max = min(50, n_customers // 5)
            adaptive_k = max(k_min, min(k_max, int(k_min + norm_entropy * (k_max - k_min))))
            
            actions, log_probs = self.model.sample_actions(
                graph, 
                k=adaptive_k,
                temperature=self.temperature
            )
            
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            
            # Apply heuristics
            candidates = set(actions.detach().cpu().tolist())
            new_solution, improved, h_name = self._apply_heuristics(solution, candidates, step)
            
            new_obj = new_solution.compute_objective()
            raw_reward = new_obj - old_obj
            
            # --- MTRPP Reward Calculation ---
            # Balance Profit Gain with Node Coverage Bonus
            # improved means LH12_Add or similar increased the visited set
            coverage_bonus = self.config.coverage_bonus if (improved or new_obj > old_obj) else 0.0
            
            # Distance Penalty (to discourage inefficient jumps)
            old_dist = sum(r.compute_travel_time() for r in solution.routes)
            new_dist = sum(r.compute_travel_time() for r in new_solution.routes)
            dist_penalty = ((new_dist - old_dist) / max_dist) * self.config.distance_penalty_weight if max_dist > 0 else 0
            
            reward = (raw_reward / total_profit) * 100.0 + coverage_bonus - dist_penalty
            
            # Exponential Reward Shaping for non-improving steps (exploration penalty)
            if raw_reward <= 0 and not improved:
                reward = (math.exp(raw_reward / (self.temperature * total_profit + 1e-6)) - 1.0) * 5.0
            
            # --- Breakthrough Bonus: World-Record Gap Closure ---
            if paper_best and paper_best > 0:
                gap_before = (paper_best - old_obj) / paper_best
                gap_after = (paper_best - new_obj) / paper_best
                gap_closure_bonus = math.exp(-10 * gap_after) - math.exp(-10 * gap_before)
                reward += gap_closure_bonus * 5.0
            
            reward = max(min(reward, clip_val), -clip_val)
            
            # Store transition
            episode_rewards.append(reward)
            episode_values.append(value)
            episode_log_probs.append(log_probs.sum())
            episode_entropies.append(entropy)
            
            if raw_reward > 0 or random.random() < 0.1:
                solution = new_solution
            
            del graph
        
        # --- A2C Backpropagation ---
        # Add terminal value estimate (0.0)
        episode_values.append(torch.tensor([0.0], device=self.device))
        
        # Compute TD(lambda) returns
        rewards_list = [r for r in episode_rewards]
        values_list = [v.item() for v in episode_values]
        td_returns = compute_td_lambda_returns(rewards_list, values_list)
        
        returns_tensor = torch.tensor(td_returns, device=self.device, dtype=torch.float32)
        values_tensor = torch.stack(episode_values[:-1]).squeeze()
        log_probs_tensor = torch.stack(episode_log_probs)
        entropies_tensor = torch.stack(episode_entropies)
        
        # Advantage: A(s,a) = G_t - V(s)
        advantages = returns_tensor - values_tensor.detach()
        
        # --- Fix 1: Dual-stream normalized advantage ---
        advantages = returns_tensor - values_tensor.detach()
        adv_std = advantages.std().clamp(min=1e-6)
        advantages_norm = ((advantages - advantages.mean()) / adv_std).clamp(-3.0, 3.0)
        
        # --- Fix 2: Adaptive entropy regularization ---
        n_customers = instance.n_customers
        max_entropy = math.log(max(n_customers, 2))
        current_entropy = entropies_tensor.mean().item()
        entropy_ratio = current_entropy / max(max_entropy, 1e-6)
        
        if entropy_ratio < self.config.entropy_floor:
            collapse_depth = (self.config.entropy_floor - entropy_ratio) / self.config.entropy_floor
            adaptive_coef = self.config.entropy_coef + collapse_depth * (
                self.config.entropy_coef_max - self.config.entropy_coef
            )
        else:
            adaptive_coef = self.config.entropy_coef
            
        actor_loss = -(log_probs_tensor * advantages_norm).mean()
        critic_loss = F.mse_loss(values_tensor, returns_tensor)
        entropy_loss = -adaptive_coef * entropies_tensor.mean()
        
        total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
        
        if self.scaler:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
            
        return sum(rewards_list), rewards_list, total_loss.item()

    def run_batch_episode(
        self,
        instances: List[Instance]
    ) -> Tuple[float, float, int]:
        """Run one episode for a batch of instances using A2C + TD(lambda)."""
        # Skip instances too large for GPU training — they belong in D&C eval
        MAX_TRAIN_NODES = 500 # Optimized for 4GB VRAM with GradScaler guard
        batch_instances = [inst for inst in instances if inst.n_customers <= MAX_TRAIN_NODES]
        if not batch_instances:
            return 0.0, 0.0, 0
            
        instances = batch_instances
        batch_size = len(instances)
        
        # Track total episodes for warmup logic
        self.episodes_run += batch_size
        is_warmup = self.episodes_run <= self.config.warmup_episodes
        
        # Initialize solutions with diverse starts
        solutions = []
        for inst in instances:
            best_init_sol = None
            best_init_obj = -float('inf')
            # Breakthrough Initialization: Best-of-3 randomized constructive starts
            for seed_offset in range(3):
                seed = abs(hash(inst.name)) % 10000 + seed_offset
                init_sol = ConstructiveHeuristic(inst).construct(seed=seed)
                init_obj = init_sol.compute_objective()
                if init_obj > best_init_obj:
                    best_init_obj = init_obj
                    best_init_sol = init_sol
            solutions.append(best_init_sol)
            
        # Per-instance trajectories
        batch_log_probs = [[] for _ in range(batch_size)]
        batch_values = [[] for _ in range(batch_size)]
        batch_rewards = [[] for _ in range(batch_size)]
        batch_entropies = [[] for _ in range(batch_size)]  # Track for adaptive boost
        batch_actions = [[] for _ in range(batch_size)]    # Indices for re-forward
        batch_graphs = [[] for _ in range(batch_size)]
        
        instance_improved = [False] * batch_size
        ks = [max(3, min(inst.n_customers // 10, 50)) for inst in instances]
        max_k = max(ks)
        
        clip_val = self.config.reward_clip
        n_workers = 2
        
        ctx = mp.get_context('spawn') if n_workers > 1 else None
        from concurrent.futures import ProcessPoolExecutor
        
        torch.cuda.empty_cache()  # Pre-episode cleanup
        
        with (ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) if n_workers > 1 else contextlib.nullcontext()) as executor:
            for step in range(self.config.max_steps_per_episode):
                if step % 20 == 0:
                    print(f"      [Step {step}/{self.config.max_steps_per_episode}]")
                
                # 1. Build and Batch graphs
                graphs = []
                old_objs = []
                for sol in solutions:
                    old_objs.append(sol.compute_objective())
                    graphs.append(build_solution_graph(sol))
                
                batch = Batch.from_data_list(graphs).to(self.device)
                
                # 2. Model forward pass (Actor-Critic)
                try:
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        logits, values = self.model(batch, return_logits=True, return_value=True)
                        if hasattr(batch, 'ptr'):
                            logits[batch.ptr[:-1]] = float('-inf')
                        
                        # Calculate entropy for batch
                        probs = pyg_softmax(logits / self.temperature, batch.batch)
                        entropy = -(probs * torch.log(probs + 1e-10)).sum() / batch_size
                        
                        actions_list, log_probs_list = self.model.sample_actions(
                            batch, k=max_k, temperature=self.temperature
                        )
                    del batch
                except torch.OutOfMemoryError:
                    print(f"      [Warning] CUDA OOM at step {step}, skipping batch forward...")
                    torch.cuda.empty_cache()
                    continue
                
                # 3. Apply heuristics (Parallel)
                if n_workers > 1 and executor is not None:
                    worker_args = [(self.heuristics, self.config.max_steps_per_episode, solutions[i], 
                                    set(actions_list[i].detach().cpu().tolist()), step) for i in range(batch_size)]
                    futures = [executor.submit(rl_heuristic_worker, *args) for args in worker_args]
                    new_solutions = [f.result() for f in futures]
                else:
                    new_solutions = []
                    for i in range(batch_size):
                        candidates = set(actions_list[i].detach().cpu().tolist())
                        new_sol, _, _ = self._apply_heuristics(solutions[i], candidates, step)
                        new_solutions.append(new_sol)
                
                # 4. Rewards and Trajectory Storage
                for i in range(batch_size):
                    new_obj = new_solutions[i].compute_objective()
                    raw_reward = new_obj - old_objs[i]
                    
                    total_profit_inst = max(sum(instances[i].profits[1:]), 1.0)
                    max_dist_inst = np.max(instances[i].travel_time) + 1e-6
                    
                    # --- MTRPP Multi-Objective Reward ---
                    # Check if coverage increased
                    old_visited = solutions[i].get_visited_count()
                    new_visited = new_solutions[i].get_visited_count()
                    
                    # Normalize bonuses by instance scale for gradient stability
                    n_cust_inst = max(instances[i].n_customers, 1)
                    coverage_bonus = (self.config.coverage_bonus / n_cust_inst) if new_visited > old_visited else 0.0
                    
                    # Distance Efficiency (Normalized)
                    old_dist = sum(r.compute_travel_time() for r in solutions[i].routes)
                    new_dist = sum(r.compute_travel_time() for r in new_solutions[i].routes)
                    dist_penalty = ((new_dist - old_dist) / max_dist_inst) * self.config.distance_penalty_weight
                    
                    reward = (raw_reward / total_profit_inst) * 100.0 + (coverage_bonus * 100.0) - (dist_penalty * 10.0)
                    
                    # Exploration penalty for non-improving steps
                    if raw_reward <= 0 and new_visited <= old_visited:
                        reward = (math.exp(raw_reward / (self.temperature * total_profit_inst + 1e-6)) - 1.0) * 5.0
                    
                    # --- SOTA Gap Bonus ---
                    paper_best = getattr(instances[i], 'final_profit', None)
                    if paper_best and paper_best > 0:
                        gap_before = (paper_best - old_objs[i]) / paper_best
                        gap_after = (paper_best - new_obj) / paper_best
                        reward += (math.exp(-10 * gap_after) - math.exp(-10 * gap_before)) * 5.0
                        
                    reward = max(min(reward, clip_val), -clip_val)
                    
                    # Improvement defined by MTRPP Goal (Profit OR Coverage)
                    if raw_reward > 0 or new_visited > old_visited:
                        instance_improved[i] = True
                        
                    # Store transition data (Memory-Safe detachment)
                    batch_rewards[i].append(reward)
                    batch_values[i].append(values[i].detach())
                    # Store action indices for gradient re-forwarding
                    batch_actions[i].append(actions_list[i].detach()) 
                    batch_entropies[i].append(entropy.detach()) 
                    batch_graphs[i].append(graphs[i])
                    
                    if raw_reward > 0 or random.random() < 0.1:
                        solutions[i] = new_solutions[i]

        # 5. Memory-Safe Consolidated Update
        import gc
        gc.collect()
        torch.cuda.empty_cache() 
        batch_actor_losses = []
        batch_critic_losses = []
        
        for i in range(batch_size):
            batch_values[i].append(torch.tensor([0.0], device=self.device))
            td_returns = compute_td_lambda_returns(batch_rewards[i], [v.item() for v in batch_values[i]])
            returns_tensor = torch.tensor(td_returns, device=self.device, dtype=torch.float32)
            
            # Re-forward to get gradients without keeping loop graphs
            step_graphs = batch_graphs[i]
            step_actions = batch_actions[i]
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                fresh_values = []
                fresh_log_probs = []
                fresh_entropies = []
                
                # Process strictly one-by-one for maximum VRAM safety
                for j in range(len(step_graphs)):
                    sub_data = Batch.from_data_list([step_graphs[j]]).to(self.device)
                    # We need the full forward pass (attached)
                    logits, v = self.model(sub_data, return_logits=True, return_value=True)
                    
                    # Calculate log_prob for the actions actually taken
                    # Mask depot (0) and infeasible nodes
                    logits[0] = float('-inf')
                    if hasattr(sub_data, 'node_feasible'):
                        logits[sub_data.node_feasible < 0.5] = float('-inf')
                        
                    probs = F.softmax(logits, dim=0) # Temperature=1.0 for training targets
                    
                    # Log-probs of the K nodes sampled
                    indices = step_actions[j].to(self.device)
                    lp = torch.log(probs[indices] + 1e-10).sum()
                    
                    # Attached Entropy for regularization
                    ent = -(probs * torch.log(probs + 1e-10)).sum()
                    
                    fresh_values.append(v)
                    fresh_log_probs.append(lp)
                    fresh_entropies.append(ent)
                    
                    if j % 5 == 0: torch.cuda.empty_cache()
                
                fresh_values = torch.cat(fresh_values).squeeze()
                fresh_log_probs = torch.stack(fresh_log_probs)
                fresh_entropies = torch.stack(fresh_entropies)
                
                advantages = returns_tensor - fresh_values.detach()
                adv_std = advantages.std().clamp(min=1e-6)
                advantages_norm = ((advantages - advantages.mean()) / adv_std).clamp(-3.0, 3.0)
                
                if not is_warmup:
                    # Policy Gradient + Entropy Regularization (Attached)
                    policy_loss = -(fresh_log_probs * advantages_norm).mean()
                    
                    # Adaptive entropy coefficient logic
                    n_cust = instances[i].n_customers
                    max_ent = math.log(max(n_cust, 2))
                    cur_ent = fresh_entropies.mean().item()
                    ent_ratio = cur_ent / max(max_ent, 1e-6)
                    
                    if ent_ratio < self.config.entropy_floor:
                        depth = (self.config.entropy_floor - ent_ratio) / self.config.entropy_floor
                        adapt_coef = self.config.entropy_coef + depth * (
                            self.config.entropy_coef_max - self.config.entropy_coef
                        )
                    else:
                        adapt_coef = self.config.entropy_coef
                        
                    batch_actor_losses.append(policy_loss - adapt_coef * fresh_entropies.mean())
                else:
                    batch_actor_losses.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                
                batch_critic_losses.append(F.mse_loss(fresh_values, returns_tensor))
        
        total_loss = torch.stack(batch_actor_losses).mean() + 0.5 * torch.stack(batch_critic_losses).mean()
        
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.did_backward_accum = True
        else:
            total_loss.backward()
            self.did_backward_accum = True
            
        torch.cuda.empty_cache() # Final cleanup
        return sum([sum(r) for r in batch_rewards]) / batch_size, total_loss.item(), sum(instance_improved)

    
    def _apply_heuristics(
        self,
        solution: Solution,
        candidates: set,
        step: int
    ) -> Tuple[Solution, bool, str]:
        """Apply all heuristics to selected customers and return the best improvement."""
        best_solution = solution.copy()
        best_obj = solution.compute_objective()
        improved = False
        best_heuristic_name = ""
        
        # Shuffle heuristics to avoid bias
        shuffled_heuristics = list(self.heuristics)
        random.shuffle(shuffled_heuristics)
        
        for heuristic in shuffled_heuristics:
            try:
                new_sol = heuristic.apply(
                    solution,
                    candidates=candidates,
                    iteration=step,
                    max_iterations=self.config.max_steps_per_episode
                )
                if new_sol is not None and new_sol.is_feasible():
                    if not new_sol.validate_ordering():
                        continue
                        
                    new_obj = new_sol.compute_objective()
                    if new_obj > best_obj:
                        best_solution = new_sol
                        best_obj = new_obj
                        improved = True
                        best_heuristic_name = heuristic.name
            except Exception:
                continue
        
        return best_solution, improved, best_heuristic_name
    
    def train_on_instances(
        self,
        instances: List[Instance],
        num_epochs: int = 100,
        start_epoch: int = 0,
        start_instance_idx: int = 0,
        existing_stats: Dict = None
    ) -> Dict[str, List[float]]:
        """Train the model on a set of instances with resume support."""
        self.model.train()
        
        stats = existing_stats or {
            'epoch_rewards': [],
            'epoch_losses': [],
            'improvements': [],
            'best_improvement_rate': 0.0,
            'best_avg_reward': -float('inf'),
        }
        
        # Ensure best_avg_reward exists for backward compatibility
        if 'best_avg_reward' not in stats:
            stats['best_avg_reward'] = -float('inf')
        
        checkpoint_dir = CONFIG.get('output_dir', 'checkpoints_rl')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_every = self.config.checkpoint_every
        
        # Cosine annealing LR schedule across all remaining epochs
        remaining_epochs = num_epochs - start_epoch
        if remaining_epochs > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_epochs,
                eta_min=1e-5
            )
            self.optimizer.zero_grad() # Ensure clean start
            # Fast-forward scheduler if resuming
            for _ in range(start_epoch - (num_epochs - remaining_epochs)):
                self.scheduler.step()
        
        best_imp_rate = stats.get('best_improvement_rate', 0.0)
        
        for epoch in range(start_epoch, num_epochs):
            epoch_reward = 0
            epoch_loss = 0
            n_episodes = 0
            total_improvements = 0
            best_avg_reward = stats.get('best_avg_reward', -float('inf'))
            
            random.shuffle(instances)
            
            # Chunk instances into batches
            batch_size = self.config.batch_size
            batches = [instances[i:i + batch_size] for i in range(0, len(instances), batch_size)]
            
            pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}")
            current_idx = 0
            
            for batch_instances in pbar:
                if epoch == start_epoch and current_idx < start_instance_idx:
                    current_idx += len(batch_instances)
                    continue
                
                # Handling episodes_per_instance
                for _ in range(self.config.episodes_per_instance):
                    batch_total_reward, batch_avg_loss, batch_improvements = self.run_batch_episode(batch_instances)
                    
                    current_batch_size = len(batch_instances)
                    epoch_reward += batch_total_reward
                    epoch_loss += batch_avg_loss 
                    
                    n_episodes += current_batch_size
                    total_improvements += batch_improvements
                
                # 5. Gradient Accumulation Step
                if (current_idx % self.config.grad_accum_steps == 0) or (current_idx >= len(instances)):
                    if hasattr(self, 'scaler') and self.scaler:
                        # Grad clipping for stability
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        
                        # Only step if gradients were actually produced in this cycle
                        if getattr(self, 'did_backward_accum', False):
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.did_backward_accum = False
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()
                        
                    self.optimizer.zero_grad()
                
                current_idx += len(batch_instances)
                
                # Decay temperature PER BATCH
                self.temperature = max(
                    self.config.min_temperature,
                    self.temperature * self.config.temperature_decay
                )
                
                # Dynamic Distance Penalty (Soft start)
                # Epochs 1-5: 0.1, Epochs 6-10: 0.3, Epochs 11+: 0.5
                if epoch < 5:
                    self.config.distance_penalty_weight = 0.1
                elif epoch < 10:
                    self.config.distance_penalty_weight = 0.3
                else:
                    self.config.distance_penalty_weight = 0.5
                
                pbar.set_postfix({
                    'reward': f'{epoch_reward/max(n_episodes,1):.2f}',
                    'loss': f'{epoch_loss/max(len(batches) * self.config.episodes_per_instance, 1):.4f}',
                    'imp_rate': f'{total_improvements/max(n_episodes,1)*100:.1f}%',
                    'temp': f'{self.temperature:.3f}'
                })
                
                # Periodic checkpoint & Mid-epoch best model check
                if checkpoint_every > 0 and (current_idx) % checkpoint_every < batch_size:
                    ckpt_path = os.path.join(checkpoint_dir, 'rl_model_periodic.pt')
                    self.save_checkpoint(ckpt_path, stats, epoch=epoch, instance_idx=current_idx)
                    
                    # Mid-epoch best check based on current running average
                    current_avg_reward = epoch_reward / max(n_episodes, 1)
                    if current_avg_reward > best_avg_reward:
                        best_avg_reward = current_avg_reward
                        stats['best_avg_reward'] = best_avg_reward
                        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
                        self.save_checkpoint(best_path, stats, epoch=epoch, instance_idx=current_idx)
                        print(f"  New best model saved mid-epoch (avg_reward={best_avg_reward:.2f})")
                
                # Cleanup GPU memory after batch
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            avg_reward = epoch_reward / max(n_episodes, 1)
            imp_rate = total_improvements / max(n_episodes, 1)
            
            stats['epoch_rewards'].append(avg_reward)
            stats['epoch_losses'].append(epoch_loss / max(len(batches), 1))
            stats['improvements'].append(imp_rate)
            
            # Step LR scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}: Avg Reward = {avg_reward:.2f}, "
                  f"Improvement Rate = {imp_rate:.2%}, "
                  f"Temp = {self.temperature:.3f}, "
                  f"LR = {current_lr:.2e}")
            
            # Track and save best model by average reward (more granular than imp_rate)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                stats['best_avg_reward'] = best_avg_reward
                best_path = os.path.join(checkpoint_dir, 'best_model.pt')
                self.save_checkpoint(best_path, stats, epoch=epoch+1, instance_idx=0)
                print(f"  New best model saved (avg_reward={avg_reward:.2f}, imp_rate={imp_rate:.2%})")
            
            # Epoch-end checkpoint
            epoch_ckpt_path = os.path.join(checkpoint_dir, f'rl_model_epoch{epoch+1}.pt')
            self.save_checkpoint(epoch_ckpt_path, stats, epoch=epoch + 1, instance_idx=0)
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return stats
    
    def save_checkpoint(self, path: str, stats: Dict = None, epoch: int = 0, instance_idx: int = 0):
        """Save model checkpoint with full state for resumption."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baselines': self.baselines,
            'temperature': self.temperature,
            'config': self.config,
            'epoch': epoch,
            'instance_idx': instance_idx,
            # Save architecture params
            'hidden_dim': CONFIG.get('hidden_dim', 128),
            'num_layers': CONFIG.get('num_layers', 6),
            'heads': CONFIG.get('num_heads', 8),
        }
        if stats:
            checkpoint['stats'] = stats
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def heuristic_worker(solver, solution, candidates, iteration):
    """Worker function for parallel heuristic application."""
    # Deterministic seeding to ensure Baseline vs GNN comparability
    # Use instance name as a stable seed offset
    import zlib
    base_seed = 42
    inst_name = getattr(solution.instance, 'name', 'unnamed')
    seed_offset = zlib.adler32(inst_name.encode())
    worker_seed = (base_seed + iteration + seed_offset) % 123456789
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    
    return solver._apply_heuristics(solution, candidates, iteration)

def rl_heuristic_worker(heuristics, config_max_steps, solution, candidates, step):
    """Worker function for parallel heuristic application during RL training."""
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    
    best_solution = solution
    best_obj = solution.compute_objective()
    
    for heuristic in heuristics:
        try:
            new_sol = heuristic.apply(
                solution,
                candidates=candidates,
                iteration=step,
                max_iterations=config_max_steps
            )
            if new_sol is not None and new_sol.is_feasible():
                new_obj = new_sol.compute_objective()
                if new_obj > best_obj:
                    best_solution = new_sol
                    best_obj = new_obj
        except Exception:
            continue
            
    return best_solution

def load_instances(dataset_paths: list, max_instances: int = None) -> list:
    """Load instances from dataset files."""
    all_instances = []
    
    for ds_path in dataset_paths:
        if not os.path.exists(ds_path):
            print(f"Warning: {ds_path} not found, skipping")
            continue
        
        print(f"Loading {ds_path}...")
        graphs = torch.load(ds_path, weights_only=False)
        
        # Optimization: Slice before expensive precomputation
        if max_instances is not None and len(all_instances) < max_instances:
            remaining = max_instances - len(all_instances)
            graphs = graphs[:remaining]
            
        for i, data in enumerate(graphs):
            if isinstance(data, Instance):
                instance = data
            else:
                name = getattr(data, 'instance_name', f'inst_{len(all_instances)}')
                instance = Instance.from_pyg_data(data, instance_name=name)
                instance._optimal_routes = getattr(data, 'routes', None)
                instance.final_profit = getattr(data, 'final_profit', None)
            
            # Ensure attributes are preserved
            if not hasattr(instance, 'final_profit'):
                instance.final_profit = getattr(data, 'final_profit', None)
            if not hasattr(instance, '_optimal_routes'):
                instance._optimal_routes = getattr(data, 'routes', None)
            
            all_instances.append(instance)
    
    print(f"Loaded {len(all_instances)} instances total")
    return all_instances


def split_instances(
    instances: List[Instance], 
    train_ratio: float = 0.8, 
    seed: int = 42
) -> Tuple[List[Instance], List[Instance]]:
    """
    Split instances into train and test sets.
    
    Args:
        instances: List of all instances
        train_ratio: Fraction of instances for training
        seed: Random seed for reproducibility
        
    Returns:
        (train_instances, test_instances)
    """
    random.seed(seed)
    shuffled = instances.copy()
    random.shuffle(shuffled)
    
    n_train = int(len(shuffled) * train_ratio)
    train_instances = shuffled[:n_train]
    test_instances = shuffled[n_train:]
    
    print(f"Split: {len(train_instances)} train, {len(test_instances)} test")
    return train_instances, test_instances

# NOTE: Old main() removed. The updated main() with train/test split,
# resume logic, and evaluation is defined below.

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# HYPER-HEURISTIC MODULE: Feedback Controller
# ==============================================================================

@dataclass
class ControllerConfig:
    """Configuration for feedback controller."""
    k_min: int = 3
    k_max: int = 10
    k_initial: int = 5
    k_increase_rate: int = 1
    k_decrease_rate: int = 1
    
    perturbation_min: float = 0.0
    perturbation_max: float = 0.3
    perturbation_initial: float = 0.05
    perturbation_increase_rate: float = 0.02
    perturbation_decrease_rate: float = 0.01
    
    low_stagnation: int = 5
    medium_stagnation: int = 15
    high_stagnation: int = 30


@dataclass
class ControllerState:
    """Current state of the feedback controller."""
    k: int = 5
    perturbation: float = 0.05
    stagnation: int = 0
    best_objective: float = 0.0


class FeedbackController:
    """Adaptive controller that adjusts search parameters based on stagnation."""
    
    def __init__(self, config: ControllerConfig = None):
        self.config = config or ControllerConfig()
        self.state = ControllerState(
            k=self.config.k_initial,
            perturbation=self.config.perturbation_initial
        )
        self.heuristic_success: Dict[str, Tuple[int, int]] = {}
    
    def update(self, improved: bool, new_objective: float = None) -> None:
        if improved:
            self.state.stagnation = 0
            if new_objective is not None:
                self.state.best_objective = max(self.state.best_objective, new_objective)
        else:
            self.state.stagnation += 1
        self._adjust_parameters()
    
    def _adjust_parameters(self) -> None:
        sigma = self.state.stagnation
        cfg = self.config
        
        if sigma < cfg.low_stagnation:
            self.state.k = max(self.state.k - cfg.k_decrease_rate, cfg.k_min)
            self.state.perturbation = max(
                self.state.perturbation - cfg.perturbation_decrease_rate,
                cfg.perturbation_min
            )
        elif sigma < cfg.medium_stagnation:
            pass  # Keep current parameters
        elif sigma < cfg.high_stagnation:
            self.state.k = min(self.state.k + cfg.k_increase_rate, cfg.k_max)
            self.state.perturbation = min(
                self.state.perturbation + cfg.perturbation_increase_rate,
                cfg.perturbation_max
            )
        else:
            self.state.k = cfg.k_max
            self.state.perturbation = cfg.perturbation_max
    
    def record_heuristic_result(self, heuristic_name: str, improved: bool) -> None:
        if heuristic_name not in self.heuristic_success:
            self.heuristic_success[heuristic_name] = (0, 0)
        successes, total = self.heuristic_success[heuristic_name]
        self.heuristic_success[heuristic_name] = (
            successes + (1 if improved else 0),
            total + 1
        )
    
    def select_heuristics(self, all_heuristics: List[Heuristic], use_stochastic: bool = True) -> List[Heuristic]:
        if not use_stochastic:
            return all_heuristics
            
        sigma = self.state.stagnation
        cfg = self.config
        
        if sigma < cfg.low_stagnation:
            if not self.heuristic_success:
                return all_heuristics
            
            def success_rate(h):
                name = h.name
                if name in self.heuristic_success:
                    s, t = self.heuristic_success[name]
                    return s / max(t, 1)
                return 0.5
            
            sorted_h = sorted(all_heuristics, key=success_rate, reverse=True)
            n_select = max(3, len(all_heuristics) // 2)
            return sorted_h[:n_select]
        else:
            return all_heuristics

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# HYPER-HEURISTIC MODULE: Enhanced Algorithm
# ==============================================================================

@dataclass
class EnhancedConfig:
    """Configuration for enhanced hyper-heuristic."""
    max_iterations: int = 1000
    use_gnn_guidance: bool = True
    use_parallel: bool = False
    use_online_learning: bool = False
    use_adaptive_k: bool = True
    use_warm_start: bool = False
    temperature: float = 0.8  # Temperature for stochastic sampling (match training)
    use_stochastic_sampling: bool = True  # Use Gumbel-softmax like training
    # Baseline specific
    nrst: int = 1
    titr: int = 100

def post_process_refinement(solution: Solution, intensive: bool = False) -> Solution:
    """
    Refinement phase. Hill-climbing (Quick) or ALNS (Intensive).
    """
    refined = solution.copy()
    
    # Phase 1: Intensive Local Search with Pruning (Always run)
    improved_global = True
    passes = 0
    while improved_global and passes < 5:
        improved_global = False
        passes += 1
        
        # 1.1 Intra-Route Intensive Hill Climbing (2-opt & Relocation)
        for r_idx in range(refined.n_routes):
            route = refined.routes[r_idx]
            if len(route) < 2: continue
            
            route_improved = True
            while route_improved:
                route_improved = False
                best_obj = route.compute_objective()
                best_time = route.compute_travel_time()
                best_customers = list(route.customers)
                
                # Try all 2-opt moves
                for i in range(len(route)):
                    for j in range(i + 1, len(route)):
                        # Simulate move
                        new_cust = list(route.customers)
                        new_cust[i:j+1] = new_cust[i:j+1][::-1]
                        
                        # Test with pruning
                        test_route = Route(refined.instance, r_idx)
                        test_route.customers = new_cust
                        test_route.prune_late_nodes()
                        
                        new_obj = test_route.compute_objective()
                        new_time = test_route.compute_travel_time()
                        if new_obj > best_obj + 1e-3 or (abs(new_obj - best_obj) <= 1e-3 and new_time < best_time - 1e-3):
                            best_obj = new_obj
                            best_time = new_time
                            best_customers = list(test_route.customers)
                            route_improved = True
                            improved_global = True
                
                # Try all single-node relocations (Or-opt 1)
                for i in range(len(route)):
                    cust = route.customers[i]
                    temp_cust = list(route.customers)
                    temp_cust.pop(i)
                    for j in range(len(temp_cust) + 1):
                        new_cust = list(temp_cust)
                        new_cust.insert(j, cust)
                        
                        test_route = Route(refined.instance, r_idx)
                        test_route.customers = new_cust
                        test_route.prune_late_nodes()
                        
                        new_obj = test_route.compute_objective()
                        new_time = test_route.compute_travel_time()
                        if new_obj > best_obj + 1e-3 or (abs(new_obj - best_obj) <= 1e-3 and new_time < best_time - 1e-3):
                            best_obj = new_obj
                            best_time = new_time
                            best_customers = list(test_route.customers)
                            route_improved = True
                            improved_global = True
                
                route.customers = best_customers
                route._invalidate_cache()

        # 1.2 Inter-Route Relocation & Swap
        for i in range(refined.n_routes):
            for j in range(refined.n_routes):
                if i == j: continue
                
                ri = refined.routes[i]
                rj = refined.routes[j]
                
                # Try moving node from i to j
                found_move = False
                if len(ri) > 0:
                    for pos_i in range(len(ri)):
                        cust = ri.customers[pos_i]
                        for pos_j in range(len(rj) + 1):
                            new_ri_cust = list(ri.customers)
                            new_ri_cust.pop(pos_i)
                            new_rj_cust = list(rj.customers)
                            new_rj_cust.insert(pos_j, cust)
                            
                            test_ri = Route(refined.instance, i); test_ri.customers = new_ri_cust
                            test_rj = Route(refined.instance, j); test_rj.customers = new_rj_cust
                            test_ri.prune_late_nodes(); test_rj.prune_late_nodes()
                            
                            new_obj_sum = test_ri.compute_objective() + test_rj.compute_objective()
                            old_obj_sum = ri.compute_objective() + rj.compute_objective()
                            delta = new_obj_sum - old_obj_sum
                            
                            new_time_sum = test_ri.compute_travel_time() + test_rj.compute_travel_time()
                            old_time_sum = ri.compute_travel_time() + rj.compute_travel_time()
                            
                            if delta > 1e-3 or (abs(delta) <= 1e-3 and new_time_sum < old_time_sum - 1e-3):
                                ri.customers = test_ri.customers
                                rj.customers = test_rj.customers
                                ri._invalidate_cache(); rj._invalidate_cache()
                                improved_global = True
                                found_move = True
                                break
                        if found_move: break
    
    # Pass 2: SOTA Destroy & Repair (ALNS Perturbation)
    # Only run during Evaluation (intensive=True)
    if intensive:
        for p_idx in range(5): # 5 perturbation rounds
            current = refined.copy()
            
            # 1. Destroy: Randomly remove 20% of customers
            removed = []
            for r_idx in range(current.n_routes):
                route = current.routes[r_idx]
                if len(route) < 5: continue
                
                n_remove = max(1, int(len(route) * 0.20))
                for _ in range(n_remove):
                    if len(route) > 0:
                        idx = random.randint(0, len(route) - 1)
                        removed.append(route.remove_at(idx))
            
            current._invalidate_visited_cache()
            
            # 2. Repair: Regret-2 prioritised reinsertion
            while removed:
                best_regret = -1.0
                best_node_idx = -1
                best_node_info = None
                
                for i, c in enumerate(removed):
                    insertions = []
                    for r_idx in range(current.n_routes):
                        route_nodes = [0] + list(current.routes[r_idx].customers) + [0]
                        for p in range(len(current.routes[r_idx]) + 1):
                            u = route_nodes[p]
                            v = route_nodes[p+1]
                            
                            travel_delta = current.instance.travel_time[u, c] + \
                                           current.instance.travel_time[c, v] - \
                                           current.instance.travel_time[u, v]
                            
                            delta = current.instance.profits[c] - travel_delta
                            
                            if delta > 0:
                                insertions.append((delta, r_idx, p))
                    
                    insertions.sort(key=lambda x: x[0], reverse=True)
                    
                    if not insertions:
                        regret = 0
                        info = (-1, -1)
                    elif len(insertions) == 1:
                        regret = insertions[0][0]
                        info = (insertions[0][1], insertions[0][2])
                    else:
                        regret = insertions[0][0] - insertions[1][0]
                        info = (insertions[0][1], insertions[0][2])
                    
                    if regret > best_regret:
                        best_regret = regret
                        best_node_idx = i
                        best_node_info = info
                
                c = removed.pop(best_node_idx)
                if best_node_info[0] != -1:
                    current.routes[best_node_info[0]].insert(best_node_info[1], c)
                    current._invalidate_visited_cache()
            
            # 3. Post-Repair Intensive Greedy Fill
            current = _greedy_extraction_pass(current)
            
            # 4. Accept strictly better
            if current.compute_objective() > refined.compute_objective() + 1e-3:
                refined = current
            
    return refined

def _greedy_extraction_pass(solution: Solution) -> Solution:
    """Exhaustively add any nodes that yield positive profit delta."""
    unvisited = list(solution.get_unvisited_set())
    # Sort by profit density or raw profit
    unvisited.sort(key=lambda c: solution.instance.profits[c], reverse=True)
    
    modified = True
    while modified:
        modified = False
        for c in unvisited:
            if solution.is_customer_visited(c): continue
            
            r, p, delta = solution.find_best_insertion_global(c)
            if delta > 1e-3:
                solution.routes[r].insert(p, c)
                solution._invalidate_visited_cache()
                modified = True
    return solution

class EnhancedHyperHeuristic:
    """
    Enhanced hyper-heuristic with GNN guidance for candidate selection.
    """
    
    def __init__(
        self,
        instance: Instance,
        config: EnhancedConfig = None,
        controller_config: ControllerConfig = None,
        gnn_model=None
    ):
        self.instance = instance
        self.config = config or EnhancedConfig()
        self.gnn_model = gnn_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.controller = FeedbackController(controller_config)
        
        # Initialize heuristics (ALL 12 - must match training!)
        self.heuristics = [
            # Intra-route heuristics (LH1-LH5)
            LH1_RemcrRoute(crn=3),
            LH2_RempRoute(),
            LH3_RemlRoute(rl=2),
            LH4_SwapRoute(),
            LH5_TwoOptRoute(),
            # Inter-route heuristics (LH6-LH10)
            LH6_RemlIroute(rl=2),
            LH7_RemcrIroute(crn=2),
            LH8_SwapIroute(),
            LH9_RemprIroute(),
            LH10_RempaddIroute(),
            # Greedy heuristics (LH11-LH12)
            LH11_Remove(),
            LH12_Add(),
            LH13_DestroyRepair(),
            LH14_SlackTriage(slack_threshold_pct=0.05, lookahead=3)
        ]
    
    def _construct_initial_solution(self, seed: Optional[int] = None) -> Solution:
        constructor = ConstructiveHeuristic(self.instance)
        return constructor.construct(seed=seed)
    def _get_adaptive_k(self, probs: torch.Tensor) -> int:
        """Compute K based on GNN confidence entropy.
        High entropy (uncertain) -> larger K (explore more candidates).
        Low entropy (confident) -> smaller K (focus on top picks).
        """
        n_customers = self.instance.n_customers
        # Entropy of the distribution over customers (exclude depot)
        probs_cust = probs[1:]  # exclude depot
        entropy = -(probs_cust * torch.log(probs_cust + 1e-10)).sum().item()
        max_entropy = math.log(n_customers)  # entropy of uniform distribution
        normalised_entropy = entropy / max(max_entropy, 1e-6)  # in [0, 1]

        k_min = max(3, n_customers // 20)    # at least 5% of customers
        k_max = min(50, n_customers // 5)    # at most 20% of customers
        # Linear interpolation: low entropy -> k_min, high entropy -> k_max
        k = int(k_min + normalised_entropy * (k_max - k_min))
        return max(k_min, min(k_max, k))

    def _get_candidates(self, solution: Solution) -> Set[int]:
        """
        Returns a set of candidate customers for heuristic application.
        """
        if self.gnn_model is not None and self.config.use_gnn_guidance:
            graph = build_solution_graph(solution).to(self.device)
            
            with torch.no_grad():
                logits = self.gnn_model(graph, return_logits=True).cpu()
            
            logits[0] = float('-inf')  # Mask depot
            if hasattr(graph, 'node_feasible'):
                logits[graph.node_feasible.cpu() < 0.5] = float('-inf')
                
            probs = F.softmax(logits / self.config.temperature, dim=0)
            
            # Dynamic K based on entropy
            k = self._get_adaptive_k(probs)
            
            if self.config.use_stochastic_sampling:
                # Stochastic sampling (Gumbel-softmax)
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
                perturbed = torch.log(probs + 1e-10) + gumbel_noise
                perturbed[0] = float('-inf')
                
                n_customers = probs.size(0) - 1
                k_gnn = min(k, n_customers)
                _, top_k_gnn = torch.topk(perturbed, k_gnn)
                return set(top_k_gnn.tolist())
            else:
                # Deterministic top-k
                scores = logits.numpy()
                n_customers = len(scores) - 1
                k_gnn = min(k, n_customers)
                top_k_gnn = np.argsort(scores)[-k_gnn:]
                return set(top_k_gnn.tolist())
        else:
            # Fallback to random if GNN is missing
            k = max(3, min(self.instance.n_customers // 10, 50))
            all_customers = list(range(1, self.instance.n_customers + 1))
            return set(random.sample(all_customers, min(k, len(all_customers))))

    def _apply_heuristics(
        self,
        solution: Solution,
        candidates: set,
        step: int
    ) -> Tuple[Solution, bool, str]:
        """Apply all heuristics to selected customers and return the best improvement."""
        best_solution = solution.copy()
        best_obj = solution.compute_objective()
        improved = False
        best_heuristic_name = ""
        
        # Shuffle heuristics to avoid bias
        shuffled_heuristics = list(self.heuristics)
        random.shuffle(shuffled_heuristics)
        
        for heuristic in shuffled_heuristics:
            try:
                new_sol = heuristic.apply(
                    solution,
                    candidates=candidates,
                    iteration=step,
                    max_iterations=self.config.max_iterations
                )
                if new_sol is not None and new_sol.is_feasible():
                    if not new_sol.validate_ordering():
                        continue
                        
                    new_obj = new_sol.compute_objective()
                    if new_obj > best_obj:
                        best_solution = new_sol
                        best_obj = new_obj
                        improved = True
                        best_heuristic_name = heuristic.name
            except Exception:
                continue
        
        return best_solution, improved, best_heuristic_name

    
    def solve(self, initial_solution: Optional[Solution] = None) -> Solution:
        # Multi-start elite pool — best-of-5 constructive starts
        if initial_solution is None:
            elite_pool = []
            for seed_offset in range(5):
                seed = abs(hash(self.instance.name)) % 10000 + seed_offset
                sol = self._construct_initial_solution(seed=seed)
                elite_pool.append((sol.compute_objective(), sol))
            elite_pool.sort(key=lambda x: x[0], reverse=True)
            current = elite_pool[0][1].copy()
        else:
            current = initial_solution.copy()
            elite_pool = [(current.compute_objective(), current.copy())]
        
        best = current.copy()
        best_obj = best.compute_objective()
        self.controller.state.best_objective = best_obj
        
        for iteration in range(1, self.config.max_iterations + 1):
            # Temperature cycling — alternate exploit/explore phases
            phase_length = max(self.config.max_iterations // 10, 1)
            if (iteration // phase_length) % 2 == 0:
                effective_temp = self.config.temperature * 0.5  # exploit
            else:
                effective_temp = self.config.temperature * 1.5  # explore
            
            # Temporarily override temperature for candidate selection
            original_temp = self.config.temperature
            self.config.temperature = effective_temp
            
            candidates = self._get_candidates(current)
            
            # Restore original temperature
            self.config.temperature = original_temp
            
            new_solution, improved, heuristic_name = self._apply_heuristics(
                current, candidates, iteration
            )
            
            new_obj = new_solution.compute_objective()
            
            if new_obj > best_obj:
                best = new_solution.copy()
                best_obj = new_obj
                improved = True
                # Update elite pool
                elite_pool.append((new_obj, new_solution.copy()))
                elite_pool.sort(key=lambda x: x[0], reverse=True)
                elite_pool = elite_pool[:5]  # Keep top 5
            
            self.controller.update(improved, new_obj)
            
            if heuristic_name:
                self.controller.record_heuristic_result(heuristic_name, improved)
            
            # SA acceptance
            if improved:
                current = new_solution
            elif self.config.use_stochastic_sampling:
                delta = new_obj - current.compute_objective()
                sa_temp = max(0.001, 1.0 - iteration / self.config.max_iterations)
                normalizer = max(abs(best_obj), 1.0)
                if random.random() < math.exp(delta / (sa_temp * normalizer)):
                    current = new_solution
            
            # Stagnation restart from elite pool
            if self.controller.state.stagnation > 30 and len(elite_pool) > 1:
                idx = random.randint(0, min(2, len(elite_pool) - 1))
                current = elite_pool[idx][1].copy()
                self.controller.state.stagnation = 0
        
        if best is not None:
            best = post_process_refinement(best)
            
        return best

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# DIVIDE & CONQUER MODULE
# ==============================================================================

class SpatialDecomposer:
    """
    Partition large MTRPP instances into balanced geographic sub-instances
    using K-Means clustering. Each sub-instance has 1 repairman.
    """
    
    def __init__(
        self,
        target_cluster_size: int = 50,
        profit_aware: bool = True,
        urgency_weight: float = 0.5,
        random_state: int = 42
    ):
        self.target_cluster_size = target_cluster_size
        self.profit_aware = profit_aware
        self.urgency_weight = urgency_weight
        self.random_state = random_state
    
    def decompose(self, instance: Instance) -> Tuple[List[Instance], np.ndarray]:
        """
        Partition customer nodes into spatial clusters.
        
        Returns:
            sub_instances: list of Instance objects (1 repairman each)
            labels: cluster assignment for each customer (0-indexed, len=n_customers)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for D&C decomposition")
        
        n = instance.n_customers
        
        # Determine number of clusters
        n_clusters = max(
            instance.n_repairmen,
            n // self.target_cluster_size
        )
        n_clusters = min(n_clusters, n)
        
        # Build feature vectors for clustering
        coords = instance.coords[1:]  # [n_customers, 2], exclude depot
        
        if self.profit_aware:
            labels = self._profit_aware_cluster(coords, instance.profits[1:], n_clusters)
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(coords)
        
        # Build sub-instances
        sub_instances = []
        for cluster_id in range(n_clusters):
            # Customer indices in parent instance (1-indexed)
            members = [i + 1 for i in range(n) if labels[i] == cluster_id]
            
            if not members:
                continue
            
            sub = self._build_sub_instance(instance, members, cluster_id)
            sub_instances.append(sub)
        
        return sub_instances, labels
    
    def _profit_aware_cluster(
        self, coords: np.ndarray, deadlines: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        """
        Weighted K-Means using [x_norm, y_norm, urgency] features.
        Groups customers with similar location AND similar deadline urgency.
        """
        coord_range = np.ptp(coords, axis=0) + 1e-6
        deadline_range = np.ptp(deadlines) + 1e-6
        
        # Urgency: inverse deadline (tighter deadline -> higher urgency)
        urgency = 1.0 / (deadlines / deadline_range + 0.1)
        
        features = np.column_stack([
            coords / coord_range,
            urgency.reshape(-1, 1) * self.urgency_weight
        ])
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        return kmeans.fit_predict(features)
    
    def _build_sub_instance(
        self,
        parent: Instance,
        members: List[int],
        cluster_id: int
    ) -> Instance:
        """Build a sub-instance containing depot + cluster members."""
        nodes = [0] + members  # depot + members
        m = len(members)
        
        sub_coords = parent.coords[nodes]
        sub_profits = parent.profits[nodes]
        
        # Sub travel-time matrix
        sub_travel = np.zeros((m + 1, m + 1))
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                sub_travel[i, j] = parent.travel_time[ni, nj]
        
        sub_instance = Instance(
            n_customers=m,
            n_repairmen=1,
            coords=sub_coords,
            profits=sub_profits,
            travel_time=sub_travel,
            instance_name=f"{parent.name}_c{cluster_id}"
        )
        
        # Store mapping: sub-instance node index -> parent node index
        sub_instance._parent_mapping = {
            sub_idx: parent_idx
            for sub_idx, parent_idx in enumerate(nodes)
        }
        sub_instance._cluster_id = cluster_id
        
        return sub_instance


class DivideAndConquerSolver:
    """
    D&C solver with GPU-batched GNN inference and CPU-parallel heuristics.
    
    Architecture:
    - Size-gates: only activates D&C for instances > dc_threshold nodes
    - Decomposes large instances into ~target_cluster_size sub-problems
    - Solves all sub-problems simultaneously:
      * GPU: single batched GNN forward pass across all sub-problems
      * CPU: ProcessPoolExecutor for parallel heuristic application
    - Merges sub-solutions and runs boundary repair
    """
    
    def __init__(
        self,
        instance: Instance,
        config: EnhancedConfig = None,
        gnn_model=None,
        target_cluster_size: int = 50,
        dc_threshold: int = 150,
        profit_aware: bool = True,
        boundary_repair_iters: int = 200,
        random_state: int = 42,
    ):
        self.instance = instance
        self.config = config or EnhancedConfig()
        self.gnn_model = gnn_model
        self.target_cluster_size = target_cluster_size
        self.dc_threshold = dc_threshold
        self.profit_aware = profit_aware
        self.boundary_repair_iters = boundary_repair_iters
        self.random_state = random_state
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def solve(self, intensive: bool = False) -> Solution:
        """
        Main entry point. Size-gates to monolithic solver if N <= threshold.
        """
        if self.instance.n_customers <= self.dc_threshold:
            return self._solve_monolithic(intensive=intensive)
        return self._solve_dc(intensive=intensive)
    
    def _solve_monolithic(self, intensive: bool = False) -> Solution:
        """Fall back to standard EnhancedHyperHeuristic for small instances."""
        solver = EnhancedHyperHeuristic(
            instance=self.instance,
            config=self.config,
            gnn_model=self.gnn_model
        )
        solution = solver.solve()
        if intensive:
            solution = post_process_refinement(solution, intensive=True)
        return solution
    
    def _solve_dc(self, intensive: bool = False) -> Solution:
        """Full D&C pipeline: decompose -> parallel solve -> merge -> repair."""
        decomposer = SpatialDecomposer(
            target_cluster_size=self.target_cluster_size,
            profit_aware=self.profit_aware,
            random_state=self.random_state
        )
        
        best_repaired = None
        best_obj = -1
        
        # Single-pass D&C (Optimized for speed)
        ts = self.target_cluster_size
        decomposer.target_cluster_size = ts
        sub_instances, labels = decomposer.decompose(self.instance)
        
        n_clusters = len(sub_instances)
        print(f"  D&C Optimized Solve: {self.instance.n_customers} nodes -> "
              f"{n_clusters} clusters (target size {ts})")
        
        # Phase 2: Solve all sub-problems in parallel
        sub_solutions = self._solve_all_parallel(sub_instances)
        
        # Phase 3: Merge sub-solutions into parent solution
        merged = self._merge_solutions(sub_instances, sub_solutions)
        
        # Phase 4: Boundary repair
        repaired = self._boundary_repair(merged, labels, intensive=intensive)
        
        return repaired
    
    def _solve_all_parallel(self, sub_instances: List[Instance]) -> List[Solution]:
        """
        Solve all sub-instances using batched GNN + parallel heuristics.
        
        Per iteration:
        1. GPU: batch all sub-problem graphs, single GNN forward pass
        2. CPU: parallel heuristic application across all sub-problems
        3. Main thread: update best solutions
        """
        n_subs = len(sub_instances)
        max_iterations = self.config.max_iterations
        use_gnn = self.config.use_gnn_guidance and self.gnn_model is not None
        
        # Create per-sub-problem solvers (for heuristic application only)
        solvers = []
        for sub_inst in sub_instances:
            solver_config = EnhancedConfig(
                max_iterations=max_iterations,
                use_gnn_guidance=False,  # GNN handled at batch level
                use_adaptive_k=True,
                temperature=self.config.temperature,
                use_stochastic_sampling=self.config.use_stochastic_sampling
            )
            solver = EnhancedHyperHeuristic(
                instance=sub_inst,
                config=solver_config,
                gnn_model=None  # GNN handled centrally
            )
            solvers.append(solver)
        
        # Construct initial solutions
        current_solutions = [s._construct_initial_solution() for s in solvers]
        best_solutions = [sol.copy() for sol in current_solutions]
        best_objs = [sol.compute_objective() for sol in best_solutions]
        
        # Precompute adaptive k for each sub-problem
        ks = [max(3, min(sub.n_customers // 10, 50)) for sub in sub_instances]
        max_k = max(ks) if ks else 5
        
        # Direct in-process execution: ProcessPoolExecutor with 'spawn' context
        # has prohibitive overhead (reimporting 160KB module + PyTorch per worker
        # per iteration). For N sub-problems × max_iterations, the IPC cost
        # dominates over actual heuristic computation.
        # Optimized GNN Frequency: only rebuild graph and infer every 2 iterations
        # Reusing actions_list from previous step significantly reduces bottleneck
        actions_list = None
        for iteration in range(1, max_iterations + 1):
            
            # --- Step A: GPU Batched GNN Inference (Optimized Frequency) ---
            if use_gnn and TORCH_GEOMETRIC_AVAILABLE and (iteration % 2 == 1 or actions_list is None):
                try:
                    graphs = [build_solution_graph(sol) for sol in current_solutions]
                    batch = Batch.from_data_list(graphs).to(self.device)
                    
                    with torch.no_grad():
                        actions_list, _ = self.gnn_model.sample_actions(
                            batch,
                            k=max_k,
                            temperature=self.config.temperature
                        )
                except RuntimeError:
                    # GPU OOM fallback: skip GNN this iteration
                    actions_list = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # --- Step B: Sequential Heuristic Application ---
            results = []
            for i, solver in enumerate(solvers):
                if actions_list is not None and i < len(actions_list):
                    acts = actions_list[i]
                    gnn_candidates = set(acts.cpu().tolist())
                    base_candidates = set(solver._get_candidates(current_solutions[i]))
                    candidates = gnn_candidates.union(base_candidates)
                else:
                    candidates = solver._get_candidates(current_solutions[i])
                
                result = solver._apply_heuristics(current_solutions[i], candidates, iteration)
                results.append(result)
            
            # --- Step C: Update Best Solutions ---
            for i, (new_sol, improved, name) in enumerate(results):
                new_obj = new_sol.compute_objective()
                
                if new_obj > best_objs[i]:
                    best_solutions[i] = new_sol.copy()
                    best_objs[i] = new_obj
                    solvers[i].controller.update(True, new_obj)
                else:
                    solvers[i].controller.update(False, new_obj)
                
                if name:
                    solvers[i].controller.record_heuristic_result(name, improved)
                
                # SA Acceptance
                if improved:
                    current_solutions[i] = new_sol
                elif self.config.use_stochastic_sampling:
                    delta = new_obj - current_solutions[i].compute_objective()
                    sa_temp = max(0.001, 1.0 - iteration / self.config.max_iterations)
                    normalizer = max(abs(best_objs[i]), 1.0)
                    if random.random() < math.exp(delta / (sa_temp * normalizer)):
                        current_solutions[i] = new_sol
        
        return best_solutions
    
    def _merge_solutions(
        self,
        sub_instances: List[Instance],
        sub_solutions: List[Solution]
    ) -> Solution:
        """
        Merge sub-solutions into a parent solution.
        
        Strategy: assign clusters to repairmen using angular sweep from depot,
        then chain cluster routes within each repairman's route.
        """
        parent = self.instance
        n_repairmen = parent.n_repairmen
        n_subs = len(sub_instances)
        
        # Compute cluster centroids for repairman assignment
        centroids = []
        for sub in sub_instances:
            centroid = np.mean(sub.coords[1:], axis=0)
            centroids.append(centroid)
        
        depot = parent.coords[0]
        
        # Sort clusters by angle from depot (angular sweep)
        angles = [
            np.arctan2(c[1] - depot[1], c[0] - depot[0])
            for c in centroids
        ]
        sorted_indices = np.argsort(angles)
        
        # Assign clusters to repairmen (round-robin by angle)
        assignments = [[] for _ in range(n_repairmen)]
        for rank, idx in enumerate(sorted_indices):
            repairman_id = rank % n_repairmen
            assignments[repairman_id].append(idx)
        
        # Build merged solution
        merged = Solution(parent)
        
        for repairman_id in range(n_repairmen):
            all_customers = []
            for sub_idx in assignments[repairman_id]:
                sub_inst = sub_instances[sub_idx]
                sub_sol = sub_solutions[sub_idx]
                
                # Map sub-solution customer indices -> parent indices
                for route in sub_sol.routes:
                    for cust in route.customers:
                        parent_cust = sub_inst._parent_mapping[cust]
                        if parent_cust != 0:  # Skip depot
                            all_customers.append(parent_cust)
            
            merged.routes[repairman_id].customers = all_customers
            merged.routes[repairman_id]._invalidate_cache()
        
        merged._invalidate_visited_cache()
        return merged
    
    def _find_best_profitable_insertion(
        self, solution: Solution, customer: int
    ) -> tuple[int, int, float]:
        """Return (route_idx, pos, realised_profit) for the insertion that maximises
        the realised profit of `customer`, not just the improvement delta."""
        best_route, best_pos, best_profit = 0, 0, -1.0
        inst = solution.instance

        for r_idx, route in enumerate(solution.routes):
            customers = route.customers
            for pos in range(len(customers) + 1):
                # Simulate full route arrival time at `customer` in this position
                t = 0.0
                prev = 0  # depot

                # Travel through prefix [0 .. pos-1]
                for k in range(pos):
                    t += inst.get_travel_time(prev, customers[k])
                    prev = customers[k]

                # Arrive at inserted customer
                t += inst.get_travel_time(prev, customer)
                profit = inst.compute_profit(customer, t)

                if profit > best_profit:
                    best_profit = profit
                    best_route = r_idx
                    best_pos = pos

        return best_route, best_pos, best_profit

    def _boundary_repair(self, solution: Solution, labels: np.ndarray, intensive: bool = False) -> Solution:
        """
        Four-phase boundary repair for profit recovery after a cluster merge.
        Phases: (0) LH14 timing cleanup, (1) zero-profit ejection,
        (2) urgency-ordered reinsertion, (3) boundary-restricted local search.
        """
        boundary = self._find_boundary_customers(labels)
        if not boundary:
            return solution

        # --- Phase 0: LH14 Timing Reset (Timing Damage Cleanup) ---
        # Run aggressive cascade pruning to clear merge-induced delays
        lh14_repair = LH14_SlackTriage(slack_threshold_pct=0.10, lookahead=5)
        current = solution.copy()
        for _ in range(3):
            result = lh14_repair.apply(current, candidates=boundary)
            if result is None:
                break
            current = result

        # --- Phase 1: Slack-prioritised removal of zero-profit nodes ---
        # Identify nodes near the seam that are now arriving too late to be profitable
        best = current.copy()
        best_obj = best.compute_objective()
        
        arrival_times = current.compute_arrival_times_dict()
        zero_profit_boundary = {
            c for c in boundary
            if c in arrival_times
            and current.instance.compute_profit(c, arrival_times[c]) < 1e-3
        }

        for c in zero_profit_boundary:
            loc = current.get_customer_position(c)
            if loc:
                r_idx, pos = loc
                current.routes[r_idx].remove_at(pos)
                current._invalidate_visited_cache()

        # --- Phase 2: Slack-sorted reinsertion with timing lookahead ---
        # Sort removed nodes by deadline ASC (most urgent first) so they claim early spots
        removed_sorted = sorted(
            list(zero_profit_boundary),
            key=lambda c: current.instance.profits[c],
            reverse=False # Smallest deadline = Most Urgent
        )
        for c in removed_sorted:
            best_route, best_pos, best_realised = self._find_best_profitable_insertion(current, c)
            if best_realised > 0:
                current.routes[best_route].insert(best_pos, c)
                current._invalidate_visited_cache()

        # --- Phase 3: Standard boundary heuristics with stagnation tolerance ---
        inter_heuristics = [
            LH6_RemlIroute(rl=2), 
            LH8_SwapIroute(),
            LH10_RempaddIroute(), 
            LH11_Remove(), 
            LH12_Add(), 
            LH13_DestroyRepair(),
            LH14_SlackTriage(slack_threshold_pct=0.05, lookahead=3)
        ]
        
        stagnation = 0
        for iteration in range(self.boundary_repair_iters):
            improved_any = False
            for heuristic in inter_heuristics:
                try:
                    new_sol = heuristic.apply(
                        current,
                        candidates=boundary,
                        iteration=iteration,
                        max_iterations=self.boundary_repair_iters
                    )
                    if new_sol is not None and new_sol.is_feasible():
                        new_obj = new_sol.compute_objective()
                        if new_obj > best_obj:
                            best = new_sol.copy()
                            best_obj = new_obj
                            improved_any = True
                        current = new_sol
                except Exception:
                    continue
            
            stagnation = 0 if improved_any else stagnation + 1
            if stagnation > 5:
                break
        
        if best is not None:
            best = post_process_refinement(best, intensive=intensive)
            
        return best
    
    def _find_boundary_customers(
        self, labels: np.ndarray, radius_pct: float = 0.20
    ) -> Set[int]:
        """
        Identify customers near cluster boundaries.
        A customer is a boundary node if its nearest neighbor in a DIFFERENT
        cluster is within radius_pct of the average intra-cluster distance.
        """
        coords = self.instance.coords[1:]  # exclude depot
        n = len(coords)
        boundary = set()
        
        # Precompute per-cluster average intra-cluster distance
        cluster_ids = set(labels)
        cluster_avg_dist = {}
        for cid in cluster_ids:
            members = np.where(labels == cid)[0]
            if len(members) < 2:
                cluster_avg_dist[cid] = float('inf')
                continue
            dists = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    dists.append(np.linalg.norm(
                        coords[members[i]] - coords[members[j]]
                    ))
            cluster_avg_dist[cid] = np.mean(dists) if dists else float('inf')
        
        # For each customer, check nearest cross-cluster neighbor
        boundary_candidates = []
        for i in range(n):
            my_cluster = labels[i]
            avg_d = cluster_avg_dist[my_cluster]
            if avg_d == float('inf'):
                continue
                
            threshold = avg_d * radius_pct
            min_cross_dist = float('inf')
            
            for j in range(n):
                if labels[j] == my_cluster:
                    continue
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < min_cross_dist:
                    min_cross_dist = dist
                    
            if min_cross_dist < threshold:
                boundary_candidates.append((i + 1, min_cross_dist / avg_d))
                
        # Cap boundary size to prevent O(B^2) explosion in inter-heuristics
        # Scale boundary cap dynamically with instance size
        boundary_candidates.sort(key=lambda x: x[1])
        max_boundary = min(max(60, self.instance.n_customers // 5), 150)
        boundary = {c[0] for c in boundary_candidates[:max_boundary]}
        
        return boundary

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_on_instances(
    instances: List[Instance],
    gnn_model: nn.Module,
    batch_size: int = 16,
    max_iterations: int = 200,
    temperature: float = 0.8,
    use_stochastic: bool = False,
    checkpoint_iters: List[int] = None,
    restart_idx: int = 0,
    intensive: bool = False
) -> Dict[str, Any]:
    """Evaluate the GNN solver on a list of instances."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if gnn_model is not None:
        gnn_model.to(device)
        gnn_model.eval()
    else:
        raise ValueError("evaluate_on_instances requires a loaded gnn_model!")
    
    results = []
    
    if batch_size > 1:
            print(f"  Using batch size: {batch_size} for evaluation")
            
            def run_batch_evaluation(batch_instances, cp_iters=None):
                solvers = []
                for inst in batch_instances:
                    use_dc = inst.n_customers > CONFIG.get('dc_threshold', 150)
                    solver_config = EnhancedConfig(
                        max_iterations=max_iterations,
                        use_gnn_guidance=not use_dc,
                        use_adaptive_k=True,
                        temperature=temperature,
                        use_stochastic_sampling=use_stochastic
                    )
                    if use_dc:
                        solver = DivideAndConquerSolver(
                            inst, 
                            config=solver_config, 
                            gnn_model=gnn_model,
                            target_cluster_size=CONFIG.get('dc_target_cluster_size', 50),
                            dc_threshold=CONFIG.get('dc_threshold', 150),
                            profit_aware=CONFIG.get('dc_profit_aware', True),
                            boundary_repair_iters=CONFIG.get('dc_boundary_repair_iters', 50)
                        )
                    else:
                        solver = EnhancedHyperHeuristic(inst, config=solver_config, gnn_model=gnn_model)
                    solvers.append(solver)
                
                current_solutions = []
                for s in solvers:
                    # Standardized Best-of-3 robust initialization (matches training)
                    best_init_sol = None
                    best_init_obj = -float('inf')
                    for seed_offset in range(3):
                        init_seed = (abs(hash(s.instance.name)) % 10000 + seed_offset + restart_idx * 100)
                        sol = s._construct_initial_solution(seed=init_seed)
                        obj = sol.compute_objective()
                        if obj > best_init_obj:
                            best_init_obj = obj
                            best_init_sol = sol
                    current_solutions.append(best_init_sol)
                    
                best_solutions = [sol.copy() for sol in current_solutions]
                best_objs = [sol.compute_objective() for sol in best_solutions]
                
                ks = [max(3, min(inst.n_customers // 10, 50)) for inst in batch_instances]
                max_k = max(ks)
                
                opts = []
                for inst in batch_instances:
                    opt_val = 0.0
                    if hasattr(inst, '_optimal_routes') and inst._optimal_routes:
                        try:
                            opt_sol = Solution(inst, inst._optimal_routes)
                            opt_val = opt_sol.compute_objective()
                        except: pass
                    opts.append(opt_val)

                checkpoint_snapshots = {}
                checkpoint_set = set(cp_iters) if cp_iters else set()
                
                actions_list = None
                for iteration in range(1, max_iterations + 1):
                    if iteration % 2 == 1 or actions_list is None:
                        graphs = [build_solution_graph(sol) for sol in current_solutions]
                        batch = Batch.from_data_list(graphs).to(device)
                        with torch.no_grad():
                            actions_list, _ = gnn_model.sample_actions(batch, k=max_k, temperature=temperature)
                        
                        # Cleanup VRAM during long eval runs
                        if iteration % 50 == 0:
                            torch.cuda.empty_cache()

                    heuristic_inputs = []
                    for i, solver in enumerate(solvers):
                        if actions_list is not None and i < len(actions_list):
                            acts = actions_list[i]
                            candidates = set(acts.cpu().tolist())
                        else:
                            candidates = solver._get_candidates(current_solutions[i])
                        heuristic_inputs.append((solver, current_solutions[i], candidates))

                    results_list = []
                    for i, (solver, sol, cands) in enumerate(heuristic_inputs):
                        results_list.append(solver._apply_heuristics(sol, cands, iteration))

                    for i, (new_sol, improved, name) in enumerate(results_list):
                        solver = solvers[i]
                        new_obj = new_sol.compute_objective()
                        if new_obj > best_objs[i]:
                            best_solutions[i] = new_sol.copy()
                            best_objs[i] = new_obj
                            solver.controller.update(True, new_obj)
                        else:
                            solver.controller.update(False, new_obj)
                        if improved or (solver.config.use_stochastic_sampling and random.random() < 0.1):
                            current_solutions[i] = new_sol
                    
                    if iteration in checkpoint_set:
                        gaps = []
                        for i, obj in enumerate(best_objs):
                            opt = opts[i]
                            gaps.append((opt - obj) / opt * 100.0 if opt > 0 else None)
                        checkpoint_snapshots[iteration] = {'objs': list(best_objs), 'gaps': gaps}
                
                # Final Intensive Polish
                best_solutions = [post_process_refinement(sol, intensive=intensive) for sol in best_solutions]
                
                return best_solutions, checkpoint_snapshots

            dc_threshold = CONFIG.get('dc_threshold', 150)
            small_instances = [inst for inst in instances if inst.n_customers <= dc_threshold]
            large_instances = [inst for inst in instances if inst.n_customers > dc_threshold]
            
            for inst in tqdm(large_instances, desc="Evaluating (D&C Large)"):
                solver_config = EnhancedConfig(
                    max_iterations=max_iterations,
                    use_gnn_guidance=EnhancedConfig.use_gnn_guidance,
                    temperature=temperature,
                    use_stochastic_sampling=use_stochastic
                )
                solver = DivideAndConquerSolver(
                    inst, 
                    config=solver_config, 
                    gnn_model=gnn_model,
                    target_cluster_size=CONFIG.get('dc_target_cluster_size', 50),
                    dc_threshold=dc_threshold,
                    profit_aware=CONFIG.get('dc_profit_aware', True),
                    boundary_repair_iters=CONFIG.get('dc_boundary_repair_iters', 50),
                    random_state=CONFIG.get('seed', 42)
                )
                start_time = time.time()
                best_sol = solver.solve(intensive=intensive)
                solve_time = time.time() - start_time
                
                obj = best_sol.compute_objective()
                optimal = getattr(inst, 'final_profit', None)
                if optimal is None and hasattr(inst, '_optimal_routes') and inst._optimal_routes:
                    try:
                        optimal = Solution(inst, inst._optimal_routes).compute_objective()
                    except Exception: pass
                
                gap = ((optimal - obj) / optimal * 100.0) if (optimal is not None and optimal > 0) else None
                
                results.append({
                    'name': inst.name if hasattr(inst, 'name') else f"Instance_{len(results)}",
                    'objective': obj,
                    'optimal': optimal,
                    'gap': gap,
                    'time': solve_time,
                    'visited': best_sol.get_visited_count(),
                    'n_customers': inst.n_customers,
                    'routes': [list(route.customers) for route in best_sol.routes],
                })
            
            all_checkpoint_results = {}
            n_small = len(small_instances)
            batches = [small_instances[i:i + batch_size] for i in range(0, n_small, batch_size)]
            
            for batch_instances in tqdm(batches, desc="Evaluating (Batched Small)"):
                start_time = time.time()
                batch_results, batch_snapshots = run_batch_evaluation(batch_instances, cp_iters=checkpoint_iters)
                per_instance_time = (time.time() - start_time) / len(batch_instances)

                if batch_snapshots:
                    for it, data in batch_snapshots.items():
                        if it not in all_checkpoint_results:
                            all_checkpoint_results[it] = {'objs': [], 'gaps': []}
                        all_checkpoint_results[it]['objs'].extend(data['objs'])
                        all_checkpoint_results[it]['gaps'].extend(data['gaps'])
                
                for i, best_sol in enumerate(batch_results):
                    inst = batch_instances[i]
                    obj = best_sol.compute_objective()
                    
                    optimal = getattr(inst, 'final_profit', None)
                    if optimal is None and hasattr(inst, '_optimal_routes') and inst._optimal_routes:
                        try:
                            optimal = Solution(inst, inst._optimal_routes).compute_objective()
                        except Exception: pass
                    
                    gap = ((optimal - obj) / optimal * 100.0) if (optimal is not None and optimal > 0) else None
                    
                    results.append({
                        'name': inst.name if hasattr(inst, 'name') else f"Instance_{len(results)}",
                        'objective': obj,
                        'optimal': optimal,
                        'gap': gap,
                        'time': per_instance_time,
                        'visited': list(best_sol.get_visited_set()),
                        'routes': [r.customers for r in best_sol.routes],
                        'n_customers': inst.n_customers,
                    })
            
            objectives = [r['objective'] for r in results]
            out = {
                'results': results,
                'avg_objective': np.mean(objectives) if objectives else 0.0,
                'median_objective': np.median(objectives) if objectives else 0.0,
                'std_objective': np.std(objectives) if objectives else 0.0,
                'min_objective': np.min(objectives) if objectives else 0.0,
                'max_objective': np.max(objectives) if objectives else 0.0,
                'avg_gap': np.mean([r['gap'] for r in results if r.get('gap') is not None]) if results else None,
                'avg_time': np.mean([r['time'] for r in results]) if results else 0.0,
            }
            if checkpoint_iters:
                checkpoint_stats = {}
                for it, data in all_checkpoint_results.items():
                    objs, gaps = data['objs'], [g for g in data['gaps'] if g is not None]
                    checkpoint_stats[it] = {
                        'avg_objective': float(np.mean(objs)),
                        'median_objective': float(np.median(objs)),
                        'avg_gap': float(np.mean(gaps)) if gaps else None,
                        'median_gap': float(np.median(gaps)) if gaps else None,
                        'results': [{'objective': o, 'gap': g} for o, g in zip(objs, data['gaps'])] 
                    }
                out['checkpoints'] = checkpoint_stats
            return out
    
    # Non-batched fallback
    for instance in tqdm(instances, desc="Evaluating"):
        use_dc = instance.n_customers > CONFIG.get('dc_threshold', 150)
        config = EnhancedConfig(
            max_iterations=max_iterations,
            use_gnn_guidance=EnhancedConfig.use_gnn_guidance,
            use_adaptive_k=True,
            temperature=temperature,
            use_stochastic_sampling=use_stochastic,
        )
        if use_dc:
            solver = DivideAndConquerSolver(
                instance=instance, config=config, gnn_model=gnn_model,
                target_cluster_size=CONFIG.get('dc_target_cluster_size', 50),
                dc_threshold=CONFIG.get('dc_threshold', 150),
                profit_aware=CONFIG.get('dc_profit_aware', True),
                boundary_repair_iters=CONFIG.get('dc_boundary_repair_iters', 50)
            )
        else:
            solver = EnhancedHyperHeuristic(instance=instance, config=config, gnn_model=gnn_model)
        
        start_time = time.time()
        solution = solver.solve()
        solve_time = time.time() - start_time
        
        obj = solution.compute_objective()
        optimal = getattr(instance, 'final_profit', None)
        if optimal is None and hasattr(instance, '_optimal_routes') and instance._optimal_routes:
            try:
                optimal = Solution(instance, instance._optimal_routes).compute_objective()
            except Exception: pass
        
        gap = ((optimal - obj) / optimal * 100.0) if (optimal is not None and optimal > 0) else None
        
        results.append({
            'name': instance.name,
            'objective': obj,
            'optimal': optimal,
            'gap': gap,
            'time': solve_time,
            'visited': [int(c) for c in solution.get_visited_set() if c != 0],
            'n_customers': instance.n_customers,
            'routes': [[int(c) for c in r.customers if c != 0] for r in solution.routes],
        })
    
    objectives = [r['objective'] for r in results]
    gaps = [r['gap'] for r in results if r['gap'] is not None]
    return {
        'results': results,
        'avg_objective': np.mean(objectives),
        'median_objective': np.median(objectives),
        'std_objective': np.std(objectives),
        'min_objective': np.min(objectives),
        'max_objective': np.max(objectives),
        'avg_time': np.mean([r['time'] for r in results]),
        'avg_gap': np.mean(gaps) if gaps else None,
        'n_instances': len(instances)
    }

def run_evaluation(
    test_instances: List[Instance],
    model_path: str = None,
    output_path: str = None,
    temperature: float = 0.8,
    max_iterations: int = 200,
    checkpoint_iters: List[int] = None,
    intensive: bool = False
):
    print("=" * 60)
    print("MTRPP GNN Evaluation (vs PDF Bounds)")
    print("=" * 60)
    print(f"Test instances: {len(test_instances)}")
    print(f"Temperature: {temperature}")
    print(f"Max iterations: {max_iterations}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not test_instances:
        print("No test instances provided!")
        return None
    
    if not model_path or not os.path.exists(model_path):
        print(f"\n[Error] Model checkpoint NOT FOUND at: {model_path}")
        return None
        
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Breakthrough: Infer architecture from state_dict for backward compatibility
    node_dim = state_dict['node_encoder.0.weight'].size(1)
    hidden_dim = state_dict['node_encoder.0.weight'].size(0)
    num_layers = checkpoint.get('num_layers', CONFIG.get('num_layers', 6))
    heads = checkpoint.get('heads', CONFIG.get('num_heads', 8))
    
    # Detect if readout uses global context (hidden_dim * 2)
    readout_weight = state_dict['readout.0.weight']
    use_global_context = (readout_weight.size(1) == hidden_dim * 2)
    
    print(f"  Detected Architecture: NodeDim={node_dim}, HiddenDim={hidden_dim}, Layers={num_layers}, Heads={heads}, GlobalContext={use_global_context}")
    
    gnn_model = create_large_model(
        node_dim=node_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        heads=heads,
        use_global_context=use_global_context
    )
    
    # Load with strict=False to allow loading legacy models without ValueHead
    gnn_model.load_state_dict(state_dict, strict=False)
    gnn_model.to(device)
    gnn_model.eval()
    print("Model loaded successfully!")
    
    n_restarts = CONFIG.get('eval_restarts', 1)
    best_gnn_stats = None
    
    for restart_idx in range(n_restarts):
        if n_restarts > 1:
            print(f"\n  --- Restart {restart_idx + 1}/{n_restarts} ---")
        if restart_idx == 0:
            set_seed(CONFIG.get('seed', 42))
            
        current_stats = evaluate_on_instances(
            test_instances,
            gnn_model=gnn_model,
            max_iterations=max_iterations,
            temperature=temperature,
            use_stochastic=(restart_idx > 0),
            checkpoint_iters=checkpoint_iters,
            restart_idx=restart_idx,
            intensive=intensive
        )
        if best_gnn_stats is None or current_stats['avg_objective'] > best_gnn_stats['avg_objective']:
            best_gnn_stats = current_stats
            if n_restarts > 1:
                print(f"  New best restart (avg_obj={current_stats['avg_objective']:.2f})")
    
    gnn_stats = best_gnn_stats
    print(f"\n{'='*60}")
    print("GNN EVALUATION SUMMARY (vs PDF Optimal)")
    print(f"{'='*60}")
    print(f"  Avg Objective:    {gnn_stats['avg_objective']:.2f}")
    print(f"  Median Objective: {gnn_stats['median_objective']:.2f}")
    print(f"  Avg Time:         {gnn_stats['avg_time']:.2f}s")
    if gnn_stats['avg_gap'] is not None:
        print(f"  Avg Gap to Opt:   {gnn_stats['avg_gap']:.2f}%")
        
    if output_path:
        json_path = output_path
        os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    else:
        out_dir = os.path.dirname(model_path) if model_path else "checkpoints_rl"
        os.makedirs(out_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(out_dir, f'evaluation_results_t{temperature}_i{max_iterations}_{timestamp}.json')
    
    out_data = {
        'config': {
            'temperature': temperature,
            'max_iterations': max_iterations,
            'n_test_instances': len(test_instances)
        },
        'gnn_guided': {
            'summary': {
                'avg_objective': gnn_stats['avg_objective'],
                'median_objective': gnn_stats['median_objective'],
                'std_objective': gnn_stats.get('std_objective', 0.0),
                'avg_time': gnn_stats['avg_time'],
                'avg_gap': gnn_stats['avg_gap']
            },
            'instances': gnn_stats['results']
        }
    }
    if 'checkpoints' in gnn_stats:
        out_data['checkpoints'] = gnn_stats['checkpoints']
        
    import json
    with open(json_path, 'w') as f:
        json.dump(out_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n Evaluation results saved to {json_path}")
    return gnn_stats

def _perturb_solution(solution, intensity=0.15):
    """Randomly eject customers to shake up the solution structure."""
    import random
    new_sol = solution.copy()
    all_customers = []
    for r in new_sol.routes:
        all_customers.extend(r.customers)
    
    if not all_customers: return new_sol
    
    num_to_remove = max(1, int(len(all_customers) * intensity))
    removed = random.sample(all_customers, num_to_remove)
    
    for r in new_sol.routes:
        r.customers = [c for c in r.customers if c not in removed]
    
    new_sol._invalidate_visited_cache()
    return _greedy_extraction_pass(new_sol)

