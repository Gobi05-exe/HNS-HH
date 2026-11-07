# MTRPP High-Performance Experimental Suite

This directory contains the fully automated, decoupled experimental pipeline for the GNN-guided ALNS Solver.

## Deployment Instructions

**Step 1:** Copy this entire `MTRPP_Experiments` folder to your target high-performance Windows system.
**Step 2:** Ensure your Python virtual environment is set up (the batch scripts assume a `venv` exists inside this directory. You can create it using `python -m venv venv` and install requirements).

## 🏃 Execution 

You can execute the pipeline in stages or all at once:

1. **`run_training_only.bat`**: 
   Starts the 48-hour overnight training phase from scratch. It aggressively tracks validation loss and incorporates early stopping if the GNN proves it can beat the ground-truth benchmark consistently.
   
2. **`run_analysis_only.bat`**: 
   Executes the exhaustive 8-phase reporting suite using the best checkpoint. It exports deep JSON structural data (containing exact routes) alongside statistical `.csv` files for:
   - Benchmark vs Avci/Lu heuristics
   - Parameter sensitivity (MSHGSM parameters like $T_{itr}$)
   - Ablation studies (stripping out components one by one)
   - Scalability checks
   - Non-parametric statistical tests

3. **`run_experiments.bat`**:
   Runs Phase 1 and Phase 2 back-to-back automatically.
