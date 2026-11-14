# HNS-HH for the Multiple Traveling Repairman Problem with Profits (MTRPP)

This repository contains the implementation, experiment artifacts, analysis scripts, and manuscript sources for the **Hybrid Neuro-Symbolic Hyper-Heuristic (HNS-HH)** framework for the Multiple Traveling Repairman Problem with Profits (MTRPP).

The codebase includes:
- the HNS-HH solver and training/inference pipeline,
- scripts for evaluation, ablation/sensitivity experiments, and figure/table generation,
- recorded experimental results used in the manuscript, and
- the LaTeX manuscript source (`paper/MTRPP_Paper/`).

## Repository structure

- `mtrpp_local.py` — core solver/model/training and evaluation entry points.
- `scripts/` — experiment runners, analysis utilities, and figure-generation scripts.
- `dataset/` — dataset splits/preprocessed inputs used by the pipeline (large `.pt` files may be excluded by `.gitignore`).
- `experiments/`
  - `experiments/results/` — recorded JSON/CSV/TXT result artifacts and analysis outputs used for figures/tables.
  - `experiments/models/` — model checkpoints (ignored by default except as configured).
  - `experiments/logs/`, `experiments/route_plots/`, `experiments/results/plots/` — logs and generated visualizations.
- `paper/MTRPP_Paper/` — manuscript source (`sn-article.tex`), bibliography, figures, and compiled PDF.
- `run_training_only.*`, `run_analysis_only.*`, `run_experiments.*` — convenience scripts for staged/full runs.
- `requirements.txt` — Python dependencies.

## Setup

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure required dataset files/checkpoints are available in `dataset/` and `experiments/models/` as needed for your workflow.

> Note: large model checkpoints and dataset binaries may be omitted from version control via `.gitignore`.

## Usage

Typical entry points:

- **Training:** `run_training_only.bat` / `run_training_only.sh` (or invoke the training routine in `mtrpp_local.py`).
- **Analysis / evaluation:** `run_analysis_only.bat` / `run_analysis_only.sh`.
- **End-to-end:** `run_experiments.bat` / `run_experiments.sh`.
- **Figure/analysis regeneration:** scripts in `scripts/` (e.g. `comprehensive_analysis.py`, `comprehensive_analysis_part2.py`, `generate_*_plot.py`, `fix_fig12_heatmap.py`) consume recorded artifacts in `experiments/results/`.

Refer to script headers and command-line arguments for configuration details.

## Manuscript

- Source: `paper/MTRPP_Paper/sn-article.tex` (Springer Nature `sn-jnl`, `sn-mathphys-num`).
- Compile (from `paper/MTRPP_Paper/`): `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`.

## Data and code availability

- **Benchmark instances:** from the cited MTRPP literature benchmark sets.
- **Recorded results:** included under `experiments/results/` (and derived analysis plots) as used by the manuscript.
- **Code:** https://github.com/Gobi05-exe/HNS-HH

## Notes

- Experimental figures and tables in the manuscript are grounded in artifacts under `experiments/results/` and the corresponding generation scripts in `scripts/`.
- This repository is organized to keep core source, reproducibility scripts, recorded results, and manuscript materials; transient caches and large binaries are excluded where appropriate.
