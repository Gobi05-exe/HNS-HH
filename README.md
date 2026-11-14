# HNS-HH for the Multiple Traveling Repairman Problem with Profits (MTRPP)

This repository contains the source code and scripts for the **Hybrid Neuro-Symbolic Hyper-Heuristic (HNS-HH)** framework for the Multiple Traveling Repairman Problem with Profits (MTRPP).

## Contents

- **Core implementation:** `mtrpp_local.py` — solver, model, training, and evaluation entry points.
- **Experiment and analysis scripts:** `scripts/` — evaluation, ablation/sensitivity experiments, statistical analysis, and figure/summary generation utilities.
- **Dependencies:** `requirements.txt`.

## Repository layout

```text
.
├─ mtrpp_local.py
├─ requirements.txt
└─ scripts/
```

> **Note:** Manuscript sources, compiled paper artifacts, datasets, model checkpoints, experiment logs/results/route plots, and reference materials are not tracked in this repository (see `.gitignore`). Running the pipeline/scripts will generate required outputs locally (e.g. under `dataset/`, `experiments/models/`, `experiments/results/`, `experiments/logs/`, `experiments/route_plots/`) as configured.

## Setup

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide or generate the required dataset files and (optionally) model checkpoints in the expected local paths before running training, evaluation, or analysis workflows.

## Usage

Typical workflows:

- **Training / evaluation:** use the training and evaluation entry points in `mtrpp_local.py`.
- **Individual workflows:** scripts in `scripts/` (e.g. `run_evaluation.py`, `ablation_study.py`, `parameter_sensitivity.py`, `scalability_analysis.py`, `statistical_tests.py`, `comprehensive_analysis.py`, `comprehensive_analysis_part2.py`).

Refer to script headers and command-line arguments for configuration and expected inputs/outputs.

## Data and code availability

- **Benchmark instances:** from the cited MTRPP literature benchmark sets.
- **Code:** https://github.com/Gobi05-exe/HNS-HH
- **Generated artifacts:** datasets, checkpoints, logs, raw/processed results, and figures are produced locally by the provided scripts and are excluded from version control.
