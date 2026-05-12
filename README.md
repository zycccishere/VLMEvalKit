## Image Replay Evaluation Fork

This branch is a cleaned, publishable VLMEvalKit fork for the image-replay experiments. It is based on `open-compass/VLMEvalKit` commit `00804217f868058f871f5ff252a7b9623c3475d9` and keeps the active replay evaluation surface used for Qwen2.5-VL, MiniCPM-4.5, and Gemma3-family experiments.

Runtime artifacts are intentionally not included: no `runs/`, `checkpoints/`, model weights, `LMUData/`, private API keys, or machine-local caches are tracked. Historical launchers are kept under `scripts_legacy/` for provenance; the maintained entrypoint is `scripts/run_benchmark.sh`.

### Quick Start

Copy `.env.example` to `.env` or export the variables in your shell:

```bash
export MODEL_ROOT=/models
export CONDA_ROOT=$HOME/miniconda3
export LMUData=$PWD/LMUData
export OPENAI_API_KEY_JUDGE=
export OPENAI_API_BASE_JUDGE=https://api.openai.com/v1
export OPENAI_COMPATIBLE_API_KEY=
export OPENAI_COMPATIBLE_API_BASE=
```

Run a dry plan before launching work:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config scripts/configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml \
  --model-config scripts/configs/models.yaml \
  --scheduler gpu_pool \
  --plan-only
```

### Active Entrypoints

- `bash scripts/run_benchmark.sh ...`: canonical matrix runner. It reads `scripts/configs/models.yaml` plus a matrix YAML, supports `--resume-infer`, and has `--scheduler gpu_pool` for mixed tensor-parallel packing.
- `bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh <node_rank> [gpu_ids]`: two-node Qwen2.5-VL plus MiniCPM-4.5 run over the four newer reasoning/perception benchmarks.
- `bash scripts/run_gemma3_family_all11_replay6_2nodes_20260422.sh <node_rank> [gpu_ids]`: two-node Gemma3-family run over the 11-benchmark replay matrix.
- `bash scripts/ssh_launch_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh [host0 host1]`: tmux launcher for the Qwen2.5/MiniCPM two-node entrypoint. Set `REMOTE_REPO` to the remote clone path.
- `bash scripts/ssh_launch_gemma3_family_all11_replay6_2nodes_20260422.sh [host0 host1]`: tmux launcher for the Gemma3 two-node entrypoint. Set `REMOTE_REPO` to the remote clone path.
- `bash scripts/run_legacy_dynamath_infer_only_all_replay.sh`: preserved legacy Dynamath infer-only entrypoint used by older table rows.

The `gpu_pool` scheduler is the expected scheduler for mixed-size model queues. It can keep an 8-GPU node filled with combinations such as one `tp=4` task plus two `tp=2` tasks when the matrix contains both 72B and 32B jobs.

### Final Table Reproduction

The public release surface has two result-relevant execution stacks. In practice, the score-relevant new/legacy differences are judge/evaluator choice, dataset/cache lineage, and a few runtime knobs rather than scheduler shape. The detailed source-group and per-cell audit remains in [`docs/FINAL_TABLE_REPRODUCTION.md`](docs/FINAL_TABLE_REPRODUCTION.md), [`docs/final_table_reproduction_entries.csv`](docs/final_table_reproduction_entries.csv), and [`docs/final_table_cell_sources.csv`](docs/final_table_cell_sources.csv).

Run the new stack with the maintained matrix runner:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config <matrix.yaml> \
  --model-config scripts/configs/models.yaml \
  --scheduler gpu_pool \
  --plan-only

bash scripts/run_benchmark.sh \
  --matrix-config <matrix.yaml> \
  --model-config scripts/configs/models.yaml \
  --scheduler gpu_pool
```

For the main two-node new matrices, launch one command per node:

```bash
bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh 0 0,1,2,3,4,5,6,7
bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh 1 0,1,2,3,4,5,6,7

bash scripts/run_gemma3_family_all11_replay6_2nodes_20260422.sh 0 0,1,2,3,4,5,6,7
bash scripts/run_gemma3_family_all11_replay6_2nodes_20260422.sh 1 0,1,2,3,4,5,6,7
```

Run the legacy stack with the preserved historical launchers:

```bash
bash scripts_legacy/<legacy_entrypoint>.sh
bash scripts/run_legacy_dynamath_infer_only_all_replay.sh
```

Use the source-stack table below to choose `New` or `Legacy` at the model-by-dataset level. It uses the majority stack across replay settings for each model and dataset; use the detailed CSVs only when exact per-cell provenance is needed.

### Final Table

The table below is copied from `assets/topics/topic-image-replay/resources/final-tables/final_table.csv`.

| Model | Setting | DynaMath | MathVision | LogicVista | VisualPuzzles | MMMU single image | We-Math | Reasoning Avg | ΔReasoning | AI2D_TEST | OCRBench | SEEDBench2_Plus | MMBench_DEV_EN_V11 | MMStar | Perception Avg | ΔPerception | ALL | ΔALL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 72B | Public | 61.2 | 39.3 | 55.7 | 42.3 |  |  | 49.63 |  | 88.7 | 88.5 | 73 |  |  |  |  |  |  |
| Qwen2.5 72B | I-Q | 66.71 | 40 | 55.93 | 43.32 | 67.68 | 49.43 | 53.85 |  | 88.63 | 88.6 | 73.69 | 88.16 | 71 | 82.02 | 0 | 66.65 |  |
| Qwen2.5 72B | Q-I | 65.51 | 40 | 54.59 | 45.55 | 66.28 | 48.95 | 53.48 | -0.36 | 82.51 | 87.8 | 69.78 | 82.59 | 61.33 | 76.8 | -5.21 | 64.08 | -2.57 |
| Qwen2.5 72B | I-I-Q | 66.03 | 39.84 | 54.81 | 44.95 | 66.51 | 48.95 | 53.52 | -0.33 | 89.15 | 89.4 | 73.52 | 87.77 | 69.73 | 81.91 | -0.1 | 66.42 | -0.23 |
| Qwen2.5 72B | I-Q-I | 68.48 | 40.79 | 57.27 | 44.01 | 68.84 | 52.1 | 55.25 | 1.4 | 88.73 | 89.4 | 73.39 | 88.62 | 71.53 | 82.33 | 0.32 | 67.56 | 0.91 |
| Qwen2.5 72B | I-Q-Q | 66.23 | 40.16 | 57.05 | 43.49 | 67.44 | 51.05 | 54.24 | 0.39 | 88.76 | 89.4 | 72.9 | 88.31 | 70.53 | 81.98 | -0.04 | 66.85 | 0.2 |
| Qwen2.5 72B | I-Q-I-Q | 67.96 | 41.78 | 60.18 | 45.38 | 66.74 | 51.14 | 55.53 | 1.69 | 88.76 | 89.1 | 73.74 | 89.71 | 71.53 | 82.57 | 0.55 | 67.82 | 1.17 |
| Qwen2.5 32B | Public | 55.5 | 37.8 | 55 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Qwen2.5 32B | I-Q | 57.5 | 38.65 | 54.36 | 38.36 | 65.81 | 48 | 50.45 |  | 83.58 | 85.5 | 72.6 | 85.53 | 66.8 | 78.8 | 0 | 63.34 |  |
| Qwen2.5 32B | Q-I | 55.55 | 38.65 | 56.82 | 41.01 | 63.36 | 43 | 49.72 | -0.72 | 80.67 | 81.8 | 70.58 | 79.33 | 62.13 | 74.9 | -3.9 | 61.17 | -2.17 |
| Qwen2.5 32B | I-I-Q | 58.5 | 38.62 | 56.6 | 39.81 | 65.81 | 50 | 51.56 | 1.11 | 85.82 | 85.6 | 72.86 | 84.29 | 67.8 | 79.27 | 0.47 | 64.16 | 0.82 |
| Qwen2.5 32B | I-Q-I | 60.24 | 39.51 | 58.84 | 41.52 | 64.99 | 50.5 | 52.6 | 2.15 | 84.65 | 84.2 | 72.86 | 85.68 | 67.33 | 78.94 | 0.14 | 64.57 | 1.24 |
| Qwen2.5 32B | I-Q-Q | 59.96 | 38.82 | 54.81 | 40.15 | 66.86 | 47.3 | 51.32 | 0.88 | 83.71 | 83.9 | 72.68 | 85.29 | 66.33 | 78.38 | -0.42 | 63.62 | 0.29 |
| Qwen2.5 32B | I-Q-I-Q | 61.5 | 40 | 57.72 | 41.95 | 67.68 | 49.4 | 53.05 | 2.6 | 84.42 | 85.8 | 71.81 | 86.07 | 67.73 | 79.17 | 0.36 | 64.92 | 1.58 |
| Qwen2.5 7B | Public | 50.7 | 25.8 | 46.1 | 33.7 | 58.6 | 36.2 | 41.85 |  | 83.9 | 86.4 | 70.4 | 82.6 | 63.9 |  |  |  |  |
| Qwen2.5 7B | I-Q | 53.89 | 25.3 | 44.74 | 32.88 | 54.26 | 36.6 | 41.27 |  | 85.07 | 88.3 | 70.88 | 83.2 | 63.93 | 78.28 | 0 | 58.09 |  |
| Qwen2.5 7B | Q-I | 52.16 | 28.49 | 43.62 | 32.62 | 50.53 | 32.4 | 39.97 | -1.31 | 74 | 85.6 | 62.23 | 71.52 | 54.8 | 69.63 | -8.65 | 53.45 | -4.64 |
| Qwen2.5 7B | I-I-Q | 52.5 | 24.8 | 46.98 | 31.16 | 52.86 | 33.5 | 40.3 | -0.97 | 84.59 | 88.2 | 70.75 | 83.98 | 63.93 | 78.29 | 0.01 | 57.57 | -0.52 |
| Qwen2.5 7B | I-Q-I | 54.23 | 26.97 | 47.87 | 33.56 | 52.98 | 37.6 | 42.21 | 0.93 | 85.17 | 88.1 | 71.19 | 83.59 | 63.67 | 78.34 | 0.07 | 58.63 | 0.54 |
| Qwen2.5 7B | I-Q-Q | 53.91 | 27.07 | 46.76 | 32.79 | 54.84 | 36.1 | 41.91 | 0.64 | 84.29 | 88.2 | 69.74 | 82.66 | 62.67 | 77.51 | -0.76 | 58.09 | 0 |
| Qwen2.5 7B | I-Q-I-Q | 53.25 | 26.02 | 45.64 | 32.79 | 54.14 | 37.6 | 41.58 | 0.3 | 84.59 | 88.3 | 70.71 | 82.43 | 61.47 | 77.5 | -0.78 | 57.91 | -0.19 |
| Qwen2.5 3B | Public | ? | 21.9 | 39.6 | ? |  |  | 30.75 |  | 81.6 | 79.7 | 67.6 |  |  |  |  |  |  |
| Qwen2.5 3B | I-Q | 43.39 | 22.24 | 37.58 | 28.51 | 48.77 | 21.4 | 33.65 |  | 81.48 | 82.3 | 68.47 | 77.09 | 55.67 | 73 | 0 | 51.54 |  |
| Qwen2.5 3B | Q-I | 39.96 | 22.24 | 39.15 | 28.17 | 48.42 | 22.3 | 33.37 | -0.28 | 66.55 | 77.9 | 56.08 | 61.46 | 41 | 60.6 | -12.4 | 45.75 | -5.79 |
| Qwen2.5 3B | I-I-Q | 42.99 | 21.58 | 40.27 | 29.02 | 49.24 | 23.3 | 34.41 | 0.75 | 82.06 | 83 | 68.95 | 76.86 | 55.47 | 73.27 | 0.27 | 52.07 | 0.53 |
| Qwen2.5 3B | I-Q-I | 43.93 | 20.03 | 38.93 | 28.42 | 50.41 | 24.57 | 34.38 | 0.73 | 81.93 | 82.5 | 68.42 | 77.01 | 55.47 | 73.07 | 0.06 | 51.97 | 0.43 |
| Qwen2.5 3B | I-Q-Q | 42.24 | 19.9 | 40.04 | 29.02 | 50.41 | 23.71 | 34.22 | 0.57 | 81.7 | 82.7 | 68.38 | 76.16 | 54.13 | 72.61 | -0.39 | 51.67 | 0.13 |
| Qwen2.5 3B | I-Q-I-Q | 42.89 | 21 | 41.83 | 28.77 | 50.99 | 24.95 | 35.07 | 1.42 | 81.77 | 82.9 | 68.86 | 77.09 | 55.6 | 73.24 | 0.24 | 52.42 | 0.88 |
| MiniCPM-V | I-Q | 64.15 | 40.95 | 55.93 | 20.98 | 67.09 | 49.9 | 49.83 |  | 86.56 | 82.4 | 69.21 | 86.22 | 70.47 | 78.97 | 0 | 63.08 |  |
| MiniCPM-V | Q-I | 64.61 | 40.33 | 55.93 | 21.23 | 67.56 | 51.43 | 50.18 | 0.35 | 86.63 | 82.1 | 69.35 | 86.38 | 70.73 | 79.04 | 0.07 | 63.3 | 0.22 |
| MiniCPM-V | I-I-Q | 64.79 | 41.35 | 55.26 | 21.92 | 67.09 | 49.62 | 50 | 0.17 | 86.63 | 82.5 | 69.35 | 85.99 | 72.13 | 79.32 | 0.35 | 63.33 | 0.25 |
| MiniCPM-V | I-Q-I | 64.35 | 40.62 | 55.93 | 21.15 | 67.33 | 50.95 | 50.05 | 0.22 | 86.66 | 82.3 | 69.3 | 85.91 | 70.53 | 78.94 | -0.03 | 63.18 | 0.11 |
| MiniCPM-V | I-Q-Q | 64.35 | 39.67 | 56.82 | 20.72 | 67.44 | 49.14 | 49.69 | -0.14 | 86.72 | 81.9 | 69.26 | 86.15 | 70.8 | 78.97 | -0.01 | 63 | -0.08 |
| MiniCPM-V | I-Q-I-Q | 63.73 | 40.59 | 56.15 | 21.83 | 67.44 | 49.43 | 49.86 | 0.03 | 86.59 | 81.6 | 69.26 | 86.61 | 71 | 79.01 | 0.04 | 63.11 | 0.03 |
| MiniCPM-o | I-Q | 63.27 | 45.76 | 53.02 | 20.98 | 69.78 | 61.14 | 52.32 |  | 82.45 | 84.5 | 65.26 | 85.29 | 70.27 | 77.55 | 0 | 63.79 |  |
| MiniCPM-o | Q-I | 63.27 | 45.33 | 53.69 | 19.43 | 68.73 | 59.9 | 51.73 | -0.6 | 82.45 | 84.9 | 65.39 | 85.45 | 69.67 | 77.57 | 0.02 | 63.47 | -0.32 |
| MiniCPM-o | I-I-Q | 63.21 | 44.84 | 52.13 | 20.72 | 71.18 | 61.14 | 52.2 | -0.12 | 82.51 | 85 | 65.48 | 85.68 | 68.8 | 77.49 | -0.06 | 63.7 | -0.09 |
| MiniCPM-o | I-Q-I | 63.59 | 45.53 | 53.91 | 20.38 | 68.49 | 59.05 | 51.82 | -0.5 | 82.48 | 84.9 | 65.48 | 85.37 | 69.6 | 77.57 | 0.01 | 63.53 | -0.27 |
| MiniCPM-o | I-Q-Q | 63.33 | 44.9 | 52.57 | 20.63 | 70.01 | 61.14 | 52.1 | -0.23 | 82.45 | 84.3 | 65.35 | 85.53 | 69.73 | 77.47 | -0.08 | 63.63 | -0.16 |
| MiniCPM-o | I-Q-I-Q | 63.77 | 45.2 | 52.8 | 21.83 | 70.95 | 62.19 | 52.79 | 0.47 | 82.55 | 85.1 | 65.48 | 85.45 | 69.13 | 77.54 | -0.01 | 64.04 | 0.25 |
| Gemma3 27B | Public |  | 32.4 |  |  | 64.9 | 37.9 |  |  | 84.5 |  |  | 77.2 | 61.7 |  |  |  |  |
| Gemma3 27B | I-Q | 50.62 | 36.74 | 47.2 | 15.24 | 57.64 | 42 | 41.57 |  | 83.32 | 75.3 | 70.05 | 79.49 | 56.47 | 72.93 | 0 | 55.82 |  |
| Gemma3 27B | Q-I | 42.77 | 38.16 | 38.26 | 17.21 | 54.49 | 40.19 | 38.51 | -3.06 | 76 | 73.7 | 64.56 | 72.06 | 47.87 | 66.84 | -6.09 | 51.39 | -4.44 |
| Gemma3 27B | I-I-Q | 49.64 | 35.63 | 46.53 | 17.29 | 56.71 | 43.14 | 41.49 | -0.08 | 83.65 | 75.8 | 70 | 79.72 | 58.07 | 73.45 | 0.52 | 56.02 | 0.19 |
| Gemma3 27B | I-Q-I | 48.14 | 36.38 | 46.31 | 18.07 | 57.29 | 39.33 | 40.92 | -0.65 | 83.78 | 74.6 | 70.62 | 80.57 | 58.27 | 73.57 | 0.64 | 55.76 | -0.06 |
| Gemma3 27B | I-Q-Q | 47.17 | 36.74 | 48.99 | 17.89 | 57.99 | 45.71 | 42.42 | 0.84 | 84.46 | 75.2 | 69.7 | 80.42 | 56.8 | 73.32 | 0.39 | 56.46 | 0.64 |
| Gemma3 27B | I-Q-I-Q | 48.08 | 37.17 | 45.86 | 16.35 | 59.04 | 46.29 | 42.13 | 0.56 | 84.55 | 74.6 | 70.18 | 81.19 | 59.2 | 73.94 | 1.02 | 56.59 | 0.77 |
| Gemma3 12B | Public |  | 28.1 |  |  | 59.6 | 33.6 |  |  | 84.2 |  |  | 71.8 | 56.1 |  |  |  |  |
| Gemma3 12B | I-Q | 30.76 | 31.15 | 36.91 | 16.44 | 51.58 | 35.33 | 33.7 |  | 80.25 | 72.9 | 67.15 | 75.23 | 54.73 | 70.05 | 0 | 50.22 |  |
| Gemma3 12B | Q-I | 32.99 | 32.53 | 34.45 | 17.64 | 50.18 | 26.67 | 32.41 | -1.28 | 74.77 | 70.1 | 62.36 | 73.53 | 47.8 | 65.71 | -4.34 | 47.55 | -2.67 |
| Gemma3 12B | I-I-Q | 32.32 | 31.84 | 40.27 | 17.55 | 50.88 | 35.62 | 34.75 | 1.05 | 81.25 | 72 | 67.28 | 75.54 | 55.53 | 70.32 | 0.27 | 50.92 | 0.7 |
| Gemma3 12B | I-Q-I | 36.85 | 32.37 | 37.58 | 16.95 | 52.16 | 31.62 | 34.59 | 0.89 | 81.02 | 70.7 | 68.07 | 78.02 | 55.73 | 70.71 | 0.66 | 51.01 | 0.79 |
| Gemma3 12B | I-Q-Q | 35.05 | 32.04 | 35.79 | 17.12 | 54.03 | 37.43 | 35.24 | 1.55 | 82.12 | 72.7 | 67.28 | 76.78 | 55.13 | 70.8 | 0.75 | 51.41 | 1.19 |
| Gemma3 12B | I-Q-I-Q | 37.43 | 31.38 | 37.14 | 16.18 | 52.98 | 35.9 | 35.17 | 1.47 | 82.22 | 71.1 | 67.41 | 79.18 | 56.8 | 71.34 | 1.29 | 51.61 | 1.39 |
| Gemma3 4B | Public |  |  |  |  | 48.8 | 26.7 |  |  | 74.8 |  |  | 67.6 | 46.1 |  |  |  |  |
| Gemma3 4B | I-Q | 25.85 | 22.2 | 30.65 | 13.78 | 42.01 | 25.71 | 26.7 |  | 73.54 | 67.9 | 61.48 | 66.1 | 45.87 | 62.98 | 0 | 43.19 |  |
| Gemma3 4B | Q-I | 23.49 | 22.37 | 26.85 | 13.78 | 44.11 | 27.52 | 26.35 | -0.35 | 64.09 | 63.9 | 55.29 | 63.24 | 41.6 | 57.62 | -5.35 | 40.57 | -2.62 |
| Gemma3 4B | I-I-Q | 25.99 | 22.99 | 27.74 | 14.47 | 42.24 | 27.43 | 26.81 | 0.11 | 73.54 | 68.2 | 61.79 | 65.4 | 45.93 | 62.97 | -0.01 | 43.25 | 0.06 |
| Gemma3 4B | I-Q-I | 23.19 | 21.97 | 27.74 | 14.73 | 42.82 | 28.48 | 26.49 | -0.21 | 75.1 | 66.5 | 61.97 | 68.65 | 46.4 | 63.72 | 0.75 | 43.41 | 0.22 |
| Gemma3 4B | I-Q-Q | 22.12 | 22.7 | 29.53 | 14.81 | 41.77 | 29.24 | 26.7 | 0 | 73.7 | 66.2 | 61.88 | 67.96 | 45.33 | 63.01 | 0.04 | 43.2 | 0.01 |
| Gemma3 4B | I-Q-I-Q | 22.61 | 22.96 | 27.29 | 16.01 | 41.89 | 29.71 | 26.75 | 0.05 | 74.71 | 65.6 | 61.48 | 69.12 | 46.33 | 63.45 | 0.47 | 43.43 | 0.24 |

### Final Table Source Stack

This model-by-dataset map uses the majority stack across replay settings for that model and dataset. Full per-cell provenance remains in `docs/final_table_cell_sources.csv`.

| Model | DynaMath | MathVision | LogicVista | VisualPuzzles | MMMU single image | We-Math | AI2D_TEST | OCRBench | SEEDBench2_Plus | MMBench_DEV_EN_V11 | MMStar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 72B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| Qwen2.5 32B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| Qwen2.5 7B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| Qwen2.5 3B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| MiniCPM-V | New | New | New | New | New | New | New | New | New | New | New |
| MiniCPM-o | New | New | New | New | New | New | New | New | New | New | New |
| Gemma3 27B | New | New | New | New | New | New | New | New | New | New | New |
| Gemma3 12B | New | New | New | New | New | New | New | New | New | New | New |
| Gemma3 4B | New | New | New | New | New | New | New | New | New | New | New |

### Main Matrices

- `scripts/configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml`: Qwen2.5-VL 3B/7B/32B/72B plus MiniCPM-V/O-4.5 on `MMMU_DEV_VAL_SINGLE_IMAGE`, `WeMath`, `MMBench_DEV_EN_V11`, and `MMStar`.
- `scripts/configs/matrix_gemma3_family_all11_replay6_2node_20260422.yaml`: Gemma3 family on `DynaMath`, `MathVision`, `LogicVista`, `VisualPuzzles`, `AI2D_TEST`, `OCRBench`, `SEEDBench2_Plus`, and the four newer benchmarks.
- `scripts/configs/matrix_qwen25vl_all4_reasoning_perception4_new_entry_20260421.yaml`: earlier Qwen2.5-only four-new-benchmark matrix.
- `scripts/configs/matrix_qwen25vl7b_minicpm45_table_20260406.yaml`: preserved legacy table matrix for Qwen2.5-7B and MiniCPM-4.5.
- `scripts/configs/matrix_legacy_dynamath_infer_only_all_replay.yaml`: preserved legacy Dynamath infer-only matrix.
- `scripts/configs/matrix_minicpm45_wemath_cot_rerun_20260429.yaml`: MiniCPM-4.5 WeMath CoT rerun matrix.
- `scripts/configs/matrix_final_table_legacy_backfill_20260512.yaml` plus `scripts/configs/task_manifests/final_table_legacy_backfill_20260512/all_tasks.csv`: exact backfill manifest for the 21 legacy standard-run cells needed to cover every non-public, non-derived `final_table.csv` benchmark cell with the release runner.

### Code Map

- Replay policy and transform routing: `vlmeval/vlm/replay_policy.py`, `vlmeval/vlm/replay_image_transform.py`.
- Qwen2/Qwen2.5 replay wrapper: `vlmeval/vlm/qwen2_vl/model.py` and `vlmeval/vlm/qwen2_vl/replay_prompt_template.py`.
- MiniCPM-4.5 replay wrapper: `vlmeval/vlm/minicpm_v_4_5_replay.py`.
- Gemma3/Gemma4 replay wrappers: `vlmeval/vlm/gemma3_replay.py`, `vlmeval/vlm/gemma4_replay.py`.
- Minimal model registries: `vlmeval/config_qwen_minimal.py`, `vlmeval/config_minicpm45_minimal.py`, `vlmeval/config_gemma3_minimal.py`, `vlmeval/config_gemma4_minimal.py`.
- Dataset/eval adaptations: `vlmeval/dataset/dynamath.py`, `vlmeval/dataset/image_mcq.py`, `vlmeval/dataset/image_vqa.py`, and `vlmeval/dataset/utils/{logicvista,mathv,visualpuzzles,wemath}.py`.
- Matrix orchestration and result collection: `scripts/run_benchmark.py`, `scripts/collect_matrix_results.py`, and `scripts/configs/`.

### Reproducibility Notes

- Active image-replay model and data locations are controlled by environment variables, not hardcoded private paths. Set `MODEL_ROOT`, `CONDA_ROOT`, `LMUData`, `LLAVA_ROOT`, and `QWEN35_PYDEPS` as needed.
- LLM-judge based datasets use `OPENAI_API_KEY_JUDGE`/`OPENAI_API_BASE_JUDGE` first, then `OPENAI_API_KEY`/`OPENAI_API_BASE`; generic non-OpenAI-compatible endpoints can be supplied with `OPENAI_COMPATIBLE_API_KEY`/`OPENAI_COMPATIBLE_API_BASE`.
- Non-default API wrappers such as `JTVLChatAPI` have no bundled private endpoint. Configure their endpoint and token explicitly with the documented environment variables in the wrapper before use.
- The release matrices plus `matrix_final_table_legacy_backfill_20260512` cover every non-public, non-derived benchmark cell in `final_table.csv`. Public/reference rows are external report numbers, and `MMStar no-reason only` is a derived subset summary rather than a separate VLMEvalKit run.
- Exact numeric reproduction still depends on matching model checkpoints, dataset/cache versions, evaluator dependencies, and the LLM judge endpoint/model used for judge-scored datasets.
- `scripts_legacy/` is intentionally retained for old final-table provenance. Treat these scripts as historical launch references rather than the preferred active interface.

---

![LOGO](https://opencompass.openxlab.space/utils/MMLB.jpg)

<b>A Toolkit for Evaluating Large Vision-Language Models. </b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

English | [简体中文](/docs/zh-CN/README_zh-CN.md) | [日本語](/docs/ja/README_ja.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OC Learderboard </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF Leaderboard</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 Evaluation Records</a> •
<a href="https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard">🤗 HF Video Leaderboard</a> •

<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 Report</a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>
</div>

**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-based answer extraction**.

## Recent Codebase Changes
- **[2025-09-12]** **Major Update: Improved Handling for Models with Thinking Mode**

    A new feature in [PR 1229](https://github.com/open-compass/VLMEvalKit/pull/1175) that improves support for models with thinking mode. VLMEvalKit now allows for the use of a custom `split_thinking` function. **We strongly recommend this for models with thinking mode to ensure the accuracy of evaluation**.  To use this new functionality, please enable the Environment Variable: `SPLIT_THINK=True`. By default, the function will parse content within `<think>...</think>` tags and store it in the `thinking` key of the output. For more advanced customization, you can also create a `split_think` function for model. Please see the InternVL implementation for an example.
- **[2025-09-12]** **Major Update: Improved Handling for Long Response(More than 16k/32k)**

    A new feature in [PR 1229](https://github.com/open-compass/VLMEvalKit/pull/1175) that improves support for models with long response outputs. VLMEvalKit can now save prediction files in TSV format. **Since individual cells in an `.xlsx` file are limited to 32,767 characters, we strongly recommend using this feature for models that generate long responses (e.g., exceeding 16k or 32k tokens) to prevent data truncation.** To use this new functionality, please enable the Environment Variable: `PRED_FORMAT=tsv`.
- **[2025-08-04]** In [PR 1175](https://github.com/open-compass/VLMEvalKit/pull/1175), we refine the `can_infer_option` and `can_infer_text`, which increasingly route the evaluation to LLM choice extractors and empirically leads to slight performance improvement for MCQ benchmarks.

## 🆕 News
- **[2025-07-07]** Supported [**SeePhys**](https://seephys.github.io/), which is a ​full spectrum multimodal benchmark for evaluating physics reasoning across different knowledge levels. thanks to [**Quinn777**](https://github.com/Quinn777) 🔥🔥🔥
- **[2025-07-02]** Supported [**OvisU1**](https://huggingface.co/AIDC-AI/Ovis-U1-3B), thanks to [**liyang-7**](https://github.com/liyang-7) 🔥🔥🔥
- **[2025-06-16]** Supported [**PhyX**](https://phyx-bench.github.io/), a benchmark aiming to assess capacity for physics-grounded reasoning in visual scenarios. 🔥🔥🔥
- **[2025-05-24]** To facilitate faster evaluations for large-scale or thinking models, **VLMEvalKit supports multi-node distributed inference** using **LMDeploy**  (supports *InternVL Series, QwenVL Series, LLaMa4*) or **VLLM**(supports *QwenVL Series, LLaMa4*). You can activate this feature by adding the ```use_lmdeploy``` or ```use_vllm``` flag to your custom model configuration in [config.py](vlmeval/config.py) . Leverage these tools to significantly speed up your evaluation workflows 🔥🔥🔥
- **[2025-05-24]** Supported Models: **InternVL3 Series, Gemini-2.5-Pro, Kimi-VL, LLaMA4, NVILA, Qwen2.5-Omni, Phi4, SmolVLM2, Grok, SAIL-VL-1.5, WeThink-Qwen2.5VL-7B, Bailingmm, VLM-R1, Taichu-VLR**. Supported Benchmarks: **HLE-Bench, MMVP, MM-AlignBench, Creation-MMBench, MM-IFEval, OmniDocBench, OCR-Reasoning, EMMA, ChaXiv，MedXpertQA, Physics, MSEarthMCQ, MicroBench, MMSci, VGRP-Bench, wildDoc, TDBench, VisuLogic, CVBench, LEGO-Puzzles, Video-MMLU, QBench-Video, MME-CoT, VLM2Bench, VMCBench, MOAT, Spatial457 Benchmark**. Please refer to [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) for more details. Thanks to all contributors 🔥🔥🔥
- **[2025-02-20]** Supported Models: **InternVL2.5 Series, Qwen2.5VL Series, QVQ-72B, Doubao-VL, Janus-Pro-7B, MiniCPM-o-2.6, InternVL2-MPO, LLaVA-CoT, Hunyuan-Standard-Vision, Ovis2, Valley, SAIL-VL, Ross, Long-VITA, EMU3, SmolVLM**. Supported Benchmarks: **MMMU-Pro, WeMath, 3DSRBench, LogicVista, VL-RewardBench, CC-OCR, CG-Bench, CMMMU, WorldSense**. Thanks to all contributors 🔥🔥🔥
- **[2024-12-11]** Supported [**NaturalBench**](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- **[2024-12-02]** Supported [**VisOnlyQA**](https://github.com/psunlpgroup/VisOnlyQA/), a benchmark for evaluating the visual perception capabilities 🔥🔥🔥
- **[2024-11-26]** Supported [**Ovis1.6-Gemma2-27B**](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-27B), thanks to [**runninglsy**](https://github.com/runninglsy) 🔥🔥🔥
- **[2024-11-25]** Create a new flag `VLMEVALKIT_USE_MODELSCOPE`. By setting this environment variable, you can download the video benchmarks supported from [**modelscope**](https://www.modelscope.cn) 🔥🔥🔥

## 🏗️ QuickStart

See [[QuickStart](/docs/en/Quickstart.md) | [快速开始](/docs/zh-CN/Quickstart.md)] for a quick start guide.

## 📊 Datasets, Models, and Evaluation Results

### Evaluation Results

**The performance numbers on our official multi-modal leaderboards can be downloaded from here!**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [**Download All DETAILED Results**](http://opencompass.openxlab.space/assets/OpenVLM.json).

Check **Supported Benchmarks** Tab in [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) to view all supported image & video benchmarks (70+).

Check **Supported LMMs** Tab in [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) to view all supported LMMs, including commercial APIs, open-source models, and more (200+).

**Transformers Version Recommendation:**

Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`.
- **Please use** `transformers==4.36.2` **for**: `Moondream1`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`, `VILA Series`, `Llama-3-MixSenseV1_1`, `Parrot-7B`, `PLLaVA Series`.
- **Please use** `transformers==4.40.0` **for**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **Please use** `transformers==4.42.0` **for**: `AKI`.
- **Please use** `transformers==4.44.0` **for**: `Moondream2`, `H2OVL series`.
- **Please use** `transformers==4.45.0` **for**: `Aria`.
- **Please use** `transformers==4.48.0` (or `4.46.0`) **for**: `LLaVA-Next series` (e.g., `llava-hf/llava-v1.6-vicuna-7b-hf`).
- **Please use** `transformers==latest` **for**: `PaliGemma-3B`, `Chameleon series`, `Video-LLaVA-7B-HF`, `Ovis series`, `Mantis series`, `MiniCPM-V2.6`, `OmChat-v2.0-13B-sinlge-beta`, `Idefics-3`, `GLM-4v-9B`, `VideoChat2-HD`, `RBDash_72b`, `Llama-3.2 series`, `Kosmos series`.
- **Please use** `transformers==4.50.3` (or `4.46.1` or `4.51` or `4.53`) **for**: `Molmo series`.
- **Please use** `transformers>=5.2.0` **for**: `Qwen3.5 series`.

**Torchvision Version Recommendation:**

Note that some VLMs may not be able to run under certain torchvision versions, we recommend the following settings to evaluate each VLM:

- **Please use** `torchvision>=0.16` **for**: `Moondream series` and `Aria`

**Flash-attn Version Recommendation:**

Note that some VLMs may not be able to run under certain flash-attention versions, we recommend the following settings to evaluate each VLM:

- **Please use** `pip install flash-attn --no-build-isolation` **for**: `Aria`

```python
# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.
```

## 🛠️ Development Guide

To develop custom benchmarks, VLMs, or simply contribute other codes to **VLMEvalKit**, please refer to [[Development_Guide](/docs/en/Development.md) | [开发指南](/docs/zh-CN/Development.md)].

**Call for contributions**

To promote the contribution from the community and share the corresponding credit (in the next report update):

- All Contributions will be acknowledged in the report.
- Contributors with 3 or more major contributions (implementing an MLLM, benchmark, or major feature) can join the author list of [VLMEvalKit Technical Report](https://www.arxiv.org/abs/2407.11691) on ArXiv. Eligible contributors can create an issue or dm kennyutc in [VLMEvalKit Discord Channel](https://discord.com/invite/evDT4GZmxN).

Here is a [contributor list](/docs/en/Contributors.md) we curated based on the records.

## 🎯 The Goal of VLMEvalKit

**The codebase is designed to:**

1. Provide an **easy-to-use**, **opensource evaluation toolkit** to make it convenient for researchers & developers to evaluate existing LVLMs and make evaluation results **easy to reproduce**.
2. Make it easy for VLM developers to evaluate their own models. To evaluate the VLM on multiple supported benchmarks, one just need to **implement a single `generate_inner()` function**, all other workloads (data downloading, data preprocessing, prediction inference, metric calculation) are handled by the codebase.

**The codebase is not designed to:**

1. Reproduce the exact accuracy number reported in the original papers of all **3rd party benchmarks**. The reason can be two-fold:
   1. VLMEvalKit uses **generation-based evaluation** for all VLMs (and optionally with **LLM-based answer extraction**). Meanwhile, some benchmarks may use different approaches (SEEDBench uses PPL-based evaluation, *eg.*). For those benchmarks, we compare both scores in the corresponding result. We encourage developers to support other evaluation paradigms in the codebase.
   2. By default, we use the same prompt template for all VLMs to evaluate on a benchmark. Meanwhile, **some VLMs may have their specific prompt templates** (some may not covered by the codebase at this time). We encourage VLM developers to implement their own prompt template in VLMEvalKit, if that is not covered currently. That will help to improve the reproducibility.

## 🖊️ Citation

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

If you use VLMEvalKit in your research or wish to refer to published OpenSource evaluation results, please use the following BibTeX entry and the BibTex entry corresponding to the specific VLM / benchmark you used.

```bib
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>

[github-contributors-link]: https://github.com/open-compass/VLMEvalKit/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/VLMEvalKit?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/VLMEvalKit/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/VLMEvalKit?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/VLMEvalKit/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/VLMEvalKit?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/VLMEvalKit/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/VLMEvalKit?color=white&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/VLMEvalKit/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/VLMEvalKit?color=ffcb47&labelColor=black&style=flat-square
