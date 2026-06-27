## Image Replay Evaluation Fork

This branch is a cleaned, publishable VLMEvalKit fork for the image-replay experiments. It is based on `open-compass/VLMEvalKit` commit `00804217f868058f871f5ff252a7b9623c3475d9` and keeps the active replay evaluation surface used for Qwen2.5-VL, MiniCPM-4.5, and Gemma3-family experiments.

Runtime artifacts are intentionally not included: no `runs/`, `checkpoints/`, model weights, `LMUData/`, private API keys, or machine-local caches are tracked. Historical launchers are kept under `scripts_legacy/` for provenance; the maintained entrypoint is `scripts/run_benchmark.sh`.

### Quick Start

Copy `.env.example` to `.env` or export the variables in your shell:

```bash
export MODEL_ROOT=/models
export CONDA_ROOT=/path/to/conda-or-venv-root
export LMUData=/path/to/LMUData
export OPENAI_API_KEY_JUDGE=
export OPENAI_API_BASE_JUDGE=https://api.openai.com/v1
export OPENAI_COMPATIBLE_API_KEY=
export OPENAI_COMPATIBLE_API_BASE=
```

Run a dry plan before launching work:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml \
  --model-config configs/models.yaml \
  --scheduler gpu_pool \
  --plan-only
```

### Active Entrypoints

- `bash scripts/run_benchmark.sh ...`: the only maintained shell entrypoint under `scripts/`. It reads `configs/models.yaml` plus a matrix YAML, supports `--resume-infer`, and has `--scheduler gpu_pool` for mixed tensor-parallel packing.

Top-level `scripts/` is intentionally limited to `scripts/run_benchmark.sh`. Reproduce main-table cells by selecting the documented matrix config and, for sharded historical source groups, passing `--nodes`, `--node-rank`, `--gpu-ids`, `--task-manifest`, and `--manifest-is-node-shard` directly to `run_benchmark.sh`. Historical shell launchers that are kept only for provenance live under `scripts_legacy/`.

The `gpu_pool` scheduler is the expected scheduler for mixed-size model queues. It can keep an 8-GPU node filled with combinations such as one `tp=4` task plus two `tp=2` tasks when the matrix contains both 72B and 32B jobs.

### Final Table Reproduction

The public release surface has two result-relevant execution stacks. In practice, the score-relevant new/legacy differences are judge/evaluator choice, dataset/cache lineage, and a few runtime knobs rather than scheduler shape. The detailed source-group and per-cell audit remains in [`docs/ACTIVE_ROUTE_MATRIX.md`](docs/ACTIVE_ROUTE_MATRIX.md), [`docs/FINAL_TABLE_REPRODUCTION.md`](docs/FINAL_TABLE_REPRODUCTION.md), [`docs/final_table_reproduction_entries.csv`](docs/final_table_reproduction_entries.csv), and [`docs/final_table_cell_sources.csv`](docs/final_table_cell_sources.csv).

Run the new stack with the maintained matrix runner:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config <matrix.yaml> \
  --model-config configs/models.yaml \
  --scheduler gpu_pool \
  --plan-only

bash scripts/run_benchmark.sh \
  --matrix-config <matrix.yaml> \
  --model-config configs/models.yaml \
  --scheduler gpu_pool
```

For the main sharded new matrices, launch one `run_benchmark.sh` command per node. For example, Qwen2.5-VL plus MiniCPM-4.5 on the four newer reasoning/perception benchmarks uses:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml \
  --model-config configs/models.yaml \
  --nodes 2 \
  --node-rank 0 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --task-manifest configs/task_manifests/qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422/node0_tasks.csv \
  --manifest-is-node-shard \
  --scheduler gpu_pool
```

Gemma3-family all-11 benchmark reproduction uses the same entrypoint with `configs/matrix_gemma3_family_all11_replay6_2node_20260422.yaml` and the matching `configs/task_manifests/gemma3_family_all11_replay6_2node_20260422/node<N>_tasks.csv`.

Run legacy shell stacks only from `scripts_legacy/`. Legacy DynaMath infer-only provenance is represented by `configs/matrix_legacy_dynamath_infer_only_all_replay.yaml` and the lower-level `python scripts_legacy/run_benchmark_task_balanced.py` path, not by a top-level `scripts/*.sh` wrapper.

Use the source-stack table below to choose `New` or `Legacy` at the model-by-dataset level. It uses the majority stack across replay settings for each model and dataset; use the detailed CSVs only when exact per-cell provenance is needed.

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

### Main Matrices

- `configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml`: Qwen2.5-VL 3B/7B/32B/72B plus MiniCPM-V/O-4.5 on `MMMU_DEV_VAL_SINGLE_IMAGE`, `WeMath`, `MMBench_DEV_EN_V11`, and `MMStar`.
- `configs/matrix_gemma3_family_all11_replay6_2node_20260422.yaml`: Gemma3 family on `DynaMath`, `MathVision`, `LogicVista`, `VisualPuzzles`, `AI2D_TEST`, `OCRBench`, `SEEDBench2_Plus`, and the four newer benchmarks.
- `configs/matrix_qwen25vl_all4_reasoning_perception4_new_entry_20260421.yaml`: earlier Qwen2.5-only four-new-benchmark matrix.
- `configs/matrix_qwen25vl7b_minicpm45_table_20260406.yaml`: preserved legacy table matrix for Qwen2.5-7B and MiniCPM-4.5.
- `configs/matrix_legacy_dynamath_infer_only_all_replay.yaml`: preserved legacy Dynamath infer-only matrix.
- `configs/matrix_minicpm45_wemath_cot_rerun_20260429.yaml`: MiniCPM-4.5 WeMath CoT rerun matrix.
- `configs/matrix_final_table_legacy_backfill_20260512.yaml` plus `configs/task_manifests/final_table_legacy_backfill_20260512/all_tasks.csv`: exact backfill manifest for the 21 legacy standard-run cells needed to cover every non-public, non-derived `final_table.csv` benchmark cell with the release runner.

### Code Map

- Replay policy and transform routing: `vlmeval/vlm/replay_policy.py`, `vlmeval/vlm/replay_image_transform.py`.
- Qwen2/Qwen2.5 replay wrapper: `vlmeval/vlm/qwen2_vl/model.py` and `vlmeval/vlm/qwen2_vl/replay_prompt_template.py`.
- MiniCPM-4.5 replay wrapper: `vlmeval/vlm/minicpm_v_4_5_replay.py`.
- Gemma3 replay wrapper: `vlmeval/vlm/gemma3_replay.py`.
- Minimal model registries: `vlmeval/config_qwen_minimal.py`, `vlmeval/config_minicpm45_minimal.py`, and `vlmeval/config_gemma3_minimal.py`.
- Dataset/eval adaptations: `vlmeval/dataset/dynamath.py`, `vlmeval/dataset/image_mcq.py`, `vlmeval/dataset/image_vqa.py`, and `vlmeval/dataset/utils/{logicvista,mathv,visualpuzzles,wemath}.py`.
- Matrix orchestration and result collection: `vlmeval/cli/run_benchmark.py`, `vlmeval/cli/collect_matrix_results.py`, and `configs/`.
- Runtime parity probes: `python -m vlmeval.probes.step7_standard_entry_parity` and `vlmeval/probes/standard_entry_parity.py`. These are validation tools for the release route, not benchmark launchers.

### Reproducibility Notes

- Active image-replay model and data locations are controlled by environment variables, not hardcoded private paths. Set `MODEL_ROOT`, `CONDA_ROOT`, and `LMUData` as needed for the active release routes.
- LLM-judge based datasets use `OPENAI_API_KEY_JUDGE`/`OPENAI_API_BASE_JUDGE` first, then `OPENAI_API_KEY`/`OPENAI_API_BASE`; generic non-OpenAI-compatible endpoints can be supplied with `OPENAI_COMPATIBLE_API_KEY`/`OPENAI_COMPATIBLE_API_BASE`.
- Non-default API wrappers such as `JTVLChatAPI` have no bundled private endpoint. Configure their endpoint and token explicitly with the documented environment variables in the wrapper before use.
- The release matrices plus `matrix_final_table_legacy_backfill_20260512` cover every non-public, non-derived benchmark cell in `final_table.csv`. Public/reference rows are external report numbers, and `MMStar no-reason only` is a derived subset summary rather than a separate VLMEvalKit run.
- Exact numeric reproduction still depends on matching model checkpoints, dataset/cache versions, evaluator dependencies, vLLM engine/runtime knobs such as `VLLM_USE_V1`, and the LLM judge endpoint/model used for judge-scored datasets.
- `scripts_legacy/` is intentionally retained for old final-table provenance. Treat these scripts as historical launch references rather than the preferred active interface.

### Upstream Baseline

This fork is derived from `open-compass/VLMEvalKit` commit `00804217f868058f871f5ff252a7b9623c3475d9`. Use the upstream project documentation for generic VLMEvalKit usage outside the active image-replay release routes documented above.
