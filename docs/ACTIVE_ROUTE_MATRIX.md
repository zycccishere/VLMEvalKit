# Active Route Matrix

This document is the active static route contract for the image-replay release surface. It intentionally covers only the active model scope: Qwen2.5-VL, MiniCPM 4.5, Gemma3, and closed-source API replay wrappers. Qwen3.5, Gemma4, LLaVA, and other historical launchers are legacy/provenance-only for this branch and are not adapted or validated here.

## Standard Entry

All active shell launches go through the single maintained entrypoint. `--matrix-config` is required on purpose; the runner must not default to an old or inactive experiment matrix:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config <matrix.yaml> \
  --model-config configs/models.yaml \
  --scheduler gpu_pool
```

For sharded historical source groups, pass `--nodes`, `--node-rank`, `--gpu-ids`, `--task-manifest`, and `--manifest-is-node-shard` directly to the same entrypoint. Runtime data/model roots are external to the repository and must be provided through `LMUData`, `MODEL_ROOT`, and `CONDA_ROOT`; the runner fails fast if `LMUData` is unset or missing.

## Replay Modes

| Table label | Runner mode |
| --- | --- |
| I-Q | `image_text` |
| Q-I | `text_image` |
| I-I-Q | `image_image_text` |
| I-Q-I | `image_text_image` |
| I-Q-Q | `image_text_text` |
| I-Q-I-Q | `image_text_image_text` |

All active main-table matrices use `replay_times: 1`, `template_on_last_replay_text: 1`, `image_copy_mode: reuse_path`, `limit_mm_per_prompt: 2`, `safe_fallback: 0`, and `strict_batch: 1` unless a matrix explicitly says otherwise. Replay failures and batch-generation failures should fail the task instead of silently degrading to another prompt or single-item path.

## Dataset Routes

| Dataset | Active route | Notes |
| --- | --- | --- |
| DynaMath | Dataset builder plus replay wrapper; standard entry sets `DYNAMATH_PROMPT_SCHEMA=legacy_two_keys` only for Qwen2.5-VL and `short_answer_only` for MiniCPM/Gemma3. | The legacy two-key JSON prompt is the `default` / `identity` Qwen2.5-VL reproduction route. The `direct` / `directly_answer` policy intentionally uses answer-only wording for every model family. Judge fallback is fail-fast when local parsing cannot settle pending rows and the judge is unavailable. |
| MathVision | Image VQA/LLM-judge route. | Uses external `LMUData`; repository does not vendor TSV/image payloads. |
| LogicVista | LogicVista dataset/eval route. | Qwen2.5-VL LogicVista is the only active route that forces vLLM v0. |
| VisualPuzzles | Custom prompt/eval route. | Included in Qwen reasoning and MiniCPM/Gemma3 matrices. |
| MMMU single image | `MMMU_DEV_VAL_SINGLE_IMAGE`; `MMMU_DEV_VAL_SINGLE` is canonicalized to this name. | Filters MMMU dev/val to single-image rows. |
| MathVista Mini | `MathVista_MINI`. | Covered by Qwen/MiniCPM legacy table matrices and Step7 probes when selected. |
| We-Math | WeMath dataset/eval route. | MiniCPM CoT behavior comes from the model wrapper, not a separate active dataset alias. |
| AI2D_TEST | Generic image MCQ route. | Rule/local extraction where applicable. |
| OCRBench | OCRBench evaluator route. | Covered by legacy/backfill and Gemma3 matrices. |
| SEEDBench2_Plus | Generic image MCQ route. | Covered by legacy/backfill and Gemma3 matrices. |
| MMBench_DEV_EN_V11 | MMBench image MCQ route. | Included in new four-benchmark matrices. |
| MMStar | Generic image MCQ route. | Included in new four-benchmark matrices. |

## Model Routes And Runtime Knobs

| Model key | Registry | Env profile | GPUs | Batch / max seqs | Context | Decode defaults |
| --- | --- | --- | ---: | --- | ---: | --- |
| `qwen25vl_3b` | `Qwen2.5-VL-3B-Instruct-Replay` | `main_vlmeval` | 1 | 64 / 64 | 32768 | vLLM by default; Qwen wrapper greedy by default; LogicVista overrides below. |
| `qwen25vl_7b` | `Qwen2.5-VL-7B-Instruct-Replay` | `main_vlmeval` | 1 | 64 / 64 | 32768 | vLLM by default; Qwen wrapper greedy by default; LogicVista overrides below. |
| `qwen25vl_32b` | `Qwen2VLChatReplay` | `main_vlmeval` | 2 | 16 / 16 | 32768 | vLLM by default; Qwen wrapper greedy by default; LogicVista overrides below. |
| `qwen25vl_72b` | `Qwen2VLChatReplay` | `main_vlmeval` | 4 | 16 / 16 | 32768 | vLLM by default; Qwen wrapper greedy by default; LogicVista overrides below. |
| `minicpm_v_45` | `MiniCPM-V-4_5-Replay` | `minicpm45_vllm` | 1 | 64 / 64 | 32768 | MiniCPM wrapper controls no-thinking/CoT policy by dataset. |
| `minicpm_o_45` | `MiniCPM-o-4_5-Replay` | `minicpm45_vllm` | 1 | 64 / 64 | 32768 | MiniCPM wrapper controls no-thinking/CoT policy by dataset. |
| `gemma3_4b` | `Gemma3-4B-Replay` | `gemma3_vllm` | 1 | 128 / 128 | 32768 | temperature 0, max new tokens 4096, seed 0. |
| `gemma3_12b` | `Gemma3-12B-Replay` | `gemma3_vllm` | 1 | 64 / 64 | 32768 | temperature 0, max new tokens 4096, seed 0. |
| `gemma3_27b` | `Gemma3-27B-Replay` | `gemma3_vllm` | 2 | 16 / 16 | 32768 | temperature 0, max new tokens 4096, seed 0. |

MiniCPM-4.5 active wrappers intentionally do not apply the old random per-image upsize augmentation. The replay path keeps image reuse controlled by `REPLAY_IMAGE_COPY_MODE=reuse_path` and model-side reasoning/no-thinking policy by dataset; image resizing should come only from deterministic processor/model requirements or explicit replay transforms, not from random upsize augmentation.

The active open-source standard inference backend is vLLM. Closed-source API models use their API wrappers instead of vLLM. Step7 diagnostic probes may deliberately use HF/forward paths when comparing logits, but that is not the standard benchmark entry.

Qwen2.5-VL LogicVista policy is special: `LOGICVISTA_QWEN25VL_FORCE_V0=1` by default, so the runner sets `VLLM_USE_V1=0`, `LOGICVISTA_QWEN25VL_LEGACY_SAMPLING=1`, `LOGICVISTA_QWEN25VL_BATCH_SIZE=128`, `LOGICVISTA_QWEN25VL_MAX_NUM_SEQS=128`, `QWEN2VL_VLLM_TEMPERATURE=0.01`, `QWEN2VL_VLLM_TOP_P=1.0`, `QWEN2VL_VLLM_TOP_K=0`, `QWEN2VL_VLLM_REPETITION_PENALTY=1.05`, `QWEN2VL_VLLM_MAX_TOKENS=2048`, and `QWEN2VL_VLLM_STOP_TOKEN_IDS=151645,151643`. Non-LogicVista Qwen2.5-VL routes default to `VLLM_USE_V1=1` and clear those legacy sampling keys.

## Matrix Coverage

| Matrix | Models | Datasets | Purpose |
| --- | --- | --- | --- |
| `matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml` | Qwen2.5-VL 3B/7B/32B/72B, MiniCPM-V/O 4.5 | MMMU single image, WeMath, MMBench V11, MMStar | New four-benchmark main-table route. |
| `matrix_gemma3_family_all11_replay6_2node_20260422.yaml` | Gemma3 4B/12B/27B | All 11 active datasets except MathVista Mini | Gemma3 main-table route. |
| `matrix_gemma3_3models_mathvistamini_replay6_8gpu_20260531.yaml` | Gemma3 4B/12B/27B | MathVista Mini | Gemma3 MathVista Mini parity/completion route. |
| `matrix_gemma3_12b_reference4_image_text_20260422.yaml` | Gemma3 12B | MathVision, WeMath, MMBench V11, MMStar | Recorded Gemma3 12B I-Q reference cells. |
| `matrix_qwen25vl_all4_reasoning4_new_entry_20260421.yaml` | Qwen2.5-VL 3B/7B/32B/72B | MathVision, DynaMath, LogicVista, VisualPuzzles | Qwen reasoning route; LogicVista uses vLLM v0 policy. |
| `matrix_qwen25vl_all4_reasoning_perception4_new_entry_20260421.yaml` | Qwen2.5-VL 7B historical subset | MMMU single image, WeMath, MMBench V11, MMStar | Qwen2.5-7B mixed-source table subset. |
| `matrix_qwen25vl7b_minicpm45_table_20260406.yaml` | Qwen2.5-VL 7B, MiniCPM-V/O 4.5 | MathVision, DynaMath, LogicVista, VisualPuzzles, MathVista Mini, AI2D, OCRBench, SEEDBench2_Plus | Legacy table matrix; current final-table provenance is MiniCPM-heavy. |
| `matrix_minicpm_default_infer_only_fresh_20260317.yaml` | MiniCPM-V/O 4.5 | AI2D, MathVista Mini, OCRBench, SEEDBench2_Plus, LogicVista, VisualPuzzles, DynaMath, MathVision, plus historical VisuLogic slice | MiniCPM infer-only provenance route; VisuLogic is historical/provenance-only for this branch. |
| `matrix_minicpm_logicvista_all_replay_eval_20260419.yaml` | MiniCPM-V/O 4.5 | LogicVista | MiniCPM LogicVista eval/realign route. |
| `matrix_minicpm_visualpuzzles_all_replay_eval_realign_20260420.yaml` | MiniCPM-V/O 4.5 | VisualPuzzles | MiniCPM VisualPuzzles eval/realign route. |
| `matrix_api_replay.yaml` | GPT/OpenAI-compatible, Claude, Gemini aliases in `configs/models.yaml` | Active table datasets including MathVista Mini | Closed-source API replay route; use explicit model filters for cost control. |
| `matrix_final_table_legacy_backfill_20260512.yaml` | Qwen2.5-VL 3B/72B | AI2D, OCRBench, SEEDBench2_Plus | Release backfill for remaining legacy standard-run cells. |
| `matrix_legacy_dynamath_infer_only_all_replay.yaml` | Qwen2.5-VL 3B/7B/32B/72B, MiniCPM-V/O 4.5 | DynaMath | Legacy DynaMath infer-only provenance through `scripts_legacy/run_benchmark_task_balanced.py`. |

## Closed-Source API Routes

Closed-source active routes are exposed through the `api_replay` profile in `configs/models.yaml`, the explicit `configs/matrix_api_replay.yaml` matrix, `vlmeval/config_api_replay_minimal.py`, and the standard replay API wrappers:

| Alias family | Wrapper | Temperature | Max tokens | Notes |
| --- | --- | ---: | ---: | --- |
| GPT/OpenAI-compatible | `GPT4VReplay` | 0 | `VLMEVAL_API_MAX_TOKENS`, default 2048 | Uses the main-repo OpenAI-compatible replay wrapper surface. |
| Claude aliases | `GPT4VReplay` | 0 | `VLMEVAL_API_MAX_TOKENS`, default 2048 | Matches the main-repo closed-source replay config. |
| Gemini aliases | `GPT4VReplay` | 0 | `VLMEVAL_API_MAX_TOKENS`, default 2048 | Matches the main-repo closed-source replay config. |

API usage logs are injected by `vlmeval/cli/run_benchmark.py` unless `VLMEVAL_API_USAGE_LOG_FILE` or `TOKEN_USAGE_LOG_FILE` is already set.

## Data Flow

1. `scripts/run_benchmark.sh` loads `.env`, requires `LMUData`, and executes `vlmeval/cli/run_benchmark.py`.
2. `vlmeval/cli/run_benchmark.py` resolves matrix/model YAML, canonicalizes dataset aliases, builds task records across model, policy, replay mode, transform, and dataset axes, and schedules them with `gpu_pool` or `model_sequential`.
3. Per task, the runner builds environment variables for model path, replay mode, replay template, safe-fallback policy, strict batch behavior, vLLM knobs, judge endpoint, prediction roots, and trace/audit roots.
4. Inference calls `run.py --mode infer --data <dataset> --model <registry> --batch-size <resolved_batch> --pred-output-dir <fixed_prediction_dir>`.
5. `run.py` imports `vlmeval.config_runtime`, which selects the minimal active registry from environment flags instead of importing the full experimental registry.
6. `vlmeval.inference` builds dataset rows, calls the selected model wrapper, writes fixed predictions, and raises on batch failures when `VLMEVAL_STRICT_BATCH=1`.
7. Evaluation calls `run.py --mode eval --pred-file <fixed_prediction_file> --eval-dir <fixed_eval_dir> --judge <judge>`, then writes eval manifests.

## Current Runtime Gate

Runtime validation was rerun after the strict `scripts/` entrypoint cleanup. The runner now lives in `vlmeval/cli`, configs live in top-level `configs/`, and the only maintained shell entrypoint under `scripts/` is `scripts/run_benchmark.sh`:

- Static route gate: `runs/final_strict_scripts_step7_context_20260627` covers 9 open-source models x 2 datasets (`DynaMath`, `LogicVista`) x 6 replay modes, all 108 checks passing. Qwen2.5-VL is the only DynaMath `legacy_two_keys` family and the only LogicVista vLLM v0 family; MiniCPM and Gemma3 use vLLM without the LogicVista v0 override.
- Payload gate: `runs/final_strict_scripts_step7_payload_20260627` covers the same 108 open-source routes with real dataset rows and model-wrapper prompt/payload serialization, all checks passing.
- API no-network gate: `runs/final_strict_scripts_api_context_20260627` covers all 11 tracked API aliases through the `api_replay` profile, all checks passing.
- Direct DynaMath gate: `runs/final_head4c3_step7_payload_direct_dynamath_20260627` covers 9 open-source models x 6 replay modes under the `directly_answer` policy, all 54 checks passing.
- Main-tree parity gate: `runs/final_head4c3_vs_main_payload_compare_20260627.json` compares the release payload gate with `/user/zyc1781/vlmevalkit`, all 108 rows passing.
- Step 8 speed profiling is recorded in `docs/STEP8_SPEED_BATCHSIZE.md` and `configs/model_speed_profiles_step8.yaml`; raw logs remain under ignored `runs/step8_speed_20260627`.
