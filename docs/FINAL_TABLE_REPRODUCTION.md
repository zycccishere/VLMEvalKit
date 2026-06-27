# Final Table Reproduction Map

This document maps the current WorkHub `final_table.csv` numbers to the best-known release entrypoints and historical source groups.

The exact-number source of truth is:

```text
assets/topics/topic-image-replay/resources/final-tables/final_table.csv
```

The provenance source used to build this map is:

```text
assets/topics/topic-image-replay/resources/table-aligned-run-artifacts-20260429/final_table_remote_manifests/
```

## Replay Mode Names

The table labels map to runner replay modes as follows:

| Table label | Runner replay mode |
|---|---|
| `I-Q` | `image_text` |
| `Q-I` | `text_image` |
| `I-I-Q` | `image_image_text` |
| `I-Q-I` | `image_text_image` |
| `I-Q-Q` | `image_text_text` |
| `I-Q-I-Q` | `image_text_image_text` |

## Reproduction Policy

At the execution-stack level, the final table has two result-relevant launch families:

| Stack | Entrypoint shape | When to use it |
|---|---|---|
| New matrix runner | `bash scripts/run_benchmark.sh --matrix-config ... --model-config ... --scheduler gpu_pool` | Preferred route for new reruns, Gemma3, Qwen/MiniCPM four-new-benchmark runs, and most cleaned/eval-realigned source groups. |
| Legacy launch shape | `scripts_legacy/*.sh`, plus documented lower-level Python runners for legacy matrix configs | Use when the source cell came from the older March/early-April launch stack and exact table provenance requires the old prompt/eval/judge shape. |

Most of the many entries below are not different experimental ideas. They are matrix partitions, model/dataset slices, or source-group aliases over those two stacks. The exception is exact historical reproduction: the source group still matters because it fixes the dataset cache, judge model, rerun/eval-realign state, and a few manually accepted caveat cells.

In this document, `mixed source` means that one apparent table family is assembled from more than one historical source run directory. It does not mean that the replay method changed inside one cell, and it usually does not mean a judge-model mismatch. For example, Qwen2.5-7B on the four new benchmarks uses 16 cells from `runs/qwen25vl_all4_reasoning_perception4_new_entry_20260421` and 8 cells from `runs/qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422`; both source groups use `gpt-4o-mini` for LLM-judged cells plus rule/local metrics for non-LLM-evaluated cells.

Use `scripts/run_benchmark.sh` and the matrix YAMLs for new reruns. Use `scripts_legacy/` when you need to match the old launch shape exactly. The legacy scripts are kept because part of the main table came from those old runs.

## Final Table Source Stack

This model-by-dataset map uses the majority stack across replay settings for that model and dataset. Full per-cell provenance remains in `docs/final_table_cell_sources.csv`.

| Model | DynaMath | MathVision | LogicVista | VisualPuzzles | MMMU single image | We-Math | AI2D_TEST | OCRBench | SEEDBench2_Plus | MMBench_DEV_EN_V11 | MMStar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 72B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| Qwen2.5 32B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| Qwen2.5 7B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| Qwen2.5 3B | Legacy | Legacy | New | New | New | New | Legacy | Legacy | Legacy | New | New |
| MiniCPM-V | New | New | New | New | New | New | New | New | New | New | New |
| MiniCPM-o | New | New | New | New | New | New | New | New | New | New | New |
| Gemma3 27B | New | New | New | New | New | New | New | New | New | New | New |
| Gemma3 12B | New | New | New | New | New | New | New | New | New | New | New |
| Gemma3 4B | New | New | New | New | New | New | New | New | New | New | New |

## Result-Relevant New vs Legacy Differences

`New` and `legacy` are execution/provenance labels, not two different replay algorithms. The scheduler, node count, tmux launcher, log directory layout, and task partitioning should not change benchmark semantics. The result-relevant differences are the following:

| Axis | New matrix runner | Preserved legacy launch shape | Result relevance |
|---|---|---|---|
| Model registry and checkpoint | `vlmeval/cli/run_benchmark.py` reads `configs/models.yaml`, sets `MODEL_PATH`, and calls `run.py --model <registry_name>`. | The standard legacy Qwen scripts export `MODEL_PATH` and call `run.py --model Qwen2VLChatReplay`. | For Qwen2.5 final-table cells, this is intended to be equivalent when `MODEL_ROOT=/models`: both use the same Qwen2.5-VL Instruct checkpoints and the same replay wrapper. Gemma3 and most MiniCPM cells are new-stack only. |
| Replay/prompt construction | The main matrices set `replay_prompt_template_name: identity`, `replay_times: 1`, `template_on_last_replay_text: 1`, `image_copy_mode: reuse_path`, and `limit_mm_per_prompt: 2`. | The main final-table legacy wrappers also force `REPLAY_PROMPT_TEMPLATE_NAME=identity`; the shared legacy helper's `directly_answer` default is not the final-table default when these wrappers are used. | Equivalent for the preserved final-table launchers. Running the lower-level legacy helper directly without the wrapper would be score-changing because it can default to `directly_answer`. |
| Image transform | The main new matrices use only `image_transforms: baseline`. | Legacy final-table standard runs have no transform axis, which is equivalent to baseline. | Equivalent for main table cells. Non-baseline ablations are separate experiments, not a new/legacy distinction. |
| Decode and vLLM runtime parameters | Runtime comes from `models.yaml`: Qwen2.5 3B/7B use `infer_batch_size=64`; 32B uses `tp=2`, `batch=8`, `max_num_seqs=8`, `max_model_len=32768`; 72B uses `tp=4`, `batch=1`, `max_num_seqs=1`, `max_model_len=32768`. The new wrappers do not pin `VLLM_USE_V1`, so the vLLM engine follows the installed vLLM/default environment unless `.env` sets it. In the recorded Cybertron Qwen new environment, this resolved to vLLM V1 by default. | Legacy Qwen scripts explicitly set `VLLM_USE_V1=${VLLM_USE_V1:-0}`, i.e. vLLM v0 by default. Legacy small Qwen scripts default to `INFER_BATCH_SIZE=32`, `VLLM_MAX_NUM_SEQS=$INFER_BATCH_SIZE`, `VLLM_MAX_MODEL_LEN=32768`. Legacy 32B/72B wrappers force `tp=2/4`, `batch=1`, `max_num_seqs=1`, and `max_model_len=32768`. | Core decoding is still deterministic vLLM sampling (`temperature=0.0`, `max_tokens=max_new_tokens`) in the Qwen wrapper, but vLLM engine version plus batch/max-seq differences can change scheduling/numerics and are the main non-judge runtime differences, especially for Qwen2.5-32B. For strict legacy reproduction, keep `VLLM_USE_V1=0`; for new-run parity, leave the Qwen new environment default unless the source log records an override. |
| Judge/evaluator | Main new matrices explicitly set `judge: gpt-4o-mini` and pass it to `run.py --mode eval --judge`. | The preserved legacy guard defaults to `JUDGE_MODEL=gpt-4o-mini`, but historical March source artifacts in `docs/final_table_cell_sources.csv` include many `gpt-4o` judge-scored cells. | This is a real score-relevant difference for LLM-judged datasets. Exact historical reproduction must follow the per-cell provenance CSV or set `JUDGE_MODEL` to the recorded judge. Rule/local metrics do not use an LLM judge. |
| Dataset/cache surface | New matrices explicitly enumerate the new four benchmarks or the 11-benchmark Gemma3 matrix and use the configured `LMUData` cache. | Legacy scripts use older `DATALIST` groups such as `AI2D_TEST DynaMath MathVista_MINI OCRBench SEEDBench2_Plus VisuLogic LogicVista VisualPuzzles MathVision`, also through VLMEvalKit dataset builders and `LMUData`. | Dataset name, dataset-code version, and local `LMUData` cache are score-relevant. The repository does not vendor dataset payloads, so exact reproduction requires matching the cache/version used by the source artifact. |
| Resume/artifact hygiene | New runner defaults `resume_infer_default: false`, cleans stale infer/eval artifacts before rerun, and can explicitly resume with `--resume-infer`. | Standard legacy scripts default `INFER_RESUME_ENABLED=0` and also clean stale artifacts; some preserved legacy matrices are infer-only and intentionally skip eval. | This should not change clean-run semantics, but it matters when a directory contains partial or stale artifacts. Exact artifact reuse should follow the recorded source group. |

The practical interpretation is: use `new` vs `legacy` mainly to recover the correct judge/evaluator, dataset/cache lineage, and a small number of runtime knobs. Do not treat them as separate replay mechanisms.

### Effective Score-Changing Checklist

The following items are the real knobs/files that can change table numbers. Scheduler layout, tmux naming, node rank, and log directory names should not change scores unless they indirectly change one of these items.

| Area | Score-sensitive details to preserve |
|---|---|
| Active model implementation | Qwen2/Qwen2.5 final-table replay cells use `vlmeval/vlm/qwen2_vl/model.py::Qwen2VLChatReplay` through `vlmeval/config_qwen_minimal.py` or the equivalent full-config registry. They do **not** use the inactive `qwen25vl_custom*`, `qwen2_5_vl_custom.py`, `modeling_qwen_custom_vl_hw.py`, or `duplex*` experimental modeling files unless a legacy script explicitly names those registries. MiniCPM cells use `vlmeval/vlm/minicpm_v_4_5_replay.py`; Gemma3 cells use `vlmeval/vlm/gemma3_replay.py`. |
| Runtime registry import | `run.py` should import `supported_VLM` from `vlmeval.config_runtime`, not directly from the full experimental `vlmeval.config`, and `vlmeval/__init__.py` should not eagerly import the full config when a minimal runtime registry flag is set. This lets `VLMEVAL_USE_QWEN_MINIMAL_CONFIG`, `VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG`, and `VLMEVAL_USE_GEMMA3_MINIMAL_CONFIG` select the intended active registry. |
| Qwen sampling | Qwen vLLM replay inference is greedy: `SamplingParams(temperature=0.0, max_tokens=max_new_tokens)`, with `max_new_tokens=2048` by default. The non-vLLM/lmdeploy fallback has different generation kwargs (`temperature=0.01`, `top_p=0.001`, `top_k=1`), so exact reproduction should stay on the recorded vLLM path. |
| Qwen vLLM engine and context | New Qwen matrix runs do not pin `VLLM_USE_V1`; on the recorded Cybertron Qwen environment this resolves to vLLM V1. Legacy Qwen scripts explicitly set `VLLM_USE_V1=0`, i.e. vLLM v0. The new runner only exports `VLLM_MAX_MODEL_LEN` when a model entry has `max_model_len`; `qwen25vl_7b` in `configs/models.yaml` has no explicit value, so the Qwen wrapper falls back to `8192`, while new `qwen25vl_3b/32b/72b` and legacy Qwen small/large wrappers set `32768`. This is a genuine reproduction-sensitive detail, especially for replay modes that lengthen the prompt. |
| Batch and sequence packing | New Qwen 3B/7B use `infer_batch_size=max_num_seqs=64`; new 32B uses `tp=2`, `batch=max_num_seqs=8`; new 72B uses `tp=4`, `batch=max_num_seqs=1`. Legacy small Qwen uses `batch=max_num_seqs=32`; legacy 32B/72B final-table wrappers use `batch=max_num_seqs=1`. Even with greedy decoding, vLLM engine/version and packing can change numerical behavior or trigger a different fallback path. |
| MiniCPM prompt and sampling | MiniCPM does not simply reuse the dataset prompt for all final-table datasets. `MiniCPM_V_4_5.use_custom_prompt()` routes MCQ/VQA/Y/N datasets through the model wrapper. Datasets in `use_long_cot()` include `MMMU`, `MMBench`, `MMStar`, `MathVision`, `DynaMath`, `LogicVista`, `VisualPuzzles`, and `WeMath`; on the vLLM path these use sampling (`temperature=0.7`, `top_p=1.0`, `top_k=0`, `num_beams=1`). AI2D/SEEDBench2_Plus disable thinking; OCRBench uses the shorter CoT branch. This is model-family behavior, not a scheduler detail. |
| Gemma3 prompt and sampling | Gemma3 replay uses `Gemma3Replay` with `temperature=0.0`, `max_new_tokens=4096`, vLLM seed `0`, and the default system prompt `You are a helpful assistant. `. It uses the same replay ordering controls as Qwen but its own message serialization and image limit handling. |
| Replay and prompt template | Main matrices use `replay_prompt_template_name: identity`, `replay_times: 1`, `template_on_last_replay_text: 1`, `image_copy_mode: reuse_path`, and `limit_mm_per_prompt: 2`. Lower-level legacy helpers may default to `directly_answer`; that strips or replaces dataset instructions for datasets such as DynaMath/VisualPuzzles and is score-changing if run without the final-table wrapper. |
| Common-prompt ablations | `REPLAY_FORCE_COMMON_PROMPT=1` changes prompt construction in `vlmeval/inference.py` and is only for explicit common-prompt probes. The main final table does not use it. The file named `matrix_minicpm45_wemath_cot_rerun_20260429.yaml` still lists dataset `WeMath`; the CoT behavior comes from MiniCPM's model wrapper, not from the `WeMath_COT` dataset alias. |
| Dataset modeling and cache | Exact dataset names matter. `MMMU_DEV_VAL_SINGLE_IMAGE` is a filtered alias over `MMMU_DEV_VAL` that keeps only single-image rows. `WeMath` and `WeMath_COT` share the same TSV but build different prompts only when the dataset alias contains `COT`. The DynaMath dataset builder defaults to `short_answer_only`; the standard runner injects `legacy_two_keys` only for Qwen2.5-VL and explicitly keeps `short_answer_only` for MiniCPM/Gemma3. VisualPuzzles appends a step-by-step and `Answer: $LETTER` suffix in the dataset prompt. The repository does not vendor TSV/image payloads, so `LMUData`/absolute dataset caches and MD5-compatible payloads must match the source artifacts. |
| Judge and extraction | New matrices pass `--judge gpt-4o-mini`. Historical legacy source groups include many `gpt-4o` judged cells. Several datasets have custom extraction before or during judge use: MMMU single-image extracts the last `Answer: [A-I]`; MMStar/MMBench/AI2D extract `Answer: [A-D]`; WeMath first tries deterministic option inference and then uses a WeMath-specific judge prompt; MathVision, LogicVista, and DynaMath rely on LLM auxiliary evaluators; VisualPuzzles can fall back to exact/local extraction. |
| Failure/fallback hygiene | The active standard runner sets `REPLAY_SAFE_FALLBACK=0` and `VLMEVAL_STRICT_BATCH=1`, so replay or batch failures fail the task instead of silently changing prompt shape. Historical legacy artifacts can still contain fallback behavior, and exact reproduction audits should check logs for `[safe-fallback]`, `[SKIPPED_OVERLONG_PROMPT]`, quota errors, invalid tokens, and disk errors before accepting a cell. |
| Answer-format report | `vlmeval/cli/postprocess_answer_format.py` is a QA/reporting pass in the maintained runners. With the current invocation it checks/extracts format statistics but does not rewrite the prediction file used by evaluation, so it should not be treated as a score-changing postprocessor unless the script is changed. |

The current provenance scan accounts for the main table as follows:

| Category | Count | Meaning |
|---|---:|---|
| Aligned source cells | 589 | Exact or accepted rounding match to `final_table.csv`. |
| Manual override cells | 2 | User-accepted source artifacts whose numeric score differs from the rounded table target. |
| Unresolved caveat cells | 3 | The current artifact scan did not find an exact source metric matching the table number. |
| Covered by release matrices/manifests | 594 | Every non-public, non-derived benchmark cell has a release-side entrypoint. |

The provenance README reports 5 unresolved cells before manual override handling. In this release map, those are represented as 2 manual override cells plus 3 remaining unresolved/near/mismatch caveat cells.

`MMStar no-reason only` is a derived analysis column and is not a separate VLMEvalKit benchmark run. Public/reference rows are external report numbers, not local replay runs.

## Main Command Chooser

| Model/dataset scope | Use this entrypoint |
|---|---|
| Gemma3 4B/12B/27B on all 11 table datasets and all 6 replay modes | `bash scripts/run_benchmark.sh --matrix-config configs/matrix_gemma3_family_all11_replay6_2node_20260422.yaml --model-config configs/models.yaml --nodes 2 --node-rank <node_rank> --gpu-ids <gpu_ids> --task-manifest configs/task_manifests/gemma3_family_all11_replay6_2node_20260422/node<node_rank>_tasks.csv --manifest-is-node-shard --scheduler gpu_pool` |
| Gemma3 12B I-Q on `MathVision`, `WeMath`, `MMBench_DEV_EN_V11`, `MMStar` when matching the recorded table provenance | `bash scripts/run_benchmark.sh --matrix-config configs/matrix_gemma3_12b_reference4_image_text_20260422.yaml --model-config configs/models.yaml --scheduler gpu_pool` |
| Qwen2.5 3B/32B/72B and MiniCPM-V/O on `MMMU_DEV_VAL_SINGLE_IMAGE`, `WeMath`, `MMBench_DEV_EN_V11`, `MMStar` | `bash scripts/run_benchmark.sh --matrix-config configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml --model-config configs/models.yaml --nodes 2 --node-rank <node_rank> --gpu-ids <gpu_ids> --task-manifest configs/task_manifests/qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422/node<node_rank>_tasks.csv --manifest-is-node-shard --scheduler gpu_pool` |
| Qwen2.5 7B on `MMMU_DEV_VAL_SINGLE_IMAGE`, `WeMath`, `MMBench_DEV_EN_V11`, `MMStar` | Mixed source: 16 cells use `bash scripts/run_benchmark.sh --matrix-config configs/matrix_qwen25vl_all4_reasoning_perception4_new_entry_20260421.yaml --model-config configs/models.yaml --nodes 4 --node-rank <node_rank> --gpu-ids <gpu_ids> --task-manifest configs/task_manifests/qwen25vl_all4_reasoning_perception4_new_entry_20260421/node<node_rank>_tasks.csv --manifest-is-node-shard --scheduler gpu_pool`, while 8 cells use `bash scripts/run_benchmark.sh --matrix-config configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml --model-config configs/models.yaml --nodes 2 --node-rank <node_rank> --gpu-ids <gpu_ids> --task-manifest configs/task_manifests/qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422/node<node_rank>_tasks.csv --manifest-is-node-shard --scheduler gpu_pool`. This is a source-run split, not a judge-model split. Use `docs/final_table_cell_sources.csv` for the exact cell list. |
| Qwen2.5 3B/7B/32B/72B on `LogicVista` and `VisualPuzzles` | `bash scripts/run_benchmark.sh --matrix-config configs/matrix_qwen25vl_all4_reasoning4_new_entry_20260421.yaml --model-config configs/models.yaml --scheduler gpu_pool --datasets LogicVista VisualPuzzles` |
| Qwen2.5 legacy cells on `AI2D_TEST`, `DynaMath`, `MathVision`, `OCRBench`, `SEEDBench2_Plus` | Use the legacy scripts listed in `docs/final_table_reproduction_entries.csv`; `bash scripts/run_benchmark.sh --matrix-config configs/matrix_final_table_legacy_backfill_20260512.yaml --model-config configs/models.yaml --scheduler gpu_pool --gpu-ids <gpu_ids> --task-manifest configs/task_manifests/final_table_legacy_backfill_20260512/all_tasks.csv` is only the 21-cell release backfill for `Qwen2.5 3B/72B` on `AI2D_TEST`, `OCRBench`, and `SEEDBench2_Plus`. |
| MiniCPM-V/O on `AI2D_TEST`, `DynaMath`, `MathVision`, `OCRBench`, `SEEDBench2_Plus` | Use `matrix_qwen25vl7b_minicpm45_table_20260406.yaml`, `matrix_minicpm_default_infer_only_fresh_20260317.yaml`, and the legacy DynaMath matrix/Python runner listed in the CSV. |
| MiniCPM-V/O on `LogicVista` | `bash scripts/run_benchmark.sh --matrix-config configs/matrix_minicpm_logicvista_all_replay_eval_20260419.yaml --model-config configs/models.yaml --scheduler gpu_pool` |
| MiniCPM-V/O on `VisualPuzzles` | `bash scripts/run_benchmark.sh --matrix-config configs/matrix_minicpm_visualpuzzles_all_replay_eval_realign_20260420.yaml --model-config configs/models.yaml --scheduler gpu_pool` |

For the complete source-group list, including cell counts, judge model observed in the source artifacts, and legacy command, see:

```text
docs/final_table_reproduction_entries.csv
```

For the direct model/dataset/setting to run-path mapping, see:

```text
docs/final_table_cell_sources.csv
```

## Judge And Evaluator Differences

The source artifacts show a mixture of judge/evaluator modes:

| Source family | Judge/eval mode seen |
|---|---|
| Gemma3 20260422 and Gemma3 12B reference | Mostly `gpt-4o-mini` where an LLM judge/extractor was used, plus rule/local metrics. |
| Qwen/MiniCPM new four-benchmark 20260421/20260422 runs | Mostly `gpt-4o-mini` where an LLM judge/extractor was used, plus rule/local metrics. |
| Legacy Qwen2.5 standard newsets from March | Mostly `gpt-4o` for judge-scored datasets, plus rule/local metrics. |
| Legacy DynaMath infer-only | `gpt-4o`. |
| MiniCPM April table/infer-only runs | Mostly `gpt-4o-mini` where an LLM judge/extractor was used, plus rule/local metrics. |

To reproduce exact historical numbers, match both the model checkpoint and the judge endpoint/model. For new public reruns, prefer the release default environment variables and record `OPENAI_API_BASE_JUDGE`, `OPENAI_API_KEY_JUDGE`, and the actual judge model in the run metadata.

The practical rule from the current provenance scan is: new-stack LLM-judged cells use `gpt-4o-mini`; March legacy standard cells mostly use `gpt-4o`; rule/local metrics have no LLM judge. There are two documented MathVision legacy-source caveat cells recorded with `gpt-4o-mini`, so do not infer the judge model from the directory name alone when auditing exact numbers.

## Known Provenance Caveats

These cells are intentionally called out because the current artifact scan does not provide a clean exact-number proof:

| Cell | Table target | Best source artifact | Recommended entrypoint |
|---|---:|---:|---|
| Qwen2.5 72B, MathVision, Q-I | 40.00 | 39.2434, manually accepted on 2026-04-29 | `scripts_legacy/run_missing_qwen25_72b_text_image_default_1node2workers_tp4.sh` or the release backfill matrix |
| Qwen2.5 32B, MathVision, I-Q-I-Q | 40.00 | 38.5197, manually accepted on 2026-04-29 | `matrix_qwen25vl32b_unified_topology_image2_20260404.yaml` |
| Qwen2.5 32B, MathVision, Q-I | 38.65 | 38.7829 near match | `scripts_legacy/run_missing_qwen25_32b_text_image_default_1node4workers_tp2.sh` or the release backfill matrix |
| Qwen2.5 3B, MathVision, Q-I | 22.24 | no exact aligned artifact found | `scripts_legacy/run_missing_qwen2_qwen25_small_text_image_default_1node8workers.sh` or the release backfill matrix |
| Qwen2.5 3B, MathVision, I-Q-I-Q | 21.00 | 19.9342 mismatch from an earlier source path | `scripts_legacy/run_standard_qwen2_qwen25_small_newsets_last1_default_2node16workers.sh` or the release backfill matrix |

These caveats should be preserved in any paper/release note. They are not code coverage gaps: the release has runnable entrypoints for them, but the historical artifact record does not prove exact equality to the table number.

## Practical Launch Examples

Plan the maintained two-node Qwen/MiniCPM new-four benchmark matrix:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml \
  --model-config configs/models.yaml \
  --scheduler gpu_pool \
  --plan-only
```

Run a single node-rank of the Gemma3 family matrix:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_gemma3_family_all11_replay6_2node_20260422.yaml \
  --model-config configs/models.yaml \
  --nodes 2 \
  --node-rank 0 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --task-manifest configs/task_manifests/gemma3_family_all11_replay6_2node_20260422/node0_tasks.csv \
  --manifest-is-node-shard \
  --scheduler gpu_pool
```

Run the release backfill manifest for legacy table cells:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_final_table_legacy_backfill_20260512.yaml \
  --model-config configs/models.yaml \
  --scheduler gpu_pool \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --task-manifest configs/task_manifests/final_table_legacy_backfill_20260512/all_tasks.csv
```

Run MiniCPM VisualPuzzles final-table parity:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config configs/matrix_minicpm_visualpuzzles_all_replay_eval_realign_20260420.yaml \
  --model-config configs/models.yaml \
  --scheduler gpu_pool
```
