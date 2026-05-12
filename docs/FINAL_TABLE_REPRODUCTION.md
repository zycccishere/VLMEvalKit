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

Use `scripts/run_benchmark.sh` and the matrix YAMLs for new reruns. Use `scripts_legacy/` when you need to match the old launch shape exactly. The legacy scripts are kept because part of the main table came from those old runs.

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
| Gemma3 4B/12B/27B on all 11 table datasets and all 6 replay modes | `bash scripts/run_gemma3_family_all11_replay6_2nodes_20260422.sh <node_rank> [gpu_ids]` |
| Gemma3 12B I-Q on `MathVision`, `WeMath`, `MMBench_DEV_EN_V11`, `MMStar` when matching the recorded table provenance | `bash scripts/run_benchmark.sh --matrix-config scripts/configs/matrix_gemma3_12b_reference4_image_text_20260422.yaml --model-config scripts/configs/models.yaml --scheduler gpu_pool` |
| Qwen2.5 3B/32B/72B and MiniCPM-V/O on `MMMU_DEV_VAL_SINGLE_IMAGE`, `WeMath`, `MMBench_DEV_EN_V11`, `MMStar` | `bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh <node_rank> [gpu_ids]` |
| Qwen2.5 7B on `MMMU_DEV_VAL_SINGLE_IMAGE`, `WeMath`, `MMBench_DEV_EN_V11`, `MMStar` | Mixed source: most cells use `bash scripts/run_qwen25vl_all4_reasoning_perception4_new_entry_4nodes_20260421.sh <node_rank> [gpu_ids]`, while 8 cells use `bash scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh <node_rank> [gpu_ids]`. Use `docs/final_table_cell_sources.csv` for the exact split. |
| Qwen2.5 3B/7B/32B/72B on `LogicVista` and `VisualPuzzles` | `bash scripts/run_benchmark.sh --matrix-config scripts/configs/matrix_qwen25vl_all4_reasoning4_new_entry_20260421.yaml --model-config scripts/configs/models.yaml --scheduler gpu_pool --datasets LogicVista VisualPuzzles` |
| Qwen2.5 legacy cells on `AI2D_TEST`, `DynaMath`, `MathVision`, `OCRBench`, `SEEDBench2_Plus` | Use the legacy scripts listed in `docs/final_table_reproduction_entries.csv`; `bash scripts/run_final_table_legacy_backfill_20260512.sh` is only the 21-cell release backfill for `Qwen2.5 3B/72B` on `AI2D_TEST`, `OCRBench`, and `SEEDBench2_Plus`. |
| MiniCPM-V/O on `AI2D_TEST`, `DynaMath`, `MathVision`, `OCRBench`, `SEEDBench2_Plus` | Use `matrix_qwen25vl7b_minicpm45_table_20260406.yaml`, `matrix_minicpm_default_infer_only_fresh_20260317.yaml`, and `run_legacy_dynamath_infer_only_all_replay.sh` as listed in the CSV. |
| MiniCPM-V/O on `LogicVista` | `bash scripts/run_benchmark.sh --matrix-config scripts/configs/matrix_minicpm_logicvista_all_replay_eval_20260419.yaml --model-config scripts/configs/models.yaml --scheduler gpu_pool` |
| MiniCPM-V/O on `VisualPuzzles` | `bash scripts/run_benchmark.sh --matrix-config scripts/configs/matrix_minicpm_visualpuzzles_all_replay_eval_realign_20260420.yaml --model-config scripts/configs/models.yaml --scheduler gpu_pool` |

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
  --matrix-config scripts/configs/matrix_qwen25vl_minicpm45_all4_reasoning_perception4_2node_20260422.yaml \
  --model-config scripts/configs/models.yaml \
  --scheduler gpu_pool \
  --plan-only
```

Run a single node-rank of the Gemma3 family matrix:

```bash
bash scripts/run_gemma3_family_all11_replay6_2nodes_20260422.sh 0 0,1,2,3,4,5,6,7
```

Run the release backfill manifest for legacy table cells:

```bash
bash scripts/run_final_table_legacy_backfill_20260512.sh
```

Run MiniCPM VisualPuzzles final-table parity:

```bash
bash scripts/run_benchmark.sh \
  --matrix-config scripts/configs/matrix_minicpm_visualpuzzles_all_replay_eval_realign_20260420.yaml \
  --model-config scripts/configs/models.yaml \
  --scheduler gpu_pool
```
