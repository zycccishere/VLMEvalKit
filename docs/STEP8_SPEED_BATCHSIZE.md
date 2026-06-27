# Step 8 Speed Batch Size Record

This record persists the real standard-entry batch-size search for active
open-source models. Raw logs are intentionally kept under ignored runtime
state and are referenced here by repo-relative paths.

## Scope

- Dataset: `MathVision`
- Samples: 512 real dataset rows
- Replay mode: `image_text_image_text` (`IQIQ`)
- Standard entry: `bash scripts/run_benchmark.sh`
- Raw run root: `runs/step8_speed_20260627`
- Candidate result log: `runs/step8_speed_20260627/candidate_results.jsonl`
- Sample manifest SHA256: `3bbec440367bb90ff2bc04664cd6697e93d9b4d7a760da3e64f073352cdf8058`
- Host: `devspace-job-imagereplay-506002-main-0`
- GPU: `NVIDIA H100 80GB HBM3`, driver `560.35.05`
- Runtime: torch `2.10.0+cu128`, transformers `4.57.6`, vLLM `0.17.1`

Closed-source API models are not part of batch-speed search. The open-source
active routes all use vLLM through the standard model config / environment
profile, not via the generic `run.py --use-vllm` flag. For this MathVision
speed run the backend is vLLM v1. The LogicVista Qwen2.5VL route remains the
separate vLLM v0 special case documented in the route matrix.

## Recommended Profiles

`max_safe_batch_size` is the largest clean 512-sample batch tested. Because the
search dataset has exactly 512 samples, batch sizes above 512 were not tested:
they would not increase real concurrent samples. `recommended_batch_size` is
the fastest clean tested batch.

| Model | TP | Max safe batch | Recommended batch | Samples/s | Peak MiB | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-VL-3B | 1 | 512 | 512 | 1.534892 | 78290 | Reached 512 cap |
| Qwen2.5-VL-7B | 1 | 512 | 512 | 1.675419 | 78290 | Reached 512 cap |
| Qwen2.5-VL-32B | 2 | 512 | 256 | 0.898735 | 77974 | 512 passes but is slightly slower |
| Qwen2.5-VL-72B | 4 | 512 | 256 | 0.903592 | 79410 | 512 passes but is slower than 256 |
| MiniCPM-V-4.5 | 1 | 512 | 512 | 0.752623 | 74232 | Reached 512 cap |
| MiniCPM-o-4.5 | 1 | 512 | 256 | 0.912878 | 62574 | 512 passes but is slightly slower |
| Gemma3-4B | 1 | 512 | 512 | 1.752788 | 75910 | Reached 512 cap |
| Gemma3-12B | 1 | 512 | 512 | 1.165110 | 75908 | Reached 512 cap |
| Gemma3-27B | 2 | 512 | 512 | 1.037543 | 75924 | Reached 512 cap |

## Batch Ladders

- `qwen25vl_3b`: 512 pass, 1.534892 samples/s
- `qwen25vl_7b`: 512 pass, 1.675419 samples/s
- `qwen25vl_32b`: 64 pass, 128 pass, 256 pass, 512 pass; best speed at 256
- `qwen25vl_72b`: 16 pass, 32 pass, 64 pass, 128 pass, 256 pass, 512 pass; best speed at 256
- `minicpm_v_45`: 256 pass, 512 pass; best speed at 512
- `minicpm_o_45`: 256 pass, 512 pass; best speed at 256
- `gemma3_4b`: 256 pass, 512 pass; best speed at 512
- `gemma3_12b`: 128 pass, 256 pass, 512 pass; best speed at 512
- `gemma3_27b`: 64 pass, 128 pass, 256 pass, 512 pass; best speed at 512

## Backend Evidence

Every candidate was launched through `scripts/run_benchmark.sh` with a
candidate-local matrix and model config. Multi-GPU vLLM runs also have
`VLLM::Worker_TP*` evidence in worker-status logs, for example:

- `runs/step8_speed_20260627/search/qwen25vl_32b/n512_bs256/results/_logs/worker_status/node0_qwen25vl_32b_slot0.log`
- `runs/step8_speed_20260627/search/qwen25vl_72b/n512_bs256/results/_logs/worker_status/node0_qwen25vl_72b_slot0.log`
- `runs/step8_speed_20260627/search/gemma3_27b/n512_bs512/results/_logs/worker_status/node0_gemma3_27b_slot0.log`

Single-GPU vLLM routes are evidenced by the standard-entry command, the
candidate-local env profile, and model config:

- MiniCPM 4.5: `minicpm45_vllm` profile with `MINICPM45_USE_VLLM=1`
- Gemma3: `gemma3_vllm` profile with `GEMMA3_USE_VLLM=1`
- Qwen2.5VL: replay config sets `use_vllm=True`

Default generation settings for these speed runs:

- Qwen2.5VL: `temperature=0.0`, `max_tokens=2048`
- MiniCPM 4.5: greedy vLLM sampling, `max_tokens=16384`
- Gemma3: `temperature=0.0`, `max_tokens=4096`
