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

### Final Table Reproduction Map

Use [`docs/FINAL_TABLE_REPRODUCTION.md`](docs/FINAL_TABLE_REPRODUCTION.md) when reproducing the current WorkHub `final_table.csv`. It maps each model/dataset family to the correct release or legacy entrypoint and records judge/evaluator differences observed in the source artifacts.

The short version is:

- Gemma3 4B/12B/27B, all 11 datasets, all replay modes: `scripts/run_gemma3_family_all11_replay6_2nodes_20260422.sh`.
- Qwen2.5 3B/32B/72B and MiniCPM-V/O on `MMMU_DEV_VAL_SINGLE_IMAGE`, `WeMath`, `MMBench_DEV_EN_V11`, `MMStar`: `scripts/run_qwen25vl_minicpm45_all4_reasoning_perception4_2nodes_20260422.sh`.
- Qwen2.5 7B on those four new benchmarks is mixed-source: most cells use `scripts/run_qwen25vl_all4_reasoning_perception4_new_entry_4nodes_20260421.sh`, while 8 cells use the later mixed Qwen/MiniCPM two-node run. Use the cell-level map before rerunning Qwen2.5-7B new-four cells.
- Qwen2.5 on `LogicVista` and `VisualPuzzles`: `scripts/run_benchmark.sh --matrix-config scripts/configs/matrix_qwen25vl_all4_reasoning4_new_entry_20260421.yaml --model-config scripts/configs/models.yaml --scheduler gpu_pool --datasets LogicVista VisualPuzzles`.
- Qwen2.5 legacy `AI2D_TEST`/`DynaMath`/`MathVision`/`OCRBench`/`SEEDBench2_Plus` cells: use the source-equivalent scripts in `scripts_legacy/`; `scripts/run_final_table_legacy_backfill_20260512.sh` is the cleaner release-runner backfill for cells missing from active matrices.
- MiniCPM-V/O core-set cells: use `matrix_qwen25vl7b_minicpm45_table_20260406.yaml`, `matrix_minicpm_default_infer_only_fresh_20260317.yaml`, and `scripts/run_legacy_dynamath_infer_only_all_replay.sh` according to the detailed CSV.
- MiniCPM-V/O `LogicVista` and `VisualPuzzles`: use `matrix_minicpm_logicvista_all_replay_eval_20260419.yaml` and `matrix_minicpm_visualpuzzles_all_replay_eval_realign_20260420.yaml`.

The source-group map is [`docs/final_table_reproduction_entries.csv`](docs/final_table_reproduction_entries.csv). The per-cell map is [`docs/final_table_cell_sources.csv`](docs/final_table_cell_sources.csv). The latter is the authoritative entrypoint map when a model/dataset family is mixed-source, and it records five MathVision provenance caveats where the current artifact scan found manual overrides, near matches, or missing exact metrics.

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
