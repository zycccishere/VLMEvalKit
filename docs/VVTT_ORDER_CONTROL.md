# VVTT input-order control

The `image_image_text_text` replay mode reuses `replay_policy.apply_replay`
with the same image/text items as SLP. `prepare_bound_run.py` binds it to the
existing final-protocol `VVTT` cells. Execution remains the existing bound
launcher, native GPU queue, model wrappers, and background official evaluator.

Equal items alone do not establish equal engine token counts. Run the bounded
native preprocessing audit in the same environment as inference:

```bash
VLMEVAL_LAZY_INIT=1 VLMEVAL_VLM_MINIMAL_IMPORT=1 VLMEVAL_API_MINIMAL_IMPORT=1 \
python scripts/probe_vvtt_payload.py --matrix /path/to/matrix.yaml \
  --models-config /path/to/models.yaml --tasks /path/to/tasks.csv \
  --models qwen25vl_32b --samples 5 --output /scratch/qwen-native30.json
```

The audit reuses `BenchmarkRunner.build_env`, `standard_entry_parity` for
lightweight wrapper initialization and native dataset prompt construction,
then the wrappers' real request builders and vLLM `HfRenderer.render_cmpl`.
It checks image bytes/sizes, text item multiset, image span lengths and order,
and reports final engine token length and token multiset equality. Preview
traces stay under the output directory, never the formal results root. Model
weights are not loaded; this verifies preprocessing, not generated accuracy.

On vLLM 0.17.1, Qwen2.5-VL-32B passed 30 paired samples (first five rows of
each of AI2D_TEST, DynaMath, LogicVista, MMBench_DEV_EN_V11, MathVision,
OCRBench): identical token counts and multisets, same images and text,
V2 moved before T1. Dataset-specific image resizing remains native. This
bounded test is not an exhaustive audit of every future question.

Two other model/version boundaries remain explicit:

- Gemma3: six sampled datasets each lose one token under native VVTT.
  `Gemma3MultiModalProcessor._apply_token_matches` merges adjacent newline
  tokens; both visual blocks and text items remain present. Do not claim
  exact engine-token matching or silently add padding.
- MiniCPM-o 4.5: the vLLM 0.17.1 string chat parser defaults to
  `interleave_mm_strings=False`. With the current wrapper, SLP and VVTT
  parse identically. Six archived B1 first-sample inputs reconstructed with
  the recorded code/version/settings confirmed this. Those traces were
  captured before parsing, not historical engine token dumps. Changing the
  native parser option requires matched SLP anchors and an explicit protocol
  decision; it is not enabled by this patch.

No model wrapper, attention kernel, sampling parameter, or installed vLLM
library is changed. The basic replay unit tests verify item construction;
they deliberately make no tokenizer-level claim.
