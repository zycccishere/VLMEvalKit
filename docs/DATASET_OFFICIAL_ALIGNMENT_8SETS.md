# 8 个数据集与官方 VLMEvalKit 行为对拍说明

**基准**：`open-compass/VLMEvalKit` 的 `main` 分支（通过 `raw.githubusercontent.com/.../VLMEvalKit/main/...` 核对）。

**范围**：当前 `image_text` 套件已跑过的 8 个数据集：`AI2D_TEST`、`SEEDBench2_Plus`、`VisualPuzzles`、`VisuLogic`、`OCRBench`、`MathVista_MINI`、`MathVision`、`LogicVista`。

---

## 1. 通用层（一改多动）

### 1.1 `multiple_choice.py` — MCQ + LLM judge

| 维度 | 官方 | 本地 |
|------|------|------|
| 送给 judge 的文本 | **完整** `item['prediction']` | **`tail_tokens_for_judge(..., max_tokens=96)`**（见 `prompt_tail.py`） |
| judge 失败兜底 | `random.choice(choices + ['Z'])` | **固定返回 `Z`** |
| 其它 | 无 | `FORCE_GPT_JUDGE_ALL` 可强制跳过 `can_infer` 预检 |

**影响数据集**：所有走 `mcq_vanilla_eval` → `extract_answer_from_item` 的 MCQ，包括 **`AI2D_TEST`、`SEEDBench2_Plus`**（以及未列入本 8 集但同路径的 MMBench 等）。

**风险**：**高** — 长 CoT 时正确答案常落在尾部，截断会系统性改变 judge 输入；失败兜底从随机改为 Z 会改变边界样本分布。

### 1.2 `image_mcq.py` — `ImageMCQDataset.evaluate`

| 维度 | 官方 | 本地 |
|------|------|------|
| 入口 | `evaluate` → `use_verifier` 则 `evaluate_verifier`，否则 **`evaluate_heuristic`** | **单一路径**（无顶层 `use_verifier` 分流；子类另说） |
| 预测预处理 | 直接对 `prediction` 评测 | 增加 **`ori_prediction`**，并对若干数据集用正则从全文抽 `Answer:\s*([A-D])` 等写入 **`prediction`**（含 **`AI2D_TEST`、`MMStar`、`MMBench_DEV_EN_V11`…**） |
| Circular | `mmbench` / `ccbench` / **`circular`** / **`mmcr`**（大小写不敏感） | 仅 **`mmbench`、`ccbench`** |
| 中间文件路径 | 多用 `get_intermediate_file_path` | 部分用 `eval_file.replace(...)` |
| 若已有 `_acc.csv` | 视数据集而定 | **若存在则直接返回**（可能跳过重算） |

**影响数据集**：**`AI2D_TEST`、`SEEDBench2_Plus`** 等共享 `ImageMCQDataset` 的 MCQ。

**风险**：**高** — `Answer:` 预过滤 + tail judge 叠加后，与官方「整条 string + 全量给 judge」不一致。

### 1.3 `prompt_tail.py`

本地新增：按「词」切分取最后 N 个 token 的尾部，供 judge / 抽取使用。**官方无此文件、无此语义。**

**风险**：**高**（作为上述所有 tail 行为的根因）。

### 1.4 `judge_util.py`

| 维度 | 官方 | 本地 |
|------|------|------|
| 后端 | 除 OpenAI 外还可 **SiliconFlow / HFChatModel** 等 | 主要为 **`OpenAIWrapper`** + 少量模型名映射 |
| 模型映射 | 含 `qwen-7b`、`deepseek` 等 | 含 **`grok-4-fast`** 等，与官方表不完全一致 |

**风险**：**中** — 同一 `gpt-4o-mini` 别名若解析到不同 endpoint/版本， judge 输出可能略有差异；与「数据集逻辑」相比属次要。

---

## 2. 按数据集专项

### 2.1 `AI2D_TEST`

- **Prompt**：与官方同属 `ImageMCQDataset.build_prompt` 体系时，大体一致；**差异主要在评测链**（见通用层）。
- **不对齐点**：`Answer:` 行预截取 + **tail judge** + judge 失败 **Z**（官方随机）。
- **数据**：`DATASET_URL` 多为本地 TSV 路径，需与 OpenCompass 源一致才可和 leaderboard 比。

**风险**：**高**（评测链）。

### 2.2 `SEEDBench2_Plus`

- 与 `AI2D_TEST` 同属 **`ImageMCQDataset`**，**不对齐点与 2.1 相同**（除非该集在本地走了完全不同的子类；当前注册关系下视为共享 MCQ 路径）。
- **数据**：本地 `SEEDBench2_Plus` 指向自定义 TSV。

**风险**：**高**（评测链 + 数据路径）。

### 2.3 `VisualPuzzles`

| 维度 | 官方 | 本地 |
|------|------|------|
| `build_prompt`（`image_mcq.py`） | 与本地片段一致 | 与官方一致 |
| `utils/visualpuzzles.py` | `extract_answer`：`\banswer\s*:\s*([A-Z])\b`，否则 **Z**；**无** tail、**无** 二次 judge | **boxed / 多模式** + **`tail_tokens_for_judge(96)`** + 可选 **`_visualpuzzles_extract_with_judge`**、`VisulPuzzles_acc(..., fallback_cache_file, ...)` |

**风险**：**高** — 评分与官方**不是同一套规则**。

### 2.4 `VisuLogic`

| 维度 | 官方 | 本地 |
|------|------|------|
| `extract_answer` | boxed 或 `extract_lang_content`，否则 **Z** | 规则集扩展 + **tail** + **judge 二次抽取** + `extracted_answer` 缓存列 |

**风险**：**高**。

### 2.5 `OCRBench`

| 维度 | 官方 | 本地 |
|------|------|------|
| 计分逻辑（`answer in predict`、分类汇总） | 一致 | **一致** |
| 缓存 | `get_intermediate_file_path(..., '_score', 'json')` | **`eval_file.replace('.xlsx', '_score.json')`** + **若已存在则直接 `return`** |

**风险**：**低** — 数值逻辑对齐；差异主要在**缓存路径与是否短路重算**。

### 2.6 `MathVista_MINI`

| 维度 | 官方 | 本地 |
|------|------|------|
| GPT 抽取 prompt | `Model respone: ` + **完整** `prediction` | **`Model response tail (last 96 tokens):`** + `tail_tokens_for_judge` |
| 其它 | 标准 `MathVista_auxeval` 流程 | 同路径但输入不同；可能还有 `post_check` 等对 list 类型的额外归一 |

**风险**：**高** — 抽取阶段与官方不一致。

### 2.7 `MathVision`

| 维度 | 官方 | 本地 |
|------|------|------|
| `build_mathv_gpt4_prompt` | 完整 `prediction`（`Model respone:`） | **tail 96 tokens** |
| `is_equal` | latex / eval 失败后 **返回 False** | 失败后额外 **`verify(asw, gt_asw)`**（`math_verify`） |
| `evaluate` | 官方亦有 `use_verifier` 分支 | 本地 **`evaluate_heuristic` / `evaluate_verifier`** 均存在；与 `ImageMCQDataset` 不同，此处与官方结构接近 |

**风险**：**高** — tail + `verify` 扩展会改变大量样本是否判对。

### 2.8 `LogicVista`

| 维度 | 官方 | 本地 |
|------|------|------|
| judge 输入 | `build_prompt_logicvista` 使用 **完整** `prediction` | **`tail_tokens_for_judge(96)`** |
| judge 输出解析 | 要求 **`res.isupper()` 且全字母** 等严格条件 | **本地化解析**（如 `_parse_logicvista_choice`、数字选项、去掉过严的 `isupper` 约束等，以当前文件为准） |

**风险**：**高** — 与官方 extractor 语义差异大。

---

## 3. 风险汇总（按严重程度）

| 级别 | 说明 |
|------|------|
| **高** | `multiple_choice` tail judge；`ImageMCQDataset` 的 `Answer:` 预处理与无 `use_verifier` 顶层分流；`VisualPuzzles` / `VisuLogic` 专用 utils；`MathVista` / `MathVision` / `LogicVista` 的 tail + 解析/verify 扩展 |
| **中** | `judge_util` 模型路由与映射差异；`MathVision` 与官方在 verifier 路径上的细节需逐行 diff |
| **低** | `OCRBench` 仅缓存/路径；各数据集 **本地 TSV 路径** 与官方 URL 不一致导致的「数据非同一分布」 |

---

## 4. 若要与官方 leaderboard 严格可比：建议的验证动作

1. **MCQ 类（AI2D、SEEDBench2_Plus）**：固定小样本（如 20～50 条），对比「官方逻辑：全量 prediction + 无 Answer 预剪」与「当前逻辑」的 `hit` 差异比例。
2. **VisualPuzzles / VisuLogic**：各抽几十条，对比官方 `extract_answer` 与本地 `VisulPuzzles_acc` / `VisuLogic_acc` 的最终选项。
3. **MathVista_MINI / MathVision**：对比「全量 response vs tail-96」的抽取结果差异。
4. **LogicVista**：对比全量 vs tail 的 judge 输出一致性。
5. **数据**：确认 TSV 与 MD5 与官方一致（或使用官方 URL 下载的同一份文件）。

---

## 5. 结论（一句话）

当前本地在这 8 个数据集上，**推理前 prompt 部分多数与官方相近或一致（如 VisualPuzzles 的 `build_prompt`）**，但 **评测链（tail、MCQ judge、专用 utils、Math 等价判定）与官方存在多处系统性差异**；若目标是「和官方 VLMEvalKit 行为对齐」，需要按上表逐项收敛，而不能只对齐模型权重或 Qwen 封装。
