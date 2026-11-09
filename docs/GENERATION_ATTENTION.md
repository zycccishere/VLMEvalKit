# LogicVista 正常生成定向干预

在 `V1,T1,V2,T2` 中，所有语言解码层禁止 **V2 读取 V1/T1**，保留
T2 与答案的正常因果注意力。它检验新增视觉块从前文获取信息的路径是否影响
正常生成；这不是“整个第二个 VT 与第一个 VT 隔离”，也不是受限选项读出。

| 条件 | 输入 | 注意力 |
| --- | --- | --- |
| `vtt` | V1,T1,T2 | 原生因果 |
| `slp` | V1,T1,V2,T2 | 原生因果 |
| `all_visible` | 同 SLP | 经过自定义 predicate，但不删除边；技术对照 |
| `no_write` | 同 SLP | 删除 V2 → V1/T1 的读取边 |

V2 范围包含 `<|vision_start|>`、全部 image-pad token、`<|vision_end|>`，
防止视觉分隔符在多层中传递被屏蔽的前文。key 从 V1 的 vision-start 开始，
到 V2 的 vision-start 前结束；开头 system/user header 保留。V2 内仍按因果
顺序可见。每题输出确切的半开区间、prompt token SHA-256 和删边数
`(query_end-query_start)*(query_start-key_start)`（每层、每头的结构计数）。

## 原生复用与版本边界

- 沿用 `LogicVista.build_prompt`、`build_standard_prompt`、`apply_replay`；
  从 `config_qwen_minimal.supported_VLM['Qwen2VLChatReplay']` 读取 prompt/
  图像默认配置：`use_custom_prompt=False`、min/max pixels 为
  `1280*28*28` / `16384*28*28`。不改问题、图像或模型权重。
- 环境：`/data/local/yicheng/venvs/prompt-replay-vlmeval`，vLLM `0.31.0+cu129`、
  PyTorch `2.13.0+cu129`、Transformers `5.17.0`。旧实验环境不改写。
- `GenerationAttentionWorker` 使用原生 `worker_extension_cls` 和
  `LLM.collective_rpc`；只在该 worker 进程内扩展 FlexAttention predicate，
  不编辑 site-packages，不实现新 kernel、KV cache 或调度器。
- 对接 vLLM 0.31 **V2 runner**：从 `req_states.add_request` 保存 prompt，
  从 `model_state.prepare_attn` 取得排序后的请求批次。不能把持久 slot 号，
  或 predicate 的 attention-batch 参数 `b`，误当成请求号。
- 使用原生 `_convert_physical_to_logical` 处理 paged KV / chunk offset，
  并要求原生 direct BlockMask 的 full blocks 为空，防止 bypass predicate。
- 启用 chunked prefill；no-write 仅在 query 覆盖 V2 时使用扩展 predicate，
  其余阶段恢复原生路径。范围只在相关 prefill 步传入 GPU，不生成 N×N dense mask。
- 当前要求 eager、无 prefix caching、同步调度；不同条件间不复用语言 KV。
  eager 避免捕获带旧条件的 CUDA graph。此版本尚不支持 request sub-batching、
  speculative decoding 或其他模型；版本/配置不符合即报错。

FlexAttention 避免 dense attention，但这份 adapter **没有额外裁剪整个 KV
block 的稀疏目录**；少了注意力边不代表 FLOPs 或耗时按同样比例下降。
运行秒数包含首次 kernel 编译和图像缓存冷热差异，不直接作为性能比较。

## 复现

从本 worktree 根目录运行。输出目录须为新的；数据/模型离线读取。

```bash
export VLMEVAL_LAZY_INIT=1 VLMEVAL_VLM_MINIMAL_IMPORT=1 VLMEVAL_API_MINIMAL_IMPORT=1
export LMUData=/data/local/yicheng/temporary/curated/prompt-replay/LMUData
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/data/local/yicheng/temporary/scratch/prompt-replay/no-write/inductor
export VLLM_CACHE_ROOT=/data/local/yicheng/temporary/scratch/prompt-replay/no-write/vllm-cache

# CUDA_VISIBLE_DEVICES 由启动时的实际空闲设备决定。
/data/local/yicheng/venvs/prompt-replay-vlmeval/bin/python -m vlmeval.probes.run_generation_attention \
  --output /data/local/yicheng/temporary/scratch/prompt-replay/no-write/NEW-RUN \
  --tp 1 --gpu-memory-utilization .92 --batch-size 1 \
  --max-model-len 8192 --max-batched-tokens 2048 --limit 4 --max-tokens 64
```

这是截短的正确性 smoke。正式 LogicVista 为 447 题；完整生成使用足够输出预算
（默认 2048），逐题检查 `finish_reason`，再复用原生 LogicVista scorer。
本入口不会调用 judge，也不会把生成完成标成准确率完成。`--prepare-only`
仅验证数据和冻结 manifest，不启动模型；`--conditions` 可限定条件。

## 验收

```bash
VLMEVAL_LAZY_INIT=1 /data/local/yicheng/venvs/prompt-replay-vlmeval/bin/python \
  -m unittest discover -s tests -p test_generation_attention.py -v
```

测试独立构造 dense 因果矩阵，检查精确屏蔽范围、system/header 保留、T2/答案
可见性；交错物理页、不同请求长度和 decode offset；真实 GPU 编译的
FlexAttention 与 dense SDPA 在 `2e-5` 容差内一致，并验证改变被禁止读取的
value 不影响相应 query 输出。

运行时检查每个 worker 的请求身份（vLLM 0.31 内部 ID 追加八位十六进制后缀）、
prompt hash、V2 边界、删边数和每个 V2 token 恰好覆盖一次。`parity.json` 要求 SLP/all-visible 全部生成 token
相同、首 token 的 top-5 集合相同且 log-prob 最大绝对差小于 `1e-3`。
`generation-complete.json` 明确写入 `scoring_status=not_scored`。

数据、代码、后端、模型 JSON 的 SHA-256 写入 manifest；权重保存文件名、
大小及 mtime，**未计算全部权重内容 hash**。模型输出和审计在 scratch，
不写入 Git。初版 smoke 的代码 hash 单独保存在运行目录的 source-snapshot。

2026-11-09 验证：4 个测试通过；随机小模型 8 题 × 4 条件的 chunked prefill /
batch 4 通过；真实 32B BF16 的 4 题 × 4 条件通过（64-token 截短 smoke）。
真实模型全放行的首 token top-5 log-prob 差为 0，no-write 改变 4 / 4 个生成序列。
额外八题完整输出验收因启动时显存不足未进入推理，全量 447 题尚未运行。
TP > 1 的真实模型验收也尚未完成。这些检查验证实现，不提供准确率或性能比较结论。

## 科学结论边界

若 SLP 相比 VTT 有收益，且 no-write 使其下降，可支持这条注意力路径对表现
有贡献。它不能单独证明 V2 保留了哪类语义内容，或排除改变注意力归一化的
一般影响。`all_visible` 只排查实现扰动，不能称为等删边 Sham。
全量配对准确率与置信区间完成前，不填写 reviewer 的实验证据回应。

## 2026-11-09 原生 greedy 与远端全量运行

用户明确跳过中间模型验收，直接在 `172.27.0.13:22` 使用四组 TP2 全量并发。
入口改为调用 `Qwen2VLChatReplay._build_vllm_sampling_params`，其实际 vLLM
默认是 temperature 0、repetition penalty 1、max_tokens 2048。初版 smoke 使用的
.01/1.05 来自其他推理入口的构造参数；历史文件不修改，历史技术 parity 仍是
同配置内部对照，不声称与 B1 解码一致。全量三臂均使用修正后的相同配置。

`--logprobs 0 --conditions slp,no_write,vtt` 关闭正式生成的逐token log-prob收集，
保留预测、token数、finish reason和逐worker mask审计。技术全放行模式仍要求
logprobs开启。随机种子按原始行偏移分配，不随分片边界重置。

`scripts/run_generation_attention_batch.py --config <config.json>` 是固定四组的
子进程驱动，每组加载一次原生LLM并依次跑三臂；不是新的通用调度器。它核对
执行主机、干净提交和八卡余量，记录每组PID、命令、日志与退出码，不自动重跑。
全部分片成功后，`score_generation_attention` 核对配置/源码/数据hash与每臂
447个原始index的唯一完整性，转换为标准预测表，再调用原生 `LogicVista.evaluate`。
判分复用 `run_bound.sh` 的凭据加载及清除 `LOCAL_LLM`/温度覆盖规则，实际模型为
gpt-4o-mini-2024-07-18，nproc 512。评分失败与推理失败分别记录，保留已完成产物。
