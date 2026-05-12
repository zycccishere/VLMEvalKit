#!/bin/bash

echo "开始并行运行四个评估任务..."

# 直接在命令前设置环境变量
CUDA_VISIBLE_DEVICES=0 SAVE_ROOT="/path/to/home/output" bash ./scripts/run_duplexthinker_eval_per_dataset.sh "DuplexThinkerS2" "MMVet OCRBench MathVistaSample MMStarSample MMBench_DEV_EN_V11 MMMU_DEV_VAL" "S0-P-DuplexThinkerS2-pretrain-epo2-sft-23000-wsd-seed-47-epo2-finevision-4M-freeze-p2" "/path/to/home/workspace/LLaMA-Factory-S2/saves/duplex/full/sft_duplex_forward_mlp_bs256_from_epo2_wsd_seed_47_epo2_finevision_4M_freeze_17M/checkpoint-23000" &

CUDA_VISIBLE_DEVICES=1 SAVE_ROOT="/path/to/home/output" bash ./scripts/run_duplexthinker_eval_per_dataset.sh "DuplexThinkerS2" "MMVet OCRBench MathVistaSample MMStarSample MMBench_DEV_EN_V11 MMMU_DEV_VAL" "S0-P-DuplexThinkerS2-pretrain-epo2-sft-24000-wsd-seed-47-epo2-finevision-4M-freeze-p2" "/path/to/home/workspace/LLaMA-Factory-S2/saves/duplex/full/sft_duplex_forward_mlp_bs256_from_epo2_wsd_seed_47_epo2_finevision_4M_freeze_17M/checkpoint-24000" &

# CUDA_VISIBLE_DEVICES=2 SAVE_ROOT="/path/to/home/output" bash ./scripts/run_duplexthinker_eval_per_dataset.sh "DuplexThinkerS2" "MMVet OCRBench MathVistaSample MMStarSample MMBench_DEV_EN_V11 MMMU_DEV_VAL" "S0-P-DuplexThinkerS2-pretrain-epo2-sft-18000-wsd-seed-47-epo2-finevision-4M-freeze-p2" "/path/to/home/workspace/LLaMA-Factory-S2/saves/duplex/full/sft_duplex_forward_mlp_bs256_from_epo2_wsd_seed_47_epo2_finevision_4M_freeze/checkpoint-18000" &

# CUDA_VISIBLE_DEVICES=3 SAVE_ROOT="/path/to/home/output" bash ./scripts/run_duplexthinker_eval_per_dataset.sh "DuplexThinkerS2" "MMVet OCRBench MathVistaSample MMStarSample MMBench_DEV_EN_V11 MMMU_DEV_VAL" "S0-P-DuplexThinkerS2-pretrain-epo2-sft-19000-wsd-seed-47-epo2-finevision-4M-freeze-p2" "/path/to/home/workspace/LLaMA-Factory-S2/saves/duplex/full/sft_duplex_forward_mlp_bs256_from_epo2_wsd_seed_47_epo2_finevision_4M_freeze/checkpoint-19000" &

# 等待所有后台任务完成
wait

echo "所有评估任务已完成!"
