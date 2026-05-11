export PATH=/usr/local/cuda/bin:$PATH

eval "$(conda shell.bash hook)"

conda activate vlmevalkit

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'
# export PYTHONPATH=$(dirname $SELF_DIR):$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

which torchrun
which python

export MODEL_PATH=$1
export FORCE_LOCAL=True

# fp16 17-18G
# int4 7-8G
MODELNAME=$2
DATALIST=$3
SAVE_PREFIX=$4

base_name=$(basename $MODEL_PATH .pt)

save_root="${SAVE_ROOT:-/data/checkpoints/vlmeval_kit}"
work_dir="$save_root/$SAVE_PREFIX/$base_name"

echo "work directory of $MODELNAME: $work_dir"

# for DATASET in $DATALIST;
# do
#     echo "Starting inference with model $MODELNAME on dataset $DATASET"
#     torchrun --master_port 29500 --nproc_per_node=8 run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --mode infer
#     torchrun --master_port 29501 --nproc_per_node=8 run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --mode infer
#     echo "Starting evaluation with model $MODELNAME on datasets $DATASET"
#     python run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --nproc 16 --verbose
# done
echo "Starting inference with model $MODELNAME on datasets $DATALIST"
torchrun --master_port 29500 --nproc_per_node=8 run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --mode infer
torchrun --master_port 29501 --nproc_per_node=8 run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --mode infer

echo "Starting evaluation with model $MODELNAME on datasets $DATALIST"
python run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --nproc 16 --verbose
