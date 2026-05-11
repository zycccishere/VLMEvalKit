set -x

export PATH=/usr/local/cuda/bin:$PATH

eval "$(conda shell.bash hook)"

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'
# export PYTHONPATH=$(dirname $SELF_DIR):$PYTHONPATH

source /opt/miniconda3/bin/activate

conda activate vlmevalkit_s2_baseline

export CUDA_VISIBLE_DEVICES="0,1,2,3"

which torchrun
which python

export FORCE_LOCAL=True

# fp16 17-18G
# int4 7-8G
MODELNAME=$1
DATALIST=$2
SAVE_PREFIX=$3
export MODEL_PATH=$4

save_root="${SAVE_ROOT:-${PWD}}"
work_dir="$save_root/$SAVE_PREFIX/output"

echo "work directory of $MODELNAME: $work_dir"

for DATASET in $DATALIST;
do
    echo "Starting inference with model $MODELNAME on dataset $DATASET"
    # torchrun --nproc-per-node 8 run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --mode infer --verbose --batch-size 8
    python run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --mode infer --verbose --batch-size 4

    echo "Starting evaluation with model $MODELNAME on datasets $DATASET"
    python run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --nproc 16 --verbose --judge gpt-4-turbo
done
# echo "Starting inference with model $MODELNAME on datasets $DATALIST"
# torchrun --nproc-per-node 4 run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --mode infer --verbose --batch-size 16

# echo "Starting evaluation with model $MODELNAME on datasets $DATALIST"
# python run.py --data $DATALIST --model $MODELNAME --work-dir ${work_dir} --nproc 16 --verbose
