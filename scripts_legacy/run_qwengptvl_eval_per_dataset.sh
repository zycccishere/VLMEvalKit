set -x

export PATH=/usr/local/cuda/bin:$PATH

eval "$(conda shell.bash hook)"

source /opt/miniconda3/bin/activate

conda activate vlmevalkit_s2_baseline

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'

export CUDA_VISIBLE_DEVICES="0"

which torchrun
which python

export FORCE_LOCAL=True

MODELNAME=$1
DATALIST=$2
SAVE_PREFIX=$3
export MODEL_PATH=$4

save_root="${SAVE_ROOT:-/path/to/vlmevalkit}"
work_dir="$save_root/$SAVE_PREFIX/output"

echo "work directory of $MODELNAME: $work_dir"

for DATASET in $DATALIST;
do
    echo "Starting inference with model $MODELNAME on dataset $DATASET"
    python run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --mode infer --verbose --batch-size 1

    echo "Starting evaluation with model $MODELNAME on datasets $DATASET"
    python run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --nproc 8 --verbose --judge gpt-4o
done
