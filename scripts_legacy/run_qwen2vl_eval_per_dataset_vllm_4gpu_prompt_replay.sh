set -x

export PATH=/usr/local/cuda/bin:$PATH

eval "$(conda shell.bash hook)"
source /opt/miniconda3/bin/activate
conda activate vlmevalkit_s2_baseline

export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
export timestamp=`date +"%Y%m%d%H%M%S"`
export OLD_VERSION='False'

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

which torchrun
which python

export FORCE_LOCAL=True

MODELNAME=$1
DATALIST=$2
SAVE_PREFIX=$3
export MODEL_PATH=$4

export REPLAY_MODE=image_text_text
export REPLAY_TIMES=1
export REPLAY_DEBUG=${REPLAY_DEBUG:-0}
export REPLAY_LIMIT_MM_PER_PROMPT=${REPLAY_LIMIT_MM_PER_PROMPT:-1}
export REPLAY_IMAGE_COPY_MODE=${REPLAY_IMAGE_COPY_MODE:-reuse_path}

save_root="${SAVE_ROOT:-/path/to/vlmevalkit}"
work_dir="$save_root/$SAVE_PREFIX/output"

echo "work directory of $MODELNAME: $work_dir"
echo "Replay settings: mode=$REPLAY_MODE times=$REPLAY_TIMES debug=$REPLAY_DEBUG limit_mm=$REPLAY_LIMIT_MM_PER_PROMPT copy_mode=$REPLAY_IMAGE_COPY_MODE"

for DATASET in $DATALIST;
do
    echo "Starting inference with model $MODELNAME on dataset $DATASET"
    python run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --mode infer --verbose --batch-size 4

    echo "Starting evaluation with model $MODELNAME on datasets $DATASET"
    python run.py --data $DATASET --model $MODELNAME --work-dir ${work_dir} --nproc 8 --verbose --judge gpt-4o
done
