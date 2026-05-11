export PATH=/usr/local/cuda/bin:$PATH
# source /path/to/home/vlmevalenv/bin/activate
# echo "Active Python virtual environment: $VIRTUAL_ENV"
pip config set global.index-url https://pypi.bj10.paratera.com/root/pypi/+simple/
pip install --upgrade pip
pip install -r requirements.txt
pip install -U seaborn tabulate
pip install -U sty
pip install -U portalocker
pip install -U pycocoevalcap
pip install -U openpyxl
pip install -U xlsxwriter
pip install -U validators
# pip uninstall -y opencv
pip install opencv-python==4.8.0.76
pip install numpy==1.26.4
pip install ruff==0.5.5
pip install tqdm==4.66.4
pip install rich
pip install pillow==9.4.0
pip install peft
pip install timm==0.9.10
pip install /path/to/wheels/starlette-0.42.0-py3-none-any.whl
pip install /path/to/wheels/torchvision-0.17.0+cu118-cp310-cp310-linux_x86_64.whl
pip install /path/to/wheels/torch-2.2.0+cu118-cp310-cp310-linux_x86_64.whl
pip install /path/to/wheels/flash_attn-2.5.9.post1+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# 运行 3o 需要降低版本，运行 2.6 之前的版本，可以更新到 4.43 以上的新版本
# pip install transformers==4.40.2
pip install transformers==4.44.2
pip list

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
