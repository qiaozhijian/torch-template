#!/bin/bash/env

export OMP_NUM_THREADS=24
# 获得第一个参数，并设默认值为1
export CUDA_VISIBLE_DEVICES=${1:-0}

echo "On node ${HOSTNAME}"
echo "CUDA VISIBLE DEVICES ${CUDA_VISIBLE_DEVICES}"

cd /data0/XXXX # 进入工作目录
python train.py --cfg_file configs/model_configs/simplenet.yaml