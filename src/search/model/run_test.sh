#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
ROOT_DIR=${CUR_DIR}/../../../
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  ARCH_NAME=amd64
else
  ARCH_NAME=arm
fi

set -e
set -x

source /data/local/conda_env.sh
# pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 transformers
conda activate test
cd ${ROOT_DIR}/${ARCH_NAME}/test/
# Arm Default 1, Must Set
export OMP_NUM_THREADS=32

python ${CUR_DIR}/test_emb.py
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3ModelEmb

python ${CUR_DIR}/test_input_layernorm.py 0
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayerInputLayer0

python ${CUR_DIR}/test_self_attn.py 0
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayerAttn0

python ${CUR_DIR}/test_layer.py 0
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayer0

python ${CUR_DIR}/test_layer.py 1
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayer1

python ${CUR_DIR}/test_rotary_emb.py
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3RotaryEmb

LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3AttentionMask

python ${CUR_DIR}/test_model.py
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3Model

python ${CUR_DIR}/test_lm.py
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3Lm

python ${CUR_DIR}/test_runner.py
LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3Runner

LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ./test_model_qwen3 --gtest_filter=Model.Qwen3RunnerCache

which nvidia-smi &> /dev/null
if [ $? -eq 0 ]; then
  LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/gpu/12.2/lib/:$LD_LIBRARY_PATH  ./test_model_qwen3_gpu_12.2 --gtest_filter=Model.Qwen3RunnerCache
fi

which npu-smi &> /dev/null
if [ $? -eq 0 ]; then
  LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch_npu/lib/:${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH  ./test_model_qwen3_npu_9.0.0 --gtest_filter=Model.Qwen3RunnerCache
fi

# vim: set expandtab ts=4 sw=4 sts=4:
