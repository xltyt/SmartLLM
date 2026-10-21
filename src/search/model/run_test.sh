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
BASE=0

set -x

source /data/local/conda_env.sh
# pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 transformers
conda activate test
cd ${ROOT_DIR}/${ARCH_NAME}/test/
# Arm Default 1, Must Set
export OMP_NUM_THREADS=32

#MODEL_NAMES=(Qwen3-0.6B Qwen3-4B)
#MODEL_NAMES=(Qwen3-4B)
MODEL_NAMES=(Qwen3-0.6B)
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  mkdir -p ${MODEL_NAME}
  cd ${MODEL_NAME}
  if [ $BASE -eq 1 ]; then
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_emb.py
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3ModelEmb
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_input_layernorm.py 0
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayerInputLayer0
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_self_attn.py 0
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayerAttn0
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_layer.py 0
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayer0
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_layer.py 1
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayer1
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_rotary_emb.py
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3RotaryEmb
    
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3AttentionMask
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_model.py
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3Model
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_lm.py
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3Lm
    
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_runner.py
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3Runner
    
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3 --gtest_filter=Model.Qwen3RunnerCache
  fi
  if [ ! -f ./test_runner.pt ] || [ ! -f ./test_runner_logits.pt ]; then
    MODEL_NAME=${MODEL_NAME} python ${CUR_DIR}/test_runner.py || exit 1
  fi
  which nvidia-smi &> /dev/null
  if [ $? -eq 0 ]; then
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch/gpu/12.2/lib/:$LD_LIBRARY_PATH ../test_model_qwen3_gpu_12.2 --gtest_filter=Model.Qwen3RunnerCache
  fi
  
  which npu-smi &> /dev/null
  if [ $? -eq 0 ]; then
    MODEL_NAME=${MODEL_NAME} LD_LIBRARY_PATH=${ROOT_DIR}/${ARCH_NAME}/local/torch_npu/lib/:${ROOT_DIR}/${ARCH_NAME}/local/torch/cpu/lib/:$LD_LIBRARY_PATH ../test_model_qwen3_npu_9.0.0 --gtest_filter=Model.Qwen3RunnerCache
  fi
  cd ..
done

# vim: set expandtab ts=4 sw=4 sts=4:
