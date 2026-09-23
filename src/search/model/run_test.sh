#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
ROOT_DIR=${CUR_DIR}/../../../

set -e
set -x

source /data/local/conda_env.sh
conda activate test
export LD_LIBRARY_PATH=${ROOT_DIR}/amd64/local/torch/gpu/12.2/lib/:$LD_LIBRARY_PATH 
cd ${ROOT_DIR}/amd64/test/

python ${CUR_DIR}/test_emb.py
./test_model_qwen3 --gtest_filter=Model.Qwen3ModelEmb

python ${CUR_DIR}/test_input_layernorm.py 0
./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayerInputLayer0

python ${CUR_DIR}/test_self_attn.py 0
./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayerAttn0

python ${CUR_DIR}/test_layer.py 0
./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayer0

python ${CUR_DIR}/test_layer.py 1
./test_model_qwen3 --gtest_filter=Model.Qwen3ModelLayer1

python ${CUR_DIR}/test_rotary_emb.py
./test_model_qwen3 --gtest_filter=Model.Qwen3RotaryEmb

./test_model_qwen3 --gtest_filter=Model.Qwen3AttentionMask

python ${CUR_DIR}/test_model.py
./test_model_qwen3 --gtest_filter=Model.Qwen3Model

python ${CUR_DIR}/test_lm.py
./test_model_qwen3 --gtest_filter=Model.Qwen3Lm

python ${CUR_DIR}/test_runner.py
./test_model_qwen3 --gtest_filter=Model.Qwen3Runner

# vim: set expandtab ts=4 sw=4 sts=4:
