#!/bin/bash

# https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md
# 
# | TorchNPU | TorchNPU Package | PyTorch | CANN |
# |------|------|------|------|
# | 26.1.1 | 2.10.0.post6 | 2.10.0 | 9.1.X |
# | 26.1.0 | 2.10.0.post4 | 2.10.0 | 9.1.0 |
# | 26.0.0 | 2.10.0 | 2.10.0 | 9.0.0 |
#

set -x
set -e

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
#CPU_COUNT=12
ARCH=$(uname -m)

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

export PATH=$DST/bin:$PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
. $CUR_DIR/../build_conf.sh

export TORCH_VERSION=2.10.0
#cp -r $CUR_DIR/pkg/ pytorch_npu
cd pytorch_npu
   
source /data/local/conda_env.sh
conda activate test
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export _GLIBCXX_USE_CXX11_ABI=1

CANN_VERSIONS=(12.2)
for CANN_VERSION in "${CANN_VERSIONS[@]}"; do
  if [ ! -d $DST/torch/npu/${CANN_VERSION}/lib ]; then
    if [ "$PLATFORM" = "arm" ]; then
      python3 build_libtorch_npu.py
      rm -rf $DST/torch_npu
      cp -r libtorch_npu/ $DST/torch_npu
    else
      echo "amd64"
    fi
  fi
done

cd ..
rm -rf pytorch_npu

# vim: set expandtab ts=4 sw=4 sts=4:
