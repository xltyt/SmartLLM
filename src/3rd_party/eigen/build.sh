#!/bin/bash

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

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

export PATH=$DST/bin:$PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
. $CUR_DIR/../build_conf.sh

export EIGEN_VERSION=3.4.0
cp -r $CUR_DIR/pkg/ eigen-${EIGEN_VERSION}
cd eigen-${EIGEN_VERSION}
mkdir build
cd build
if [ "$PLATFORM" = "arm" ]; then
  cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$DST \
  -DCMAKE_INSTALL_PREFIX=$DST \
  -DEIGEN_BUILD_BLAS=OFF \
  -DEIGEN_BUILD_LAPACK=OFF \
  -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
  -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ \
  -DCMAKE_LINKER=${CROSS_COMPILE}ld \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_SYSTEM_NAME=Linux \
  ..
else
  cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$DST \
  -DCMAKE_INSTALL_PREFIX=$DST \
  -DEIGEN_BUILD_BLAS=OFF \
  -DEIGEN_BUILD_LAPACK=OFF \
  ..
fi
make VERBOSE=1 -j${CPU_COUNT}
make install
cd ..
cd ..
rm -rf eigen-${EIGEN_VERSION}


# vim: set expandtab ts=4 sw=4 sts=4:
