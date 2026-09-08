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

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

export PATH=$DST/bin:$PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
. $CUR_DIR/../build_conf.sh

cp -r $CUR_DIR/pkg/ nlohmann
cd nlohmann
mkdir build
cd build
if [ "$PLATFORM" = "arm" ]; then
  cmake \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_PREFIX_PATH=$DST \
  -DCMAKE_INSTALL_PREFIX=$DST \
  -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
  -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ \
  -DCMAKE_LINKER=${CROSS_COMPILE}ld \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_SYSTEM_NAME=Linux \
  ..
else
  cmake \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_PREFIX_PATH=$DST \
  -DCMAKE_INSTALL_PREFIX=$DST \
  ..
fi
make VERBOSE=1 -j${CPU_COUNT}
make install
cd ..
cd ..
rm -rf nlohmann

# vim: set expandtab ts=4 sw=4 sts=4:
