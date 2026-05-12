#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
#CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
CPU_COUNT=8

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

export OPENBLAS_VERSION=0.3.23
. $CUR_DIR/../build_conf.sh

cp -r $CUR_DIR/pkg/ OpenBLAS-${OPENBLAS_VERSION}
cd OpenBLAS-${OPENBLAS_VERSION}
if [ "$PLATFORM" = "arm" ]; then
  make TARGET=ARMV8 HOSTCC=gcc BINARY=64 CC=${CROSS_PREFIX}-gcc FC=${CROSS_PREFIX}-gfortran NO_SHARED=1
  TARGET=ARMV8 NO_SHARED=1 PREFIX=$DST/openblas make install
  rm -rf $DST/include/openblas
  mv $DST/openblas/include $DST/include/openblas
  mv $DST/openblas/lib/*.a $DST/lib/
  mv $DST/openblas/lib/cmake/* $DST/lib/cmake/
  mv $DST/openblas/lib/pkgconfig/* $DST/lib/pkgconfig/
  rm -rf $DST/openblas
else
  mkdir build
  cd build
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-fPIC -I$DST/include" \
    -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" \
    -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" \
    -DCMAKE_STATIC_LINKER_FLAGS="" \
    -DCMAKE_INSTALL_LIBDIR=$DST/lib \
    -DCMAKE_INSTALL_PREFIX=$DST/ \
    -DNO_AVX512=1 \
    -DUSE_OPENMP=ON \
    ../
  make VERBOSE=1 -j${CPU_COUNT}
  make install
  cd ..
fi
cd ..
rm -rf OpenBLAS-${OPENBLAS_VERSION}

# vim: set expandtab ts=4 sw=4 sts=4:
