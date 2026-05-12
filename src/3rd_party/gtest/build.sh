#!/bin/bash

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

export GTEST_VERSION=1.13.0
cp -r $CUR_DIR/pkg googletest-release-${GTEST_VERSION}
cd googletest-release-${GTEST_VERSION}
sed -i 's/int dummy;/int dummy = 0;/g' googletest/src/gtest-death-test.cc
mkdir build
cd build
cmake ${CMAKE_PARAM} -DCMAKE_C_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../
make
make install
cd ..
cd ..
rm -rf googletest-release-${GTEST_VERSION}

# vim: set expandtab ts=4 sw=4 sts=4:
