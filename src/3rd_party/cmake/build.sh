#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`

set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
export PATH=$DST/bin:$PATH
#export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
#export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

# apt install mesa-common-dev
export CMAKE_VERSION=3.26.4

cp -r $CUR_DIR/pkg/ cmake-${CMAKE_VERSION}
cd cmake-${CMAKE_VERSION}
#CFLAGS="-DCMAKE_USE_OPENSSL=OFF" CPPFLAGS="-DCMAKE_USE_OPENSSL=OFF" CXXFLAGS="-DCMAKE_USE_OPENSSL=OFF" ./configure --prefix=$DST --parallel=$CPU_COUNT
./configure --prefix=$DST --parallel=$CPU_COUNT #--no-qt-gui
make -j$CPU_COUNT
make install
cd ..
rm -rf cmake-${CMAKE_VERSION}

# vim: set expandtab ts=4 sw=4 sts=4:
