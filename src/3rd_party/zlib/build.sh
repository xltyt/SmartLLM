#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
#CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
CPU_COUNT=4

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

export PATH=$DST/bin:$PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
. $CUR_DIR/../build_conf.sh

cp -r $CUR_DIR/pkg/ zlib-1.2.11
cd zlib-1.2.11
CFLAGS="-I$DST/include/ $PARAM" CPPFLAGS="-I$DST/include/ $PARAM" CXXFLAGS="-I$DST/include/ $PARAM" LDFLAGS="-L$DST/lib $PARAM_LD -lz -lpthread -ldl" CC=${CROSS_COMPILE}gcc CXX=${CROSS_COMPILE}g++ AR=${CROSS_COMPILE}ar STRIP=${CROSS_COMPILE}strip RANLIB=${CROSS_COMPILE}ranlib LD=${CROSS_COMPILE}ld ./configure --prefix=$DST --static
make
make install
cd ..
rm -rf zlib-1.2.11

# vim: set expandtab ts=4 sw=4 sts=4:
