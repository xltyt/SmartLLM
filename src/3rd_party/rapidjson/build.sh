#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`

set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
#export PATH=$DST/bin:$PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

export RAPIDJSON_VERSION=1.1.0

cp -r $CUR_DIR/pkg/ rapidjson-${RAPIDJSON_VERSION}
cd rapidjson-${RAPIDJSON_VERSION}
rm -rf $DST/include/rapidjson
mv include/rapidjson $DST/include
cd ..
rm -rf rapidjson-${RAPIDJSON_VERSION}

# vim: set expandtab ts=4 sw=4 sts=4:
