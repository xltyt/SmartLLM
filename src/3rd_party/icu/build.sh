#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
CPU_COUNT=4

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build
. $CUR_DIR/../build_conf.sh

export ICU_VERSION=78.3
#export ICU_VERSION=73-2
#export ICU_VERSION=60-3

if [ "$PLATFORM" = "arm" ]; then
  #tar zxf $CUR_DIR/pkg/icu-release-${ICU_VERSION}.tar.gz
  cp -r $CUR_DIR/pkg/ icu-release-${ICU_VERSION}
  cd icu-release-${ICU_VERSION}/icu4c/source
  CFLAGS="-fPIC" CPPFLAGS="-fPIC" CXXFLAGS="-fPIC" LDFLAGS="-lpthread -pthread" ./configure --enable-shared=no --enable-static=yes
  make -j$CPU_COUNT
  cd ../../../
  mv icu-release-${ICU_VERSION} icu-release-${ICU_VERSION}-amd64
fi

cp -r $CUR_DIR/pkg/ icu-release-${ICU_VERSION}
cd icu-release-${ICU_VERSION}/icu4c/source
if [ "$PLATFORM" = "arm" ]; then
  CROSS_HOST_PARAM="${CROSS_HOST_PARAM} --with-cross-build --with-cross-build=${CUR_DIR}/../../../build/icu-release-${ICU_VERSION}-amd64/icu4c/source"
fi
CFLAGS="-fPIC" CPPFLAGS="-fPIC" CXXFLAGS="-fPIC" LDFLAGS="-lpthread -pthread" ./configure ${CROSS_HOST_PARAM} --prefix=$DST --enable-shared=no --enable-static=yes
make -j$CPU_COUNT
make install
cd ../../../
rm -rf icu-release-${ICU_VERSION}

rm -rf icu-release-${ICU_VERSION}-amd64

# vim: set expandtab ts=4 sw=4 sts=4:
