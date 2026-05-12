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
. $CUR_DIR/../build_conf.sh

#export OPENSSL_VERSION=1.1.0f
export OPENSSL_VERSION=1.1.1f     # For appweb SSL_SESSION_get_ticket

cp -r $CUR_DIR/pkg/ openssl-${OPENSSL_VERSION}
cd openssl-${OPENSSL_VERSION}
sed -i 's/qw\/glob/qw\/:glob/g' Configure
sed -i 's/qw\/glob/qw\/:glob/g' test/build.info
if [ "$PLATFORM" = "arm" ]; then
  ./Configure no-shared no-asm --openssldir=$DST --prefix=$DST $PARAM -fPIC --cross-compile-prefix=${CROSS_COMPILE} linux-aarch64
else
  ./Configure no-shared no-asm --openssldir=$DST --prefix=$DST -fPIC linux-x86_64
fi
make -j1
make -j1 install_sw
cd ..
rm -rf openssl-${OPENSSL_VERSION}

# vim: set expandtab ts=4 sw=4 sts=4:
