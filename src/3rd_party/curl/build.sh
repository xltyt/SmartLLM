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

#tar zxf $CUR_DIR/pkg/libssh2-1.10.0.tar.gz
#cd libssh2-1.10.0
#./configure --prefix=$DST --enable-shared=no --enable-static=yes --with-pic
#make -j$CPU_COUNT
#make install
#cd ..
#rm -rf libssh2-1.10.0

cp -r $CUR_DIR/pkg/ curl-7.84.0
cd curl-7.84.0
./buildconf
#./configure --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-openssl --with-libssh2=$DST
if [ "$PLATFORM" = "arm" ]; then
  CFLAGS="-I$DST/include/ $PARAM" CPPFLAGS="-I$DST/include/ $PARAM" CXXFLAGS="-I$DST/include/ $PARAM" LDFLAGS="-L$DST/lib $PARAM_LD" ./configure ${CROSS_HOST_PARAM} --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-openssl=$DST
else
  ./configure --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-openssl=$DST
fi
make -j$CPU_COUNT
make install
cd ..
rm -rf curl-7.84.0
