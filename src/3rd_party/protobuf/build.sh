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

#export PROTOBUF_VERSION=4.25.0
#cp -r $CUR_DIR/pkg protobuf-${PROTOBUF_VERSION}
#cd protobuf-${PROTOBUF_VERSION}
#mkdir build
#cd build
#if [ "$PLATFORM" = "arm" ]; then
#  cmake \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DCMAKE_INSTALL_LIBDIR=$DST/lib \
#    -DCMAKE_INSTALL_PREFIX=$DST/ \
#    -DCMAKE_PREFIX_PATH=$DST \
#    -DBUILD_SHARED_LIBS=OFF \
#    -Dprotobuf_USE_EXTERNAL_GTEST=ON \
#    -DBUILD_GMOCK=OFF \
#    -DCMAKE_CXX_STANDARD=17 \
#    -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
#    -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ \
#    -DCMAKE_LINKER=${CROSS_COMPILE}ld \
#    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
#    -DCMAKE_SYSTEM_NAME=Linux \
#    -Dprotobuf_BUILD_TESTS=OFF \
#    ../
#else
#  cmake \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DCMAKE_INSTALL_LIBDIR=$DST/lib \
#    -DCMAKE_INSTALL_PREFIX=$DST/ \
#    -DCMAKE_PREFIX_PATH=$DST \
#    -DBUILD_SHARED_LIBS=OFF \
#    -Dprotobuf_USE_EXTERNAL_GTEST=ON \
#    -DBUILD_GMOCK=OFF \
#    -DCMAKE_CXX_STANDARD=17 \
#    ../
#fi
#make -j${CPU_COUNT}
#make install
#cd ..
#cd ..
#rm -rf protobuf-${PROTOBUF_VERSION}

export PROTOBUF_VERSION=3.6.1.3
cp -r $CUR_DIR/pkg protobuf-${PROTOBUF_VERSION}
cd protobuf-${PROTOBUF_VERSION}
sh autogen.sh
if [ "$PLATFORM" = "arm" ]; then
  CFLAGS="-I$DST/include/ $PARAM" CPPFLAGS="-I$DST/include/ $PARAM" CXXFLAGS="-I$DST/include/ $PARAM" LDFLAGS="-L$DST/lib $PARAM_LD -lz" ./configure ${CROSS_HOST_PARAM} --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-zlib=$DST
else
  CFLAGS="-Wno-error -I$DST/include/" CPPFLAGS="-Wno-error -I$DST/include/" LDFLAGS="-L$DST/lib -lz"  ./configure --prefix=$DST --enable-shared=no --with-pic=yes --with-zlib=$DST
fi
make -j$CPU_COUNT
make install
cd ..
rm -rf protobuf-${PROTOBUF_VERSION}

if [ ! -d $DST/amd64/protobuf ]; then
  cp -r $CUR_DIR/pkg protobuf-${PROTOBUF_VERSION}
  cd protobuf-${PROTOBUF_VERSION}
  sh autogen.sh
  CFLAGS="-Wno-error -I$DST/include/" CPPFLAGS="-Wno-error -I$DST/include/" LDFLAGS="-L$DST/lib -lz" ./configure --prefix=$DST/amd64/protobuf --enable-shared=no --with-pic=yes --with-zlib=$DST
  make -j$CPU_COUNT
  make install
  cd ..
  rm -rf protobuf-${PROTOBUF_VERSION}
fi

# vim: set expandtab ts=4 sw=4 sts=4:
