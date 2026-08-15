#!/bin/bash

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`

set -e
set -x

# yum install mesa-libGL-devel
# yum install lapack-devel

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
export PATH=$DST/bin:$PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
. $CUR_DIR/../build_conf.sh
mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

#export PCRE_VERSION=8.40
#
#if [ "$PLATFORM" = "arm" ]; then
#  tar zxf $CUR_DIR/pkg/bzip2-1.0.6.tar.gz
#  cd bzip2-1.0.6
#  sed -i '20d' Makefile
#  sed -i '19d' Makefile
#  sed -i '18d' Makefile
#  sed -i 's/bzip2recover test/bzip2recover/g' Makefile
#  CC=${CROSS_COMPILE}gcc CXX=${CROSS_COMPILE}g++ AR=${CROSS_COMPILE}ar STRIP=${CROSS_COMPILE}strip RANLIB=${CROSS_COMPILE}ranlib LD=${CROSS_COMPILE}ld make
#  CC=${CROSS_COMPILE}gcc CXX=${CROSS_COMPILE}g++ AR=${CROSS_COMPILE}ar STRIP=${CROSS_COMPILE}strip RANLIB=${CROSS_COMPILE}ranlib LD=${CROSS_COMPILE}ld make install PREFIX=$DST
#  cd ..
#  rm -rf bzip2-1.0.6
#fi
#
#tar zxf $CUR_DIR/pkg/pcre-${PCRE_VERSION}.tar.gz
#cd pcre-${PCRE_VERSION}
#mkdir build
#cd build
#cmake $CMAKE_PARAM -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_STATIC_LINKER_FLAGS="" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST -DCMAKE_PREFIX_PATH=$DST -DPCRE_BUILD_TESTS=OFF ../
#make VERBOSE=1 -j${CPU_COUNT}
#make install
#cd ..
#cd ..
#rm -rf pcre-${PCRE_VERSION}

export PCRE2_VERSION=10.45
cp -r $CUR_DIR/pkg/ pcre2-${PCRE2_VERSION}
cd pcre2-${PCRE2_VERSION}
mkdir build
cd build
cmake $CMAKE_PARAM -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_STATIC_LINKER_FLAGS="" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST -DCMAKE_PREFIX_PATH=$DST -DPCRE2_BUILD_TESTS=OFF -DPCRE2_BUILD_PCRE2_8=ON -DPCRE2_BUILD_PCRE2_16=ON -DPCRE2_BUILD_PCRE2_32=ON ../
make VERBOSE=1 -j${CPU_COUNT}
make install
cd ..
cd ..
rm -rf pcre2-${PCRE2_VERSION}

# vim: set expandtab ts=4 sw=4 sts=4:
