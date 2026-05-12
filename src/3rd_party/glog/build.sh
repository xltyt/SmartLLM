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

export GFLAGS_VERSION=2.2.1
cp -r $CUR_DIR/../gflags/pkg gflags-${GFLAGS_VERSION}
cd gflags-${GFLAGS_VERSION}
mkdir build
cd build
if [ "$PLATFORM" = "arm" ]; then
  cmake -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../
else
  if [ $BULID_SHARE -eq 1 ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$DST ../
  else
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../
  fi
fi
make
make install
cd ..
cd ..
rm -rf gflags-${GFLAGS_VERSION}

export GLOG_VERSION=0.4.0
cp -r $CUR_DIR/pkg/glog glog-${GLOG_VERSION}
cd glog-${GLOG_VERSION}
#sed -i '1000,1003s/^/\/\//' src/logging.cc
sh autogen.sh
if [ "$PLATFORM" = "arm" ]; then
  CFLAGS="-I$DST/include/ $PARAM" CPPFLAGS="-I$DST/include/ $PARAM" CXXFLAGS="-I$DST/include/ $PARAM" LDFLAGS="-L$DST/lib $PARAM_LD" ./configure ${CROSS_HOST_PARAM} --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-gflags=$DST
else
  #CPPFLAGS="-DGFLAGS_NAMESPACE=gflags" ./configure --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-gflags=$DST
  ./configure --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-gflags=$DST
fi
make
make install
cd ..
rm -rf glog-${GLOG_VERSION}

exit 0

# vim: set expandtab nu ts=4 sw=4 sts=4:
