#!/bin/bash
CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
#export PATH=$DST/bin:$PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build
. $CUR_DIR/../build_conf.sh

export BOOST_VERSION=1_69_0

set +e
cp -r $CUR_DIR/pkg/ boost_${BOOST_VERSION}
cd boost_${BOOST_VERSION}
./bootstrap.sh --prefix=$DST --libdir=$DST/lib
if [ "$PLATFORM" = "arm" ]; then
  sed -i "s/using gcc/using gcc : aarch64 : ${CROSS_COMPILE}gcc/g" project-config.jam
  ./bjam install --without-python link=static runtime-link=static threading=multi cxxflags="$PARAM" cflags="$PARAM" linkflags="$PARAM" release -d2 -j$CPU_COUNT --prefix=$DST --libdir=$DST/lib
else
  ./bjam install --without-python link=static runtime-link=static threading=multi cxxflags="-fPIC" cflags="-fPIC" release -d2 -j$CPU_COUNT --prefix=$DST --libdir=$DST/lib
fi
cd ..
set -e 
rm -rf boost_${BOOST_VERSION}

exit 0

# vim: set expandtab nu ts=4 sw=4 sts=4:
