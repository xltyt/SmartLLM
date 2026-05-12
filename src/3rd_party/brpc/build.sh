#!/bin/bash
CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
set -e
set -x
. $CUR_DIR/build_conf.sh
BULID_SHARE=0

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
#CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
CPU_COUNT=4

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

#tar zxf $CUR_DIR/pkg/cmake-${CMAKE_VERSION}.tar.gz
#cd cmake-${CMAKE_VERSION}
#./configure --prefix=$DST --parallel=$CPU_COUNT
#make -j$CPU_COUNT
#make install
#cd ..
#rm -rf cmake-${CMAKE_VERSION}

export PATH=$DST/bin:$PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
. $CUR_DIR/../build_conf.sh

cp -r $CUR_DIR/pkg/sqlite sqlite-autoconf-${SQLITE_VERSION}
cd sqlite-autoconf-${SQLITE_VERSION}
if [ "$PLATFORM" = "arm" ]; then
  cp -f $CUR_DIR/../config.guess $CUR_DIR/../config.sub .
  CFLAGS="-I$DST/include/ $PARAM" CPPFLAGS="-I$DST/include/ $PARAM" CXXFLAGS="-I$DST/include/ $PARAM" LDFLAGS="-L$DST/lib $PARAM_LD" ./configure ${CROSS_HOST_PARAM} --prefix=$DST $SHARED_STATIC_SWITCH
else
  if [ $BULID_SHARE -eq 1 ]; then
    ./configure --prefix=$DST --enable-static=no
  else
    ./configure --prefix=$DST --enable-shared=no
  fi
fi
make -j$CPU_COUNT
make install
cd ..
rm -rf sqlite-autoconf-${SQLITE_VERSION}

cp -r $CUR_DIR/pkg/lz4 lz4-${LZ4_VERSION}
cd lz4-${LZ4_VERSION}
if [ $BULID_SHARE -eq 1 ]; then
  sed -i 's/BUILD_STATIC:=yes/BUILD_STATIC:=no/g' lib/Makefile
else
  sed -i 's/BUILD_SHARED:=yes/BUILD_SHARED:=no/g' lib/Makefile
fi
if [ "$PLATFORM" = "arm" ]; then
  CC=${CROSS_COMPILE}gcc CXX=${CROSS_COMPILE}g++ AR=${CROSS_COMPILE}ar STRIP=${CROSS_COMPILE}strip RANLIB=${CROSS_COMPILE}ranlib LD=${CROSS_COMPILE}ld MOREFLAGS="-fPIC" make
  CC=${CROSS_COMPILE}gcc CXX=${CROSS_COMPILE}g++ AR=${CROSS_COMPILE}ar STRIP=${CROSS_COMPILE}strip RANLIB=${CROSS_COMPILE}ranlib LD=${CROSS_COMPILE}ld make -C lib install PREFIX=$DST
else
  MOREFLAGS="-fPIC" make
  make -C lib install PREFIX=$DST
fi
cd ..
rm -rf lz4-${LZ4_VERSION}

cp -r $CUR_DIR/pkg/snappy snappy-${SNAPPY_VERSION}
cd snappy-${SNAPPY_VERSION}
mkdir build
cd build
if [ "$PLATFORM" = "arm" ]; then
  cmake -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST -DCMAKE_PREFIX_PATH=$DST -DSNAPPY_BUILD_TESTS=OFF ../
else
  cmake -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST -DCMAKE_PREFIX_PATH=$DST -DSNAPPY_BUILD_TESTS=OFF ../
fi
make VERBOSE=1
make install
cd ..
cd ..
rm -rf snappy-${SNAPPY_VERSION}

cp -r $CUR_DIR/pkg/leveldb leveldb-${LEVELDB_VERSION}
cd leveldb-${LEVELDB_VERSION}
mkdir -p build
cd build
if [ "$PLATFORM" = "arm" ]; then
  cmake -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=." -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=." -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=." -DCMAKE_STATIC_LINKER_FLAGS="" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ..
else
  if [ $BULID_SHARE -eq 1 ]; then
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=$DST/lib" -DCMAKE_STATIC_LINKER_FLAGS="" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$DST ..
  else
    #sed -i '/tcmalloc malloc/iset(CMAKE_REQUIRED_LIBRARIES unwind lzma)' ../CMakeLists.txt
    #sed -i 's/target_link_libraries(leveldb tcmalloc)/target_link_libraries(leveldb tcmalloc unwind lzma)/g' ../CMakeLists.txt
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=." -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=." -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lstdc++ -lm -lpthread -ldl -Wl,-rpath=." -DCMAKE_STATIC_LINKER_FLAGS="" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ..
  fi
fi
make -j$CPU_COUNT
make install
cd ..
cd ..
rm -rf leveldb-${LEVELDB_VERSION}
#unzip $CUR_DIR/pkg/leveldb-${LEVELDB_VERSION}.zip
#cd leveldb-${LEVELDB_VERSION}
#CFLAGS="-fPIC" make -j${CPU_COUNT}
#cp ./out-static/libleveldb.a ${DST}/lib
#rm -rf ${DST}/include/leveldb
#cp -r include/leveldb $DST/include
#cd ..
#rm -rf leveldb-${LEVELDB_VERSION}

#tar zxf $CUR_DIR/pkg/gflags-${GFLAGS_VERSION}.tar.gz
#cd gflags-${GFLAGS_VERSION}
#mkdir build
#cd build
#if [ "$PLATFORM" = "arm" ]; then
#  cmake -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../
#else
#  if [ $BULID_SHARE -eq 1 ]; then
#    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$DST ../
#  else
#    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../
#  fi
#fi
#make
#make install
#cd ..
#cd ..
#rm -rf gflags-${GFLAGS_VERSION}

#tar zxf $CUR_DIR/pkg/glog-${GLOG_VERSION}.tar.gz
#cd glog-${GLOG_VERSION}
##sed -i '1000,1003s/^/\/\//' src/logging.cc
#sh autogen.sh
#if [ "$PLATFORM" = "arm" ]; then
#  CFLAGS="-I$DST/include/ $PARAM" CPPFLAGS="-I$DST/include/ $PARAM" CXXFLAGS="-I$DST/include/ $PARAM" LDFLAGS="-L$DST/lib $PARAM_LD" ./configure ${CROSS_HOST_PARAM} --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-gflags=$DST
#else
#  #CPPFLAGS="-DGFLAGS_NAMESPACE=gflags" ./configure --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-gflags=$DST
#  ./configure --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-gflags=$DST
#fi
#make
#make install
#cd ..
#rm -rf glog-${GLOG_VERSION}

#tar zxf $CUR_DIR/pkg/libunwind-${LIBUNWIND_VERSION}.tar.gz
#cd libunwind-${LIBUNWIND_VERSION}
#NOCONFIGURE=1 sh autogen.sh
#if [ $BULID_SHARE -eq 1 ]; then
#  ./configure --prefix=$DST --enable-static=no --with-pic=yes
#else
#  ./configure --prefix=$DST --enable-shared=no --with-pic=yes
#fi
#make -j$CPU_COUNT
#make install
#cd ..
#rm -rf libunwind-${LIBUNWIND_VERSION}

##https://groups.google.com/g/protobuf/c/xYF_KRg0o3Q
#unzip $CUR_DIR/pkg/protobuf-${PROTOBUF_VERSION}.zip
#cd protobuf-${PROTOBUF_VERSION}
#sed -i 's/make_pair<string, string>(virtual_path/make_pair(virtual_path/g' src/google/protobuf/compiler/command_line_interface.cc
#tar zxf ${CUR_DIR}/pkg/googletest-release-1.3.0.tar.gz
#mv googletest-release-1.3.0 gtest
#sh autogen.sh
#CFLAGS="-Wno-error" CPPFLAGS="-Wno-error" ./configure --prefix=$DST --enable-shared=no --with-pic=yes
#make -j$CPU_COUNT
#make install
#cd ..
#rm -rf protobuf-${PROTOBUF_VERSION}

#tar zxf $CUR_DIR/pkg/protobuf-${PROTOBUF_VERSION}.tar.gz
#cd protobuf-${PROTOBUF_VERSION}
#mkdir build
#cd build
#cmake -DCMAKE_C_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST -Dprotobuf_BUILD_TESTS=OFF ../
#make
#make install
#cd ..
#cd ..
#rm -rf protobuf-${PROTOBUF_VERSION}

#cp -r $CUR_DIR/pkg/protobuf protobuf-${PROTOBUF_VERSION}
#cd protobuf-${PROTOBUF_VERSION}
#sh autogen.sh
#if [ "$PLATFORM" = "arm" ]; then
#  CFLAGS="-I$DST/include/ $PARAM" CPPFLAGS="-I$DST/include/ $PARAM" CXXFLAGS="-I$DST/include/ $PARAM" LDFLAGS="-L$DST/lib $PARAM_LD -lz" ./configure ${CROSS_HOST_PARAM} --prefix=$DST --enable-shared=no --enable-static=yes --with-pic --with-zlib=$DST
#else
#  CFLAGS="-Wno-error -I$DST/include/" CPPFLAGS="-Wno-error -I$DST/include/" LDFLAGS="-L$DST/lib -lz"  ./configure --prefix=$DST --enable-shared=no --with-pic=yes --with-zlib=$DST
#fi
#make -j$CPU_COUNT
#make install
#cd ..
#rm -rf protobuf-${PROTOBUF_VERSION}
#
#if [ ! -d $DST/amd64/protobuf ]; then
#  cp -r $CUR_DIR/pkg/protobuf protobuf-${PROTOBUF_VERSION}
#  cd protobuf-${PROTOBUF_VERSION}
#  sh autogen.sh
#  CFLAGS="-Wno-error -I$DST/include/" CPPFLAGS="-Wno-error -I$DST/include/" LDFLAGS="-L$DST/lib -lz" ./configure --prefix=$DST/amd64/protobuf --enable-shared=no --with-pic=yes --with-zlib=$DST
#  make -j$CPU_COUNT
#  make install
#  cd ..
#  rm -rf protobuf-${PROTOBUF_VERSION}
#fi

#tar zxf ${CUR_DIR}/pkg/incubator-brpc-${BRPC_VERSION}.tar.gz
#cd incubator-brpc-${BRPC_VERSION}
cp -r ${CUR_DIR}/pkg/brpc brpc-${BRPC_VERSION}
cd brpc-${BRPC_VERSION}
patch -p1 < ${CUR_DIR}/brpc-${BRPC_VERSION}.patch
#sed -i 's/^set(CMAKE_CPP_FLAGS "-DBRPC_WITH_GLOG=/set(CMAKE_CPP_FLAGS "${CMAKE_CPP_FLAGS} -DBRPC_WITH_GLOG=/g' CMakeLists.txt
#sed -i 's/^set(CMAKE_CPP_FLAGS "-DGFLAGS_NS/set(CMAKE_CPP_FLAGS "${CMAKE_CPP_FLAGS} -DGFLAGS_NS/g' tools/CMakeLists.txt
sed -i 's/ dl/ snappy dl pthread ssl crypto\n/g' CMakeLists.txt
sed -i 's/-DBRPC_WITH_GLOG=/${CMAKE_CPP_FLAGS} -DBRPC_WITH_GLOG=/g' CMakeLists.txt
sed -i 's/-DBRPC_WITH_GLOG=/${CMAKE_CPP_FLAGS} -DBRPC_WITH_GLOG=/g' tools/CMakeLists.txt
mkdir -p build
cd build
if [ "$PLATFORM" = "arm" ]; then
  sed -i 's/-msse4.2//g' ../CMakeLists.txt
  sed -i 's/-msse4//g' ../CMakeLists.txt
  cmake -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -lpthread -I$DST/include -Wno-error -Wno-narrowing" -DCMAKE_CXX_FLAGS="-fPIC -lpthread -I$DST/include -Wno-error -Wno-narrowing" -DCMAKE_CPP_FLAGS="-fPIC -lpthread -I$DST/include -Wno-error -Wno-narrowing" -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lm -lpthread -ldl" -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lm -lpthread -ldl" -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lm -lpthread -ldl" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../ -DCMAKE_PREFIX_PATH=$DST -DWITH_GLOG=ON -DProtobuf_PROTOC_EXECUTABLE=$DST/amd64/protobuf/bin/protoc
else
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_CPP_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -lm -lpthread -ldl" -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -lm -lpthread -ldl" -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -lm -lpthread -ldl" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../ -DCMAKE_PREFIX_PATH=$DST -DWITH_GLOG=ON
fi
make -j${CPU_COUNT} VERBOSE=1
make install
cd ..
cd ..
#rm -rf incubator-brpc-${BRPC_VERSION}
rm -rf brpc-${BRPC_VERSION}
rm -f $DST/lib/libbrpc.so

#tar zxf $CUR_DIR/pkg/googletest-release-${GTEST_VERSION}.tar.gz
#cd googletest-release-${GTEST_VERSION}
#mkdir build
#cd build
#cmake ${CMAKE_PARAM} -DCMAKE_C_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_CXX_FLAGS="-fPIC -lpthread -I$DST/include" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../
#make
#make install
#cd ..
#cd ..
#rm -rf googletest-release-${GTEST_VERSION}

#tar zxf $CUR_DIR/pkg/googletest-release-${GTEST_VERSION}.tar.gz
#cd googletest-release-${GTEST_VERSION}
#sed -i 's/-Werror//g' googletest/cmake/internal_utils.cmake
#mkdir build
#cd build
#cmake -DCMAKE_C_FLAGS="-fPIC -lpthread -I$DST/include -Wno-error" -DCMAKE_CXX_FLAGS="-fPIC -lpthread -I$DST/include -Wno-error" -DCMAKE_INSTALL_LIBDIR=$DST/lib -DCMAKE_INSTALL_PREFIX=$DST ../
#make VERBOSE=1
#make install
#cd ..
#cd ..
#rm -rf googletest-release-${GTEST_VERSION}

exit 0

# vim: set expandtab nu ts=4 sw=4 sts=4:
