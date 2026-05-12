#!/bin/bash

set -x
set -e

CUR_FILE=`readlink -f $0`
CUR_DIR=`dirname $CUR_FILE`
set -e
set -x

DST=$CUR_DIR/../../../${PLATFORM}/local
DST_C=$(echo $DST | sed 's/\//\\\//g')
CPU_COUNT=$(cat /proc/cpuinfo | grep "processor" | awk -F": " '{print $2}' | wc -l)
#CPU_COUNT=12

mkdir -p ${CUR_DIR}/../../../build
cd ${CUR_DIR}/../../../build

export PATH=$DST/bin:$PATH
export LD_LIBRARY_PATH=$DST/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$DST/lib/pkgconfig:$PKG_CONFIG_PATH
. $CUR_DIR/../build_conf.sh

export TORCH_VERSION=2.5.1
#tar zxf $CUR_DIR/pkg/pytorch-${TORCH_VERSION}-git.tar.gz
cp -r $CUR_DIR/pkg/ pytorch
cd pytorch
#export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
#export GLIBCXX_USE_CXX11_ABI=0
#sed -i '71aset(GLIBCXX_USE_CXX11_ABI 0)' CMakeLists.txt
#sed -i 's/PYTHON_EXECUTABLE={sys.executable}/PYTHON_EXECUTABLE=python3/g' third_party/onnx/setup.py
#sed -i 's/add_library/#add_library/g' cmake/public/glog.cmake

if [ ! -d $DST/torch/cpu/lib ]; then
  mkdir -p build_cpu
  cd build_cpu
  #USE_CUDA=False \
  #BUILD_TEST=False \
  #BUILD_ONNX_PYTHON=False \
  #BUILD_SHARED_LIBS=True \
  #MAX_JOBS=8 \
  #BUILD_CUSTOM_PROTOBUF=OFF \
  #CMAKE_PREFIX_PATH=$DST \
  #CMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
  #CMAKE_MODULE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
  #CMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
  #python3 ../tools/build_libtorch.py
  #rm -rf $DST/torch/cpu
  #mkdir -p $DST/torch/cpu/lib
  #cp -r ../torch/include $DST/torch/cpu/include
  #cp build/lib/*.so $DST/torch/cpu/lib
  #mkdir -p $DST/torch/cpu/share
  #cp -r ../torch/share/cmake/ $DST/torch/cpu/share/cmake
  #mkdir -p $DST/torch/cpu/bin
  #cp -f build/sleef/bin/* $DST/torch/cpu/bin/
  if [ "$PLATFORM" = "arm" ]; then
    #sed -i '545,546s/^/#/' ../CMakeLists.txt
    cmake \
    -DUSE_CUDA=False \
    -DBUILD_TEST=False \
    -DBUILD_PYTHON=False \
    -DBUILD_ONNX_PYTHON=False \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CUSTOM_PROTOBUF=OFF \
    -DUSE_GLOG=OFF \
    -DCMAKE_PREFIX_PATH=$DST \
    -DCMAKE_EXE_LINKER_FLAGS="-L$DST/lib -static-libstdc++ -static-libgcc -lprotobuf -l:libgfortran.a -Wl,--no-as-needed -lgfortran" \
    -DCMAKE_MODULE_LINKER_FLAGS="-L$DST/lib -static-libstdc++ -static-libgcc -lprotobuf -l:libgfortran.a -Wl,--no-as-needed -lgfortran" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$DST/lib -static-libstdc++ -static-libgcc -lprotobuf -l:libgfortran.a -Wl,--no-as-needed -lgfortran" \
    -DCMAKE_INSTALL_PREFIX=$DST/torch/cpu \
    -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
    -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ \
    -DCMAKE_LINKER=${CROSS_COMPILE}ld \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DGLIBCXX_USE_CXX11_ABI=1 \
    -DNATIVE_BUILD_DIR=$DST/../../amd64/local/torch/cpu/ \
    -DONNX_CUSTOM_PROTOC_EXECUTABLE=$DST/../../amd64/local/bin/protoc \
    -DProtobuf_PROTOC_EXECUTABLE=$DST/../../amd64/local/bin/protoc \
    ..
    make VERBOSE=1 -j${CPU_COUNT}
    make install
    #cp lib/libgloo.a $DST/torch/cpu/lib/
    #cp lib/libonnx_proto.a $DST/torch/cpu/lib/
    #cp lib/libonnx.a $DST/torch/cpu/lib/
    #cp lib/libclog.a $DST/torch/cpu/lib/
  else
    cmake \
    -DUSE_CUDA=False \
    -DBUILD_TEST=False \
    -DBUILD_PYTHON=False \
    -DBUILD_ONNX_PYTHON=False \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CUSTOM_PROTOBUF=OFF \
    -DUSE_GLOG=OFF \
    -DCMAKE_PREFIX_PATH=$DST \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    -DCMAKE_MODULE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    -DCMAKE_INSTALL_PREFIX=$DST/torch/cpu \
    ..
    make VERBOSE=1 -j${CPU_COUNT}
    make install
    #cp lib/libgloo.a $DST/torch/cpu/lib/
    #cp lib/libonnx_proto.a $DST/torch/cpu/lib/
    #cp lib/libonnx.a $DST/torch/cpu/lib/
    #cp lib/libclog.a $DST/torch/cpu/lib/
    mkdir -p $DST/torch/cpu/bin
    cp -f sleef/bin/* $DST/torch/cpu/bin/
  fi
  cd ..
  rm -rf build_cpu
fi

CUDA_VERSIONS=(12.2)
for CUDA_VERSION in "${CUDA_VERSIONS[@]}"; do
  if [ ! -d $DST/torch/gpu/${CUDA_VERSION}/lib ]; then
    mkdir -p build_${CUDA_VERSION}
    cd build_${CUDA_VERSION}
    ##-DTORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0 9.0a"
    ##CUDA_HOST_COMPILER=cc
    #PYTHON_EXECUTABLE=/usr/bin/python3 \
    #LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:/usr/local/lib:$LD_LIBRARY_PATH \
    #PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH \
    #CUDA_BIN_PATH=/usr/local/cuda-${CUDA_VERSION}/bin \
    #CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}/ \
    #CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA_VERSION}/ \
    #CUDNN_LIB_DIR=/usr/local/cuda-${CUDA_VERSION}/lib64 \
    #USE_CUDA=True \
    #BUILD_TEST=False \
    #TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0 9.0a" \
    #TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    #USE_SYSTEM_NCCL=False \
    #USE_NCCL=False \
    #BUILD_ONNX_PYTHON=False \
    #BUILD_SHARED_LIBS=True \
    #MAX_JOBS=2 \
    #python3 ../tools/build_libtorch.py
    #rm -rf $DST/torch/gpu/${CUDA_VERSION}
    #mkdir -p $DST/torch/gpu/${CUDA_VERSION}/lib
    #cp -r ../torch/include $DST/torch/gpu/${CUDA_VERSION}/include
    #cp build/lib/*.so $DST/torch/gpu/${CUDA_VERSION}/lib
    #mkdir -p $DST/torch/gpu/${CUDA_VERSION}/share
    #cp -r ../torch/share/cmake/ $DST/torch/gpu/${CUDA_VERSION}/share/cmake
    if [ "$PLATFORM" = "arm" ]; then
      echo "ARM"
    else
      LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:/usr/local/lib:$LD_LIBRARY_PATH \
      PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH \
      cmake \
      -DCUDA_BIN_PATH=/usr/local/cuda-${CUDA_VERSION}/bin \
      -DCUDA_HOME=/usr/local/cuda-${CUDA_VERSION}/ \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA_VERSION}/ \
      -DCUDNN_LIB_DIR=/usr/local/cuda-${CUDA_VERSION}/lib64 \
      -DTORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9 9.0 9.0a" \
      -DTORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
      -DUSE_SYSTEM_NCCL=False \
      -DUSE_NCCL=False \
      -DUSE_CUDA=True \
      -DBUILD_TEST=False \
      -DBUILD_PYTHON=False \
      -DBUILD_ONNX_PYTHON=False \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_CUSTOM_PROTOBUF=OFF \
      -DUSE_GLOG=OFF \
      -DCMAKE_PREFIX_PATH=$DST \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
      -DCMAKE_MODULE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
      -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
      -DCMAKE_INSTALL_PREFIX=$DST/torch/gpu/${CUDA_VERSION} \
      ..
    make VERBOSE=1 -j4
    make install
    fi
    cd ..
    rm -rf build_${CUDA_VERSION}
  fi
done

cd ..
rm -rf pytorch

# vim: set expandtab ts=4 sw=4 sts=4:
