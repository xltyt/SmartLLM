#!/bin/bash

ARCH=$(uname -m)
if [ "$PLATFORM" = "arm" ]; then
  PARAM="-fsigned-char -fPIC -Wno-error"
  PARAM_LD="-fsigned-char -fPIC -Wno-error -Wl,-rpath=$DST/lib/"
  PARAM_LD_C="-fsigned-char -fPIC -Wno-error -Wl,-rpath=$DST_C\\/lib\\/"
  if [ "$ARCH" = "x86_64" ]; then
    CROSS_PREFIX=aarch64-linux-gnu
    CROSS_COMPILE=${CROSS_PREFIX}-
    CROSS_HOST_PARAM="--host=${CROSS_PREFIX}"
    CMAKE_PARAM="-DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ -DCMAKE_LINKER=${CROSS_COMPILE}ld -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux"
  else
    CROSS_PREFIX=
    CROSS_COMPILE=
    CROSS_HOST_PARAM=
    CMAKE_PARAM=
  fi
else
  PARAM="-fPIC -Wno-error"
  PARAM_LD="-fPIC -Wno-error -Wl,-rpath=$DST/lib/"
  PARAM_LD_C="-fPIC -Wno-error -Wl,-rpath=$DST_C\\/lib\\/"
  CROSS_PREFIX=
  CROSS_COMPILE=
  CROSS_HOST_PARAM=
  CMAKE_PARAM=
fi
SHARED_STATIC_SWITCH="--enable-shared=no --enable-static=yes"


# vim: set expandtab ts=4 sw=4 sts=4:
