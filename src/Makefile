CURRENT_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(shell dirname $(CURRENT_PATH))

all : prep

.PHONY: prep

# apt install lrzsz git vim git-lfs build-essential cmake perl autoconf libtool pip python-is-python3 gcc-aarch64-linux-gnu g++-aarch64-linux-gnu gfortran-aarch64-linux-gnu libssl-dev tcl
# pip install setuptools PyYAML typing_extensions --index-url https://mirrors.aliyun.com/pypi/simple/
#
prep:
	@echo "\033[32m=====> Build amd64...\033[0m"
	make -C 3rd_party
	make -C common
	make -C search
	MODE=gpu make -j16 -C search
	@echo "\033[32m=====> Build arm...\033[0m"
	PLATFORM=arm make -C 3rd_party
	PLATFORM=arm make -C common
	PLATFORM=arm make -C search
	@echo "\033[32m=====> Build Done\033[0m"

clean:
	make clean -C common
	make clean -C search
	MODE=gpu make clean -C search
	PLATFORM=arm make clean -C common
	PLATFORM=arm make clean -C search

distclean:
	make clean
	make clean -C 3rd_party
	PLATFORM=arm make clean -C 3rd_party
