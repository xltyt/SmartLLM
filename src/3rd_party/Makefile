CURRENT_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(shell dirname $(CURRENT_PATH))

THIRD_DIR := $(CURRENT_DIR)/../../local

all : prep

.PHONY: prep

PLATFORM ?= amd64

prep:
	@mkdir -p ../../$(PLATFORM)/bin
	@mkdir -p ../../$(PLATFORM)/lib
	@if [ ! -d ../../$(PLATFORM)/local/include/openssl ]; then            cd openssl && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/rapidjson ]; then          cd rapidjson && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -f ../../$(PLATFORM)/local/include/zlib.h ]; then             cd zlib && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/unicode ]; then            cd icu && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/curl ]; then               cd curl && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/boost ]; then              cd boost && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/gtest ]; then              cd gtest && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/glog ]; then               cd glog && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/google/protobuf ]; then    cd protobuf && PLATFORM=$(PLATFORM) sh build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/brpc ]; then               cd brpc && PLATFORM=$(PLATFORM) bash build.sh; fi
	@if [ ! -d ../../$(PLATFORM)/local/include/openblas ]; then           cd blas && PLATFORM=$(PLATFORM) bash build.sh; fi
ifeq ($(PLATFORM),amd64)
	@if [ ! -d ../../$(PLATFORM)/local/torch/gpu/12.2/include ]; then     cd torch && PLATFORM=$(PLATFORM) bash build.sh; fi
else
	@if [ ! -d ../../$(PLATFORM)/local/torch/cpu/include ]; then          cd torch && PLATFORM=$(PLATFORM) bash build.sh; fi
endif
	@if [ ! -d ../../$(PLATFORM)/local/include/eigen3 ]; then             cd eigen && PLATFORM=$(PLATFORM) bash build.sh; fi
	@rm -rf ../../build

clean:
	rm -rf ../../$(PLATFORM)/local
