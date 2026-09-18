CURRENT_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(shell dirname $(CURRENT_PATH))
DATE = $(shell date "+%Y%m%d%H")

MODE ?= cpu
CUDA_VERSION ?= 12.2
PLATFORM ?= amd64

VERSION=$(shell awk -F'|' '{print $$2}' ../../README.md | sed '/^ *$$/d' | awk -F'`' '{print $$2}' | tail -1)
ifeq ($(PLATFORM),arm)
  BUILD_MODULE_NAME=llm_service_arm
else
  ifeq ($(MODE),gpu)
	BUILD_MODULE_NAME=llm_service_gpu_$(CUDA_VERSION)
  else
    BUILD_MODULE_NAME=llm_service
  endif
endif

ifeq ($(PLATFORM),arm)
	CC=aarch64-linux-gnu-gcc
	CXX=aarch64-linux-gnu-g++
	STRIP=aarch64-linux-gnu-strip
else
	STRIP=strip
endif

MAJOR=2
MINOR=3

APP_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/bin
OBJS_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/build/$(BUILD_MODULE_NAME)
THIRD_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/local
TEST_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/test
ifeq ($(PLATFORM),arm)
  THIRD_HOST_DIR := $(THIRD_DIR)/../../amd64/local
else
  THIRD_HOST_DIR := $(THIRD_DIR)
endif

ifeq ($(MODE),gpu)
  TORCH_ROOT = $(THIRD_DIR)/torch/gpu/$(CUDA_VERSION)
  TORCH_LIB = -Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed -lc10 -ltorch_cpu -ltorch
else
  TORCH_ROOT = $(THIRD_DIR)/torch/cpu
  TORCH_LIB = -ltorch_cpu -ltorch -lc10
endif

CUDA_INCLUDE ?= 
ifeq ($(MODE),gpu)
  CUDA_INCLUDE = -I/usr/local/cuda-$(CUDA_VERSION)/include
endif
CUDA_LIB ?= 
ifeq ($(MODE),gpu)
  CUDA_LIB = -L/usr/local/cuda-$(CUDA_VERSION)/lib64/ -lcudart_static -lcublas_static -lcublasLt_static
  ifeq ($(CUDA_VERSION),11.3)
    CUDA_LIB += -lculibos
  else ifeq ($(CUDA_VERSION),11.4)
    CUDA_LIB += -lculibos
  else ifeq ($(CUDA_VERSION),12.4)
    CUDA_LIB += -lculibos
  endif
  CUDA_LIB += -Wl,-rpath=/usr/local/cuda-$(CUDA_VERSION)/lib64/ 
  #CUDA_LIB += /usr/local/cuda-$(CUDA_VERSION)/lib64/stubs/libnvidia-ml.so
endif
INCLUDES += \
  -I. \
  -I../common \
  -I$(OBJS_DIR) \
  -I$(TORCH_ROOT)/include/ \
  -I$(TORCH_ROOT)/include/torch/csrc/api/include/ \
  $(CUDA_INCLUDE) \
  -I$(THIRD_DIR)/include/eigen3/ \
  -I$(THIRD_DIR)/include/

ABSL_LIB = -labsl_cord -labsl_cordz_info -labsl_cord_internal -labsl_cordz_functions -labsl_cordz_handle -labsl_crc_cord_state -labsl_crc32c -labsl_crc_internal -labsl_crc_cpu_detect -labsl_str_format_internal -labsl_raw_hash_set -labsl_hash -labsl_city -labsl_bad_variant_access -labsl_low_level_hash -labsl_hashtablez_sampler -labsl_exponential_biased -labsl_synchronization -labsl_graphcycles_internal -labsl_kernel_timeout_internal -labsl_stacktrace -labsl_symbolize -labsl_debugging_internal -labsl_demangle_internal -labsl_malloc_internal -labsl_time -labsl_civil_time -labsl_strings -labsl_strings_internal -labsl_string_view -labsl_base -lrt -labsl_spinlock_wait -labsl_int128 -labsl_time_zone -labsl_bad_optional_access -labsl_throw_delegate -labsl_raw_logging_internal -labsl_log_severity

BASE_LIBS += $(THIRD_DIR)/../lib/libmycommon.a
BASE_LIBS += \
  -L$(TORCH_ROOT)/lib $(TORCH_LIB) \
  $(CUDA_LIB) \
  -L$(THIRD_DIR)/lib \
  $(ABSL_LIB) \
  -lsafetensors_cpp \
  -lboost_system -lboost_filesystem -lboost_regex \
  -lpcre2-32 -lpcre2-16 -lpcre2-8 \
  -licuuc -licuio -licudata \
  -lprotoc -lprotobuf \
  -lleveldb -llz4 -lsnappy \
  -lglog \
  -lgtest \
  -lgflags \
  -lcurl \
  -lssl -lcrypto \
  -lz \
  -lm \
  -lrt \
  -ldl \
  -lpthread

LIBS += \
  -L. \
  $(BASE_LIBS)

ifeq ($(MODE),gpu)
  CFLAGS += -DUSE_GPU=1 -DMY_CUDA_VERSION=\"$(CUDA_VERSION)\" -DGLIBCXX_USE_CXX11_ABI=1
endif
CFLAGS    += -O2 -g -std=c++17 -fopenmp -static-libstdc++ -static-libgcc
ifeq ($(PLATFORM),amd64)
  CFLAGS += -mavx -mavx2
  #CFLAGS += -march=native
endif
CFLAGS	  += -DNDEBUG -DFORTIFY_SOURCE=2 -fPIC
CFLAGS	  += -Wno-deprecated-declarations
#CFLAGS    += -DMKL_ILP64 -m64 -I${MKLROOT}/include
LIBPATHS  +=
CPPFLAGS  = $(CFLAGS) -std=c++17
CXXFLAGS  = $(CFLAGS) -std=c++17

SOURCES += \
  utils/model_utils.o \
  layers/activation.o \
  layers/layer_norm.o \
  layers/rotary_embedding.o \
  layers/sampler.o \
  model/qwen3.o \
  token/core_bpe.cpp \
  token/tiktoken.cpp \
  token/qwen_token.cpp \
  flags.cpp \
  main.cpp
OBJS := $(addprefix $(OBJS_DIR)/,$(patsubst %.cpp,%.o,$(SOURCES)))
OBJS := $(patsubst %.c,%.o,$(OBJS))

OBJS_PB := \
    $(OBJS_DIR)/search.pb.o

BINS = \
	$(APP_DIR)/$(BUILD_MODULE_NAME)

all : prep $(BINS) test

.PHONY: prep

prep:
	@mkdir -p $(OBJS_DIR)

$(APP_DIR)/$(BUILD_MODULE_NAME) : $(OBJS_PB) $(OBJS) $(OBJS_DIR)/main.o
ifeq ($(MODE),gpu)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(THIRD_DIR)/lib/libbrpc.a $(LIBS) -Wl,-rpath=../lib -Wl,-rpath=lib -Wl,-rpath=/usr/local/cuda-$(CUDA_VERSION)/lib64/
else
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(THIRD_DIR)/lib/libbrpc.a $(LIBS) -Wl,-rpath=../lib -Wl,-rpath=lib
endif
	$(STRIP) $@
  
ifeq ($(MODE),gpu)
  TEST_LAYERS_MODULE_NAME=test_layers_gpu_$(CUDA_VERSION)
else
  TEST_LAYERS_MODULE_NAME=test_layers
endif
TEST_MODEL_UTILS_MODULE_NAME=test_module_utils
TEST_MODEL_QWEN3_MODULE_NAME=test_model_qwen3

test: \
	$(TEST_DIR)/$(TEST_LAYERS_MODULE_NAME) \
	$(TEST_DIR)/$(TEST_MODEL_UTILS_MODULE_NAME) \
	$(TEST_DIR)/$(TEST_MODEL_QWEN3_MODULE_NAME) \
	$(TEST_DIR)/test_token

$(TEST_DIR)/$(TEST_LAYERS_MODULE_NAME) : \
	$(OBJS_DIR)/layers/test_layers.o \
	$(OBJS_DIR)/layers/activation.o \
	$(OBJS_DIR)/layers/layer_norm.o \
	$(OBJS_DIR)/layers/rotary_embedding.o \
	$(OBJS_DIR)/layers/sampler.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) -Wl,-rpath=../lib -Wl,-rpath=lib -lgtest
	$(STRIP) $@

$(TEST_DIR)/$(TEST_MODEL_UTILS_MODULE_NAME) : \
	$(OBJS_DIR)/utils/test_model_utils.o \
	$(OBJS_DIR)/utils/model_utils.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) -Wl,-rpath=../lib -Wl,-rpath=lib -lgtest
	$(STRIP) $@

$(TEST_DIR)/$(TEST_MODEL_QWEN3_MODULE_NAME) : \
	$(OBJS_DIR)/model/test_qwen3.o \
	$(OBJS_DIR)/model/qwen3.o \
	$(OBJS_DIR)/layers/layer_norm.o \
	$(OBJS_DIR)/utils/model_utils.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) -Wl,-rpath=../lib -Wl,-rpath=lib -lgtest
	$(STRIP) $@

$(OBJS_DIR)/search.pb.o: proto/search.proto
	$(THIRD_HOST_DIR)/bin/protoc --cpp_out=$(OBJS_DIR) --proto_path=$(CURRENT_DIR)/proto $(CURRENT_DIR)/proto/search.proto
	$(CC) $(CPPFLAGS) $(INCLUDES) -c $(OBJS_DIR)/search.pb.cc -o $(OBJS_DIR)/search.pb.o

$(TEST_DIR)/test_token : \
	$(OBJS_DIR)/token/test_token.o \
	$(OBJS_DIR)/token/core_bpe.o \
	$(OBJS_DIR)/token/tiktoken.o \
	$(OBJS_DIR)/token/qwen_token.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) $(WEB_LIBPATHS) $(WEB_LIBS) -Wl,-rpath=../lib -Wl,-rpath=lib -lgtest
	$(STRIP) $@

$(OBJS_DIR)/%.o:%.cpp
	@mkdir -p $(shell dirname $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $? -o $@
$(OBJS_DIR)/%.o:%.cc
	@mkdir -p $(shell dirname $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $? -o $@

clean:
	rm -f $(OBJS)
	rm -f $(OBJS_DIR)/search.pb.*
	rm -f $(OBJS_DIR)/main.o
	rm -f $(OBJS_DIR)/layers/test_layers.o
	rm -f $(OBJS_DIR)/token/test_token.o
	rm -f $(OBJS_DIR)/utils/test_model_utils.o
	rm -f $(OBJS_DIR)/model/test_qwen3.o
	rm -rf rpc_data

distclean:
	make clean
	rm -rf $(APP_DIR)
