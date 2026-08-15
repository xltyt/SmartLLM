CURRENT_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIR := $(shell dirname $(CURRENT_PATH))

PLATFORM ?= amd64

BUILD_MODULE_NAME=mycommon
APP_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/lib
OBJS_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/build/$(BUILD_MODULE_NAME)
THIRD_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/local
TEST_DIR := $(CURRENT_DIR)/../../$(PLATFORM)/test

ifeq ($(PLATFORM),arm)
  CC=aarch64-linux-gnu-gcc
  CXX=aarch64-linux-gnu-g++
  AR=aarch64-linux-gnu-ar
  STRIP=aarch64-linux-gnu-strip
else
  STRIP=strip
endif

INCLUDES += \
  -I. \
  -I$(OBJS_DIR) \
  -I/usr/include/libxml2/ \
  -I$(THIRD_DIR)/include

#-lprotoc -lprotobuf
BASE_LIBS += \
  -L$(THIRD_DIR)/lib \
  -lglog -lgflags -lcurl \
  -lssl -lcrypto \
  -lz \
  -lm \
  -lrt \
  -ldl \
  -lpthread

LIBS += \
  -L. \
  $(THIRD_DIR)/../lib/libmycommon.a \
  -lboost_system -lboost_filesystem \
  -lgtest \
  $(BASE_LIBS)

#CFLAGS    += -O2 -g -std=c++11 -fstack-protector --param=ssp-buffer-size=4 -Wformat -Wformat-security -Wl,-z,relro,-z,now -Wl,--as-needed -Wl,--no-copy-dt-needed-entries -Wl,-z,noexecstatck -Wl,-z,noexecheap -static-libstdc++ -static-libgcc -w
CFLAGS    += -O2 -g -std=c++17 -static-libstdc++ -static-libgcc -Wno-narrowing
CFLAGS    += -DNDEBUG -DFORTIFY_SOURCE=2 -fPIC
LIBPATHS  +=
CPPFLAGS  = $(CFLAGS) -std=c++17
CXXFLAGS  = $(CFLAGS) -std=c++17

#etcd_utils.cpp
SOURCES += \
  system_utils.cpp \
  base64.cpp \
  url_utils.cpp \
  crypt_utils.cpp \
  timer.cpp \
  murmurhash2.cpp \
  hash_utils.cpp \
  file_utils.cpp \
  string_utils.cpp \
  net_utils.cpp

OBJS := $(addprefix $(OBJS_DIR)/,$(patsubst %.cpp,%.o,$(SOURCES)))
OBJS := $(patsubst %.c,%.o,$(OBJS))

all : prep $(APP_DIR)/lib$(BUILD_MODULE_NAME).a test

.PHONY: prep

prep:
	@mkdir -p $(OBJS_DIR)
	@mkdir -p $(APP_DIR)
	@mkdir -p $(TEST_DIR)

$(APP_DIR)/lib$(BUILD_MODULE_NAME).a : $(OBJS)
	$(AR) -r $@ $^

test: $(TEST_DIR)/test_common

$(TEST_DIR)/test_common : $(OBJS_DIR)/test_common.o $(APP_DIR)/lib$(BUILD_MODULE_NAME).a
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(WEB_LIBPATHS) $(WEB_LIBS) $(LIBS) -lgtest
	$(STRIP) $@

$(OBJS_DIR)/%.o:%.cpp
	@mkdir -p $(shell dirname $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $? -o $@
$(OBJS_DIR)/%.o:%.cc
	@mkdir -p $(shell dirname $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $? -o $@
$(OBJS_DIR)/%.o:%.c
	@mkdir -p $(shell dirname $@)
	$(CC) $(CFLAGS) $(INCLUDES) $(WEB_CFLAGS) -c $? -o $@

clean:
	rm -f $(OBJS)
	rm -f $(OBJS_DIR)/test_common.o
	rm -rf rpc_data

distclean:
	make clean
	rm -rf $(APP_DIR)
