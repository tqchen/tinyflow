ROOTDIR = $(CURDIR)

ifndef CUDA_PATH
	CUDA_PATH = /usr/local/cuda
endif

ifndef NNVM_PATH
	NNVM_PATH = $(ROOTDIR)/../nnvm
endif

ifndef DMLC_CORE_PATH
	DMLC_CORE_PATH = $(ROOTDIR)/../dmlc-core
endif

export CFLAGS =  -std=c++11 -Wall -O2 -msse2  -Wno-unknown-pragmas -funroll-loops\
     -Iinclude -fPIC -I$(CUDA_PATH)/include -I$(DMLC_CORE_PATH)/include -I$(NNVM_PATH)/include

.PHONY: clean all

all: lib/libnnvm-rtc.a

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC))
ALL_DEP = $(ALL_OBJ)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libnnvm-rtc.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test

-include build/*.d
-include build/*/*.d
