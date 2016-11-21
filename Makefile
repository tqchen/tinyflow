TORCH_PATH=${TORCH_HOME}

ROOTDIR = $(CURDIR)

ifndef CUDA_PATH
	CUDA_PATH = /usr/local/cuda
endif

ifndef NNVM_PATH
	NNVM_PATH = $(ROOTDIR)/nnvm
endif

export LDFLAGS = -pthread -lm
export CFLAGS =  -std=c++11 -Wall -O2 -msse2  -Wno-unknown-pragmas -funroll-loops\
	  -fPIC -Iinclude -Idmlc-core/include -I$(NNVM_PATH)/include

# whether use fusion
USE_FUSION = 0
ifeq ($(USE_FUSION), 1)
  ifndef NNVM_FUSION_PATH
  	NNVM_FUSION_PATH = $(NNVM_PATH)/plugin/nnvm-fusion/
  endif
  CFLAGS  += -DTINYFLOW_USE_FUSION=1 -I$(NNVM_FUSION_PATH)/include -I$(CUDA_PATH)/include
  LDFLAGS += -L$(CUDA_PATH)/lib64 -lcuda -lnvrtc -lcudart
endif

.PHONY: clean all test lint doc

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	WHOLE_ARCH= -all_load
	NO_WHOLE_ARCH= -noall_load
	CFLAGS  += -I$(TORCH_PATH)/install/include -I$(TORCH_PATH)/install/include/TH
	LDFLAGS += -L$(TORCH_PATH)/install/lib -llua -lluaT -lTH
else
	WHOLE_ARCH= --whole-archive
	NO_WHOLE_ARCH= --no-whole-archive
	CFLAGS  += -I$(TORCH_PATH)/install/include -I$(TORCH_PATH)/install/include/TH \
			   -I$(TORCH_PATH)/install/include/THC/
	LDFLAGS += -L$(TORCH_PATH)/install/lib -lluajit -lluaT -lTH -lTHC
endif

SRC = $(wildcard src/*.cc src/*/*.cc src/*/*/*.cc)
OBJ = $(patsubst %.cc, build/%.o, $(SRC))
CUSRC = $(wildcard src/*.cu src/*/*.cu src/*/*/*.cu)
CUOBJ = $(patsubst %.cu, build/%_gpu.o, $(CUSRC))

LIB_DEP = $(NNVM_PATH)/lib/libnnvm.a
ALL_DEP = $(OBJ) $(LIB_DEP)

all: lib/libtinyflow.so

build/src/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) -std=c++11 $(CFLAGS) -MM -MT build/src/$*.o $< >build/src/$*.d
	$(CXX) -std=c++11 -c $(CFLAGS) -c $< -o $@

build/src/%_gpu.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -M -MT build/src/$*_gpu.o $< >build/src/$*_gpu.d
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $<

lib/libtinyflow.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o, $^) \
	-Wl,${WHOLE_ARCH} $(filter %.a, $^) -Wl,${NO_WHOLE_ARCH} $(LDFLAGS)

$(NNVM_PATH)/lib/libnnvm.a:
	+ cd $(NNVM_PATH); make lib/libnnvm.a; cd $(ROOTDIR)

lint:
	python2 dmlc-core/scripts/lint.py tinyflow cpp include src

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
