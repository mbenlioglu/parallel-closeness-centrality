###############################################################
# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-8.0

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
	$(info WARNING - x86_64 variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=x86_64 instead)
	TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
	$(info WARNING - ARMv7 variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=armv7l instead)
	TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
	$(info WARNING - aarch64 variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=aarch64 instead)
	TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
	$(info WARNING - ppc64le variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=ppc64le instead)
	TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
	$(info WARNING - GCC variable has been deprecated)
	$(info WARNING - please use HOST_COMPILER=$(GCC) instead)
	HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
	$(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
	ifneq ($(TARGET_ARCH),$(HOST_ARCH))
		ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
			TARGET_SIZE := 64
		else ifneq (,$(filter $(TARGET_ARCH),armv7l))
			TARGET_SIZE := 32
		endif
	else
		TARGET_SIZE := $(shell getconf LONG_BIT)
	endif
else
	$(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
		$(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
	endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
	TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
	$(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
	ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
		HOST_COMPILER ?= clang++
	endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
		ifeq ($(TARGET_OS),linux)
			HOST_COMPILER ?= arm-linux-gnueabihf-g++
		else ifeq ($(TARGET_OS),qnx)
			ifeq ($(QNX_HOST),)
				$(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
			endif
			ifeq ($(QNX_TARGET),)
				$(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
			endif
			export QNX_HOST
			export QNX_TARGET
			HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
		else ifeq ($(TARGET_OS),android)
			HOST_COMPILER ?= arm-linux-androideabi-g++
		endif
	else ifeq ($(TARGET_ARCH),aarch64)
		ifeq ($(TARGET_OS), linux)
			HOST_COMPILER ?= aarch64-linux-gnu-g++
		else ifeq ($(TARGET_OS),qnx)
			ifeq ($(QNX_HOST),)
				$(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
			endif
			ifeq ($(QNX_TARGET),)
				$(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
			endif
			export QNX_HOST
			export QNX_TARGET
			HOST_COMPILER ?= $(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++
		else ifeq ($(TARGET_OS), android)
			HOST_COMPILER ?= aarch64-linux-android-g++
		endif
	else ifeq ($(TARGET_ARCH),ppc64le)
		HOST_COMPILER ?= powerpc64le-linux-gnu-g++
	endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE} -O3
CCFLAGS     := -O3
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
	LDFLAGS += -rpath $(CUDA_PATH)/lib
	CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
	LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
	CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
	LDFLAGS += -pie
	CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
		ifneq ($(TARGET_FS),)
			GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
			ifeq ($(GCCVERSIONLTEQ46),1)
				CCFLAGS += --sysroot=$(TARGET_FS)
			endif
			LDFLAGS += --sysroot=$(TARGET_FS)
			LDFLAGS += -rpath-link=$(TARGET_FS)/lib
			LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
			LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
		endif
	endif
endif

# Debug build flags
ifeq ($(dbg),1)
		NVCCFLAGS += -g -G
		BUILD_TYPE := debug
else
		BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  :=
LIBRARIES := -lcuda -lcudart -L/usr/local/cuda/lib64/

################################################################################

# Gencode arguments
SMS ?= 20 30 35 37 50 52 60

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

build: closenessCentrality

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
		@echo "Sample will be waived due to the above missing dependencies"
else
		@echo "Sample is ready - all dependencies have been met"
endif

closenessCentrality.o:./src/closeness_centrality.cu
		$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

graphio.o:./src/graphio.c
		$(EXEC) $(HOST_COMPILER) -o $@ -c $< -O3 -fpermissive

mmio.o:./src/mmio.c
		$(EXEC) $(HOST_COMPILER) -o $@ -c $< -O3 -fpermissive

main.o:./src/main.cpp
		$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -Xcompiler -std=c++14 $(GENCODE_FLAGS) -o $@ -c $+

closenessCentrality:closenessCentrality.o main.o graphio.o mmio.o
		$(EXEC) $(HOST_COMPILER) -o $@ $+ $(LIBRARIES) -fpermissive -O3 -std=c++14
		$(EXEC) mkdir -p ./bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
		$(EXEC) cp $@ ./bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

run: build
		$(EXEC) ./closenessCentrality 

clean:
		rm -f closenessCentrality closenessCentrality.o graphio.o mmio.o main.o
		rm -rf ./bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/closenessCentrality

clobber: clean




#coloring: ./src/graphio.c ./src/mmio.c ./src/closeness_centrality.cpp ./src/closeness_centrality.cpp
#	gcc ./src/graphio.c -c -O3
#	gcc ./src/mmio.c -c -O3
#	nvcc -O3 -c ./src/closeness_centrality.cu -Xcompiler -O3
#	g++ -o closenessCentrality ./src/closeness_centrality.cpp closeness_centrality.o mmio.o graphio.o -lcuda -L/usr/local/cuda/lib64/ -fpermissive -fopenmp -O3 -std=c++14
#clean:
#	rm closenessCentrality *.o
