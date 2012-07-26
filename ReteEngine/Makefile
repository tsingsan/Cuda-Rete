################################################################################
#
# Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OSX and Linux Platforms)
#
################################################################################

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_FLAGS   := $(GENCODE_SM20)

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH)64 -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      TARGET    := release
endif


# Common includes and paths for CUDA
INCLUDES      := -I. -I.. -I../../common/inc -I$(CUDA_INC_PATH) -I../../../shared/inc

# Target rules
all: build

build: libReteEngine.so

#xmlutils
RDFAdaptor.o: RDFAdaptor.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

tinystr.o: tinystr.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

tinyxml.o: tinyxml.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

tinyxmlerror.o: tinyxmlerror.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

tinyxmlparser.o: tinyxmlparser.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

#ReteNode
ClauseFilter.o: ClauseFilter.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

ReteQueue.o: ReteQueue.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

ReteTerminal.o: ReteTerminal.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

#Triple
NodeAny.o: NodeAny.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

NodeFactory.o: NodeFactory.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

TripleContainer.o: TripleContainer.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

TriplePattern.o: TriplePattern.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

#Engine
ReteEngine.o: ReteEngine.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

ReteRuntime.o: ReteRuntime.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

#RuleReason
Compiler.o: Compiler.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

Rule.o: Rule.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

RuleParser.o: RuleParser.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

RuleReason.o: RuleReason.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

#Utils
common.o: common.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

ctrans.o: ctrans.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

GlobalMemPool.o: GlobalMemPool.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

String.o: String.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c $<

Kernel.o: Kernel.cu
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

libReteEngine.so: RDFAdaptor.o tinystr.o tinyxml.o tinyxmlerror.o tinyxmlparser.o ClauseFilter.o ReteQueue.o ReteTerminal.o NodeAny.o NodeFactory.o TripleContainer.o TriplePattern.o ReteEngine.o ReteRuntime.o Compiler.o Rule.o RuleParser.o RuleReason.o common.o ctrans.o GlobalMemPool.o String.o Kernel.o
	$(GCC) -shared -fPCI $(CCFLAGS) -o $@ $+ $(LDFLAGS) $(EXTRA_LDFLAGS)

clean:
	rm -f libReteEngine.so *.o
