CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc

INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart
OBJS = GaussianBlurFilter.cu.o main.cpp.o
TARGET = exec
IMAGE_MAGICK = -O2 -L/usr/X11R6/lib -lm -lpthread -lX11 
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA) $(IMAGE_MAGICK)

.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
    
%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
