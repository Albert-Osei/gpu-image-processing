COMPILER     = nvcc
IDIR         = ./
CFLAGS       = -I$(IDIR) -I/usr/local/cuda/include
LDFLAGS      = -lcudart -lm
STD_FLAG     = --std=c++17
SRC_DIR		 = src
BIN_DIR		 = bin
TARGET       = image_processing

# Extend with additional -gencode entries for other GPU generations.
ARCH = \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_86,code=compute_86

.PHONY: all build clean run generate_data

all: build

build: $(SRC_DIR)/$(TARGET).cu $(SRC_DIR)/$(TARGET).h
	$(COMPILER) $(CFLAGS) $(STD_FLAG) $(ARCH) $(LDFLAGS) \
	    -Wno-deprecated-gpu-targets \
	    $(SRC_DIR)/$(TARGET).cu -o $(BIN_DIR)/$(TARGET).exe

run: build generate_data
	./$(BIN_DIR)/$(TARGET).exe ./data/images ./output ./results.csv

generate_data:
	@mkdir -p data/images output
	python3 $(SRC_DIR)/generate_images.py 200 ./data/images

clean:
	rm -f $(BIN_DIR)/$(TARGET).exe
	rm -f results.csv
	rm -rf output