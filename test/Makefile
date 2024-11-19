# Makefile

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++11

# Target executable
TARGET = test

# Source files
SOURCES = implementation.cu test.cpp

# Object files
OBJECTS = implementation.o test.o

# Default rule
all: $(TARGET)

# Compile implementation.cu to object file
implementation.o: implementation.cu
	$(NVCC) $(NVCC_FLAGS) -c implementation.cu -o implementation.o

# Compile test.cpp to object file
test.o: test.cpp
	$(NVCC) $(NVCC_FLAGS) -c test.cpp -o test.o

# Link object files to create the executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET) $(OBJECTS)
