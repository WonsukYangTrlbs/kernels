TARGET ?= matmul
SRC    ?= $(TARGET).cu
BINARIES := $(patsubst %.cu,%,$(wildcard *.cu))

compile:
	echo "Compiling $(TARGET)"
	nvcc -o $(TARGET) $(SRC)

run: compile
	./$(TARGET) $(ARGS)

profile: compile
	nsys profile -t cuda --stats=true -o $(TARGET)_profile ./$(TARGET) $(ARGS)

clean:
	rm -f $(BINARIES)
