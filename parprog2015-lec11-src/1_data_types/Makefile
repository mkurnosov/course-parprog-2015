prog := mpi_data_type
prog_objs := mpi_data_type.o

CC := mpicc
CFLAGS := -std=c99 -Wall -O2
LDFLAGS :=

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

mpi_data_type.o: mpi_data_type.c

clean:
	@rm -rf *.o $(prog)
