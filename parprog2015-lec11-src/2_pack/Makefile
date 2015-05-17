prog := mpi_pack
prog_objs := mpi_pack.o

CC := mpicc
CFLAGS := -std=c99 -Wall -O2
LDFLAGS :=

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

mpi_pack.o: mpi_pack.c

clean:
	@rm -rf *.o $(prog)
