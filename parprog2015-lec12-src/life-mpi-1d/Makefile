life := life-mpi-1d
life_objs := life-mpi-1d.o

CC := mpicc
CFLAGS := -std=c99 -Wall -O2
LDFLAGS :=

.PHONY: all clean

all: $(life) $(mk_life_grid)

$(life): $(life_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

life-mpi-1d.o: life-mpi-1d.c

clean:
	@rm -rf *.o $(life)
