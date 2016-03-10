prog := vec
prog_objs := vec.o

CC := gcc
CFLAGS := -Wall -std=c99 -O2 -march=native -ftree-vectorize -fopt-info-vec -fopt-info-vec-missed
#-fopt-info-vec-missed
LDFLAGS := 

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

vec.o: vec.c

clean:
	@rm -rf *.o $(prog)
