prog := dgemm
prog_objs := dgemm.o

CC := gcc
CFLAGS := -std=c99 -g -Wall -O2 -fopenmp
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

dgemm.o: dgemm.c

clean:
	@rm -rf *.o $(prog)
