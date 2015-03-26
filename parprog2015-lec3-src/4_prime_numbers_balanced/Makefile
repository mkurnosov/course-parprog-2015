prog := primes
prog_objs := primes.o

CC := gcc
CFLAGS := -std=c99 -Wall -O2 -fopenmp 
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

primes.o: primes.c

clean:
	@rm -rf *.o $(prog)
