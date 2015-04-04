prog := sum
prog_objs := sum.o

CC := gcc
CFLAGS := -std=c99 -g -O2 -fopenmp
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

sum.o: sum.c

clean:
	@rm -rf *.o $(prog)
