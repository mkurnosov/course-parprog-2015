prog := quicksort
prog_objs := quicksort.o

CC := suncc
CFLAGS := -std=c99 -g -O2 -fopenmp
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

quicksort.o: quicksort.c

clean:
	@rm -rf *.o $(prog)
