prog := matvec
prog_objs := matvec.o

CC := gcc
CFLAGS := -std=c99 -g -Wall -O2 -fopenmp
LDFLAGS := -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

matvec.o: matvec.c

clean:
	@rm -rf *.o $(prog)
