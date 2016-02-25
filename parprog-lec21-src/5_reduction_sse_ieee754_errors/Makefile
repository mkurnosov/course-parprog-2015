prog := reduction
prog_objs := reduction.o

CC := gcc
CFLAGS := -Wall -O2 -std=c99
LDFLAGS := -lm

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

reduction.o: reduction.c

clean:
	@rm -rf *.o $(prog)
