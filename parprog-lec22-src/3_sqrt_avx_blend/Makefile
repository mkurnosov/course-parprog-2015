prog := sqrt
prog_objs := sqrt.o

CC := gcc
CFLAGS := -Wall -O2 -std=c99 -mavx
LDFLAGS := -lm

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

sqrt.o: sqrt.c

clean:
	@rm -rf *.o $(prog)
