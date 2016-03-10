prog := max
prog_objs := max.o

CC := gcc
CFLAGS := -Wall -O2 -std=c99
LDFLAGS := -lm

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

max.o: max.c

clean:
	@rm -rf *.o $(prog)
