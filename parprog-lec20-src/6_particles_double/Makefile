prog := particles
prog_objs := particles.o

CC := gcc
CFLAGS := -std=c99 -Wall -O2 -msse
LDFLAGS := -lm

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

particles.o: particles.c

clean:
	@rm -rf *.o $(prog)
