prog := integrate
prog_objs := integrate.o

CC := gcc
CFLAGS := -std=c99 -O2 -fopenmp -g
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

integrate.o: integrate.c

clean:
	@rm -rf *.o $(prog)
