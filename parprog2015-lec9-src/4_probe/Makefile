prog := probe
prog_objs := probe.o

CC := mpicc
CFLAGS := -g -std=c99 -Wall -O2
LDFLAGS :=

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

probe.o: probe.c

clean:
	@rm -rf *.o $(prog)
