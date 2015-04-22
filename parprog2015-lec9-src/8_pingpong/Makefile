prog := pingpong
prog_objs := pingpong.o

CC := mpicc
CFLAGS := -g -std=c99 -Wall -O2
LDFLAGS :=

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

pingpong.o: pingpong.c

clean:
	@rm -rf *.o $(prog)
