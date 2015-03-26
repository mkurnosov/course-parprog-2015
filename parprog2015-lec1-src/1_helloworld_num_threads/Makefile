prog := hello
prog_objs := hello.o

CC := gcc
CFLAGS := -Wall -O2 -fopenmp
LDFLAGS := -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

hello.o: hello.c

clean:
	@rm -rf *.o $(prog)
