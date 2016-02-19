prog := daxpy
prog_objs := daxpy.o

CC := gcc
CFLAGS := -Wall -O2
LDFLAGS := -lm

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

daxpy.o: daxpy.c

clean:
	@rm -rf *.o $(prog)
