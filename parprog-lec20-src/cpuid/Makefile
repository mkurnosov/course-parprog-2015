prog := cpuid
prog_objs := cpuid.o

CC := gcc
CFLAGS := -Wall
LDFLAGS :=

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

cpuid.o: cpuid.c

clean:
	@rm -rf *.o $(prog)
