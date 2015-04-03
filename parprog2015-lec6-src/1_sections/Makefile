prog := sections
prog_objs := sections.o

CC := gcc
CFLAGS := -std=c99 -g -Wall -O2 -fopenmp
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

sections.o: sections.c

clean:
	@rm -rf *.o $(prog)
