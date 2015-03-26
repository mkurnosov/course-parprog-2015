prog := sched
prog_objs := sched.o

CC := icc
CFLAGS := -std=c99 -Wall -O2 -fopenmp
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

sched.o: sched.c

clean:
	@rm -rf *.o $(prog)
